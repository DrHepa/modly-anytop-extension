from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from anytop_modly import assets, constants, dependencies, state
from anytop_modly.constants import EXTENSION_ID, REVISION_ID, AssetSpec
from anytop_modly.paths import (
    PathContractError,
    owned_snapshot_directory,
    resolve_models_root,
    safe_snapshot_file,
    snapshot_paths,
)
from anytop_modly.state import StateError, read_runtime_config, write_runtime_config


def select_plan(payload: dict[str, object], abi: str = "cp311") -> dependencies.DependencyPlan:
    return dependencies.select_dependency_plan({**payload, "python_abi": abi})


class Response(io.BytesIO):
    def __init__(self, payload: bytes, status: int, headers: dict[str, str]) -> None:
        super().__init__(payload)
        self.status = status
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _no_api(*_args, **_kwargs):
    raise OSError("offline")


def test_models_dir_resolution_order_and_owned_layout(tmp_path: Path) -> None:
    models = tmp_path / "configured-models"
    models.mkdir()
    extension = tmp_path / "modly" / "extensions" / EXTENSION_ID
    extension.mkdir(parents=True)
    assert resolve_models_root(
        {"models_dir": str(models)}, extension, environ={}, opener=_no_api
    ) == models
    env_models = tmp_path / "env-models"
    env_models.mkdir()
    assert resolve_models_root(
        {},
        extension,
        environ={"MODLY_MODELS_DIR": str(env_models)},
        opener=_no_api,
    ) == env_models
    revision = owned_snapshot_directory(models, create=True)
    assert revision == models / EXTENSION_ID / "anytop" / "revisions" / REVISION_ID
    paths = snapshot_paths(revision)
    assert paths.anytop_source == revision / "source" / "AnyTop"
    assert paths.motion_source == revision / "source" / "Motion"
    assert paths.t5 == revision / "t5-base"


def test_payload_alias_conflict_fails_closed(tmp_path: Path) -> None:
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()
    with pytest.raises(PathContractError, match="PATH_MODELS_CONFLICT"):
        resolve_models_root(
            {"models_dir": str(first), "modelsDir": str(second)},
            tmp_path,
            environ={},
            opener=_no_api,
        )


def test_api_precedes_generic_models_dir(tmp_path: Path) -> None:
    api_models = tmp_path / "api"
    shell_models = tmp_path / "shell"
    api_models.mkdir()
    shell_models.mkdir()

    def opener(_request, timeout):
        assert timeout > 0
        body = json.dumps({"models_dir": str(api_models)}).encode()
        return Response(body, 200, {"Content-Length": str(len(body))})

    assert resolve_models_root(
        {}, tmp_path, environ={"MODELS_DIR": str(shell_models)}, opener=opener
    ) == api_models


def test_safe_snapshot_file_rejects_escape_and_alias(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    with pytest.raises(PathContractError, match="PATH_RELATIVE_INVALID"):
        safe_snapshot_file(snapshot, "../escape", create_parent=True)
    target = snapshot / "target"
    target.write_text("x")
    alias = snapshot / "alias"
    try:
        alias.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(PathContractError, match="PATH_ASSET_INVALID"):
        safe_snapshot_file(snapshot, "alias", create_parent=False)


def test_runtime_config_roundtrip_is_atomic_and_secret_free(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    revision = owned_snapshot_directory(models, create=True)
    extension = tmp_path / "extension"
    extension.mkdir()
    config_path = write_runtime_config(
        extension,
        models,
        revision,
        extra={
            "extension_id": EXTENSION_ID,
            "source_root": str(revision / "source"),
            "python_abi": state.current_python_abi(),
        },
    )
    parsed = read_runtime_config(extension)
    assert config_path.name == "runtime_config.json"
    assert parsed.models_dir == models.resolve()
    assert parsed.revision_root == revision.resolve()
    assert parsed.payload["extension_id"] == EXTENSION_ID
    assert not list(extension.glob("*.tmp"))
    with pytest.raises(StateError, match="STATE_SECRET_REJECTED"):
        write_runtime_config(extension, models, revision, extra={"hf_token": "do-not-store"})


def test_runtime_config_rejects_stale_python_abi(tmp_path: Path, monkeypatch) -> None:
    models = tmp_path / "models"
    models.mkdir()
    revision = owned_snapshot_directory(models, create=True)
    extension = tmp_path / "extension"
    extension.mkdir()
    write_runtime_config(
        extension,
        models,
        revision,
        extra={"python_abi": "cp311"},
    )
    monkeypatch.setattr(state, "current_python_abi", lambda: "cp312")
    with pytest.raises(StateError, match="STATE_PYTHON_ABI_MISMATCH"):
        read_runtime_config(extension)


@pytest.mark.parametrize("abi", ("cp311", "cp312"))
def test_runtime_config_accepts_matching_supported_python_abi(
    tmp_path: Path, monkeypatch, abi: str
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    revision = owned_snapshot_directory(models, create=True)
    extension = tmp_path / "extension"
    extension.mkdir()
    write_runtime_config(extension, models, revision, extra={"python_abi": abi})
    monkeypatch.setattr(state, "current_python_abi", lambda: abi)
    parsed = read_runtime_config(extension)
    assert parsed.payload["python_abi"] == abi


def test_resumable_download_hashes_and_promotes(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    content = b"pinned-anytop-asset"
    spec = AssetSpec(
        "weights/test.bin",
        "https://example.invalid/test.bin",
        len(content),
        hashlib.sha256(content).hexdigest(),
        "test asset",
    )
    destination = safe_snapshot_file(snapshot, spec.relative_path, create_parent=True)
    part = destination.with_name(destination.name + ".part")
    prefix = content[:7]
    part.write_bytes(prefix)
    calls: list[str] = []

    def opener(request, timeout):
        assert timeout > 0
        calls.append(request.headers.get("Range", ""))
        body = content[len(prefix) :]
        return Response(
            body,
            206,
            {
                "Content-Length": str(len(body)),
                "Content-Range": f"bytes {len(prefix)}-{len(content) - 1}/{len(content)}",
            },
        )

    assert assets._ensure_asset(snapshot, spec, opener=opener, log=lambda _m: None, timeout=1)
    assert destination.read_bytes() == content
    assert not part.exists()
    assert calls == [f"bytes={len(prefix)}-"]
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("valid asset must not access network")

    assert not assets._ensure_asset(
        snapshot, spec, opener=forbidden, log=lambda _m: None, timeout=1
    )
    assert called is False


def _write_tar(path: Path, root: str, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for relative, content in members.items():
            info = tarfile.TarInfo(f"{root}/{relative}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def _release_digest(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for relative, content in sorted(files.items()):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).digest())
        digest.update(b"\0")
    return digest.hexdigest()


def test_source_extraction_applies_exact_matplotlib_patch(tmp_path: Path) -> None:
    old = (
        b".grid(b=None)\nax = p3.Axes3D(fig)\n" * 5
    )
    new = old.replace(b".grid(b=None)", b".grid(visible=None)").replace(
        b"ax = p3.Axes3D(fig)", b"ax = fig.add_subplot(111, projection=\"3d\")"
    )
    relative = "data_loaders/truebones/truebones_utils/plot_script.py"
    archive = tmp_path / "source.tar.gz"
    _write_tar(archive, "PinnedRoot", {relative: old})
    source_parent = tmp_path / "source"
    source_parent.mkdir()
    destination = source_parent / "AnyTop"
    assets._extract_source(
        archive,
        destination,
        kind="anytop",
        archive_root="PinnedRoot",
        expected_digest=_release_digest({relative: new}),
        expected_count=1,
        expected_bytes=len(new),
    )
    assert (destination / relative).read_bytes() == new


def test_source_archive_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("PinnedRoot/../sample/generate.py")
        info.size = 1
        handle.addfile(info, io.BytesIO(b"x"))
    source_parent = tmp_path / "source"
    source_parent.mkdir()
    with pytest.raises(assets.AssetError, match="SOURCE_ARCHIVE_PATH"):
        assets._extract_source(
            archive,
            source_parent / "AnyTop",
            kind="anytop",
            archive_root="PinnedRoot",
            expected_digest="0" * 64,
            expected_count=0,
            expected_bytes=0,
        )


def test_interrupted_source_backup_is_restored(tmp_path: Path) -> None:
    parent = tmp_path / "source"
    parent.mkdir()
    destination = parent / "AnyTop"
    backup = parent / ".AnyTop.backup.interrupted"
    backup.mkdir()
    (backup / "LICENSE").write_bytes(b"old-verified-tree")
    assets._recover_source_transaction(destination)
    assert (destination / "LICENSE").read_bytes() == b"old-verified-tree"
    assert not backup.exists()


def test_ready_snapshot_reuse_never_calls_network(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / assets.READY_MARKER_FILENAME).write_text(
        json.dumps(assets.ready_payload()), encoding="utf-8"
    )
    monkeypatch.setattr(assets, "verify_snapshot", lambda *_a, **_k: [])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("network must not be called for a ready revision")

    assert assets.ensure_snapshot(tmp_path, opener=forbidden, log=lambda _m: None) == tmp_path


def test_revision_identity_covers_inventory_but_not_wrapper_version(monkeypatch) -> None:
    original = constants.ASSETS[0]
    changed = AssetSpec(
        original.relative_path,
        original.url,
        original.size,
        "0" * 64,
        original.role,
    )
    assert constants._asset_revision_digest((changed, *constants.ASSETS[1:])) != (
        constants.ASSET_REVISION_DIGEST
    )
    assert constants._asset_revision_digest(tuple(reversed(constants.ASSETS))) == (
        constants.ASSET_REVISION_DIGEST
    )
    assert constants.REVISION_ID.endswith(constants.ASSET_REVISION_DIGEST[:24])

    marker_before = assets.ready_payload()
    plan = select_plan(
        {"platform": "linux", "arch": "x64", "accelerator": "cpu", "gpu_sm": 0}
    )
    state_before = dependencies.dependency_state_payload(plan, {"version": [3, 11]})
    monkeypatch.setattr(constants, "EXTENSION_VERSION", "99.0.0")
    monkeypatch.setattr(assets, "EXTENSION_VERSION", "99.0.0", raising=False)
    monkeypatch.setattr(dependencies, "EXTENSION_VERSION", "99.0.0", raising=False)
    assert assets.ready_payload() == marker_before
    assert dependencies.dependency_state_payload(plan, {"version": [3, 11]}) == state_before
    assert "extension_version" not in marker_before
    assert "extension_version" not in state_before


def test_incompatible_published_revision_is_never_relabelled_or_downloaded(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "AnyTop" / "sentinel.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"old-wrapper-source")
    marker = tmp_path / assets.READY_MARKER_FILENAME
    marker.write_bytes(b'{"schema_version":0}\n')
    before_marker = marker.read_bytes()
    before_source = source.read_bytes()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a published revision must not access the network")

    with pytest.raises(assets.AssetError, match="ASSET_REVISION_CONFLICT"):
        assets.ensure_snapshot(tmp_path, opener=forbidden, log=lambda _m: None)
    assert marker.read_bytes() == before_marker
    assert source.read_bytes() == before_source


def test_corrupt_published_revision_repairs_only_invalid_files_and_keeps_marker(
    tmp_path: Path, monkeypatch
) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    good_content = b"already-valid"
    bad_content = b"restored-content"
    good = AssetSpec(
        "weights/good.bin",
        "https://example.invalid/good.bin",
        len(good_content),
        hashlib.sha256(good_content).hexdigest(),
        "valid test weight",
    )
    bad = AssetSpec(
        "weights/bad.bin",
        "https://example.invalid/bad.bin",
        len(bad_content),
        hashlib.sha256(bad_content).hexdigest(),
        "corrupt test weight",
    )
    monkeypatch.setattr(assets, "ASSETS", (good, bad))
    good_path = weights / "good.bin"
    bad_path = weights / "bad.bin"
    good_path.write_bytes(good_content)
    bad_path.write_bytes(b"corrupt")
    good_identity = good_path.stat().st_ino
    marker = tmp_path / assets.READY_MARKER_FILENAME
    marker.write_text(json.dumps(assets.ready_payload()), encoding="utf-8")
    before_marker = marker.read_bytes()

    def lightweight_verify(snapshot: Path, *, require_ready: bool = True):
        failures = []
        for spec in assets.ASSETS:
            valid, reason = assets.verify_asset(snapshot / spec.relative_path, spec)
            if not valid:
                failures.append(f"{spec.relative_path}: {reason}")
        if require_ready and not assets._read_ready_marker(snapshot)[0]:
            failures.append("ready.json: invalid")
        return failures

    monkeypatch.setattr(assets, "verify_snapshot", lightweight_verify)
    monkeypatch.setattr(assets, "_ensure_sources", lambda *_a, **_k: False)
    requests: list[str] = []

    def opener(request, timeout):
        assert timeout > 0
        requests.append(request.full_url)
        assert request.full_url == bad.url
        return Response(
            bad_content,
            200,
            {"Content-Length": str(len(bad_content))},
        )

    assets.ensure_snapshot(tmp_path, opener=opener, log=lambda _m: None)
    assert marker.read_bytes() == before_marker
    assert good_path.read_bytes() == good_content
    assert good_path.stat().st_ino == good_identity
    assert bad_path.read_bytes() == bad_content
    assert requests == [bad.url]


def test_failed_repair_never_deletes_a_published_marker(tmp_path: Path, monkeypatch) -> None:
    marker = tmp_path / assets.READY_MARKER_FILENAME
    marker.write_text(json.dumps(assets.ready_payload()), encoding="utf-8")
    before = marker.read_bytes()
    results = iter(
        [
            ["AnyTop source: corrupt before repair"],
            [],
            ["AnyTop source: changed during final verification"],
        ]
    )
    monkeypatch.setattr(assets, "verify_snapshot", lambda *_a, **_k: next(results))
    monkeypatch.setattr(assets, "_ensure_asset", lambda *_a, **_k: False)
    monkeypatch.setattr(assets, "_ensure_sources", lambda *_a, **_k: False)

    with pytest.raises(assets.AssetError, match="ASSET_READY_FAILED"):
        assets.ensure_snapshot(tmp_path, log=lambda _m: None)
    assert marker.read_bytes() == before

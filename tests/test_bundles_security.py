from __future__ import annotations

import ctypes
import hashlib
import hmac
import json
import os
from pathlib import Path
from unittest import mock

import pytest

from anytop_modly import bundles


AUTH_KEY = b"A" * bundles.BUNDLE_AUTH_KEY_BYTES


def _files(root: Path) -> dict[str, Path]:
    values = {
        "preview": root / "motion.glb",
        "motion": root / "motion-00.npy",
        "bvh": root / "motion-00.bvh",
        "video": root / "motion-00.mp4",
        "condition": root / "condition.npy",
    }
    for name, path in values.items():
        path.write_bytes(f"trusted-{name}".encode("ascii"))
    return values


def _write_valid(root: Path, key: bytes = AUTH_KEY) -> tuple[Path, dict[str, Path]]:
    values = _files(root)
    bundles.write_bundle(
        preview=values["preview"],
        operation="generate-custom",
        object_name="Custom-Rig",
        condition_kind="custom",
        condition=values["condition"],
        files={name: values[name] for name in ("motion", "bvh", "video")},
        parameters={"seed": 10},
        provenance={"modelFamily": "unified"},
        authentication_key=key,
    )
    return values["preview"], values


def _manifest(preview: Path) -> dict[str, object]:
    return json.loads(preview.with_suffix(bundles.MANIFEST_SUFFIX).read_text(encoding="utf-8"))


def _resign(raw: dict[str, object], key: bytes = AUTH_KEY) -> None:
    raw.pop("authentication", None)
    raw["authentication"] = {
        "algorithm": bundles.AUTHENTICATION_ALGORITHM,
        "tag": hmac.new(key, bundles._canonical_json(raw), hashlib.sha256).hexdigest(),
    }


def _save(preview: Path, raw: dict[str, object]) -> None:
    bundles.atomic_json(preview.with_suffix(bundles.MANIFEST_SUFFIX), raw)


def test_per_installation_key_is_private_persistent_and_not_rotated(tmp_path: Path) -> None:
    cache = tmp_path / "runtime-cache"
    cache.mkdir()
    path = bundles.ensure_bundle_auth_key(cache)
    first = bundles.load_bundle_auth_key(cache)
    assert len(first) == bundles.BUNDLE_AUTH_KEY_BYTES
    assert bundles.ensure_bundle_auth_key(cache) == path
    assert bundles.load_bundle_auth_key(cache) == first
    if os.name != "nt":
        assert path.stat().st_mode & 0o077 == 0


def test_unsafe_key_permissions_are_rejected(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("POSIX mode bits are not Windows ACLs")
    cache = tmp_path / "runtime-cache"
    cache.mkdir()
    path = bundles.ensure_bundle_auth_key(cache)
    path.chmod(0o644)
    with pytest.raises(bundles.BundleError, match="unsafe"):
        bundles.load_bundle_auth_key(cache)


def _fake_dpapi_protect(value: bytes) -> bytes:
    digest = hashlib.sha256(b"test-dpapi-user" + value).digest()
    # The LF is deliberate: a Windows descriptor accidentally opened in text
    # mode expands it to CRLF and corrupts the length-prefixed envelope.
    return b"DPAPI\n" + value[::-1] + digest


def _fake_dpapi_unprotect(value: bytes) -> bytes:
    if len(value) != 6 + bundles.BUNDLE_AUTH_KEY_BYTES + 32 or not value.startswith(b"DPAPI\n"):
        raise bundles.BundleError("mock DPAPI rejected ciphertext")
    plaintext = value[6 : 6 + bundles.BUNDLE_AUTH_KEY_BYTES][::-1]
    expected = hashlib.sha256(b"test-dpapi-user" + plaintext).digest()
    if not hmac.compare_digest(value[-32:], expected):
        raise bundles.BundleError("mock DPAPI rejected ciphertext")
    return plaintext


def _windows_dpapi_mocks():
    return (
        mock.patch.object(bundles, "_uses_windows_key_protection", return_value=True),
        mock.patch.object(bundles, "_dpapi_protect", side_effect=_fake_dpapi_protect),
        mock.patch.object(bundles, "_dpapi_unprotect", side_effect=_fake_dpapi_unprotect),
    )


def test_windows_key_uses_versioned_current_user_dpapi_envelope(tmp_path: Path) -> None:
    cache = tmp_path / "runtime-cache"
    cache.mkdir()
    windows, protect, unprotect = _windows_dpapi_mocks()
    with windows, protect as protect_mock, unprotect as unprotect_mock:
        path = bundles.ensure_bundle_auth_key(cache)
        first = bundles.load_bundle_auth_key(cache)
        assert bundles.ensure_bundle_auth_key(cache) == path
        assert bundles.load_bundle_auth_key(cache) == first

    stored = path.read_bytes()
    assert len(first) == bundles.BUNDLE_AUTH_KEY_BYTES
    assert stored.startswith(bundles.BUNDLE_KEY_ENVELOPE_MAGIC)
    assert stored != first
    assert first not in stored
    assert protect_mock.call_count == 1
    assert unprotect_mock.call_count == 4


def test_windows_rejects_legacy_raw_key_without_replacing_it(tmp_path: Path) -> None:
    cache = tmp_path / "runtime-cache"
    cache.mkdir()
    path = cache / bundles.BUNDLE_AUTH_KEY_FILENAME
    legacy = b"L" * bundles.BUNDLE_AUTH_KEY_BYTES
    path.write_bytes(legacy)
    windows, protect, unprotect = _windows_dpapi_mocks()
    with windows, protect as protect_mock, unprotect as unprotect_mock:
        with pytest.raises(bundles.BundleError, match="unprotected"):
            bundles.load_bundle_auth_key(cache)
        with pytest.raises(bundles.BundleError, match="unprotected"):
            bundles.ensure_bundle_auth_key(cache)
    assert path.read_bytes() == legacy
    protect_mock.assert_not_called()
    unprotect_mock.assert_not_called()


@pytest.mark.parametrize(
    ("version", "provider", "reserved", "declared_delta"),
    [
        (2, bundles.BUNDLE_KEY_PROVIDER_DPAPI_CURRENT_USER, 0, 0),
        (bundles.BUNDLE_KEY_ENVELOPE_VERSION, 99, 0, 0),
        (bundles.BUNDLE_KEY_ENVELOPE_VERSION, bundles.BUNDLE_KEY_PROVIDER_DPAPI_CURRENT_USER, 1, 0),
        (bundles.BUNDLE_KEY_ENVELOPE_VERSION, bundles.BUNDLE_KEY_PROVIDER_DPAPI_CURRENT_USER, 0, 1),
    ],
)
def test_windows_rejects_unknown_or_malformed_envelope_before_dpapi(
    tmp_path: Path,
    version: int,
    provider: int,
    reserved: int,
    declared_delta: int,
) -> None:
    cache = tmp_path / "runtime-cache"
    cache.mkdir()
    protected = b"ciphertext"
    envelope = bundles.BUNDLE_KEY_ENVELOPE_HEADER.pack(
        bundles.BUNDLE_KEY_ENVELOPE_MAGIC,
        version,
        provider,
        reserved,
        len(protected) + declared_delta,
    ) + protected
    path = cache / bundles.BUNDLE_AUTH_KEY_FILENAME
    path.write_bytes(envelope)
    with mock.patch.object(
        bundles, "_uses_windows_key_protection", return_value=True
    ), mock.patch.object(bundles, "_dpapi_unprotect") as unprotect:
        with pytest.raises(bundles.BundleError, match="envelope"):
            bundles.load_bundle_auth_key(cache)
    unprotect.assert_not_called()


def test_windows_detects_dpapi_ciphertext_tamper_without_replacement(tmp_path: Path) -> None:
    cache = tmp_path / "runtime-cache"
    cache.mkdir()
    windows, protect, unprotect = _windows_dpapi_mocks()
    with windows, protect, unprotect:
        path = bundles.ensure_bundle_auth_key(cache)
    damaged = bytearray(path.read_bytes())
    damaged[-1] ^= 0x01
    path.write_bytes(damaged)
    before = path.read_bytes()

    windows, protect, unprotect = _windows_dpapi_mocks()
    with windows, protect as protect_mock, unprotect:
        with pytest.raises(bundles.BundleError, match="could not be unlocked"):
            bundles.load_bundle_auth_key(cache)
        with pytest.raises(bundles.BundleError, match="could not be unlocked"):
            bundles.ensure_bundle_auth_key(cache)
    assert path.read_bytes() == before
    protect_mock.assert_not_called()


def test_windows_dpapi_protect_failure_leaves_no_key_or_temporary(tmp_path: Path) -> None:
    cache = tmp_path / "runtime-cache"
    cache.mkdir()
    with mock.patch.object(
        bundles, "_uses_windows_key_protection", return_value=True
    ), mock.patch.object(
        bundles,
        "_dpapi_protect",
        side_effect=bundles.BundleError("DPAPI unavailable"),
    ):
        with pytest.raises(bundles.BundleError, match="DPAPI unavailable"):
            bundles.ensure_bundle_auth_key(cache)
    assert not (cache / bundles.BUNDLE_AUTH_KEY_FILENAME).exists()
    assert not list(cache.iterdir())


def test_dpapi_native_calls_are_current_user_and_ui_forbidden() -> None:
    buffers: list[object] = []
    calls: list[tuple[str, bytes, bytes, int]] = []
    frees: list[object] = []

    class NativeFunction:
        def __init__(self, callback):
            self.callback = callback
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.callback(*args)

    def blob_value(pointer) -> bytes:
        blob = ctypes.cast(pointer, ctypes.POINTER(bundles._DataBlob)).contents
        return ctypes.string_at(blob.pbData, blob.cbData)

    def output_value(pointer, value: bytes) -> None:
        backing = (ctypes.c_ubyte * len(value)).from_buffer_copy(value)
        buffers.append(backing)
        blob = ctypes.cast(pointer, ctypes.POINTER(bundles._DataBlob)).contents
        blob.cbData = len(value)
        blob.pbData = ctypes.cast(backing, ctypes.POINTER(ctypes.c_ubyte))

    def protect(input_pointer, _description, entropy_pointer, _reserved, _prompt, flags, output):
        calls.append(("protect", blob_value(input_pointer), blob_value(entropy_pointer), int(flags)))
        output_value(output, b"native-ciphertext")
        return 1

    def unprotect(input_pointer, _description, entropy_pointer, _reserved, _prompt, flags, output):
        calls.append(("unprotect", blob_value(input_pointer), blob_value(entropy_pointer), int(flags)))
        output_value(output, b"U" * bundles.BUNDLE_AUTH_KEY_BYTES)
        return 1

    crypt32 = type(
        "Crypt32",
        (),
        {
            "CryptProtectData": NativeFunction(protect),
            "CryptUnprotectData": NativeFunction(unprotect),
        },
    )()
    kernel32 = type(
        "Kernel32",
        (),
        {"LocalFree": NativeFunction(lambda pointer: frees.append(pointer) or None)},
    )()

    def library(name: str):
        return crypt32 if name.casefold() == "crypt32.dll" else kernel32

    with mock.patch.object(bundles, "_windows_library", side_effect=library):
        assert bundles._dpapi_protect(b"plaintext") == b"native-ciphertext"
        assert bundles._dpapi_unprotect(b"native-ciphertext") == b"U" * bundles.BUNDLE_AUTH_KEY_BYTES

    assert [call[0] for call in calls] == ["protect", "unprotect"]
    assert calls[0][1] == b"plaintext"
    assert calls[1][1] == b"native-ciphertext"
    assert calls[0][2] == calls[1][2] == bundles._dpapi_entropy()
    assert all(call[3] == bundles.CRYPTPROTECT_UI_FORBIDDEN for call in calls)
    assert all(call[3] & 0x4 == 0 for call in calls)  # CRYPTPROTECT_LOCAL_MACHINE is absent.
    assert len(frees) == 2


def test_forged_pickled_condition_and_updated_sha256_are_rejected_before_hashing(
    tmp_path: Path,
) -> None:
    preview, values = _write_valid(tmp_path)
    values["condition"].write_bytes(b"malicious-pickle-payload")
    raw = _manifest(preview)
    condition = raw["condition"]
    assert isinstance(condition, dict)
    record = condition["file"]
    assert isinstance(record, dict)
    record["size"] = values["condition"].stat().st_size
    record["sha256"] = hashlib.sha256(values["condition"].read_bytes()).hexdigest()
    _save(preview, raw)

    with mock.patch.object(bundles, "sha256_file", side_effect=AssertionError("must not hash")):
        with pytest.raises(bundles.BundleError, match="authentication failed"):
            bundles.verify_bundle(preview, authentication_key=AUTH_KEY)


def test_bundle_from_another_installation_is_rejected(tmp_path: Path) -> None:
    preview, _ = _write_valid(tmp_path)
    with pytest.raises(bundles.BundleError, match="authentication failed"):
        bundles.verify_bundle(preview, authentication_key=b"B" * bundles.BUNDLE_AUTH_KEY_BYTES)


def test_object_name_traversal_is_rejected_on_write_and_verify(tmp_path: Path) -> None:
    values = _files(tmp_path)
    with pytest.raises(bundles.BundleError, match="object name"):
        bundles.write_bundle(
            preview=values["preview"],
            operation="generate-custom",
            object_name="../../escape",
            condition_kind="custom",
            condition=values["condition"],
            files={name: values[name] for name in ("motion", "bvh", "video")},
            parameters={},
            provenance={},
            authentication_key=AUTH_KEY,
        )

    preview, _ = _write_valid(tmp_path)
    raw = _manifest(preview)
    raw["objectName"] = "../../escape"
    _resign(raw)
    _save(preview, raw)
    with pytest.raises(bundles.BundleError, match="object name"):
        bundles.verify_bundle(preview, authentication_key=AUTH_KEY)


def test_operation_and_file_roles_are_allowlisted(tmp_path: Path) -> None:
    values = _files(tmp_path)
    with pytest.raises(bundles.BundleError, match="operation"):
        bundles.write_bundle(
            preview=values["preview"],
            operation="execute",
            object_name="Custom",
            condition_kind="custom",
            condition=values["condition"],
            files={name: values[name] for name in ("motion", "bvh", "video")},
            parameters={},
            provenance={},
            authentication_key=AUTH_KEY,
        )
    with pytest.raises(bundles.BundleError, match="roles|role"):
        bundles.write_bundle(
            preview=values["preview"],
            operation="generate-custom",
            object_name="Custom",
            condition_kind="custom",
            condition=values["condition"],
            files={
                "motion": values["motion"],
                "bvh": values["bvh"],
                "video": values["video"],
                "payload": values["motion"],
            },
            parameters={},
            provenance={},
            authentication_key=AUTH_KEY,
        )


def test_item_count_limit_rejects_authenticated_manifest_before_file_io(tmp_path: Path) -> None:
    preview, _ = _write_valid(tmp_path)
    raw = _manifest(preview)
    file_values = raw["files"]
    assert isinstance(file_values, dict)
    template = file_values["motion"]
    assert isinstance(template, dict)
    for index in range(1, bundles.MAX_BUNDLE_ITEMS["generate-custom"] + 1):
        suffix = f"_{index:02d}"
        for family in ("motion", "bvh", "video"):
            record = dict(template)
            record["role"] = family + suffix
            record["path"] = f"missing-{family}-{index:02d}.bin"
            file_values[family + suffix] = record
    _resign(raw)
    _save(preview, raw)

    with mock.patch.object(bundles, "sha256_file", side_effect=AssertionError("must not hash")):
        with pytest.raises(bundles.BundleError, match="incomplete"):
            bundles.verify_bundle(preview, authentication_key=AUTH_KEY)


def test_declared_size_limit_rejects_before_hashing(tmp_path: Path) -> None:
    preview, _ = _write_valid(tmp_path)
    raw = _manifest(preview)
    file_values = raw["files"]
    assert isinstance(file_values, dict)
    motion = file_values["motion"]
    assert isinstance(motion, dict)
    motion["size"] = bundles.MAX_FILE_BYTES["motion"] + 1
    _resign(raw)
    _save(preview, raw)

    with mock.patch.object(bundles, "sha256_file", side_effect=AssertionError("must not hash")):
        with pytest.raises(bundles.BundleError, match="size"):
            bundles.verify_bundle(preview, authentication_key=AUTH_KEY)

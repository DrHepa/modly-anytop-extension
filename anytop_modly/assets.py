"""Pinned, resumable and offline-reusable AnyTop asset provisioning."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import time
from typing import Callable, Mapping
from urllib.request import Request, urlopen
import uuid

from .constants import (
    ANYTOP_COMMIT,
    ANYTOP_HF_REVISION,
    ANYTOP_SOURCE_RELATIVE,
    ANYTOP_SOURCE_TREE_SHA256,
    ASSET_REVISION_DIGEST,
    ASSETS,
    EXTENSION_ID,
    EXTENSION_VERSION,
    MOTION_COMMIT,
    MOTION_SOURCE_RELATIVE,
    MOTION_SOURCE_TREE_SHA256,
    READY_MARKER_FILENAME,
    READY_SCHEMA_VERSION,
    REVISION_ID,
    SOURCE_PATCHSET,
    T5_REVISION,
    AssetSpec,
)
from .paths import PathContractError, safe_snapshot_directory, safe_snapshot_file


LogFunction = Callable[[str], None]
OpenFunction = Callable[..., object]
CHUNK_SIZE = 1024 * 1024
MAX_MARKER_BYTES = 128 * 1024
MAX_TAR_ENTRIES = 4096
MAX_TAR_MEMBER_BYTES = 64 * 1024 * 1024
MAX_TAR_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
WINDOWS_REPARSE_ATTRIBUTE = 0x400

ANYTOP_ARCHIVE_PATH = "archives/AnyTop.tar.gz"
MOTION_ARCHIVE_PATH = "archives/Motion.tar.gz"
ANYTOP_ARCHIVE_ROOT = f"Anytop-{ANYTOP_COMMIT}"
MOTION_ARCHIVE_ROOT = f"Motion-{MOTION_COMMIT}"
ANYTOP_ALLOWED_ROOT_FILES = frozenset({"LICENSE", "README.md", "environment.yaml"})
ANYTOP_ALLOWED_DIRECTORIES = frozenset(
    {"sample", "model", "diffusion", "data_loaders", "utils", "visualization"}
)
MOTION_ALLOWED_FILES = frozenset(
    {
        "BVH.py",
        "Animation.py",
        "AnimationStructure.py",
        "InverseKinematics.py",
        "Pivots.py",
        "Quaternions.py",
        "TimeWarp.py",
        "visualizations.py",
        "README.md",
        "requirements",
        "setup.py",
        "__init__.py",
    }
)
ANYTOP_EXPECTED_FILE_COUNT = 42
ANYTOP_EXPECTED_BYTES = 334_641
MOTION_EXPECTED_FILE_COUNT = 14
MOTION_EXPECTED_BYTES = 105_111


class AssetError(RuntimeError):
    def __init__(self, code: str, public_message: str) -> None:
        self.code = code
        self.public_message = public_message
        super().__init__(f"{code}: {public_message}")


def _is_alias(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & WINDOWS_REPARSE_ATTRIBUTE
    )


def sha256_file(path: Path) -> str:
    before = path.lstat()
    if _is_alias(before) or not stat.S_ISREG(before.st_mode) or getattr(before, "st_nlink", 1) != 1:
        raise OSError("asset is not an owned regular file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(opened.st_mode)
            or getattr(opened, "st_nlink", 1) != 1
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
        ):
            raise OSError("asset identity changed before hashing")
        while block := handle.read(CHUNK_SIZE):
            digest.update(block)
        after_open = os.fstat(handle.fileno())
    after = path.lstat()
    identities = {
        (x.st_dev, x.st_ino, x.st_size, getattr(x, "st_mtime_ns", None), getattr(x, "st_nlink", 1))
        for x in (before, opened, after_open, after)
    }
    if len(identities) != 1 or _is_alias(after) or not stat.S_ISREG(after.st_mode):
        raise OSError("asset changed while hashing")
    return digest.hexdigest()


def verify_asset(path: Path, spec: AssetSpec) -> tuple[bool, str]:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False, "file is missing"
    except OSError as exc:
        return False, f"metadata is unavailable ({exc})"
    if _is_alias(info) or not stat.S_ISREG(info.st_mode) or getattr(info, "st_nlink", 1) != 1:
        return False, "path is not a regular local file"
    if info.st_size != spec.size:
        return False, f"size is {info.st_size}; expected {spec.size}"
    try:
        digest = sha256_file(path)
    except OSError as exc:
        return False, f"file could not be hashed ({exc})"
    if digest != spec.sha256:
        return False, "SHA-256 does not match the pinned asset"
    return True, "valid"


def inventory_payload() -> list[dict[str, object]]:
    return [
        {
            "path": spec.relative_path,
            "size": spec.size,
            "sha256": spec.sha256,
        }
        for spec in sorted(ASSETS, key=lambda item: item.relative_path)
    ]


def ready_payload() -> dict[str, object]:
    return {
        "schema_version": READY_SCHEMA_VERSION,
        "extension_id": EXTENSION_ID,
        "revision_id": REVISION_ID,
        "asset_revision_digest": ASSET_REVISION_DIGEST,
        "upstream": {
            "anytop_commit": ANYTOP_COMMIT,
            "anytop_hf_revision": ANYTOP_HF_REVISION,
            "motion_commit": MOTION_COMMIT,
            "t5_revision": T5_REVISION,
            "source_patchset": SOURCE_PATCHSET,
        },
        "source_trees": {
            "anytop": ANYTOP_SOURCE_TREE_SHA256,
            "motion": MOTION_SOURCE_TREE_SHA256,
        },
        "inventory": inventory_payload(),
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode(
        "utf-8"
    )
    if len(encoded) > MAX_MARKER_BYTES:
        raise AssetError("ASSET_MARKER_TOO_LARGE", "the readiness marker exceeds its limit")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise AssetError("ASSET_MARKER_WRITE_FAILED", "the readiness marker could not be written") from exc


def _read_ready_marker(snapshot_dir: Path) -> tuple[bool, str]:
    marker = snapshot_dir / READY_MARKER_FILENAME
    try:
        info = marker.lstat()
    except FileNotFoundError:
        return False, "readiness marker is missing"
    except OSError as exc:
        return False, f"readiness marker cannot be inspected ({exc})"
    if _is_alias(info) or not stat.S_ISREG(info.st_mode) or info.st_size > MAX_MARKER_BYTES:
        return False, "readiness marker is unsafe"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False, "readiness marker is invalid"
    return (True, "valid") if payload == ready_payload() else (
        False,
        "readiness marker does not match this immutable revision",
    )


def _response_status(response: object) -> int:
    status = getattr(response, "status", None)
    if isinstance(status, int):
        return status
    getter = getattr(response, "getcode", None)
    result = getter() if callable(getter) else None
    if not isinstance(result, int):
        raise AssetError("ASSET_HTTP_STATUS_MISSING", "the response has no HTTP status")
    return result


def _header(response: object, name: str) -> str:
    headers = getattr(response, "headers", None)
    getter = getattr(headers, "get", None)
    return str(getter(name, "")) if callable(getter) else ""


def _validate_response(response: object, spec: AssetSpec, existing_size: int) -> tuple[str, int]:
    status = _response_status(response)
    if existing_size:
        if status == 200:
            mode, expected = "wb", spec.size
        elif status == 206:
            match = re.fullmatch(
                r"bytes (\d+)-(\d+)/(\d+)", _header(response, "Content-Range").strip()
            )
            if not match:
                raise AssetError("ASSET_RANGE_INVALID", "resume returned an invalid Content-Range")
            start, end, total = (int(value) for value in match.groups())
            if start != existing_size or total != spec.size or end < start or end >= total:
                raise AssetError("ASSET_RANGE_INVALID", "resume range does not match the pinned file")
            mode, expected = "ab", spec.size - existing_size
        else:
            raise AssetError("ASSET_HTTP_STATUS", f"download returned HTTP {status}")
    else:
        if status != 200:
            raise AssetError("ASSET_HTTP_STATUS", f"full download returned HTTP {status}")
        mode, expected = "wb", spec.size
    content_length = _header(response, "Content-Length").strip()
    if content_length:
        try:
            declared = int(content_length)
        except ValueError as exc:
            raise AssetError("ASSET_LENGTH_INVALID", "download returned an invalid length") from exc
        if declared != expected:
            raise AssetError("ASSET_LENGTH_INVALID", "download length does not match the pinned file")
    return mode, 0 if mode == "wb" else existing_size


def _stream_download(
    spec: AssetSpec,
    part_path: Path,
    *,
    opener: OpenFunction,
    log: LogFunction,
    timeout: float,
) -> None:
    try:
        original = part_path.lstat()
    except FileNotFoundError:
        original = None
    except OSError as exc:
        raise AssetError("ASSET_PART_INVALID", "a partial download cannot be inspected") from exc
    if original is not None and (
        _is_alias(original)
        or not stat.S_ISREG(original.st_mode)
        or getattr(original, "st_nlink", 1) != 1
    ):
        raise AssetError("ASSET_PART_INVALID", "a partial download path is unsafe")
    existing = original.st_size if original is not None else 0
    if existing >= spec.size:
        part_path.unlink(missing_ok=True)
        original, existing = None, 0
    headers = {"User-Agent": f"Modly-AnyTop/{EXTENSION_VERSION}"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        log(f"Resuming {spec.relative_path} at {existing} bytes")
    request = Request(spec.url, headers=headers)
    with opener(request, timeout=timeout) as response:  # type: ignore[operator]
        mode, downloaded = _validate_response(response, spec, existing)
        if mode == "wb":
            if original is not None:
                current = part_path.lstat()
                if (
                    _is_alias(current)
                    or not stat.S_ISREG(current.st_mode)
                    or current.st_dev != original.st_dev
                    or current.st_ino != original.st_ino
                ):
                    raise AssetError("ASSET_PART_INVALID", "a partial download path is unsafe")
                part_path.unlink()
            file_mode = "xb"
        else:
            file_mode = "r+b"
        with part_path.open(file_mode) as handle:
            if mode == "ab":
                opened = os.fstat(handle.fileno())
                current = part_path.lstat()
                if (
                    original is None
                    or opened.st_dev != original.st_dev
                    or opened.st_ino != original.st_ino
                    or current.st_dev != original.st_dev
                    or current.st_ino != original.st_ino
                    or opened.st_size != existing
                ):
                    raise AssetError("ASSET_PART_INVALID", "a partial download changed before resume")
                handle.seek(0, os.SEEK_END)
            while block := response.read(CHUNK_SIZE):
                if not isinstance(block, bytes):
                    raise AssetError("ASSET_STREAM_INVALID", "download returned non-binary data")
                downloaded += len(block)
                if downloaded > spec.size:
                    raise AssetError("ASSET_SIZE_EXCEEDED", "download exceeded the pinned size")
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
    if downloaded != spec.size:
        raise AssetError("ASSET_SIZE_MISMATCH", "download ended before the pinned size")


def _ensure_asset(
    snapshot_dir: Path,
    spec: AssetSpec,
    *,
    opener: OpenFunction,
    log: LogFunction,
    timeout: float,
) -> bool:
    try:
        destination = safe_snapshot_file(snapshot_dir, spec.relative_path, create_parent=True)
    except PathContractError as exc:
        raise AssetError(exc.code, exc.public_message) from exc
    valid, _ = verify_asset(destination, spec)
    if valid:
        return False
    part = destination.with_name(destination.name + ".part")
    log(f"Downloading {spec.role} ({spec.size / (1024**2):.1f} MiB)")
    _stream_download(spec, part, opener=opener, log=log, timeout=timeout)
    part_valid, reason = verify_asset(part, spec)
    if not part_valid:
        part.unlink(missing_ok=True)
        raise AssetError("ASSET_INTEGRITY_FAILED", f"{spec.relative_path}: {reason}")
    if destination.exists() or destination.is_symlink():
        info = destination.lstat()
        if _is_alias(info) or not stat.S_ISREG(info.st_mode) or getattr(info, "st_nlink", 1) != 1:
            raise AssetError("ASSET_DESTINATION_UNSAFE", "an asset destination is unsafe")
    try:
        os.replace(part, destination)
    except OSError as exc:
        raise AssetError("ASSET_PROMOTION_FAILED", "a verified asset could not be promoted") from exc
    valid, reason = verify_asset(destination, spec)
    if not valid:
        raise AssetError("ASSET_PROMOTION_FAILED", f"promoted asset failed verification: {reason}")
    return True


def source_tree_digest(root: Path) -> tuple[str, int, int]:
    """Digest sorted regular files using the release-authority algorithm."""

    try:
        root_info = root.lstat()
    except OSError as exc:
        raise AssetError("SOURCE_TREE_MISSING", "an extracted source tree is missing") from exc
    if _is_alias(root_info) or not stat.S_ISDIR(root_info.st_mode):
        raise AssetError("SOURCE_TREE_UNSAFE", "an extracted source root is unsafe")
    inventory: list[tuple[str, bytes, int]] = []
    try:
        for current, directories, files in os.walk(root, topdown=True, followlinks=False):
            parent = Path(current)
            for name in directories:
                info = (parent / name).lstat()
                if _is_alias(info) or not stat.S_ISDIR(info.st_mode):
                    raise AssetError("SOURCE_TREE_UNSAFE", "an extracted source contains an alias")
            for name in files:
                path = parent / name
                info = path.lstat()
                if _is_alias(info) or not stat.S_ISREG(info.st_mode) or getattr(info, "st_nlink", 1) != 1:
                    raise AssetError("SOURCE_TREE_UNSAFE", "an extracted source contains a special file")
                relative = path.relative_to(root).as_posix()
                digest = bytes.fromhex(sha256_file(path))
                inventory.append((relative, digest, info.st_size))
    except AssetError:
        raise
    except OSError as exc:
        raise AssetError("SOURCE_TREE_UNREADABLE", "an extracted source tree cannot be read") from exc
    combined = hashlib.sha256()
    total = 0
    for relative, digest, size in sorted(inventory):
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest)
        combined.update(b"\0")
        total += size
    return combined.hexdigest(), len(inventory), total


def verify_source_tree(
    root: Path, expected_digest: str, expected_count: int, expected_bytes: int
) -> tuple[bool, str]:
    try:
        digest, count, size = source_tree_digest(root)
    except AssetError as exc:
        return False, exc.public_message
    if count != expected_count:
        return False, f"contains {count} files; expected {expected_count}"
    if size != expected_bytes:
        return False, f"contains {size} bytes; expected {expected_bytes}"
    if digest != expected_digest:
        return False, "tree digest does not match the pinned patched source"
    return True, "valid"


def _selected_member(kind: str, relative: PurePosixPath) -> bool:
    if not relative.parts:
        return False
    if kind == "anytop":
        return (
            len(relative.parts) == 1 and relative.name in ANYTOP_ALLOWED_ROOT_FILES
        ) or relative.parts[0] in ANYTOP_ALLOWED_DIRECTORIES
    return (
        len(relative.parts) == 1 and relative.name in MOTION_ALLOWED_FILES
    ) or relative.parts[0] == "quaternion"


def _patch_source(kind: str, staging: Path) -> None:
    if kind != "anytop":
        return
    path = staging / "data_loaders" / "truebones" / "truebones_utils" / "plot_script.py"
    try:
        source = path.read_bytes()
    except OSError as exc:
        raise AssetError("SOURCE_PATCH_MISSING", "the pinned Matplotlib patch target is missing") from exc
    if source.count(b".grid(b=None)") != 5 or source.count(b"ax = p3.Axes3D(fig)") != 5:
        raise AssetError("SOURCE_PATCH_MISMATCH", "the pinned Matplotlib patch no longer applies exactly")
    patched = source.replace(b".grid(b=None)", b".grid(visible=None)").replace(
        b"ax = p3.Axes3D(fig)", b"ax = fig.add_subplot(111, projection=\"3d\")"
    )
    try:
        path.write_bytes(patched)
    except OSError as exc:
        raise AssetError("SOURCE_PATCH_FAILED", "the Matplotlib compatibility patch could not be written") from exc


def _remove_owned_tree(path: Path, parent: Path) -> None:
    if path.parent != parent or not path.name or path.name in {".", ".."}:
        raise AssetError("SOURCE_PATH_INVALID", "refusing to remove an unexpected source path")
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if _is_alias(info):
        path.unlink() if stat.S_ISLNK(info.st_mode) else path.rmdir()
    elif stat.S_ISDIR(info.st_mode):
        shutil.rmtree(path)
    else:
        path.unlink()


def _recover_source_transaction(destination: Path) -> None:
    """Recover an interrupted source-tree rename without deleting the backup."""

    parent = destination.parent
    staging_paths = sorted(parent.glob(f".{destination.name}.staging.*"))
    backup_paths = sorted(parent.glob(f".{destination.name}.backup.*"))
    for stale in staging_paths:
        _remove_owned_tree(stale, parent)
    if destination.exists() or destination.is_symlink():
        for stale in backup_paths:
            _remove_owned_tree(stale, parent)
        return
    if len(backup_paths) > 1:
        raise AssetError(
            "SOURCE_TRANSACTION_AMBIGUOUS",
            "multiple interrupted source backups exist; preserve models_dir and run Repair after inspection",
        )
    if backup_paths:
        try:
            os.replace(backup_paths[0], destination)
        except OSError as exc:
            raise AssetError(
                "SOURCE_ROLLBACK_FAILED", "an interrupted source backup could not be restored"
            ) from exc


def _extract_source(
    archive: Path,
    destination: Path,
    *,
    kind: str,
    archive_root: str,
    expected_digest: str,
    expected_count: int,
    expected_bytes: int,
) -> None:
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    _recover_source_transaction(destination)
    valid, _ = verify_source_tree(destination, expected_digest, expected_count, expected_bytes)
    if valid:
        return
    staging = parent / f".{destination.name}.staging.{uuid.uuid4().hex}"
    backup = parent / f".{destination.name}.backup.{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        try:
            handle = tarfile.open(archive, mode="r:gz")
        except (OSError, tarfile.TarError) as exc:
            raise AssetError("SOURCE_ARCHIVE_INVALID", "a pinned source archive cannot be opened") from exc
        total = 0
        count = 0
        with handle:
            members = handle.getmembers()
            if len(members) > MAX_TAR_ENTRIES:
                raise AssetError("SOURCE_ARCHIVE_LIMIT", "a source archive has too many entries")
            for member in members:
                raw = member.name
                pure = PurePosixPath(raw)
                if (
                    pure.is_absolute()
                    or "\\" in raw
                    or any(part in {"", ".", ".."} for part in pure.parts)
                    or not pure.parts
                    or pure.parts[0] != archive_root
                ):
                    raise AssetError("SOURCE_ARCHIVE_PATH", "a source archive contains an unsafe path")
                relative = PurePosixPath(*pure.parts[1:])
                if not relative.parts or not _selected_member(kind, relative):
                    continue
                if member.isdir():
                    (staging / Path(*relative.parts)).mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile() or member.issym() or member.islnk():
                    raise AssetError("SOURCE_ARCHIVE_ENTRY", "a selected source entry is not a regular file")
                if member.size > MAX_TAR_MEMBER_BYTES:
                    raise AssetError("SOURCE_ARCHIVE_LIMIT", "a source member exceeds its size limit")
                total += member.size
                count += 1
                if total > MAX_TAR_UNCOMPRESSED_BYTES:
                    raise AssetError("SOURCE_ARCHIVE_LIMIT", "a source archive exceeds its expanded limit")
                target = staging / Path(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise AssetError("SOURCE_ARCHIVE_INVALID", "a source member cannot be read")
                with extracted, target.open("xb") as output:
                    copied = 0
                    while block := extracted.read(CHUNK_SIZE):
                        copied += len(block)
                        if copied > member.size:
                            raise AssetError("SOURCE_ARCHIVE_LIMIT", "a source member expanded past its size")
                        output.write(block)
                    if copied != member.size:
                        raise AssetError("SOURCE_ARCHIVE_INVALID", "a source member ended prematurely")
        _patch_source(kind, staging)
        valid, reason = verify_source_tree(
            staging, expected_digest, expected_count, expected_bytes
        )
        if not valid:
            raise AssetError("SOURCE_TREE_INTEGRITY", f"{kind} source: {reason}")
        previous = False
        try:
            if destination.exists() or destination.is_symlink():
                os.replace(destination, backup)
                previous = True
            os.replace(staging, destination)
        except OSError as exc:
            if previous and not destination.exists() and backup.exists():
                os.replace(backup, destination)
            raise AssetError("SOURCE_PROMOTION_FAILED", "verified source could not be promoted") from exc
        if previous:
            _remove_owned_tree(backup, parent)
    finally:
        if staging.exists() or staging.is_symlink():
            _remove_owned_tree(staging, parent)
        if (backup.exists() or backup.is_symlink()) and (
            destination.exists() or destination.is_symlink()
        ):
            # A destination exists after promotion or rollback, so this is an
            # obsolete previous tree.  If rollback itself failed, preserve the
            # sole backup for the next Repair instead of destroying it.
            _remove_owned_tree(backup, parent)


def _ensure_sources(snapshot_dir: Path, log: LogFunction) -> bool:
    source_root = safe_snapshot_directory(snapshot_dir, "source", create=True)
    specifications = (
        (
            "anytop",
            snapshot_dir.joinpath(*ANYTOP_ARCHIVE_PATH.split("/")),
            snapshot_dir.joinpath(*ANYTOP_SOURCE_RELATIVE.split("/")),
            ANYTOP_ARCHIVE_ROOT,
            ANYTOP_SOURCE_TREE_SHA256,
            ANYTOP_EXPECTED_FILE_COUNT,
            ANYTOP_EXPECTED_BYTES,
        ),
        (
            "motion",
            snapshot_dir.joinpath(*MOTION_ARCHIVE_PATH.split("/")),
            snapshot_dir.joinpath(*MOTION_SOURCE_RELATIVE.split("/")),
            MOTION_ARCHIVE_ROOT,
            MOTION_SOURCE_TREE_SHA256,
            MOTION_EXPECTED_FILE_COUNT,
            MOTION_EXPECTED_BYTES,
        ),
    )
    changed = False
    for kind, archive, destination, archive_root, digest, count, size in specifications:
        valid, _ = verify_source_tree(destination, digest, count, size)
        if valid:
            continue
        log(f"Extracting and validating pinned {kind} source")
        _extract_source(
            archive,
            destination,
            kind=kind,
            archive_root=archive_root,
            expected_digest=digest,
            expected_count=count,
            expected_bytes=size,
        )
        changed = True
    # Referencing the verified parent prevents an unnoticed alias introduced
    # between path validation and extraction/promotion.
    if source_root.resolve(strict=True).parent != snapshot_dir.resolve(strict=True):
        raise AssetError("SOURCE_PATH_INVALID", "source storage escaped the AnyTop revision")
    return changed


def verify_snapshot(snapshot_dir: Path, *, require_ready: bool = True) -> list[str]:
    failures: list[str] = []
    for spec in ASSETS:
        try:
            path = safe_snapshot_file(snapshot_dir, spec.relative_path, create_parent=False)
            valid, reason = verify_asset(path, spec)
        except (PathContractError, OSError) as exc:
            valid, reason = False, str(exc)
        if not valid:
            failures.append(f"{spec.relative_path}: {reason}")
    source_specs = (
        (
            snapshot_dir.joinpath(*ANYTOP_SOURCE_RELATIVE.split("/")),
            ANYTOP_SOURCE_TREE_SHA256,
            ANYTOP_EXPECTED_FILE_COUNT,
            ANYTOP_EXPECTED_BYTES,
            "AnyTop source",
        ),
        (
            snapshot_dir.joinpath(*MOTION_SOURCE_RELATIVE.split("/")),
            MOTION_SOURCE_TREE_SHA256,
            MOTION_EXPECTED_FILE_COUNT,
            MOTION_EXPECTED_BYTES,
            "Motion source",
        ),
    )
    for root, digest, count, size, label in source_specs:
        valid, reason = verify_source_tree(root, digest, count, size)
        if not valid:
            failures.append(f"{label}: {reason}")
    if require_ready:
        valid, reason = _read_ready_marker(snapshot_dir)
        if not valid:
            failures.append(f"{READY_MARKER_FILENAME}: {reason}")
    return failures


def ensure_snapshot(
    snapshot_dir: Path,
    *,
    opener: OpenFunction = urlopen,
    log: LogFunction = print,
    timeout: float = 120.0,
) -> Path:
    """Download an unpublished snapshot and publish it exactly once.

    A readiness marker is the commit record for an immutable *content
    contract*.  Setup may restore a file that no longer matches that contract
    (using verified temporary files and atomic promotion), but it never deletes
    or relabels a compatible marker.  Wrapper-only updates therefore reuse the
    published paths, while an intentional content change selects a new
    ``REVISION_ID`` and leaves the prior release untouched.
    """

    marker = snapshot_dir / READY_MARKER_FILENAME
    published = marker.exists() or marker.is_symlink()
    if published:
        valid_marker, _reason = _read_ready_marker(snapshot_dir)
        if not valid_marker:
            raise AssetError(
                "ASSET_REVISION_CONFLICT",
                "a published AnyTop revision has incompatible metadata; "
                "the extension must use a new content revision",
            )
        failures = verify_snapshot(snapshot_dir, require_ready=False)
        if not failures:
            log("Pinned AnyTop snapshot already verified; skipped downloads")
            return snapshot_dir
        log("Restoring files that no longer match the published AnyTop revision")
    for spec in ASSETS:
        _ensure_asset(snapshot_dir, spec, opener=opener, log=log, timeout=timeout)
    _ensure_sources(snapshot_dir, log)
    failures = verify_snapshot(snapshot_dir, require_ready=False)
    if failures:
        raise AssetError(
            "ASSET_SNAPSHOT_INCOMPLETE",
            "the pinned snapshot failed final validation: " + "; ".join(failures[:3]),
        )
    if not published:
        _atomic_json(marker, ready_payload())
    failures = verify_snapshot(snapshot_dir, require_ready=True)
    if failures:
        if not published:
            marker.unlink(missing_ok=True)
        raise AssetError("ASSET_READY_FAILED", "the ready snapshot failed final validation")
    log("All AnyTop assets and patched source trees are verified")
    return snapshot_dir

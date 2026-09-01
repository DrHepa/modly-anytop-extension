"""Verified, relocatable sidecar bundles for AnyTop workflow nodes.

The upstream ``cond.npy`` and edited-motion dictionaries require NumPy pickle
loading.  This module is the trust boundary that makes sure the worker only
ever receives either an installed, checksummed asset or a condition/motion
created by this extension and described by a checksummed bundle.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import stat
import struct
from typing import Any, Iterable, Mapping
import uuid

from .constants import EXTENSION_ID, REVISION_ID


BUNDLE_SCHEMA = "modly-anytop-bundle"
BUNDLE_SCHEMA_VERSION = 2
MANIFEST_SUFFIX = ".anytop.json"
AUTHENTICATION_ALGORITHM = "HMAC-SHA256"
BUNDLE_AUTH_KEY_FILENAME = "bundle-auth-v1.key"
BUNDLE_AUTH_KEY_BYTES = 32
BUNDLE_KEY_ENVELOPE_MAGIC = b"MATK"
BUNDLE_KEY_ENVELOPE_VERSION = 1
BUNDLE_KEY_PROVIDER_DPAPI_CURRENT_USER = 1
BUNDLE_KEY_ENVELOPE_HEADER = struct.Struct(">4sBBHI")
MAX_PROTECTED_KEY_BYTES = 64 * 1024
CRYPTPROTECT_UI_FORBIDDEN = 0x1
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
HASH_CHUNK_BYTES = 1024 * 1024
MAX_BUNDLE_ITEMS = {"preprocess": 4096, "generate": 16, "generate-custom": 16, "edit": 16}
MAX_BUNDLE_FILES = 1 + (4 * MAX_BUNDLE_ITEMS["preprocess"])
MAX_BUNDLE_TOTAL_BYTES = 32 * 1024 * 1024 * 1024
MAX_FILE_BYTES = {
    "preview": 512 * 1024 * 1024,
    "motion": 1024 * 1024 * 1024,
    "bvh": 512 * 1024 * 1024,
    "video": 2 * 1024 * 1024 * 1024,
    "upstream_edit": 1024 * 1024 * 1024,
    "condition": 512 * 1024 * 1024,
}
OBJECT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9-]{0,63}$")
HASH_VALUE = re.compile(r"^[0-9a-f]{64}$")
INDEXED_ROLE = re.compile(r"^(motion|bvh|video)(?:_([0-9]{2,4}))?$")
EDIT_ROLE = re.compile(r"^upstream_edit_([0-9]{2,4})$")
TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "schemaVersion",
        "extensionId",
        "revisionId",
        "operation",
        "objectName",
        "condition",
        "files",
        "parameters",
        "provenance",
        "authentication",
    }
)


class BundleError(ValueError):
    """Raised when a workflow artifact is not a verified AnyTop bundle."""


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


@dataclass(frozen=True)
class BundleFile:
    role: str
    path: Path
    sha256: str
    size: int


@dataclass(frozen=True)
class VerifiedBundle:
    preview: Path
    manifest: Path
    operation: str
    object_name: str
    motion: Path | None
    bvh: Path | None
    video: Path | None
    condition: Path | None
    condition_kind: str
    files: Mapping[str, BundleFile]
    raw: Mapping[str, Any]


def _require_authentication_key(value: object) -> bytes:
    if not isinstance(value, bytes) or len(value) != BUNDLE_AUTH_KEY_BYTES:
        raise BundleError("bundle authentication key is invalid")
    return value


def _uses_windows_key_protection() -> bool:
    return os.name == "nt"


def _data_blob(value: bytes) -> tuple[_DataBlob, object]:
    # Keep the backing array alive for the duration of the native call. DPAPI
    # accepts an empty blob, but AnyTop always passes non-empty key/entropy data.
    backing = (ctypes.c_ubyte * len(value)).from_buffer_copy(value)
    return _DataBlob(len(value), ctypes.cast(backing, ctypes.POINTER(ctypes.c_ubyte))), backing


def _dpapi_entropy() -> bytes:
    return hashlib.sha256(
        f"{EXTENSION_ID}\0{REVISION_ID}\0bundle-auth-v1".encode("utf-8")
    ).digest()


def _windows_library(name: str) -> object:
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        raise BundleError("Windows DPAPI is unavailable")
    try:
        return loader(name, use_last_error=True)
    except OSError as exc:
        raise BundleError("Windows DPAPI is unavailable") from exc


def _local_free(pointer: object) -> None:
    if not pointer:
        return
    kernel32 = _windows_library("Kernel32.dll")
    local_free = kernel32.LocalFree
    local_free.argtypes = [ctypes.c_void_p]
    local_free.restype = ctypes.c_void_p
    local_free(ctypes.cast(pointer, ctypes.c_void_p))


def _dpapi_protect(value: bytes) -> bytes:
    """Protect bytes for the current Windows user without permitting UI."""

    input_blob, input_backing = _data_blob(value)
    entropy_blob, entropy_backing = _data_blob(_dpapi_entropy())
    output_blob = _DataBlob()
    crypt32 = _windows_library("Crypt32.dll")
    protect = crypt32.CryptProtectData
    protect.argtypes = [
        ctypes.POINTER(_DataBlob),
        wintypes.LPCWSTR,
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    protect.restype = wintypes.BOOL
    # The backing references are intentionally kept in scope across this call.
    _ = input_backing, entropy_backing
    if not protect(
        ctypes.byref(input_blob),
        None,
        ctypes.byref(entropy_blob),
        None,
        None,
        CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(output_blob),
    ):
        error = int(getattr(ctypes, "get_last_error", lambda: 0)())
        raise BundleError(f"Windows DPAPI could not protect the workflow key ({error})")
    try:
        protected = ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        _local_free(output_blob.pbData)
    if not protected or len(protected) > MAX_PROTECTED_KEY_BYTES:
        raise BundleError("Windows DPAPI returned an invalid protected key")
    return protected


def _dpapi_unprotect(value: bytes) -> bytes:
    """Unprotect bytes for the current Windows user without permitting UI."""

    input_blob, input_backing = _data_blob(value)
    entropy_blob, entropy_backing = _data_blob(_dpapi_entropy())
    output_blob = _DataBlob()
    crypt32 = _windows_library("Crypt32.dll")
    unprotect = crypt32.CryptUnprotectData
    unprotect.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    unprotect.restype = wintypes.BOOL
    _ = input_backing, entropy_backing
    if not unprotect(
        ctypes.byref(input_blob),
        None,
        ctypes.byref(entropy_blob),
        None,
        None,
        CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(output_blob),
    ):
        error = int(getattr(ctypes, "get_last_error", lambda: 0)())
        raise BundleError(f"Windows DPAPI could not unlock the workflow key ({error})")
    try:
        plaintext = ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        _local_free(output_blob.pbData)
    return plaintext


def _encode_windows_key(value: bytes) -> bytes:
    protected = _dpapi_protect(_require_authentication_key(value))
    if not protected or len(protected) > MAX_PROTECTED_KEY_BYTES:
        raise BundleError("Windows DPAPI returned an invalid protected key")
    header = BUNDLE_KEY_ENVELOPE_HEADER.pack(
        BUNDLE_KEY_ENVELOPE_MAGIC,
        BUNDLE_KEY_ENVELOPE_VERSION,
        BUNDLE_KEY_PROVIDER_DPAPI_CURRENT_USER,
        0,
        len(protected),
    )
    return header + protected


def _decode_windows_key(value: bytes) -> bytes:
    if len(value) < BUNDLE_KEY_ENVELOPE_HEADER.size:
        raise BundleError("unprotected Windows workflow keys are not accepted")
    try:
        magic, version, provider, reserved, protected_size = BUNDLE_KEY_ENVELOPE_HEADER.unpack_from(
            value
        )
    except struct.error as exc:
        raise BundleError("Windows workflow-key envelope is invalid") from exc
    if magic != BUNDLE_KEY_ENVELOPE_MAGIC:
        raise BundleError("unprotected Windows workflow keys are not accepted")
    if (
        version != BUNDLE_KEY_ENVELOPE_VERSION
        or provider != BUNDLE_KEY_PROVIDER_DPAPI_CURRENT_USER
        or reserved != 0
        or protected_size <= 0
        or protected_size > MAX_PROTECTED_KEY_BYTES
        or len(value) != BUNDLE_KEY_ENVELOPE_HEADER.size + protected_size
    ):
        raise BundleError("Windows workflow-key envelope is invalid")
    try:
        return _require_authentication_key(
            _dpapi_unprotect(value[BUNDLE_KEY_ENVELOPE_HEADER.size :])
        )
    except BundleError as exc:
        raise BundleError("Windows workflow-key envelope could not be unlocked") from exc


def bundle_auth_key_path(runtime_cache: Path) -> Path:
    """Return the fixed per-installation key path inside the owned cache."""

    cache = require_directory(runtime_cache)
    return cache / BUNDLE_AUTH_KEY_FILENAME


def _read_key_file(path: Path) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise BundleError("bundle authentication key is unavailable") from exc
    windows_protected = _uses_windows_key_protection()
    maximum_size = (
        BUNDLE_KEY_ENVELOPE_HEADER.size + MAX_PROTECTED_KEY_BYTES
        if windows_protected
        else BUNDLE_AUTH_KEY_BYTES
    )
    minimum_size = BUNDLE_KEY_ENVELOPE_HEADER.size + 1 if windows_protected else BUNDLE_AUTH_KEY_BYTES
    if (
        path.is_symlink()
        or _is_reparse_point(path)
        or not stat.S_ISREG(before.st_mode)
        or getattr(before, "st_nlink", 1) != 1
        or not minimum_size <= before.st_size <= maximum_size
        or (not windows_protected and stat.S_IMODE(before.st_mode) & 0o077)
    ):
        raise BundleError("bundle authentication key is unsafe")
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            value = handle.read(maximum_size + 1)
        after = path.lstat()
    except OSError as exc:
        raise BundleError("bundle authentication key cannot be read") from exc
    identity = lambda info: (
        info.st_dev,
        info.st_ino,
        info.st_size,
        getattr(info, "st_mtime_ns", None),
    )
    if identity(before) != identity(opened) or identity(before) != identity(after):
        raise BundleError("bundle authentication key changed while reading")
    if windows_protected:
        return _decode_windows_key(value)
    return _require_authentication_key(value)


def load_bundle_auth_key(runtime_cache: Path) -> bytes:
    return _read_key_file(bundle_auth_key_path(runtime_cache))


def ensure_bundle_auth_key(runtime_cache: Path) -> Path:
    """Create the persistent per-installation HMAC key, or validate the existing key."""

    path = bundle_auth_key_path(runtime_cache)
    if path.exists() or path.is_symlink():
        _read_key_file(path)
        return path
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        key = secrets.token_bytes(BUNDLE_AUTH_KEY_BYTES)
        stored = _encode_windows_key(key) if _uses_windows_key_protection() else key
        written = os.write(descriptor, stored)
        if written != len(stored):
            raise OSError("short authentication-key write")
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if path.exists() or path.is_symlink():
            temporary.unlink(missing_ok=True)
            _read_key_file(path)
            return path
        os.replace(temporary, path)
        if not _uses_windows_key_protection():
            os.chmod(path, 0o600)
        _read_key_file(path)
        return path
    except (OSError, BundleError) as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        temporary.unlink(missing_ok=True)
        if isinstance(exc, BundleError):
            raise
        raise BundleError("bundle authentication key could not be created") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(HASH_CHUNK_BYTES)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    except OSError:
        return True
    return bool(attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)))


def require_regular_file(path: Path, *, max_bytes: int | None = None) -> Path:
    """Return a resolved, non-link regular file or raise ``BundleError``."""

    candidate = Path(path)
    try:
        info = candidate.lstat()
    except OSError as exc:
        raise BundleError("required file is unavailable") from exc
    if candidate.is_symlink() or _is_reparse_point(candidate):
        raise BundleError("linked files are not accepted")
    if not stat.S_ISREG(info.st_mode):
        raise BundleError("expected a regular file")
    if max_bytes is not None and info.st_size > max_bytes:
        raise BundleError("file exceeds the allowed size")
    return candidate.resolve(strict=True)


def require_directory(path: Path) -> Path:
    candidate = Path(path)
    try:
        info = candidate.lstat()
    except OSError as exc:
        raise BundleError("required directory is unavailable") from exc
    if candidate.is_symlink() or _is_reparse_point(candidate):
        raise BundleError("linked directories are not accepted")
    if not stat.S_ISDIR(info.st_mode):
        raise BundleError("expected a directory")
    return candidate.resolve(strict=True)


def _relative_file(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise BundleError("bundle file path is invalid")
    raw = Path(value)
    if raw.is_absolute() or raw.anchor or ".." in raw.parts:
        raise BundleError("bundle file path must be relative")
    candidate = require_regular_file(root / raw)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise BundleError("bundle file escapes its directory") from exc
    return candidate


def _file_record(role: str, path: Path, root: Path) -> dict[str, object]:
    family = _role_family(role)
    checked = require_regular_file(path, max_bytes=MAX_FILE_BYTES[family])
    try:
        relative = checked.relative_to(root)
    except ValueError as exc:
        raise BundleError("bundle output is outside the run directory") from exc
    return {
        "role": role,
        "path": relative.as_posix(),
        "sha256": sha256_file(checked),
        "size": checked.stat().st_size,
    }


def _canonical_json(value: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise BundleError("bundle manifest is not canonical JSON") from exc


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError, RecursionError) as exc:
        raise BundleError("bundle JSON is invalid") from exc
    if len(payload.encode("utf-8")) > MAX_MANIFEST_BYTES:
        raise BundleError("bundle JSON exceeds the allowed size")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_bundle(
    *,
    preview: Path,
    operation: str,
    object_name: str,
    condition_kind: str,
    condition: Path | None,
    files: Mapping[str, Path],
    parameters: Mapping[str, object],
    provenance: Mapping[str, object],
    authentication_key: bytes,
) -> Path:
    """Write a same-stem, relocatable manifest after hashing every sidecar."""

    auth_key = _require_authentication_key(authentication_key)
    preview = require_regular_file(preview, max_bytes=MAX_FILE_BYTES["preview"])
    root = require_directory(preview.parent)
    if preview.suffix.lower() != ".glb":
        raise BundleError("the bundle preview must be a GLB")
    if condition_kind not in {"builtin", "custom"}:
        raise BundleError("unknown condition kind")
    if condition_kind == "custom" and condition is None:
        raise BundleError("custom bundles require a condition sidecar")
    if condition_kind == "builtin" and condition is not None:
        raise BundleError("built-in bundles must not include a condition sidecar")
    _validate_identity(operation, object_name, condition_kind)
    _validate_file_roles(operation, ("preview", *files.keys()))

    records: dict[str, object] = {"preview": _file_record("preview", preview, root)}
    for role, value in files.items():
        if role == "preview" or not isinstance(role, str) or not role:
            raise BundleError("invalid bundle file role")
        records[role] = _file_record(role, value, root)
    condition_record: dict[str, object] = {"kind": condition_kind}
    if condition is not None:
        condition_record["file"] = _file_record("condition", condition, root)
    record_values = list(records.values())
    if condition is not None:
        record_values.append(condition_record["file"])
    paths = [str(record["path"]) for record in record_values if isinstance(record, dict)]
    if len(paths) != len(record_values) or len(paths) != len(set(paths)):
        raise BundleError("bundle file paths must be unique")
    if sum(int(record["size"]) for record in record_values if isinstance(record, dict)) > MAX_BUNDLE_TOTAL_BYTES:
        raise BundleError("bundle files exceed the total size limit")

    manifest = preview.with_suffix(MANIFEST_SUFFIX)
    document: dict[str, object] = {
        "schema": BUNDLE_SCHEMA,
        "schemaVersion": BUNDLE_SCHEMA_VERSION,
        "extensionId": EXTENSION_ID,
        "revisionId": REVISION_ID,
        "operation": operation,
        "objectName": object_name,
        "condition": condition_record,
        "files": records,
        "parameters": dict(parameters),
        "provenance": dict(provenance),
    }
    document["authentication"] = {
        "algorithm": AUTHENTICATION_ALGORITHM,
        "tag": hmac.new(auth_key, _canonical_json(document), hashlib.sha256).hexdigest(),
    }
    atomic_json(manifest, document)
    return manifest


def _read_manifest(path: Path) -> Mapping[str, Any]:
    manifest = require_regular_file(path, max_bytes=MAX_MANIFEST_BYTES)
    try:
        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in pairs:
                if key in value:
                    raise BundleError("bundle manifest contains duplicate fields")
                value[key] = item
            return value

        value = json.loads(manifest.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    except BundleError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise BundleError("bundle manifest is invalid") from exc
    if not isinstance(value, dict):
        raise BundleError("bundle manifest must contain an object")
    return value


def _role_family(key: str) -> str:
    if key == "preview" or key == "condition":
        return key
    match = INDEXED_ROLE.fullmatch(key)
    if match:
        return match.group(1)
    if EDIT_ROLE.fullmatch(key):
        return "upstream_edit"
    raise BundleError("bundle file role is not allowed")


def _role_index(key: str) -> int | None:
    match = INDEXED_ROLE.fullmatch(key)
    if match:
        suffix = match.group(2)
        if suffix is None:
            return 0
        if suffix == "00" or (suffix.startswith("0") and len(suffix) > 2):
            raise BundleError("bundle file role is not canonical")
        return int(suffix, 10)
    match = EDIT_ROLE.fullmatch(key)
    if match:
        suffix = match.group(1)
        if suffix.startswith("0") and len(suffix) > 2:
            raise BundleError("bundle file role is not canonical")
        return int(suffix, 10)
    return None


def _validate_identity(operation: object, object_name: object, condition_kind: object) -> None:
    if not isinstance(operation, str) or operation not in MAX_BUNDLE_ITEMS:
        raise BundleError("bundle operation is not allowed")
    if not isinstance(object_name, str) or not OBJECT_NAME.fullmatch(object_name):
        raise BundleError("bundle object name is invalid")
    expected_conditions = {
        "preprocess": {"custom"},
        "generate": {"builtin"},
        "generate-custom": {"custom"},
        "edit": {"builtin", "custom"},
    }
    if not isinstance(condition_kind, str) or condition_kind not in expected_conditions[operation]:
        raise BundleError("bundle condition does not match its operation")


def _validate_file_roles(operation: object, roles: Iterable[object]) -> None:
    if not isinstance(operation, str) or operation not in MAX_BUNDLE_ITEMS:
        raise BundleError("bundle operation is not allowed")
    role_list = list(roles)
    if len(role_list) > MAX_BUNDLE_FILES or any(not isinstance(role, str) for role in role_list):
        raise BundleError("bundle contains too many files")
    role_set = set(role_list)
    if len(role_set) != len(role_list) or "preview" not in role_set:
        raise BundleError("bundle file roles are invalid")
    indexed: dict[str, set[int]] = {"motion": set(), "bvh": set(), "video": set()}
    edits: set[int] = set()
    for role in role_set - {"preview"}:
        family = _role_family(role)
        index = _role_index(role)
        if index is None:
            raise BundleError("bundle file role is invalid")
        if family in indexed:
            indexed[family].add(index)
        elif family == "upstream_edit":
            edits.add(index)
    item_indices = indexed["motion"]
    maximum = MAX_BUNDLE_ITEMS[operation]
    if (
        not item_indices
        or any(values != item_indices for values in indexed.values())
        or item_indices != set(range(len(item_indices)))
        or len(item_indices) > maximum
    ):
        raise BundleError("bundle motion sidecar roles are incomplete")
    if operation == "edit":
        if edits != item_indices:
            raise BundleError("edited bundles require one upstream-edit sidecar per motion")
    elif edits:
        raise BundleError("upstream-edit sidecars are not valid for this operation")


def _verify_record(root: Path, key: str, value: object) -> BundleFile:
    if not isinstance(value, dict) or set(value) != {"role", "path", "sha256", "size"}:
        raise BundleError("bundle file record is invalid")
    family = _role_family(key)
    maximum = MAX_FILE_BYTES[family]
    expected_hash = value.get("sha256")
    expected_size = value.get("size")
    role = value.get("role")
    if role != key or not isinstance(expected_hash, str) or not HASH_VALUE.fullmatch(expected_hash):
        raise BundleError("bundle file metadata is invalid")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
        or expected_size > maximum
    ):
        raise BundleError("bundle file size is invalid")
    path = _relative_file(root, value.get("path"))
    if path.stat().st_size != expected_size or sha256_file(path) != expected_hash.lower():
        raise BundleError("bundle file integrity check failed")
    return BundleFile(role=key, path=path, sha256=expected_hash.lower(), size=expected_size)


def _verify_authentication(raw: Mapping[str, Any], authentication_key: bytes) -> None:
    if set(raw) != TOP_LEVEL_FIELDS:
        raise BundleError("bundle manifest fields are invalid")
    authentication = raw.get("authentication")
    if not isinstance(authentication, dict) or set(authentication) != {"algorithm", "tag"}:
        raise BundleError("bundle authentication metadata is invalid")
    tag = authentication.get("tag")
    if (
        authentication.get("algorithm") != AUTHENTICATION_ALGORITHM
        or not isinstance(tag, str)
        or not HASH_VALUE.fullmatch(tag)
    ):
        raise BundleError("bundle authentication metadata is invalid")
    unsigned = dict(raw)
    del unsigned["authentication"]
    expected = hmac.new(
        _require_authentication_key(authentication_key),
        _canonical_json(unsigned),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(tag, expected):
        raise BundleError("bundle authentication failed")


def verify_bundle(preview_path: Path, *, authentication_key: bytes) -> VerifiedBundle:
    """Verify a wrapper-created preview and every referenced workflow sidecar."""

    preview = require_regular_file(Path(preview_path), max_bytes=MAX_FILE_BYTES["preview"])
    if preview.suffix.lower() != ".glb":
        raise BundleError("AnyTop nodes require an AnyTop preview GLB")
    manifest = preview.with_suffix(MANIFEST_SUFFIX)
    raw = _read_manifest(manifest)
    if raw.get("schema") != BUNDLE_SCHEMA or raw.get("schemaVersion") != BUNDLE_SCHEMA_VERSION:
        raise BundleError("unsupported AnyTop bundle schema")
    if raw.get("extensionId") != EXTENSION_ID or raw.get("revisionId") != REVISION_ID:
        raise BundleError("bundle was created by a different AnyTop revision")
    _verify_authentication(raw, authentication_key)
    operation = raw.get("operation")
    object_name = raw.get("objectName")

    file_values = raw.get("files")
    if not isinstance(file_values, dict) or "preview" not in file_values:
        raise BundleError("bundle file index is missing")
    condition_value = raw.get("condition")
    if not isinstance(condition_value, dict):
        raise BundleError("bundle condition metadata is missing")
    condition_kind = condition_value.get("kind")
    _validate_identity(operation, object_name, condition_kind)
    _validate_file_roles(operation, file_values.keys())
    if set(condition_value) not in ({"kind"}, {"kind", "file"}):
        raise BundleError("bundle condition metadata is invalid")
    if not isinstance(raw.get("parameters"), dict) or not isinstance(raw.get("provenance"), dict):
        raise BundleError("bundle metadata sections are invalid")

    # All semantic/count/declared-size checks happen before opening and hashing
    # sidecars. Authentication happened earlier, so an untrusted JSON file
    # cannot turn verification into arbitrary filesystem work.
    declared_total = 0
    for role, value in file_values.items():
        if not isinstance(value, dict):
            raise BundleError("bundle file record is invalid")
        declared_size = value.get("size")
        family = _role_family(role)
        if (
            isinstance(declared_size, bool)
            or not isinstance(declared_size, int)
            or declared_size < 0
            or declared_size > MAX_FILE_BYTES[family]
        ):
            raise BundleError("bundle file size is invalid")
        declared_total += declared_size
    condition_declared = condition_value.get("file")
    if isinstance(condition_declared, dict):
        size = condition_declared.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size <= MAX_FILE_BYTES["condition"]:
            raise BundleError("bundle condition size is invalid")
        declared_total += size
    if declared_total > MAX_BUNDLE_TOTAL_BYTES:
        raise BundleError("bundle files exceed the total size limit")

    files = {key: _verify_record(preview.parent, key, value) for key, value in file_values.items()}
    if files["preview"].path != preview:
        raise BundleError("bundle preview does not match the input GLB")

    condition: Path | None = None
    if condition_kind == "custom":
        if set(condition_value) != {"kind", "file"}:
            raise BundleError("bundle condition metadata is incomplete")
        condition_record = condition_value.get("file")
        condition_file = _verify_record(preview.parent, "condition", condition_record)
        condition = condition_file.path
    elif set(condition_value) != {"kind"}:
        raise BundleError("built-in conditions must not include a sidecar")

    all_paths = [item.path for item in files.values()]
    if condition is not None:
        all_paths.append(condition)
    if len(set(all_paths)) != len(all_paths):
        raise BundleError("bundle file paths must be unique")
    if sum(item.size for item in files.values()) + (
        condition_file.size if condition is not None else 0
    ) > MAX_BUNDLE_TOTAL_BYTES:
        raise BundleError("bundle files exceed the total size limit")

    return VerifiedBundle(
        preview=preview,
        manifest=require_regular_file(manifest),
        operation=operation,
        object_name=object_name,
        motion=files.get("motion").path if files.get("motion") else None,
        bvh=files.get("bvh").path if files.get("bvh") else None,
        video=files.get("video").path if files.get("video") else None,
        condition=condition,
        condition_kind=str(condition_kind),
        files=files,
        raw=raw,
    )


def verify_hash(path: Path, expected_sha256: str) -> Path:
    checked = require_regular_file(path)
    if sha256_file(checked) != expected_sha256.lower():
        raise BundleError("installed asset integrity check failed")
    return checked


def file_records(paths: Iterable[Path]) -> list[dict[str, object]]:
    """Return deterministic standalone file records for text manifests."""

    records = []
    for path in sorted((require_regular_file(item) for item in paths), key=lambda item: item.name):
        records.append({"path": path.name, "sha256": sha256_file(path), "size": path.stat().st_size})
    return records

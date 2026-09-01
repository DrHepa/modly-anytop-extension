"""Atomic, secret-free setup and runtime state for the AnyTop process."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
from typing import Mapping
import uuid

from .constants import EXTENSION_ID, REVISION_ID, RUNTIME_CONFIG_FILENAME


RUNTIME_CONFIG_SCHEMA_VERSION = 1
DEPENDENCY_STATE_SCHEMA_VERSION = 1
MAX_STATE_BYTES = 128 * 1024
WINDOWS_REPARSE_ATTRIBUTE = 0x400
RUNTIME_RESERVED = frozenset({"schema_version", "models_dir", "revision_root"})


class StateError(RuntimeError):
    def __init__(self, code: str, public_message: str) -> None:
        self.code = code
        self.public_message = public_message
        super().__init__(f"{code}: {public_message}")


@dataclass(frozen=True)
class RuntimeConfig:
    models_dir: Path
    revision_root: Path
    payload: Mapping[str, object]

    @property
    def revision_dir(self) -> Path:
        """Compatibility alias for integrations written against older state."""

        return self.revision_root


def _is_alias(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & WINDOWS_REPARSE_ATTRIBUTE
    )


def _secret_key(key: object) -> bool:
    lowered = str(key).casefold()
    return any(part in lowered for part in ("token", "secret", "password", "authorization"))


def _contains_secret(value: object, key: object = "") -> bool:
    if _secret_key(key):
        return True
    if isinstance(value, dict):
        return any(_contains_secret(item, name) for name, item in value.items())
    if isinstance(value, list):
        return any(_contains_secret(item) for item in value)
    return False


def _atomic_json(path: Path, payload: Mapping[str, object]) -> Path:
    if _contains_secret(payload):
        raise StateError("STATE_SECRET_REJECTED", "generated runtime state must not contain credentials")
    try:
        encoded = (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise StateError("STATE_JSON_INVALID", "generated state is not JSON serializable") from exc
    if len(encoded) > MAX_STATE_BYTES:
        raise StateError("STATE_TOO_LARGE", "generated state exceeds its size limit")
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if _is_alias(info) or not stat.S_ISREG(info.st_mode) or getattr(info, "st_nlink", 1) != 1:
            raise StateError("STATE_FILE_INVALID", "an existing generated-state path is unsafe")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise StateError("STATE_WRITE_FAILED", "generated state could not be written atomically") from exc
    return path


def _read_json(path: Path) -> dict[str, object]:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise StateError("STATE_MISSING", "generated state is missing; run Repair") from exc
    except OSError as exc:
        raise StateError("STATE_READ_FAILED", "generated state cannot be inspected") from exc
    if (
        _is_alias(info)
        or not stat.S_ISREG(info.st_mode)
        or getattr(info, "st_nlink", 1) != 1
        or info.st_size > MAX_STATE_BYTES
    ):
        raise StateError("STATE_FILE_INVALID", "generated state is unsafe; run Repair")
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or getattr(opened, "st_nlink", 1) != 1
                or (opened.st_dev, opened.st_ino, opened.st_size)
                != (info.st_dev, info.st_ino, info.st_size)
            ):
                raise StateError("STATE_FILE_INVALID", "generated state changed before reading")
            raw = handle.read(MAX_STATE_BYTES + 1)
            after_open = os.fstat(handle.fileno())
        after = path.lstat()
        if (
            len(raw) > MAX_STATE_BYTES
            or _is_alias(after)
            or (after.st_dev, after.st_ino, after.st_size, getattr(after, "st_mtime_ns", None))
            != (
                info.st_dev,
                info.st_ino,
                info.st_size,
                getattr(info, "st_mtime_ns", None),
            )
            or (after_open.st_dev, after_open.st_ino, after_open.st_size)
            != (info.st_dev, info.st_ino, info.st_size)
        ):
            raise StateError("STATE_FILE_INVALID", "generated state changed while reading")
        payload = json.loads(raw.decode("utf-8"))
    except StateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StateError("STATE_JSON_INVALID", "generated state is unreadable; run Repair") from exc
    if not isinstance(payload, dict) or _contains_secret(payload):
        raise StateError("STATE_JSON_INVALID", "generated state is invalid; run Repair")
    return payload


def runtime_config_payload(
    models_dir: Path,
    revision_root: Path,
    *,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    try:
        models = models_dir.resolve(strict=True)
        revision = revision_root.resolve(strict=True)
    except OSError as exc:
        raise StateError("STATE_PATH_MISSING", "configured model paths must already exist") from exc
    expected_suffix = Path(EXTENSION_ID) / "anytop" / "revisions" / REVISION_ID
    try:
        relative = revision.relative_to(models)
    except ValueError as exc:
        raise StateError("STATE_PATH_INVALID", "the AnyTop revision is outside models_dir") from exc
    if not models.is_dir() or not revision.is_dir() or relative != expected_suffix:
        raise StateError("STATE_REVISION_INVALID", "the AnyTop revision path does not match this release")
    additions = dict(extra or {})
    if RUNTIME_RESERVED.intersection(additions):
        raise StateError("STATE_FIELD_CONFLICT", "extra runtime state overrides a reserved field")
    payload: dict[str, object] = {
        "schema_version": RUNTIME_CONFIG_SCHEMA_VERSION,
        "models_dir": str(models),
        "revision_root": str(revision),
    }
    payload.update(additions)
    if _contains_secret(payload):
        raise StateError("STATE_SECRET_REJECTED", "runtime state must not contain credentials")
    return payload


def write_runtime_config(
    extension_dir: Path,
    models_dir: Path,
    revision_root: Path,
    *,
    extra: Mapping[str, object] | None = None,
) -> Path:
    try:
        extension = extension_dir.resolve(strict=True)
        info = extension.lstat()
    except OSError as exc:
        raise StateError("STATE_EXTENSION_MISSING", "the extension directory is unavailable") from exc
    if _is_alias(info) or not stat.S_ISDIR(info.st_mode):
        raise StateError("STATE_EXTENSION_INVALID", "the extension directory is unsafe")
    return _atomic_json(
        extension / RUNTIME_CONFIG_FILENAME,
        runtime_config_payload(models_dir, revision_root, extra=extra),
    )


def read_runtime_config(extension_dir: Path) -> RuntimeConfig:
    payload = _read_json(extension_dir / RUNTIME_CONFIG_FILENAME)
    if payload.get("schema_version") != RUNTIME_CONFIG_SCHEMA_VERSION:
        raise StateError("STATE_SCHEMA_MISMATCH", "runtime state is stale; run Repair")
    models_raw = payload.get("models_dir")
    revision_raw = payload.get("revision_root")
    if not isinstance(models_raw, str) or not isinstance(revision_raw, str):
        raise StateError("STATE_PATH_INVALID", "runtime state lacks model paths; run Repair")
    models = Path(models_raw)
    revision = Path(revision_raw)
    if not models.is_absolute() or not revision.is_absolute():
        raise StateError("STATE_PATH_INVALID", "runtime model paths must be absolute; run Repair")
    extras = {key: value for key, value in payload.items() if key not in RUNTIME_RESERVED}
    expected = runtime_config_payload(models, revision, extra=extras)
    if payload != expected:
        raise StateError("STATE_PATH_INVALID", "runtime model paths are stale; run Repair")
    return RuntimeConfig(models, revision, expected)


def write_dependency_state(path: Path, payload: Mapping[str, object]) -> Path:
    state = dict(payload)
    state.setdefault("schema_version", DEPENDENCY_STATE_SCHEMA_VERSION)
    return _atomic_json(path, state)


def dependency_state_matches(path: Path, expected: Mapping[str, object]) -> bool:
    candidate = dict(expected)
    candidate.setdefault("schema_version", DEPENDENCY_STATE_SCHEMA_VERSION)
    try:
        return _read_json(path) == candidate
    except StateError:
        return False

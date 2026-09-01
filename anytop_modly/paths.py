"""Resolve Modly storage and keep every AnyTop asset inside its owned root."""

from __future__ import annotations

from dataclasses import dataclass
import json
import ntpath
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import stat
import sys
from typing import Mapping, Sequence
from urllib.request import Request, urlopen

from .constants import (
    ANYTOP_SOURCE_RELATIVE,
    BUILTIN_COND_RELATIVE,
    EXTENSION_ID,
    MOTION_SOURCE_RELATIVE,
    READY_MARKER_FILENAME,
    REVISION_ID,
    T5_SOURCE_RELATIVE,
)


MODLY_PATHS_URL = "http://127.0.0.1:8765/settings/paths"
SETUP_MODELS_PAYLOAD_KEYS = ("models_dir", "modelsDir")
RUNTIME_MODELS_PAYLOAD_KEYS = ("modelsDir", "models_dir")
MODELS_ENVIRONMENT_KEYS = ("MODLY_MODELS_DIR", "MODELS_DIR")
MAX_SETTINGS_RESPONSE_BYTES = 64 * 1024
WINDOWS_REPARSE_ATTRIBUTE = 0x400


class PathContractError(RuntimeError):
    """A fail-closed model-storage error with a stable UI-safe code."""

    def __init__(self, code: str, public_message: str) -> None:
        self.code = code
        self.public_message = public_message
        super().__init__(f"{code}: {public_message}")


@dataclass(frozen=True)
class SnapshotPaths:
    root: Path
    anytop_source: Path
    motion_source: Path
    checkpoints: Path
    builtin_cond: Path
    t5: Path
    ready_marker: Path


def normalize_platform_name(value: object) -> str:
    raw = str(value or "").strip().casefold()
    if raw.startswith("linux"):
        return "linux"
    if raw in {"windows", "win32"}:
        return "win32"
    return raw


def current_platform_name() -> str:
    return normalize_platform_name(sys.platform)


def normalize_configured_directory_path(
    value: object, label: str, platform_name: object
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PathContractError("PATH_ABSOLUTE_REQUIRED", f"{label} must be an absolute path")
    if "\0" in value:
        raise PathContractError("PATH_NULL_BYTE", f"{label} must not contain null bytes")
    system = normalize_platform_name(platform_name)
    if system == "win32":
        if not re.match(r"^(?:[A-Za-z]:[\\/]|[\\/]{2}[^\\/]+[\\/][^\\/]+)", value):
            raise PathContractError(
                "PATH_ABSOLUTE_REQUIRED", f"{label} must be a fully-qualified Windows path"
            )
        return ntpath.normpath(value)
    if system == "linux":
        if not posixpath.isabs(value):
            raise PathContractError("PATH_ABSOLUTE_REQUIRED", f"{label} must be absolute")
        return posixpath.normpath(value)
    raise PathContractError("PATH_PLATFORM_UNSUPPORTED", "only Windows and Linux are supported")


def _is_alias(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & WINDOWS_REPARSE_ATTRIBUTE
    )


def _native_directory(
    value: object,
    label: str,
    platform_name: str,
    *,
    require_existing: bool,
) -> Path:
    normalized = normalize_configured_directory_path(value, label, platform_name)
    if platform_name != current_platform_name():
        raise PathContractError(
            "PATH_PLATFORM_MISMATCH", "the configured path flavor does not match this host"
        )
    path = Path(normalized)
    if require_existing and not path.is_dir():
        raise PathContractError("PATH_DIRECTORY_MISSING", f"{label} is unavailable")
    return path


def _comparison_key(path: Path, platform_name: str) -> str:
    raw = os.fspath(path)
    return (
        ntpath.normcase(ntpath.normpath(raw))
        if platform_name == "win32"
        else posixpath.normpath(raw)
    )


def _candidate_from_mapping(
    values: Mapping[str, object],
    keys: Sequence[str],
    platform_name: str,
    *,
    require_existing: bool,
) -> Path | None:
    present = [key for key in keys if key in values]
    if not present:
        return None
    candidates = [
        _native_directory(values[key], key, platform_name, require_existing=require_existing)
        for key in present
    ]
    first = _comparison_key(candidates[0], platform_name)
    if any(_comparison_key(item, platform_name) != first for item in candidates[1:]):
        raise PathContractError(
            "PATH_MODELS_CONFLICT",
            "multiple model-directory settings identify different locations",
        )
    return candidates[0]


def _response_status(response: object) -> int | None:
    status = getattr(response, "status", None)
    if isinstance(status, int):
        return status
    getter = getattr(response, "getcode", None)
    result = getter() if callable(getter) else None
    return result if isinstance(result, int) else None


def _models_root_from_modly_api(
    platform_name: str,
    *,
    require_existing: bool,
    opener: object,
    timeout: float,
) -> Path | None:
    request = Request(
        MODLY_PATHS_URL,
        headers={"Accept": "application/json", "User-Agent": "Modly-AnyTop/1.0"},
    )
    try:
        with opener(request, timeout=timeout) as response:  # type: ignore[operator]
            if _response_status(response) != 200:
                return None
            raw = response.read(MAX_SETTINGS_RESPONSE_BYTES + 1)
    except (OSError, TimeoutError, ValueError):
        return None
    if len(raw) > MAX_SETTINGS_RESPONSE_BYTES:
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or "models_dir" not in payload:
        return None
    # A value returned by Modly is authoritative.  An unsafe/stale value must
    # be repaired in Modly settings, never silently redirected elsewhere.
    return _native_directory(
        payload["models_dir"],
        "Modly settings models_dir",
        platform_name,
        require_existing=require_existing,
    )


def resolve_models_root(
    payload: Mapping[str, object],
    extension_dir: Path,
    platform_name: object | None = None,
    *,
    payload_keys: Sequence[str] = SETUP_MODELS_PAYLOAD_KEYS,
    environ: Mapping[str, str] | None = None,
    opener: object = urlopen,
    api_timeout: float = 1.5,
    require_existing: bool = True,
) -> Path:
    """Resolve models_dir in payload, environment, API, sibling order."""

    system = normalize_platform_name(platform_name or current_platform_name())
    explicit = _candidate_from_mapping(
        payload, payload_keys, system, require_existing=require_existing
    )
    if explicit is not None:
        return explicit

    environment: Mapping[str, object] = os.environ if environ is None else environ
    modly_env = _candidate_from_mapping(
        environment, ("MODLY_MODELS_DIR",), system, require_existing=require_existing
    )
    if modly_env is not None:
        return modly_env

    # MODELS_DIR is authoritative ahead of the API only when Modly exported
    # its related storage boundaries too.  A generic shell MODELS_DIR remains
    # a lower-priority manual fallback.
    if "MODELS_DIR" in environment and (
        "WORKSPACE_DIR" in environment or "EXTENSIONS_DIR" in environment
    ):
        host_env = _candidate_from_mapping(
            environment, ("MODELS_DIR",), system, require_existing=require_existing
        )
        if host_env is not None:
            return host_env

    api_value = _models_root_from_modly_api(
        system, require_existing=require_existing, opener=opener, timeout=api_timeout
    )
    if api_value is not None:
        return api_value

    manual = _candidate_from_mapping(
        environment, ("MODELS_DIR",), system, require_existing=require_existing
    )
    if manual is not None:
        return manual

    extension = extension_dir.absolute() if not extension_dir.is_absolute() else extension_dir
    extensions_root = extension.parent
    if extensions_root.name.casefold() != "extensions":
        raise PathContractError(
            "PATH_MODELS_UNAVAILABLE",
            "Modly's models directory could not be discovered; keep Modly running or set MODLY_MODELS_DIR",
        )
    inferred = extensions_root.parent / "models"
    try:
        return _native_directory(
            str(inferred),
            "conventional Modly models directory",
            system,
            require_existing=require_existing,
        )
    except PathContractError as exc:
        if exc.code != "PATH_DIRECTORY_MISSING":
            raise
        raise PathContractError(
            "PATH_MODELS_UNAVAILABLE",
            "the conventional Modly models directory is unavailable; run Repair with Modly open",
        ) from exc


def _ensure_directory_beneath(root: Path, parts: Sequence[str], *, create: bool) -> Path:
    if not root.is_dir():
        raise PathContractError("PATH_MODELS_MISSING", "the configured models directory is unavailable")
    try:
        canonical_root = root.resolve(strict=True)
    except OSError as exc:
        raise PathContractError("PATH_MODELS_INVALID", "the models directory cannot be verified") from exc
    current = root
    for part in parts:
        pure = PurePosixPath(part)
        if part in {"", ".", ".."} or pure.is_absolute() or len(pure.parts) != 1:
            raise PathContractError("PATH_RELATIVE_INVALID", "an owned model path is unsafe")
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if not create:
                raise PathContractError("PATH_OWNED_MISSING", "the owned model snapshot is unavailable")
            try:
                current.mkdir()
                info = current.lstat()
            except OSError as exc:
                raise PathContractError(
                    "PATH_OWNED_CREATE_FAILED", "an owned model directory could not be created"
                ) from exc
        except OSError as exc:
            raise PathContractError("PATH_OWNED_INVALID", "an owned model path cannot be inspected") from exc
        if _is_alias(info) or not stat.S_ISDIR(info.st_mode):
            raise PathContractError("PATH_OWNED_ALIAS", "an owned model path must be a regular directory")
        try:
            resolved = current.resolve(strict=True)
        except OSError as exc:
            raise PathContractError("PATH_OWNED_INVALID", "an owned model path cannot be verified") from exc
        if canonical_root not in resolved.parents:
            raise PathContractError("PATH_OWNED_ESCAPE", "an owned model path escapes models_dir")
    return current


def owned_snapshot_directory(models_root: Path, *, create: bool) -> Path:
    return _ensure_directory_beneath(
        models_root,
        (EXTENSION_ID, "anytop", "revisions", REVISION_ID),
        create=create,
    )


def safe_snapshot_directory(snapshot_dir: Path, relative_path: str, *, create: bool) -> Path:
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "\\" in relative_path
    ):
        raise PathContractError("PATH_RELATIVE_INVALID", "an owned directory path is unsafe")
    return _ensure_directory_beneath(snapshot_dir, tuple(pure.parts), create=create)


def safe_snapshot_file(snapshot_dir: Path, relative_path: str, *, create_parent: bool) -> Path:
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "\\" in relative_path
    ):
        raise PathContractError("PATH_RELATIVE_INVALID", "an asset path is unsafe")
    parent = snapshot_dir
    if pure.parts[:-1]:
        parent = _ensure_directory_beneath(
            snapshot_dir, tuple(pure.parts[:-1]), create=create_parent
        )
    candidate = parent / pure.name
    if candidate.exists() or candidate.is_symlink():
        info = candidate.lstat()
        if _is_alias(info) or not stat.S_ISREG(info.st_mode):
            raise PathContractError("PATH_ASSET_INVALID", "an asset path is not a regular file")
    try:
        canonical_root = snapshot_dir.resolve(strict=True)
        resolved_parent = candidate.parent.resolve(strict=True)
    except OSError as exc:
        raise PathContractError("PATH_ASSET_INVALID", "an asset path cannot be verified") from exc
    if resolved_parent != canonical_root and canonical_root not in resolved_parent.parents:
        raise PathContractError("PATH_ASSET_ESCAPE", "an asset path escapes the snapshot")
    return candidate


def snapshot_paths(snapshot_dir: Path) -> SnapshotPaths:
    return SnapshotPaths(
        root=snapshot_dir,
        anytop_source=snapshot_dir.joinpath(*PurePosixPath(ANYTOP_SOURCE_RELATIVE).parts),
        motion_source=snapshot_dir.joinpath(*PurePosixPath(MOTION_SOURCE_RELATIVE).parts),
        checkpoints=snapshot_dir / "checkpoints",
        builtin_cond=snapshot_dir.joinpath(*PurePosixPath(BUILTIN_COND_RELATIVE).parts),
        t5=snapshot_dir.joinpath(*PurePosixPath(T5_SOURCE_RELATIVE).parts),
        ready_marker=snapshot_dir / READY_MARKER_FILENAME,
    )

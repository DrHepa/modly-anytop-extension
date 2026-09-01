"""Strict Modly PROCESS protocol and AnyTop workflow adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
from typing import Any, Mapping, TextIO
import uuid

from .bundles import (
    BUNDLE_SCHEMA_VERSION,
    BundleError,
    VerifiedBundle,
    atomic_json,
    file_records,
    load_bundle_auth_key,
    require_directory,
    require_regular_file,
    verify_bundle,
    write_bundle,
)
from .constants import (
    ASSETS,
    BUILTIN_COND_RELATIVE,
    BUILTIN_SKELETONS,
    CHECKPOINTS,
    EXTENSION_ID,
    EXTENSION_VERSION,
    NODE_IDS,
    READY_MARKER_FILENAME,
    REVISION_ID,
    RUNTIME_CONFIG_FILENAME,
    SPECIALIZED_FAMILY,
)
from .paths import snapshot_paths


ROOT = Path(__file__).resolve().parents[1]
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_CONFIG_BYTES = 1024 * 1024
MAX_READY_BYTES = 4 * 1024 * 1024
MAX_INPUT_BYTES = 2 * 1024 * 1024 * 1024
MAX_BVH_FILES = 4096
MAX_WORKER_DIAGNOSTIC_BYTES = 16 * 1024
MAX_WORKER_DIAGNOSTIC_CHARS = 2000
MODEL_FAMILIES = frozenset(CHECKPOINTS)
DEVICE_MODES = frozenset({"auto", "cpu", "cuda"})
OBJECT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9-]{0,63}$")
_VERIFIED_SNAPSHOT_IDENTITIES: set[tuple[str, int, int, int, int]] = set()
_SNAPSHOT_VERIFICATION_LOCK = threading.Lock()


ERRORS: dict[str, str] = {
    "REQUEST_INVALID": "AnyTop received an invalid Modly process request. Recreate the node and try again.",
    "NODE_INVALID": "This AnyTop node is not available. Update or repair the extension.",
    "PARAM_INVALID": "One or more AnyTop parameters are invalid. Reset the node parameters and retry.",
    "INPUT_REQUIRED": "This AnyTop node needs its documented input connection or path parameter.",
    "INPUT_INVALID": "The input is not a verified, supported AnyTop artifact. Check the node input and retry.",
    "SETUP_REQUIRED": "AnyTop setup is missing or inconsistent. Run Repair for this extension and retry.",
    "ASSET_INVALID": "A pinned AnyTop model asset failed validation. Run Repair before inference.",
    "DEVICE_UNAVAILABLE": "The requested compute device is unavailable. Select Auto or CPU, or repair the CUDA environment.",
    "WORKER_FAILED": "AnyTop upstream processing failed. Check the input and parameters, then retry.",
    "OUTPUT_INVALID": "AnyTop did not produce a complete workflow artifact. Repair the extension and retry.",
    "UNEXPECTED": "AnyTop processing failed unexpectedly. Run Repair and try again.",
}


class ProcessFailure(RuntimeError):
    def __init__(self, code: str, *, diagnostic: str | None = None) -> None:
        self.code = code if code in ERRORS else "UNEXPECTED"
        self.diagnostic = diagnostic
        super().__init__(self.code)

    def public_message(self) -> str:
        return f"[{self.code}] {ERRORS[self.code]}"


class ProtocolEmitter:
    """Emit complete NDJSON records, monotonic progress, and one terminal."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.last_progress = -1
        self.terminal = False
        self.failed = False

    def _write(self, value: Mapping[str, object], *, terminal: bool = False) -> None:
        if self.terminal or self.failed:
            raise RuntimeError("protocol channel is closed")
        if terminal:
            self.terminal = True
        try:
            line = json.dumps(value, ensure_ascii=True, separators=(",", ":")) + "\n"
            written = self.stream.write(line)
            if not isinstance(written, int) or written != len(line):
                raise OSError("short protocol write")
            self.stream.flush()
        except BaseException:
            self.failed = True
            raise

    def progress(self, percent: int, label: str) -> None:
        value = max(self.last_progress, min(100, max(0, int(percent))))
        self.last_progress = value
        self._write({"type": "progress", "percent": value, "label": str(label)[:200]})

    def log(self, message: str) -> None:
        self._write({"type": "log", "message": str(message)[:MAX_WORKER_DIAGNOSTIC_CHARS]})

    def done_file(self, path: Path) -> None:
        self._write({"type": "done", "result": {"filePath": str(path)}}, terminal=True)

    def done_text(self, value: str) -> None:
        self._write({"type": "done", "result": {"text": value}}, terminal=True)

    def error(self, message: str) -> None:
        self._write({"type": "error", "message": message}, terminal=True)


@dataclass(frozen=True)
class RuntimeState:
    models_dir: Path
    revision_root: Path
    source_root: Path
    motion_source: Path
    checkpoints_root: Path
    builtin_cond: Path
    t5_path: Path
    ready_marker: Path
    bundle_auth_key: bytes
    available_devices: frozenset[str]
    default_device: str


@dataclass(frozen=True)
class Request:
    node_id: str
    input_path: Path | None
    text: str | None
    params: Mapping[str, Any]
    workspace: Path
    temp: Path
    state: RuntimeState


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _read_json_file(path: Path, maximum: int) -> Mapping[str, Any]:
    try:
        checked = require_regular_file(path, max_bytes=maximum)
        value = json.loads(checked.read_text(encoding="utf-8"))
    except (BundleError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProcessFailure("SETUP_REQUIRED") from exc
    if not isinstance(value, dict):
        raise ProcessFailure("SETUP_REQUIRED")
    return value


def _config_path(value: object, key: str, *, directory: bool) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ProcessFailure("SETUP_REQUIRED")
    path = Path(value)
    if not path.is_absolute():
        raise ProcessFailure("SETUP_REQUIRED")
    try:
        return require_directory(path) if directory else require_regular_file(path)
    except BundleError as exc:
        raise ProcessFailure("SETUP_REQUIRED") from exc


def _owned_revision_root(models_dir: Path) -> Path:
    """Walk the immutable owned suffix without accepting links/junctions."""

    current = models_dir
    for part in (EXTENSION_ID, "anytop", "revisions", REVISION_ID):
        candidate = current / part
        try:
            checked = require_directory(candidate)
        except BundleError as exc:
            raise ProcessFailure("SETUP_REQUIRED") from exc
        # ``current`` is already canonical.  Inequality means an ancestor or
        # the candidate itself resolved through an alias/reparse boundary.
        if checked != candidate:
            raise ProcessFailure("SETUP_REQUIRED")
        current = checked
    return current


def _canonical_snapshot_path(
    revision_root: Path,
    expected: Path,
    *,
    directory: bool,
) -> Path:
    """Resolve an expected snapshot path while checking every component."""

    try:
        relative = expected.relative_to(revision_root)
    except ValueError as exc:
        raise ProcessFailure("SETUP_REQUIRED") from exc
    if not relative.parts:
        return revision_root
    current = revision_root
    for index, part in enumerate(relative.parts):
        candidate = current / part
        final = index == len(relative.parts) - 1
        try:
            checked = (
                require_directory(candidate)
                if not final or directory
                else require_regular_file(candidate)
            )
        except BundleError as exc:
            raise ProcessFailure("SETUP_REQUIRED") from exc
        if checked != candidate:
            raise ProcessFailure("SETUP_REQUIRED")
        current = checked
    return current


def _verify_snapshot_once(revision_root: Path, ready_marker: Path) -> None:
    """Fully hash a configured immutable snapshot once per processor process.

    The ready marker makes normal state loading cheap, while this first-use
    audit also covers the patched source trees and the large offline T5 files.
    The marker's filesystem identity is part of the cache key, so replacing it
    within a long-lived process triggers a fresh audit.
    """

    try:
        info = ready_marker.lstat()
        identity = (
            str(revision_root.resolve(strict=True)),
            int(info.st_dev),
            int(info.st_ino),
            int(info.st_size),
            int(getattr(info, "st_mtime_ns", 0)),
        )
    except OSError as exc:
        raise ProcessFailure("SETUP_REQUIRED") from exc
    with _SNAPSHOT_VERIFICATION_LOCK:
        if identity in _VERIFIED_SNAPSHOT_IDENTITIES:
            return
        try:
            from .assets import verify_snapshot

            failures = verify_snapshot(revision_root, require_ready=True)
        except BaseException as exc:
            raise ProcessFailure("ASSET_INVALID") from exc
        if failures:
            raise ProcessFailure("ASSET_INVALID")
        _VERIFIED_SNAPSHOT_IDENTITIES.add(identity)


def load_state(config_path: Path | None = None) -> RuntimeState:
    config = _read_json_file(config_path or (ROOT / RUNTIME_CONFIG_FILENAME), MAX_CONFIG_BYTES)
    if config.get("extension_id") != EXTENSION_ID or config.get("revision_id") != REVISION_ID:
        raise ProcessFailure("SETUP_REQUIRED")
    models_dir = _config_path(config.get("models_dir"), "models_dir", directory=True)
    revision_root = _config_path(config.get("revision_root"), "revision_root", directory=True)
    source_root = _config_path(config.get("source_root"), "source_root", directory=True)
    motion_source = _config_path(config.get("motion_source"), "motion_source", directory=True)
    checkpoints_root = _config_path(config.get("checkpoints_root"), "checkpoints_root", directory=True)
    builtin_cond = _config_path(config.get("builtin_cond"), "builtin_cond", directory=False)
    t5_path = _config_path(config.get("t5_path"), "t5_path", directory=True)
    ready_marker = _config_path(config.get("ready_marker"), "ready_marker", directory=False)
    runtime_cache = _config_path(config.get("runtime_cache_dir"), "runtime_cache_dir", directory=True)
    canonical_revision = _owned_revision_root(models_dir)
    if revision_root != canonical_revision:
        raise ProcessFailure("SETUP_REQUIRED")
    paths = snapshot_paths(revision_root)
    expected_paths = {
        "source_root": _canonical_snapshot_path(
            revision_root, paths.anytop_source, directory=True
        ),
        "motion_source": _canonical_snapshot_path(
            revision_root, paths.motion_source, directory=True
        ),
        "checkpoints_root": _canonical_snapshot_path(
            revision_root, paths.checkpoints, directory=True
        ),
        "builtin_cond": _canonical_snapshot_path(
            revision_root, paths.builtin_cond, directory=False
        ),
        "t5_path": _canonical_snapshot_path(revision_root, paths.t5, directory=True),
        "ready_marker": _canonical_snapshot_path(
            revision_root, paths.ready_marker, directory=False
        ),
        "runtime_cache": _canonical_snapshot_path(
            revision_root, revision_root / "runtime-cache", directory=True
        ),
    }
    configured_paths = {
        "source_root": source_root,
        "motion_source": motion_source,
        "checkpoints_root": checkpoints_root,
        "builtin_cond": builtin_cond,
        "t5_path": t5_path,
        "ready_marker": ready_marker,
        "runtime_cache": runtime_cache,
    }
    if configured_paths != expected_paths:
        raise ProcessFailure("SETUP_REQUIRED")
    if ready_marker.name != READY_MARKER_FILENAME:
        raise ProcessFailure("SETUP_REQUIRED")
    ready = _read_json_file(ready_marker, MAX_READY_BYTES)
    try:
        from .assets import ready_payload

        expected_ready = ready_payload()
    except BaseException as exc:
        raise ProcessFailure("SETUP_REQUIRED") from exc
    if ready != expected_ready:
        raise ProcessFailure("SETUP_REQUIRED")
    if ready.get("extension_id") != EXTENSION_ID or ready.get("revision_id") != REVISION_ID:
        raise ProcessFailure("SETUP_REQUIRED")
    _verify_snapshot_once(revision_root, ready_marker)
    try:
        bundle_auth_key = load_bundle_auth_key(runtime_cache)
    except BundleError as exc:
        raise ProcessFailure("SETUP_REQUIRED") from exc

    available_raw = config.get("available_devices")
    if not isinstance(available_raw, list) or not available_raw:
        raise ProcessFailure("SETUP_REQUIRED")
    available = frozenset(item for item in available_raw if isinstance(item, str))
    if "cpu" not in available or not available.issubset({"cpu", "cuda"}):
        raise ProcessFailure("SETUP_REQUIRED")
    default = config.get("default_device")
    if default not in {"auto", "cpu", "cuda"}:
        raise ProcessFailure("SETUP_REQUIRED")
    if default == "cuda" and "cuda" not in available:
        raise ProcessFailure("SETUP_REQUIRED")
    return RuntimeState(
        models_dir=models_dir,
        revision_root=revision_root,
        source_root=source_root,
        motion_source=motion_source,
        checkpoints_root=checkpoints_root,
        builtin_cond=builtin_cond,
        t5_path=t5_path,
        ready_marker=ready_marker,
        bundle_auth_key=bundle_auth_key,
        available_devices=available,
        default_device=str(default),
    )


def _payload_directory(value: object) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ProcessFailure("REQUEST_INVALID")
    path = Path(value)
    if not path.is_absolute():
        raise ProcessFailure("REQUEST_INVALID")
    try:
        return require_directory(path)
    except BundleError as exc:
        raise ProcessFailure("REQUEST_INVALID") from exc


def validate_request(payload: object, *, state_loader: Any = load_state) -> Request:
    if not isinstance(payload, dict):
        raise ProcessFailure("REQUEST_INVALID")
    input_value = payload.get("input")
    params = payload.get("params")
    if not isinstance(input_value, dict) or not isinstance(params, dict):
        raise ProcessFailure("REQUEST_INVALID")
    node_id = payload.get("nodeId") or input_value.get("nodeId")
    if not isinstance(node_id, str) or node_id not in NODE_IDS:
        raise ProcessFailure("NODE_INVALID")
    text = input_value.get("text")
    if text is not None and not isinstance(text, str):
        raise ProcessFailure("REQUEST_INVALID")
    input_path: Path | None = None
    raw_path = input_value.get("filePath")
    if raw_path is not None:
        if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
            raise ProcessFailure("REQUEST_INVALID")
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            raise ProcessFailure("REQUEST_INVALID")
        try:
            input_path = require_regular_file(candidate, max_bytes=MAX_INPUT_BYTES)
        except BundleError as exc:
            raise ProcessFailure("INPUT_INVALID") from exc
    workspace = _payload_directory(payload.get("workspaceDir"))
    temp = _payload_directory(payload.get("tempDir"))
    return Request(
        node_id=node_id,
        input_path=input_path,
        text=text,
        params=params,
        workspace=workspace,
        temp=temp,
        state=state_loader(),
    )


def _param(params: Mapping[str, Any], key: str, default: Any) -> Any:
    return params[key] if key in params else default


def _select(params: Mapping[str, Any], key: str, default: str, choices: frozenset[str]) -> str:
    value = _param(params, key, default)
    if not isinstance(value, str) or value not in choices:
        raise ProcessFailure("PARAM_INVALID")
    return value


def _integer(params: Mapping[str, Any], key: str, default: int, minimum: int, maximum: int) -> int:
    value = _param(params, key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ProcessFailure("PARAM_INVALID")
    return value


def _number(params: Mapping[str, Any], key: str, default: float, minimum: float, maximum: float) -> float:
    value = _param(params, key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProcessFailure("PARAM_INVALID")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ProcessFailure("PARAM_INVALID")
    return result


def _string(params: Mapping[str, Any], key: str, default: str = "", maximum: int = 4096) -> str:
    value = _param(params, key, default)
    if not isinstance(value, str) or "\x00" in value or len(value) > maximum:
        raise ProcessFailure("PARAM_INVALID")
    return value.strip()


def _device_params(request: Request) -> dict[str, object]:
    mode = _select(request.params, "device_mode", "auto", DEVICE_MODES)
    index = _integer(request.params, "cuda_device", 0, 0, 31)
    if mode == "cuda" and "cuda" not in request.state.available_devices:
        raise ProcessFailure("DEVICE_UNAVAILABLE")
    return {"device_mode": mode, "cuda_device": index}


def _family(params: Mapping[str, Any], object_name: str | None, *, correspondence: bool = False) -> str:
    selected = _select(params, "model_family", "auto", frozenset({"auto", *MODEL_FAMILIES}))
    if selected != "auto":
        return selected
    if correspondence or object_name is None:
        return "unified"
    return SPECIALIZED_FAMILY.get(object_name, "unified")


def _asset(relative: str) -> Any:
    for spec in ASSETS:
        if spec.relative_path == relative:
            return spec
    raise ProcessFailure("SETUP_REQUIRED")


def _checkpoint(state: RuntimeState, family: str) -> Path:
    metadata = CHECKPOINTS[family]
    relative = f"checkpoints/{metadata['directory']}/{metadata['filename']}"
    args_relative = f"checkpoints/{metadata['directory']}/{metadata['args']}"
    checkpoint = state.checkpoints_root / str(metadata["directory"]) / str(metadata["filename"])
    args = checkpoint.with_name(str(metadata["args"]))
    from .assets import verify_asset

    checkpoint_valid, _ = verify_asset(checkpoint, _asset(relative))
    args_valid, _ = verify_asset(args, _asset(args_relative))
    if not checkpoint_valid or not args_valid:
        raise ProcessFailure("ASSET_INVALID")
    return checkpoint.resolve()


def _builtin_condition(state: RuntimeState) -> Path:
    from .assets import verify_asset

    valid, _ = verify_asset(state.builtin_cond, _asset(BUILTIN_COND_RELATIVE))
    if not valid:
        raise ProcessFailure("ASSET_INVALID")
    return state.builtin_cond.resolve()


def _upper_body_roots(params: Mapping[str, Any]) -> list[int]:
    value = _param(params, "upper_body_root", "0")
    if isinstance(value, int) and not isinstance(value, bool):
        values = [value]
    elif isinstance(value, str):
        parts = [part for part in re.split(r"[\s,]+", value.strip()) if part]
        try:
            values = [int(part, 10) for part in parts]
        except ValueError as exc:
            raise ProcessFailure("PARAM_INVALID") from exc
    elif isinstance(value, list) and all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        values = list(value)
    else:
        raise ProcessFailure("PARAM_INVALID")
    if not values or len(values) > 64 or len(set(values)) != len(values):
        raise ProcessFailure("PARAM_INVALID")
    if any(item < 0 or item > 1023 for item in values):
        raise ProcessFailure("PARAM_INVALID")
    return values


def _edit_controls(params: Mapping[str, Any]) -> tuple[str, float, float, list[int]]:
    mode = _select(params, "edit_mode", "in_between", frozenset({"in_between", "upper_body"}))
    if mode == "in_between":
        prefix = _number(params, "prefix_end", 0.25, 0.0, 1.0)
        suffix = _number(params, "suffix_start", 0.75, 0.0, 1.0)
        if prefix >= suffix:
            raise ProcessFailure("PARAM_INVALID")
        return mode, prefix, suffix, [0]
    # Hidden in-between controls may retain stale UI state; upstream does not
    # read them in upper-body mode.  The converse applies to upper_body_root.
    return mode, 0.25, 0.75, _upper_body_roots(params)


def _safe_user_directory(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ProcessFailure("PARAM_INVALID")
    try:
        return require_directory(path)
    except BundleError as exc:
        raise ProcessFailure("INPUT_INVALID") from exc


def _preprocess_inputs(request: Request) -> tuple[str, Path, list[str], Path | None]:
    object_name = (request.text or "").strip()
    if not OBJECT_NAME.fullmatch(object_name):
        raise ProcessFailure("INPUT_REQUIRED" if not object_name else "PARAM_INVALID")
    directory_value = _string(request.params, "bvh_directory")
    if not directory_value:
        raise ProcessFailure("INPUT_REQUIRED")
    directory = _safe_user_directory(directory_value)
    bvh_files = []
    try:
        for child in directory.iterdir():
            if child.suffix.lower() == ".bvh":
                bvh_files.append(require_regular_file(child, max_bytes=MAX_INPUT_BYTES))
    except (OSError, BundleError) as exc:
        raise ProcessFailure("INPUT_INVALID") from exc
    if len(bvh_files) < 2 or len(bvh_files) > MAX_BVH_FILES:
        raise ProcessFailure("INPUT_INVALID")
    joints = [
        _string(request.params, "right_hip", "RLeg1", 128),
        _string(request.params, "left_hip", "LLeg1", 128),
        _string(request.params, "right_shoulder", "RArm1", 128),
        _string(request.params, "left_shoulder", "LArm1", 128),
    ]
    if any(not joint for joint in joints) or len(set(joints)) != 4:
        raise ProcessFailure("PARAM_INVALID")
    tpos_value = _string(request.params, "tpos_bvh", "", 4096)
    tpos: Path | None = None
    if tpos_value:
        raw = Path(tpos_value)
        candidate = raw if raw.is_absolute() else directory / raw
        try:
            tpos = require_regular_file(candidate, max_bytes=MAX_INPUT_BYTES)
        except BundleError as exc:
            raise ProcessFailure("INPUT_INVALID") from exc
        if not _inside(tpos, directory) or tpos.suffix.lower() != ".bvh":
            raise ProcessFailure("INPUT_INVALID")
    return object_name, directory, joints, tpos


def _common_inference(request: Request, object_name: str | None, *, correspondence: bool = False) -> tuple[str, dict[str, object]]:
    family = _family(request.params, object_name, correspondence=correspondence)
    values: dict[str, object] = {
        "model_family": family,
        "seed": _integer(request.params, "seed", 10, 0, 4_294_967_295),
        **_device_params(request),
    }
    return family, values


def _run_name(operation: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{operation}-{stamp}-{uuid.uuid4().hex[:10]}"


@dataclass
class OutputRun:
    staging: Path
    final: Path

    @classmethod
    def create(cls, workspace: Path, operation: str) -> "OutputRun":
        workflows = workspace / "Workflows"
        root = workflows / "AnyTop"
        try:
            workflows.mkdir(exist_ok=True)
            root.mkdir(exist_ok=True)
            workflows_checked = require_directory(workflows)
            root_checked = require_directory(root)
        except (OSError, BundleError) as exc:
            raise ProcessFailure("OUTPUT_INVALID") from exc
        if not _inside(root_checked, workspace) or not _inside(root_checked, workflows_checked):
            raise ProcessFailure("OUTPUT_INVALID")
        name = _run_name(operation)
        final = root_checked / name
        staging = root_checked / f".{name}.partial"
        try:
            staging.mkdir()
        except OSError as exc:
            raise ProcessFailure("OUTPUT_INVALID") from exc
        return cls(staging=staging, final=final)

    def commit(self) -> Path:
        if self.final.exists() or not self.staging.is_dir():
            raise ProcessFailure("OUTPUT_INVALID")
        try:
            self.staging.rename(self.final)
        except OSError as exc:
            raise ProcessFailure("OUTPUT_INVALID") from exc
        return self.final

    def cleanup(self) -> None:
        try:
            root = self.staging.parent.resolve(strict=True)
            candidate = self.staging.resolve(strict=True)
            if candidate.parent == root and candidate.name.startswith(".") and candidate.name.endswith(".partial"):
                shutil.rmtree(candidate)
        except (OSError, ValueError):
            pass


def _worker_environment() -> dict[str, str]:
    names = {
        "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT",
        # NVIDIA container/SBSA hosts commonly expose libcuda.so and companion
        # runtime libraries only through these loader/search paths.
        "LD_LIBRARY_PATH", "LIBRARY_PATH",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LANG", "LC_ALL",
        "LC_CTYPE", "TZ", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
    }
    environment = {key: value for key, value in os.environ.items() if key in names or key.startswith(("CUDA_", "NVIDIA_", "ROCR_", "HIP_", "HSA_"))}
    environment.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "MPLBACKEND": "Agg",
        }
    )
    return environment


def _sanitized_worker_tail(path: Path) -> str | None:
    """Return a bounded, path/credential-redacted stderr tail for Modly logs."""

    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - MAX_WORKER_DIAGNOSTIC_BYTES))
            text = handle.read(MAX_WORKER_DIAGNOSTIC_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return None
    if not text.strip():
        return None
    text = re.sub(r'File\s+"[^"]+"', 'File "<path>"', text)
    text = re.sub(
        r"(?i)(?:[A-Z]:[\\/]|/)[^\r\n|]*[\\/][^\r\n|]*",
        "<path>",
        text,
    )
    text = re.sub(r"(?i)\b(?:[A-Z]:[\\/]|/)(?:[^\s:'\"]+[\\/])+[^\s:'\"]*", "<path>", text)
    text = re.sub(
        r"(?i)\b(token|secret|password|passwd|authorization|api[_-]?key)\b\s*[:=]\s*[^\s,;]+",
        lambda match: f"{match.group(1)}=<redacted>",
        text,
    )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sanitized = " | ".join(lines)
    if len(sanitized) > MAX_WORKER_DIAGNOSTIC_CHARS:
        sanitized = sanitized[-MAX_WORKER_DIAGNOSTIC_CHARS:]
    return f"AnyTop worker diagnostic: {sanitized}" if sanitized else None


def run_worker(request: dict[str, object], staging: Path, temp: Path) -> Mapping[str, Any]:
    result_path = staging / ".worker-result.json"
    request = {**request, "result_path": str(result_path)}
    log_root = temp / f"anytop-{uuid.uuid4().hex}"
    try:
        log_root.mkdir()
        stdout_path = log_root / "stdout.log"
        stderr_path = log_root / "stderr.log"
        creationflags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) if os.name == "nt" else 0
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                [sys.executable, "-B", "-m", "anytop_modly.worker"],
                cwd=ROOT,
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                env=_worker_environment(),
                creationflags=creationflags,
                start_new_session=os.name != "nt",
            )
            payload = (json.dumps(request, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")
            try:
                process.communicate(payload)
            except BaseException:
                try:
                    if os.name == "nt":
                        process.terminate()
                    else:
                        os.killpg(process.pid, signal.SIGTERM)
                except OSError:
                    pass
                raise
        if process.returncode != 0:
            raise ProcessFailure(
                "WORKER_FAILED",
                diagnostic=_sanitized_worker_tail(stderr_path),
            )
        try:
            value = json.loads(require_regular_file(result_path, max_bytes=MAX_CONFIG_BYTES).read_text(encoding="utf-8"))
        except (BundleError, OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProcessFailure("WORKER_FAILED") from exc
        if not isinstance(value, dict) or value.get("ok") is not True or not isinstance(value.get("result"), dict):
            raise ProcessFailure("WORKER_FAILED")
        result_path.unlink()
        return value["result"]
    finally:
        try:
            shutil.rmtree(log_root)
        except OSError:
            pass


def _worker_file(value: object, staging: Path) -> Path:
    if not isinstance(value, str):
        raise ProcessFailure("OUTPUT_INVALID")
    try:
        path = require_regular_file(Path(value), max_bytes=MAX_INPUT_BYTES)
    except BundleError as exc:
        raise ProcessFailure("OUTPUT_INVALID") from exc
    if not _inside(path, staging):
        raise ProcessFailure("OUTPUT_INVALID")
    return path


def _worker_files(value: object, staging: Path) -> list[Path]:
    if not isinstance(value, list):
        raise ProcessFailure("OUTPUT_INVALID")
    return [_worker_file(item, staging) for item in value]


def _copy(path: Path, destination: Path) -> Path:
    try:
        checked = require_regular_file(path, max_bytes=MAX_INPUT_BYTES)
        shutil.copyfile(checked, destination)
        return require_regular_file(destination, max_bytes=MAX_INPUT_BYTES)
    except (BundleError, OSError) as exc:
        raise ProcessFailure("OUTPUT_INVALID") from exc


def _condition_for_bundle(bundle: VerifiedBundle, state: RuntimeState) -> Path:
    if bundle.condition_kind == "custom":
        if bundle.condition is None:
            raise ProcessFailure("INPUT_INVALID")
        return bundle.condition
    return _builtin_condition(state)


def _package_motion_bundle(
    *,
    run: OutputRun,
    result: Mapping[str, Any],
    operation: str,
    condition_kind: str,
    condition_source: Path | None,
    parameters: Mapping[str, object],
    provenance: Mapping[str, object],
    motion_source: Path,
    authentication_key: bytes,
) -> Path:
    object_name = result.get("object_name")
    items = result.get("items")
    if not isinstance(object_name, str) or not isinstance(items, list) or not items:
        raise ProcessFailure("OUTPUT_INVALID")
    bundle_files: dict[str, Path] = {}
    first_bvh: Path | None = None
    first_motion: Path | None = None
    first_video: Path | None = None
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ProcessFailure("OUTPUT_INVALID")
        motion = _copy(_worker_file(item.get("motion"), run.staging), run.staging / f"motion-{index:02d}.npy")
        bvh = _copy(_worker_file(item.get("bvh"), run.staging), run.staging / f"motion-{index:02d}.bvh")
        video = _copy(_worker_file(item.get("video"), run.staging), run.staging / f"motion-{index:02d}.mp4")
        suffix = "" if index == 0 else f"_{index:02d}"
        bundle_files[f"motion{suffix}"] = motion
        bundle_files[f"bvh{suffix}"] = bvh
        bundle_files[f"video{suffix}"] = video
        upstream_edit = item.get("upstream_edit")
        if upstream_edit is not None:
            bundle_files[f"upstream_edit_{index:02d}"] = _copy(
                _worker_file(upstream_edit, run.staging),
                run.staging / f"motion-{index:02d}.upstream-edit.npy",
            )
        if index == 0:
            first_motion, first_bvh, first_video = motion, bvh, video
    if first_bvh is None or first_motion is None or first_video is None:
        raise ProcessFailure("OUTPUT_INVALID")
    condition: Path | None = None
    if condition_kind == "custom":
        if condition_source is None:
            raise ProcessFailure("OUTPUT_INVALID")
        condition = _copy(condition_source, run.staging / "condition.npy")
    preview = run.staging / "motion.glb"
    try:
        from .glb import bvh_to_glb

        bvh_to_glb(
            first_bvh,
            preview,
            extras={
                "schemaVersion": BUNDLE_SCHEMA_VERSION,
                "manifest": "motion.anytop.json",
                "canonicalMotion": first_bvh.name,
                "features": first_motion.name,
                "previewVideo": first_video.name,
                "objectName": object_name,
                "operation": operation,
            },
            motion_source=motion_source,
        )
        require_regular_file(preview)
    except (OSError, ValueError, BundleError) as exc:
        raise ProcessFailure("OUTPUT_INVALID") from exc
    try:
        write_bundle(
            preview=preview,
            operation=operation,
            object_name=object_name,
            condition_kind=condition_kind,
            condition=condition,
            files=bundle_files,
            parameters=parameters,
            provenance=provenance,
            authentication_key=authentication_key,
        )
    except BundleError as exc:
        raise ProcessFailure("OUTPUT_INVALID") from exc
    upstream = run.staging / "upstream"
    if upstream.exists() and upstream.is_dir() and not upstream.is_symlink():
        shutil.rmtree(upstream)
    if any(run.staging.rglob("model*.pt")):
        raise ProcessFailure("OUTPUT_INVALID")
    final = run.commit()
    return final / preview.name


def _operation_preprocess(request: Request, emitter: ProtocolEmitter) -> Path:
    object_name, directory, joints, tpos = _preprocess_inputs(request)
    run = OutputRun.create(request.workspace, "preprocess")
    try:
        emitter.progress(15, "Preprocessing skeleton")
        worker_request: dict[str, object] = {
            "operation": "preprocess",
            "source_root": str(request.state.source_root),
            "motion_source": str(request.state.motion_source),
            "t5_path": str(request.state.t5_path),
            "output_dir": str(run.staging / "upstream"),
            "object_name": object_name,
            "bvh_directory": str(directory),
            "face_joints": joints,
            "tpos_bvh": str(tpos) if tpos else "",
        }
        result = run_worker(worker_request, run.staging, request.temp)
        condition_source = _worker_file(result.get("condition"), run.staging)
        items = result.get("items")
        if not isinstance(items, list) or not items:
            raise ProcessFailure("OUTPUT_INVALID")
        emitter.progress(85, "Building AnyTop bundle")
        return _package_motion_bundle(
            run=run,
            result={"object_name": object_name, "items": items},
            operation="preprocess",
            condition_kind="custom",
            condition_source=condition_source,
            parameters={
                "objectName": object_name,
                "faceJoints": joints,
                "tposBvh": tpos.name if tpos else None,
                "sourceBvhCount": len(
                    [entry for entry in directory.iterdir() if entry.suffix.lower() == ".bvh"]
                ),
            },
            provenance={"upstream": "AnyTop process_new_skeleton"},
            motion_source=request.state.motion_source,
            authentication_key=request.state.bundle_auth_key,
        )
    except BaseException:
        run.cleanup()
        raise


def _operation_generate(request: Request, emitter: ProtocolEmitter, *, custom: bool) -> Path:
    condition_kind = "custom" if custom else "builtin"
    bundle: VerifiedBundle | None = None
    if custom:
        if request.input_path is None:
            raise ProcessFailure("INPUT_REQUIRED")
        try:
            bundle = verify_bundle(
                request.input_path,
                authentication_key=request.state.bundle_auth_key,
            )
        except BundleError as exc:
            raise ProcessFailure("INPUT_INVALID") from exc
        if bundle.condition_kind != "custom" or bundle.condition is None:
            raise ProcessFailure("INPUT_INVALID")
        object_name = bundle.object_name
        condition = bundle.condition
    else:
        object_name = (request.text or "").strip()
        if not object_name:
            raise ProcessFailure("INPUT_REQUIRED")
        if object_name not in BUILTIN_SKELETONS:
            raise ProcessFailure("INPUT_INVALID")
        condition = _builtin_condition(request.state)
    family, common = _common_inference(request, object_name)
    params = {
        **common,
        "motion_length": _number(request.params, "motion_length", 6.0, 0.1, 9.8),
        "num_repetitions": _integer(request.params, "num_repetitions", 3, 1, 16),
    }
    checkpoint = _checkpoint(request.state, family)
    operation = "generate-custom" if custom else "generate"
    run = OutputRun.create(request.workspace, operation)
    try:
        emitter.progress(15, "Generating AnyTop motion")
        result = run_worker(
            {
                "operation": "generate_custom" if custom else "generate",
                "source_root": str(request.state.source_root),
                "motion_source": str(request.state.motion_source),
                "t5_path": str(request.state.t5_path),
                "output_dir": str(run.staging / "upstream"),
                "checkpoint": str(checkpoint),
                "condition": str(condition),
                "object_name": object_name,
                **params,
            },
            run.staging,
            request.temp,
        )
        emitter.progress(85, "Building AnyTop bundle")
        return _package_motion_bundle(
            run=run,
            result=result,
            operation=operation,
            condition_kind=condition_kind,
            condition_source=condition if custom else None,
            parameters=params,
            provenance={"modelFamily": family, "checkpoint": checkpoint.name},
            motion_source=request.state.motion_source,
            authentication_key=request.state.bundle_auth_key,
        )
    except BaseException:
        run.cleanup()
        raise


def _operation_edit(request: Request, emitter: ProtocolEmitter) -> Path:
    if request.input_path is None:
        raise ProcessFailure("INPUT_REQUIRED")
    try:
        bundle = verify_bundle(
            request.input_path,
            authentication_key=request.state.bundle_auth_key,
        )
    except BundleError as exc:
        raise ProcessFailure("INPUT_INVALID") from exc
    if bundle.motion is None:
        raise ProcessFailure("INPUT_INVALID")
    family, common = _common_inference(request, bundle.object_name)
    edit_mode, prefix, suffix, upper_body_root = _edit_controls(request.params)
    params = {
        **common,
        "edit_mode": edit_mode,
        "prefix_end": prefix,
        "suffix_start": suffix,
        "upper_body_root": upper_body_root,
        "num_repetitions": _integer(request.params, "num_repetitions", 3, 1, 16),
    }
    condition = _condition_for_bundle(bundle, request.state)
    checkpoint = _checkpoint(request.state, family)
    run = OutputRun.create(request.workspace, "edit")
    try:
        emitter.progress(15, "Editing AnyTop motion")
        result = run_worker(
            {
                "operation": "edit",
                "source_root": str(request.state.source_root),
                "motion_source": str(request.state.motion_source),
                "t5_path": str(request.state.t5_path),
                "output_dir": str(run.staging / "upstream"),
                "checkpoint": str(checkpoint),
                "condition": str(condition),
                "input_motion": str(bundle.motion),
                "object_name": bundle.object_name,
                **params,
            },
            run.staging,
            request.temp,
        )
        emitter.progress(85, "Building AnyTop bundle")
        return _package_motion_bundle(
            run=run,
            result=result,
            operation="edit",
            condition_kind=bundle.condition_kind,
            condition_source=condition if bundle.condition_kind == "custom" else None,
            parameters=params,
            provenance={"modelFamily": family, "checkpoint": checkpoint.name, "sourceBundle": bundle.manifest.name},
            motion_source=request.state.motion_source,
            authentication_key=request.state.bundle_auth_key,
        )
    except BaseException:
        run.cleanup()
        raise


def _bundle_param_path(value: str, workspace: Path) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else workspace / raw
    try:
        return require_regular_file(candidate, max_bytes=MAX_INPUT_BYTES)
    except BundleError as exc:
        raise ProcessFailure("INPUT_INVALID") from exc


def _operation_correspondence(request: Request, emitter: ProtocolEmitter) -> str:
    reference_value = _string(request.params, "reference_bundle", "", 4096)
    if not reference_value:
        raise ProcessFailure("INPUT_REQUIRED")
    target_path = request.input_path
    if target_path is None:
        target_alias = _string(request.params, "target_bundle", "", 4096)
        if not target_alias:
            raise ProcessFailure("INPUT_REQUIRED")
        target_path = _bundle_param_path(target_alias, request.workspace)
    try:
        reference = verify_bundle(
            _bundle_param_path(reference_value, request.workspace),
            authentication_key=request.state.bundle_auth_key,
        )
        target = verify_bundle(
            target_path,
            authentication_key=request.state.bundle_auth_key,
        )
    except BundleError as exc:
        raise ProcessFailure("INPUT_INVALID") from exc
    if reference.motion is None or target.motion is None:
        raise ProcessFailure("INPUT_INVALID")
    family, common = _common_inference(request, None, correspondence=True)
    params = {
        **common,
        "dift_type": _select(request.params, "dift_type", "spatial", frozenset({"spatial", "temporal"})),
        "layer": _integer(request.params, "layer", 0, 0, 3),
        "timestep": _integer(request.params, "timestep", 90, 0, 99),
        "num_repetitions": 1,
    }
    checkpoint = _checkpoint(request.state, family)
    run = OutputRun.create(request.workspace, "correspondence")
    try:
        emitter.progress(15, "Computing AnyTop correspondence")
        result = run_worker(
            {
                "operation": "correspondence",
                "source_root": str(request.state.source_root),
                "motion_source": str(request.state.motion_source),
                "t5_path": str(request.state.t5_path),
                "output_dir": str(run.staging / "upstream"),
                "checkpoint": str(checkpoint),
                "reference": {
                    "motion": str(reference.motion),
                    "condition": str(_condition_for_bundle(reference, request.state)),
                    "object_name": reference.object_name,
                },
                "target": {
                    "motion": str(target.motion),
                    "condition": str(_condition_for_bundle(target, request.state)),
                    "object_name": target.object_name,
                },
                **params,
            },
            run.staging,
            request.temp,
        )
        mappings = _worker_files(result.get("mappings"), run.staging)
        videos = _worker_files(result.get("videos"), run.staging)
        if not mappings or not videos:
            raise ProcessFailure("OUTPUT_INVALID")
        copied = []
        for index, path in enumerate(mappings):
            copied.append(_copy(path, run.staging / f"mapping-{index:02d}.npy"))
        for index, path in enumerate(videos):
            copied.append(_copy(path, run.staging / f"correspondence-{index:02d}.mp4"))
        upstream = run.staging / "upstream"
        if upstream.exists() and upstream.is_dir() and not upstream.is_symlink():
            shutil.rmtree(upstream)
        if any(run.staging.rglob("model*.pt")):
            raise ProcessFailure("OUTPUT_INVALID")
        manifest_name = "correspondence.json"
        document: dict[str, object] = {
            "schema": "modly-anytop-correspondence",
            "schemaVersion": 1,
            "extensionId": EXTENSION_ID,
            "revisionId": REVISION_ID,
            "operation": "correspondence",
            "manifestPath": str(run.final / manifest_name),
            "referenceObject": reference.object_name,
            "targetObject": target.object_name,
            "diftType": params["dift_type"],
            "modelFamily": family,
            "parameters": params,
            "files": file_records(copied),
        }
        atomic_json(run.staging / manifest_name, document)
        emitter.progress(90, "Saving correspondence manifest")
        run.commit()
        return json.dumps(document, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except BaseException:
        run.cleanup()
        raise


def process(request: Request, emitter: ProtocolEmitter) -> tuple[str, Path | str]:
    if request.node_id == "anytop-preprocess":
        return "file", _operation_preprocess(request, emitter)
    if request.node_id == "anytop-generate":
        return "file", _operation_generate(request, emitter, custom=False)
    if request.node_id == "anytop-generate-custom":
        return "file", _operation_generate(request, emitter, custom=True)
    if request.node_id == "anytop-edit":
        return "file", _operation_edit(request, emitter)
    if request.node_id == "anytop-correspondence":
        return "text", _operation_correspondence(request, emitter)
    raise ProcessFailure("NODE_INVALID")


def _read_request(stream: TextIO) -> object:
    line = stream.readline(MAX_REQUEST_BYTES + 1)
    if not line:
        raise ProcessFailure("REQUEST_INVALID")
    if len(line.encode("utf-8")) > MAX_REQUEST_BYTES:
        raise ProcessFailure("REQUEST_INVALID")
    remainder = stream.read()
    if remainder.strip():
        raise ProcessFailure("REQUEST_INVALID")
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProcessFailure("REQUEST_INVALID") from exc


def run_protocol(stdin: TextIO, stdout: TextIO) -> int:
    emitter = ProtocolEmitter(stdout)
    try:
        payload = _read_request(stdin)
        emitter.progress(0, "Validating AnyTop request")
        request = validate_request(payload)
        emitter.progress(8, "Validating installed AnyTop assets")
        result_type, result = process(request, emitter)
        emitter.progress(100, "AnyTop complete")
        if result_type == "file":
            emitter.done_file(Path(result))
        else:
            emitter.done_text(str(result))
        return 0
    except ProcessFailure as exc:
        if not emitter.terminal and not emitter.failed:
            try:
                if exc.diagnostic:
                    emitter.log(exc.diagnostic)
                emitter.error(exc.public_message())
            except BaseException:
                pass
        return 1
    except BaseException:
        if not emitter.terminal and not emitter.failed:
            try:
                emitter.error(ProcessFailure("UNEXPECTED").public_message())
            except BaseException:
                pass
        return 1

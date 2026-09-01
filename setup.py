"""Install the complete AnyTop runtime into Modly-owned storage.

Modly 0.4.2+ invokes setup with one JSON object.  The older positional form
(``python_exe ext_dir gpu_sm [cuda_version]``) remains supported so Repair can
upgrade an existing extension.  Immutable code, checkpoints and T5 weights
are stored below Modly's models_dir and survive extension updates; only the
isolated virtual environment and generated configuration live in this repo.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import json
import os
import platform
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
from typing import BinaryIO, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anytop_modly import dependencies as deps
from anytop_modly.assets import ensure_snapshot, verify_asset, verify_snapshot
from anytop_modly.bundles import BundleError, ensure_bundle_auth_key
from anytop_modly.constants import (
    ASSETS,
    EXTENSION_ID,
    EXTENSION_VERSION,
    READY_MARKER_FILENAME,
    REVISION_ID,
    RUNTIME_CONFIG_FILENAME,
    SETUP_LOCK_FILENAME,
    SETUP_STATE_FILENAME,
)
from anytop_modly.paths import (
    SETUP_MODELS_PAYLOAD_KEYS,
    current_platform_name,
    normalize_platform_name,
    owned_snapshot_directory,
    resolve_models_root,
    safe_snapshot_directory,
    snapshot_paths,
)
from anytop_modly.state import write_runtime_config


VENV_NAME = "venv"
VENV_STAGING_NAME = "venv.__modly_staging"
VENV_BACKUP_NAME = "venv.__modly_backup"
STATE_STAGING_FILENAME = f"{SETUP_STATE_FILENAME}.__modly_staging"
STATE_BACKUP_FILENAME = f"{SETUP_STATE_FILENAME}.__modly_backup"
CONFIG_BACKUP_FILENAME = f"{RUNTIME_CONFIG_FILENAME}.__modly_backup"
SETUP_LOCK_TIMEOUT_SECONDS = 30.0
SETUP_LOCK_POLL_SECONDS = 0.25
COMMAND_TIMEOUT_SECONDS = 4 * 60 * 60
GIB = 1024**3
ASSET_HEADROOM_BYTES = 2 * GIB
ENVIRONMENT_FREE_BYTES = {
    "cpu": 8 * GIB,
    "cu124": 18 * GIB,
    "cu126": 18 * GIB,
    "cu128": 20 * GIB,
    "cu130": 22 * GIB,
}
WINDOWS_REPARSE_ATTRIBUTE = 0x400

INTERPRETER_PROBE = r"""
import json
import platform
import struct
import sys
import sysconfig
print(json.dumps({
    "implementation": sys.implementation.name,
    "version": list(sys.version_info[:2]),
    "cache_tag": sys.implementation.cache_tag,
    "soabi": sysconfig.get_config_var("SOABI"),
    "platform": sysconfig.get_platform().lower(),
    "machine": platform.machine().lower(),
    "pointer_bits": struct.calcsize("P") * 8,
}, sort_keys=True))
"""


class SetupFailure(RuntimeError):
    def __init__(self, code: str, public_message: str) -> None:
        self.code = code
        self.public_message = public_message
        super().__init__(f"{code}: {public_message}")


@dataclass(frozen=True)
class SetupContext:
    python_exe: Path
    ext_dir: Path
    gpu_sm: int
    cuda_version: int
    accelerator: str
    platform_name: str
    arch: str
    payload: Mapping[str, object]
    host_fingerprint: Mapping[str, object]


@dataclass(frozen=True)
class EnvironmentResult:
    python: Path
    reused: bool
    smoke: Mapping[str, object]
    promotion: "EnvironmentPromotion | None" = None


@dataclass(frozen=True)
class EnvironmentPromotion:
    had_previous_venv: bool
    had_previous_state: bool
    had_previous_config: bool


def log(message: str) -> None:
    print(f"[AnyTop setup] {message}", flush=True)


def error_log(message: str) -> None:
    print(f"[AnyTop setup] {message}", file=sys.stderr, flush=True)


def _is_alias(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & WINDOWS_REPARSE_ATTRIBUTE
    )


def _integer(value: object, label: str, *, minimum: int = 0, maximum: int = 999) -> int:
    if isinstance(value, bool):
        raise SetupFailure("SETUP_ARGUMENT_INVALID", f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SetupFailure("SETUP_ARGUMENT_INVALID", f"{label} must be an integer") from exc
    if str(value).strip() not in {str(parsed), f"{parsed}.0"} or not minimum <= parsed <= maximum:
        raise SetupFailure(
            "SETUP_ARGUMENT_INVALID", f"{label} must be between {minimum} and {maximum}"
        )
    return parsed


def parse_args(argv: Sequence[str]) -> dict[str, object]:
    if len(argv) == 2:
        try:
            payload = json.loads(argv[1])
        except json.JSONDecodeError as exc:
            raise SetupFailure("SETUP_JSON_INVALID", "Modly supplied malformed setup metadata") from exc
        if not isinstance(payload, dict):
            raise SetupFailure("SETUP_JSON_INVALID", "Modly setup metadata must be a JSON object")
        return payload
    # Keep the legacy positional contract explicit for older Modly Repair
    # launchers: python_exe ext_dir gpu_sm [cuda_version].
    if len(argv) >= 4 and len(argv) <= 5:
        gpu_sm = _integer(argv[3], "gpu_sm")
        cuda_version = _integer(argv[4], "cuda_version") if len(argv) == 5 else 0
        return {
            "python_exe": argv[1],
            "ext_dir": argv[2],
            "gpu_sm": gpu_sm,
            "cuda_version": cuda_version,
            "accelerator": "cuda" if gpu_sm else "cpu",
            "platform": sys.platform,
            "arch": platform.machine(),
        }
    raise SetupFailure(
        "SETUP_ARGUMENTS_INVALID",
        "expected one Modly JSON argument or legacy python/ext_dir/gpu_sm arguments",
    )


def _normalize_arch(value: object) -> str:
    raw = str(value or "").strip().casefold().replace("-", "_")
    if raw in {"x86_64", "amd64", "x64"}:
        return "x64"
    if raw in {"aarch64", "arm64"}:
        return "arm64"
    return raw


def interpreter_fingerprint(python: Path) -> dict[str, object]:
    try:
        completed = subprocess.run(
            [str(python), "-I", "-S", "-c", INTERPRETER_PROBE],
            check=True,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=30,
            env=deps.sanitize_subprocess_environment(),
        )
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise SetupFailure("PYTHON_PROBE_FAILED", "Modly's Python interpreter could not be inspected") from exc
    if not isinstance(payload, dict):
        raise SetupFailure("PYTHON_PROBE_INVALID", "Python returned invalid ABI metadata")
    return payload


def validate_context(payload: Mapping[str, object], root: Path = ROOT) -> SetupContext:
    python_raw = payload.get("python_exe")
    if not isinstance(python_raw, str) or not python_raw.strip():
        raise SetupFailure("PYTHON_MISSING", "Modly did not provide its Python executable")
    python_exe = Path(python_raw).expanduser().resolve()
    if not python_exe.is_file():
        raise SetupFailure("PYTHON_MISSING", f"Modly Python is unavailable: {python_exe}")
    ext_raw = payload.get("ext_dir") or str(root)
    if not isinstance(ext_raw, str) or not ext_raw.strip():
        raise SetupFailure("EXTENSION_PATH_INVALID", "Modly supplied an invalid ext_dir")
    ext_dir = Path(ext_raw).expanduser().resolve()
    if ext_dir != root.resolve(strict=True):
        raise SetupFailure("EXTENSION_PATH_MISMATCH", "ext_dir does not identify this extension")
    system = normalize_platform_name(payload.get("platform") or sys.platform)
    if system != current_platform_name():
        raise SetupFailure("PLATFORM_MISMATCH", "Modly's platform metadata does not match this host")
    arch = _normalize_arch(payload.get("arch") or platform.machine())
    if arch != _normalize_arch(platform.machine()):
        raise SetupFailure("ARCH_MISMATCH", "Modly's architecture metadata does not match this host")
    gpu_sm = _integer(payload.get("gpu_sm", 0), "gpu_sm")
    cuda_version = _integer(payload.get("cuda_version", 0), "cuda_version")
    accelerator = str(
        payload.get("accelerator") or ("cuda" if gpu_sm else "cpu")
    ).strip().casefold()
    fingerprint = interpreter_fingerprint(python_exe)
    try:
        python_abi = deps.python_abi_from_fingerprint(fingerprint)
    except deps.DependencyError as exc:
        version = fingerprint.get("version")
        if isinstance(version, list) and len(version) >= 2:
            reported = f"CPython {version[0]}.{version[1]}"
        else:
            reported = str(version or fingerprint.get("implementation") or "unknown Python")
        raise SetupFailure(
            "PYTHON_ABI_UNSUPPORTED",
            "AnyTop requires Modly's 64-bit CPython 3.11 or 3.12 runtime; "
            f"this host reported {reported}. Use Modly's bundled CPython 3.11.9, "
            "or a Modly runtime based on 64-bit CPython 3.12, then run Repair.",
        ) from exc
    normalized = dict(payload)
    normalized.update(
        {
            "python_exe": str(python_exe),
            "ext_dir": str(ext_dir),
            "gpu_sm": gpu_sm,
            "cuda_version": cuda_version,
            "accelerator": accelerator,
            "platform": system,
            "arch": arch,
            "python_abi": python_abi,
            "python_version": list(deps.SUPPORTED_PYTHON_ABIS[python_abi]),
            "host_python": dict(fingerprint),
        }
    )
    return SetupContext(
        python_exe,
        ext_dir,
        gpu_sm,
        cuda_version,
        accelerator,
        system,
        arch,
        normalized,
        fingerprint,
    )


def _lock_would_block(exc: OSError) -> bool:
    return exc.errno in {errno.EACCES, errno.EAGAIN, getattr(errno, "EDEADLK", -1)} or getattr(
        exc, "winerror", None
    ) in {33, 36}


@contextmanager
def setup_lock(
    extension_dir: Path,
    *,
    timeout: float = SETUP_LOCK_TIMEOUT_SECONDS,
    poll_interval: float = SETUP_LOCK_POLL_SECONDS,
    platform_name: str | None = None,
) -> Iterator[None]:
    system = normalize_platform_name(platform_name or current_platform_name())
    if system not in {"linux", "win32"}:
        raise SetupFailure("SETUP_LOCK_UNSUPPORTED", "setup locking requires Windows or Linux")
    if timeout < 0 or poll_interval <= 0:
        raise ValueError("setup lock timeout must be non-negative and poll interval positive")
    path = extension_dir / SETUP_LOCK_FILENAME
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if _is_alias(info) or not stat.S_ISREG(info.st_mode) or getattr(info, "st_nlink", 1) != 1:
            raise SetupFailure("SETUP_LOCK_UNSAFE", "the setup lock path is unsafe")
    handle: BinaryIO | None = None
    try:
        handle = path.open("a+b")
        opened = os.fstat(handle.fileno())
        current = path.lstat()
        if (
            _is_alias(current)
            or not stat.S_ISREG(current.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or getattr(current, "st_nlink", 1) != 1
            or getattr(opened, "st_nlink", 1) != 1
            or (
                getattr(current, "st_ino", 0)
                and getattr(opened, "st_ino", 0)
                and (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
            )
        ):
            handle.close()
            raise SetupFailure("SETUP_LOCK_UNSAFE", "the setup lock path is unsafe")
        if opened.st_size == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
            after_open = os.fstat(handle.fileno())
            after_path = path.lstat()
            if (
                _is_alias(after_path)
                or not stat.S_ISREG(after_open.st_mode)
                or not stat.S_ISREG(after_path.st_mode)
                or (after_open.st_dev, after_open.st_ino)
                != (after_path.st_dev, after_path.st_ino)
            ):
                handle.close()
                raise SetupFailure("SETUP_LOCK_UNSAFE", "the setup lock path changed while opening")
    except SetupFailure:
        raise
    except OSError as exc:
        if handle is not None and not handle.closed:
            handle.close()
        raise SetupFailure("SETUP_LOCK_OPEN_FAILED", "the setup lock could not be opened") from exc
    if handle is None:
        raise SetupFailure("SETUP_LOCK_OPEN_FAILED", "the setup lock could not be opened")
    deadline = time.monotonic() + timeout
    acquired = False
    try:
        while True:
            handle.seek(0)
            try:
                if system == "win32":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError as exc:
                if not _lock_would_block(exc):
                    raise SetupFailure("SETUP_LOCK_FAILED", "the setup lock could not be acquired") from exc
                if time.monotonic() >= deadline:
                    raise SetupFailure(
                        "SETUP_LOCK_TIMEOUT", "another AnyTop Install/Repair is still running"
                    )
                time.sleep(poll_interval)
        yield
    finally:
        if acquired:
            handle.seek(0)
            if system == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def venv_python(venv: Path, platform_name: str | None = None) -> Path:
    return venv / ("Scripts/python.exe" if (platform_name or current_platform_name()) == "win32" else "bin/python")


def _remove_venv(venv: Path, extension_dir: Path) -> None:
    if venv.parent.resolve(strict=True) != extension_dir.resolve(strict=True) or venv.name not in {
        VENV_NAME,
        VENV_STAGING_NAME,
        VENV_BACKUP_NAME,
    }:
        raise SetupFailure("VENV_PATH_INVALID", "refusing to remove an unexpected path")
    try:
        info = venv.lstat()
    except FileNotFoundError:
        return
    if _is_alias(info):
        venv.unlink() if stat.S_ISLNK(info.st_mode) else venv.rmdir()
    elif stat.S_ISDIR(info.st_mode):
        shutil.rmtree(venv)
    else:
        venv.unlink()


def _remove_generated_file(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if _is_alias(info) or not stat.S_ISREG(info.st_mode):
        raise SetupFailure("STATE_FILE_INVALID", f"{label} is unsafe")
    path.unlink()


def _move_generated_file(source: Path, destination: Path, label: str) -> bool:
    try:
        info = source.lstat()
    except FileNotFoundError:
        return False
    if _is_alias(info) or not stat.S_ISREG(info.st_mode):
        raise SetupFailure("STATE_FILE_INVALID", f"{label} is unsafe")
    if destination.exists() or destination.is_symlink():
        raise SetupFailure("STATE_TRANSACTION_CONFLICT", f"a stale {label} transaction exists")
    os.replace(source, destination)
    return True


def _replace_owned_venv(source: Path, destination: Path, extension_dir: Path) -> None:
    allowed = {VENV_NAME, VENV_STAGING_NAME, VENV_BACKUP_NAME}
    if (
        source.parent != extension_dir
        or destination.parent != extension_dir
        or source.name not in allowed
        or destination.name not in allowed
    ):
        raise SetupFailure("VENV_PATH_INVALID", "refusing to move an unexpected venv path")
    if destination.exists() or destination.is_symlink():
        raise SetupFailure("VENV_PROMOTION_CONFLICT", "a venv transaction path already exists")
    info = source.lstat()
    if _is_alias(info) or not stat.S_ISDIR(info.st_mode):
        raise SetupFailure("VENV_PATH_INVALID", "the source venv is unsafe")
    os.replace(source, destination)


def _recover_transaction(context: SetupContext) -> None:
    extension = context.ext_dir
    venv = extension / VENV_NAME
    staging = extension / VENV_STAGING_NAME
    backup = extension / VENV_BACKUP_NAME
    state = extension / SETUP_STATE_FILENAME
    state_staging = extension / STATE_STAGING_FILENAME
    state_backup = extension / STATE_BACKUP_FILENAME
    config = extension / RUNTIME_CONFIG_FILENAME
    config_backup = extension / CONFIG_BACKUP_FILENAME
    had_state_staging = state_staging.exists() or state_staging.is_symlink()
    if staging.exists() or staging.is_symlink():
        _remove_venv(staging, extension)
    if had_state_staging:
        _remove_generated_file(state_staging, STATE_STAGING_FILENAME)
    if backup.exists() or backup.is_symlink():
        # An interrupted Repair may have promoted an uncommitted new venv.
        if venv.exists() or venv.is_symlink():
            _remove_venv(venv, extension)
        _replace_owned_venv(backup, venv, extension)
        if state_backup.exists() or state_backup.is_symlink():
            if state.exists() or state.is_symlink():
                _remove_generated_file(state, SETUP_STATE_FILENAME)
            _move_generated_file(state_backup, state, STATE_BACKUP_FILENAME)
        elif not had_state_staging and (state.exists() or state.is_symlink()):
            # No previous state existed and the staged state was already
            # promoted.  It belongs to the uncommitted venv generation.
            _remove_generated_file(state, SETUP_STATE_FILENAME)
        if config.exists() or config.is_symlink():
            _remove_generated_file(config, RUNTIME_CONFIG_FILENAME)
        if config_backup.exists() or config_backup.is_symlink():
            _move_generated_file(config_backup, config, CONFIG_BACKUP_FILENAME)
        return

    # The config is moved before the old venv.  A crash in that narrow window
    # leaves no venv backup, an active old venv, and only config_backup.  Restore
    # missing metadata; if the active counterpart exists, venv promotion had
    # already committed and these are merely cleanup leftovers.
    for active, saved, active_label, saved_label in (
        (state, state_backup, SETUP_STATE_FILENAME, STATE_BACKUP_FILENAME),
        (config, config_backup, RUNTIME_CONFIG_FILENAME, CONFIG_BACKUP_FILENAME),
    ):
        if not (saved.exists() or saved.is_symlink()):
            continue
        if active.exists() or active.is_symlink():
            _remove_generated_file(saved, saved_label)
        else:
            _move_generated_file(saved, active, saved_label)


def _create_venv(context: SetupContext, destination: Path) -> Path:
    log("Creating the isolated AnyTop virtual environment")
    try:
        subprocess.run(
            [str(context.python_exe), "-m", "venv", str(destination)],
            check=True,
            stdin=subprocess.DEVNULL,
            timeout=15 * 60,
            env=deps.sanitize_subprocess_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SetupFailure("VENV_CREATE_FAILED", "Modly Python could not create the AnyTop venv") from exc
    python = venv_python(destination, context.platform_name)
    if not python.is_file():
        raise SetupFailure("VENV_CREATE_INCOMPLETE", "venv creation produced no Python executable")
    if interpreter_fingerprint(python) != dict(context.host_fingerprint):
        raise SetupFailure("VENV_ABI_MISMATCH", "the AnyTop venv does not match Modly Python")
    return python


def _environment_smoke(
    python: Path,
    plan: deps.DependencyPlan,
    revision: Path,
) -> dict[str, object]:
    paths = snapshot_paths(revision)
    return deps.verify_dependencies(
        python,
        plan,
        revision,
        paths.anytop_source,
        paths.motion_source,
        paths.t5,
        paths.checkpoints,
    )


def _reusable_environment(
    context: SetupContext,
    plan: deps.DependencyPlan,
    revision: Path,
    expected_state: Mapping[str, object],
) -> EnvironmentResult | None:
    venv = context.ext_dir / VENV_NAME
    python = venv_python(venv, context.platform_name)
    state = context.ext_dir / SETUP_STATE_FILENAME
    try:
        info = venv.lstat()
    except OSError:
        return None
    if _is_alias(info) or not stat.S_ISDIR(info.st_mode) or not python.is_file():
        return None
    if not deps.state_matches(state, expected_state):
        return None
    try:
        if interpreter_fingerprint(python) != dict(context.host_fingerprint):
            return None
        smoke = _environment_smoke(python, plan, revision)
    except Exception as exc:
        log(f"Existing venv failed {getattr(exc, 'code', type(exc).__name__)}; rebuilding")
        return None
    log("Verified existing AnyTop venv; skipped dependency installation")
    return EnvironmentResult(python, True, smoke)


def _promote_environment(
    context: SetupContext,
    plan: deps.DependencyPlan,
    revision: Path,
    staging: Path,
    state_staging: Path,
    smoke: Mapping[str, object],
) -> EnvironmentResult:
    extension = context.ext_dir
    venv = extension / VENV_NAME
    backup = extension / VENV_BACKUP_NAME
    state = extension / SETUP_STATE_FILENAME
    state_backup = extension / STATE_BACKUP_FILENAME
    config = extension / RUNTIME_CONFIG_FILENAME
    config_backup = extension / CONFIG_BACKUP_FILENAME
    old_venv = old_state = old_config = False
    new_venv = new_state = False
    try:
        old_config = _move_generated_file(config, config_backup, RUNTIME_CONFIG_FILENAME)
        if venv.exists() or venv.is_symlink():
            info = venv.lstat()
            if _is_alias(info) or not stat.S_ISDIR(info.st_mode):
                _remove_venv(venv, extension)
            else:
                _replace_owned_venv(venv, backup, extension)
                old_venv = True
        _replace_owned_venv(staging, venv, extension)
        new_venv = True
        promoted_python = venv_python(venv, context.platform_name)
        if interpreter_fingerprint(promoted_python) != dict(context.host_fingerprint):
            raise SetupFailure(
                "VENV_PROMOTION_ABI_MISMATCH",
                "the promoted AnyTop venv no longer matches Modly Python",
            )
        smoke = _environment_smoke(promoted_python, plan, revision)
        old_state = _move_generated_file(state, state_backup, SETUP_STATE_FILENAME)
        os.replace(state_staging, state)
        new_state = True
    except BaseException:
        if new_venv and (venv.exists() or venv.is_symlink()):
            _remove_venv(venv, extension)
        if old_venv and (backup.exists() or backup.is_symlink()):
            _replace_owned_venv(backup, venv, extension)
        if new_state and (state.exists() or state.is_symlink()):
            _remove_generated_file(state, SETUP_STATE_FILENAME)
        if old_state and state_backup.exists():
            os.replace(state_backup, state)
        if old_config and config_backup.exists():
            os.replace(config_backup, config)
        raise
    return EnvironmentResult(
        venv_python(venv, context.platform_name),
        False,
        smoke,
        EnvironmentPromotion(old_venv, old_state, old_config),
    )


def _install_environment(
    context: SetupContext,
    plan: deps.DependencyPlan,
    revision: Path,
    expected_state: Mapping[str, object],
) -> EnvironmentResult:
    extension = context.ext_dir
    staging = extension / VENV_STAGING_NAME
    state_staging = extension / STATE_STAGING_FILENAME
    if staging.exists() or staging.is_symlink():
        _remove_venv(staging, extension)
    if state_staging.exists() or state_staging.is_symlink():
        _remove_generated_file(state_staging, STATE_STAGING_FILENAME)
    required = ENVIRONMENT_FREE_BYTES[plan.torch_lane]
    if _available_bytes(extension) < required:
        raise SetupFailure(
            "DISK_SPACE_INSUFFICIENT",
            f"the transactional {plan.torch_lane} environment needs about "
            f"{required / GIB:.0f} GiB free beside the extension",
        )
    try:
        python = _create_venv(context, staging)
        paths = snapshot_paths(revision)
        deps.install_dependencies(python, plan, paths.motion_source, log=log)
        smoke = _environment_smoke(python, plan, revision)
        deps.write_state(state_staging, expected_state)
        return _promote_environment(context, plan, revision, staging, state_staging, smoke)
    except BaseException:
        if staging.exists() or staging.is_symlink():
            _remove_venv(staging, extension)
        if state_staging.exists() or state_staging.is_symlink():
            _remove_generated_file(state_staging, STATE_STAGING_FILENAME)
        raise


def install_or_reuse_environment(
    context: SetupContext, plan: deps.DependencyPlan, revision: Path
) -> EnvironmentResult:
    _recover_transaction(context)
    expected = deps.dependency_state_payload(plan, context.host_fingerprint)
    reusable = _reusable_environment(context, plan, revision, expected)
    return reusable or _install_environment(context, plan, revision, expected)


def _rollback_promotion(context: SetupContext, promotion: EnvironmentPromotion) -> None:
    extension = context.ext_dir
    venv = extension / VENV_NAME
    backup = extension / VENV_BACKUP_NAME
    state = extension / SETUP_STATE_FILENAME
    state_backup = extension / STATE_BACKUP_FILENAME
    config = extension / RUNTIME_CONFIG_FILENAME
    config_backup = extension / CONFIG_BACKUP_FILENAME
    if config.exists() or config.is_symlink():
        _remove_generated_file(config, RUNTIME_CONFIG_FILENAME)
    if state.exists() or state.is_symlink():
        _remove_generated_file(state, SETUP_STATE_FILENAME)
    if venv.exists() or venv.is_symlink():
        _remove_venv(venv, extension)
    if promotion.had_previous_venv and (backup.exists() or backup.is_symlink()):
        _replace_owned_venv(backup, venv, extension)
    if promotion.had_previous_state and state_backup.exists():
        os.replace(state_backup, state)
    if promotion.had_previous_config and config_backup.exists():
        os.replace(config_backup, config)
    # Backups whose generation was not declared restorable are stale and must
    # not survive a failed first install.
    if backup.exists() or backup.is_symlink():
        _remove_venv(backup, extension)
    for stale, label in (
        (state_backup, STATE_BACKUP_FILENAME),
        (config_backup, CONFIG_BACKUP_FILENAME),
    ):
        if stale.exists() or stale.is_symlink():
            _remove_generated_file(stale, label)


def _commit_promotion(context: SetupContext, promotion: EnvironmentPromotion) -> None:
    extension = context.ext_dir
    backup = extension / VENV_BACKUP_NAME
    if backup.exists() or backup.is_symlink():
        _remove_venv(backup, extension)
    for stale, label in (
        (extension / STATE_BACKUP_FILENAME, STATE_BACKUP_FILENAME),
        (extension / CONFIG_BACKUP_FILENAME, CONFIG_BACKUP_FILENAME),
    ):
        if stale.exists() or stale.is_symlink():
            _remove_generated_file(stale, label)


def _available_bytes(path: Path) -> int:
    try:
        return shutil.disk_usage(path).free
    except OSError as exc:
        raise SetupFailure("DISK_CHECK_FAILED", f"free space could not be checked for {path}") from exc


def _remaining_asset_bytes(revision: Path) -> int:
    remaining = 0
    for spec in ASSETS:
        destination = revision.joinpath(*spec.relative_path.split("/"))
        valid, _ = verify_asset(destination, spec)
        if valid:
            continue
        part = destination.with_name(destination.name + ".part")
        try:
            info = part.lstat()
        except OSError:
            info = None
        if (
            info is not None
            and not _is_alias(info)
            and stat.S_ISREG(info.st_mode)
            and 0 < info.st_size < spec.size
        ):
            remaining += spec.size - info.st_size
        else:
            remaining += spec.size
    return remaining


def _preflight_storage(models_root: Path, revision: Path) -> None:
    remaining = _remaining_asset_bytes(revision)
    if remaining and _available_bytes(models_root) < remaining + ASSET_HEADROOM_BYTES:
        required = (remaining + ASSET_HEADROOM_BYTES) / GIB
        raise SetupFailure(
            "DISK_SPACE_INSUFFICIENT",
            f"the pinned AnyTop/T5 snapshot needs about {required:.1f} GiB free in models_dir",
        )


def _run_setup_locked(context: SetupContext) -> Path:
    # Dependency installation ends with the pinned plan's literal ``pip check``
    # command before the stronger import/model/device smoke is accepted.
    plan = deps.select_dependency_plan(context.payload)
    host_runtime = deps.validate_host_runtime(plan)
    log(
        f"host={plan.platform}/{plan.arch} accelerator={plan.accelerator} SM={plan.gpu_sm} "
        f"driver_cuda={plan.cuda_version or 'unknown'} torch={plan.torch_version}/{plan.torch_lane}"
    )
    models_root = resolve_models_root(
        context.payload,
        context.ext_dir,
        context.platform_name,
        payload_keys=SETUP_MODELS_PAYLOAD_KEYS,
        require_existing=True,
    ).resolve(strict=True)
    revision = owned_snapshot_directory(models_root, create=True).resolve(strict=True)
    runtime_cache = safe_snapshot_directory(revision, "runtime-cache", create=True)
    try:
        ensure_bundle_auth_key(runtime_cache)
    except BundleError as exc:
        raise SetupFailure(
            "BUNDLE_AUTH_FAILED",
            "the per-installation AnyTop workflow key is missing or unsafe; repair models_dir permissions",
        ) from exc
    _preflight_storage(models_root, revision)
    ensure_snapshot(revision, log=log)
    failures = verify_snapshot(revision, require_ready=True)
    if failures:
        raise SetupFailure(
            "ASSET_VERIFY_FAILED", "the pinned AnyTop snapshot is incomplete: " + "; ".join(failures[:3])
        )
    paths = snapshot_paths(revision)
    environment = install_or_reuse_environment(context, plan, revision)
    try:
        config = write_runtime_config(
            context.ext_dir,
            models_root,
            revision,
            extra={
                "extension_id": EXTENSION_ID,
                "extension_version": EXTENSION_VERSION,
                "revision_id": REVISION_ID,
                "source_root": str(paths.anytop_source),
                "motion_source": str(paths.motion_source),
                "checkpoints_root": str(paths.checkpoints),
                "builtin_cond": str(paths.builtin_cond),
                "t5_path": str(paths.t5),
                "ready_marker": str(paths.ready_marker),
                "runtime_cache_dir": str(runtime_cache),
                "platform": plan.platform,
                "arch": plan.arch,
                "torch_lane": plan.torch_lane,
                "torch_version": plan.torch_version,
                "available_devices": list(plan.available_devices),
                "default_device": plan.default_device,
                "gpu_sm": plan.gpu_sm,
                "cuda_version": plan.cuda_version,
                "dependency_support_level": plan.support_level,
                "python_abi": plan.python_abi,
                "python_runtime": plan.python_label,
                "host_python": dict(context.host_fingerprint),
                "host_runtime": host_runtime,
                "dependency_lock_digest": deps.requirements_digest(plan),
                "environment_reused": environment.reused,
            },
        )
    except BaseException:
        if environment.promotion is not None:
            _rollback_promotion(context, environment.promotion)
        raise
    if environment.promotion is not None:
        _commit_promotion(context, environment.promotion)
    log("Setup complete: all five checkpoints, offline T5, sources and dependencies are verified")
    return config


def run_setup(payload: Mapping[str, object], root: Path = ROOT) -> Path:
    context = validate_context(payload, root)
    with setup_lock(context.ext_dir, platform_name=context.platform_name):
        return _run_setup_locked(context)


def _known_failure(exc: BaseException) -> tuple[str, str] | None:
    code = getattr(exc, "code", None)
    message = getattr(exc, "public_message", None)
    return (code, message) if isinstance(code, str) and isinstance(message, str) else None


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_setup(parse_args(list(sys.argv if argv is None else argv)))
        return 0
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        known = _known_failure(exc)
        if known:
            error_log(f"ERROR [{known[0]}] {known[1]}")
        else:
            error_log(f"ERROR [SETUP_UNEXPECTED] {type(exc).__name__}: {exc}; run Repair")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

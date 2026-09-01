"""Pinned CPython 3.11/3.12 dependency plans for AnyTop inference and preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
from typing import Callable, Mapping, Sequence

from .constants import (
    ANYTOP_SOURCE_TREE_SHA256,
    EXTENSION_ID,
    MOTION_SOURCE_TREE_SHA256,
    REVISION_ID,
    SOURCE_PATCHSET,
)
from .state import dependency_state_matches, write_dependency_state


PYPI_INDEX = "https://pypi.org/simple"
PYTORCH_INDEXES = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu128": "https://download.pytorch.org/whl/cu128",
    "cu130": "https://download.pytorch.org/whl/cu130",
}
TORCH_VERSIONS = {
    "cpu": "2.4.1",
    "cu124": "2.4.1",
    "cu126": "2.6.0",
    "cu128": "2.7.1",
    "cu130": "2.9.1",
}
TORCH_CUDA_RUNTIMES = {
    "cpu": None,
    "cu124": "12.4",
    "cu126": "12.6",
    "cu128": "12.8",
    "cu130": "13.0",
}
SUPPORTED_PYTHON_ABIS = {
    "cp311": (3, 11),
    "cp312": (3, 12),
}
PYTHON_ABI_LABELS = {
    "cp311": "CPython 3.11",
    "cp312": "CPython 3.12",
}
BLACKWELL_LANES = {
    100: "cu128",
    120: "cu128",
    103: "cu130",
    110: "cu130",
    121: "cu130",
}
BOOTSTRAP_REQUIREMENTS = (
    "pip==25.1.1",
    "setuptools==75.3.2",
    "wheel==0.45.1",
)

# Complete, pinned direct + transitive closure of upstream's inference,
# skeleton-preprocessing and preview stack for CPython 3.11.  Matplotlib is the
# first compatible release line used with the deterministic source patch.
# Motion declares PyMEL as a package dependency, but the AnyTop inference and
# preprocessing paths never import or execute its Maya authoring helpers.
CP311_INFERENCE_REQUIREMENTS = (
    "blis==0.7.11",
    "blobfile==3.0.0",
    "catalogue==2.0.10",
    "certifi==2025.1.31",
    "charset-normalizer==3.4.1",
    "click==8.1.8",
    "cloudpathlib==0.16.0",
    "colorama==0.4.6",
    "confection==0.1.5",
    "contourpy==1.1.1",
    "cycler==0.12.1",
    "cymem==2.0.11",
    "decorator==4.4.2",
    "docopt==0.6.2",
    "filelock==3.16.1",
    "fonttools==4.57.0",
    "fsspec==2025.3.0",
    "future==1.0.0",
    "huggingface-hub==0.30.1",
    "idna==3.10",
    "imageio==2.35.1",
    "imageio-ffmpeg==0.5.1",
    "importlib-resources==6.4.5",
    "jinja2==3.1.6",
    "kiwisolver==1.4.7",
    "langcodes==3.4.1",
    "language-data==1.3.0",
    "lxml==5.3.2",
    "marisa-trie==1.2.1",
    "markupsafe==2.1.5",
    "matplotlib==3.7.5",
    "moviepy==1.0.3",
    "mpmath==1.3.0",
    "murmurhash==1.0.12",
    "networkx==3.1",
    "num2words==0.5.14",
    "numpy==1.24.4",
    "packaging==24.2",
    "pathlib-abc==0.1.1",
    "pathy==0.11.0",
    "pillow==10.4.0",
    "preshed==3.0.9",
    "proglog==0.1.11",
    "psutil==7.0.0",
    "pycryptodomex==3.22.0",
    "pydantic==1.10.26",
    "pyparsing==3.1.4",
    "python-dateutil==2.9.0.post0",
    "pyyaml==6.0.2",
    "regex==2024.11.6",
    "requests==2.32.3",
    "safetensors==0.5.3",
    "scipy==1.10.1",
    "sentencepiece==0.2.0",
    "six==1.17.0",
    "smart-open==6.4.0",
    "spacy==3.7.2",
    "spacy-legacy==3.0.12",
    "spacy-loggers==1.0.5",
    "srsly==2.4.8",
    "thinc==8.1.8",
    "tokenizers==0.20.3",
    "tqdm==4.67.1",
    "transformers==4.46.3",
    "typer==0.4.2",
    "typing-extensions==4.12.2",
    "urllib3==2.2.3",
    "wasabi==0.10.1",
    "weasel==0.3.4",
    "zipp==3.20.2",
)

# CPython 3.12 keeps the validated local setup closure.  Only the packages with
# ABI-sensitive wheel constraints differ from the CPython 3.11 lock above.
CP312_INFERENCE_REQUIREMENTS = tuple(
    {
        "contourpy==1.1.1": "contourpy==1.2.1",
        "matplotlib==3.7.5": "matplotlib==3.8.4",
        "numpy==1.24.4": "numpy==1.26.4",
        "scipy==1.10.1": "scipy==1.11.4",
        "spacy==3.7.2": "spacy==3.7.5",
        "thinc==8.1.8": "thinc==8.2.5",
    }.get(requirement, requirement)
    for requirement in CP311_INFERENCE_REQUIREMENTS
)
INFERENCE_REQUIREMENTS_BY_ABI = {
    "cp311": CP311_INFERENCE_REQUIREMENTS,
    "cp312": CP312_INFERENCE_REQUIREMENTS,
}

# Torch 2.6.0 declares sympy==1.13.1 on Python >=3.9, while the other
# selected releases use the newer closure. Keeping this lane-specific avoids
# both an unnecessary reinstall and a deterministic ``pip check`` failure on
# Linux ARM64 CUDA hosts.
TORCH_LANE_REQUIREMENTS = {
    "cpu": ("sympy==1.13.3",),
    "cu124": ("sympy==1.13.3",),
    "cu126": ("sympy==1.13.1",),
    "cu128": ("sympy==1.13.3",),
    "cu130": ("sympy==1.13.3",),
}

COMMAND_TIMEOUT_SECONDS = 4 * 60 * 60
_SENSITIVE_ENV = re.compile(
    r"TOKEN|SECRET|PASSWORD|PASSWD|AUTH|API_KEY|ACCESS_KEY|PRIVATE_KEY|CREDENTIAL|COOKIE",
    re.IGNORECASE,
)
_SAFE_EXACT = frozenset(
    {
        "APPDATA",
        "COMSPEC",
        "HOME",
        "HOMEDRIVE",
        "HOMEPATH",
        "LANG",
        "LANGUAGE",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "LOCALAPPDATA",
        "NUMBER_OF_PROCESSORS",
        "OS",
        "PATH",
        "PATHEXT",
        "PIP_CACHE_DIR",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PYTHONNOUSERSITE",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TZ",
        "USERPROFILE",
        "WINDIR",
    }
)
_SAFE_PREFIXES = ("CUDA_", "LC_", "MODLY_", "OMP_", "XDG_")
_NETWORK_ENV = frozenset(
    {
        "ALL_PROXY",
        "CURL_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
    }
)


class DependencyError(RuntimeError):
    def __init__(self, code: str, public_message: str) -> None:
        self.code = code
        self.public_message = public_message
        super().__init__(f"{code}: {public_message}")


@dataclass(frozen=True)
class DependencyPlan:
    platform: str
    arch: str
    accelerator: str
    gpu_sm: int
    cuda_version: int
    torch_lane: str
    torch_version: str
    torch_index: str
    support_level: str
    python_abi: str

    @property
    def torch_requirement(self) -> str:
        return f"torch=={self.torch_version}"

    @property
    def torch_cuda_runtime(self) -> str | None:
        return TORCH_CUDA_RUNTIMES[self.torch_lane]

    @property
    def available_devices(self) -> tuple[str, ...]:
        return ("cpu", "cuda") if self.accelerator == "cuda" else ("cpu",)

    @property
    def default_device(self) -> str:
        return "cuda" if self.accelerator == "cuda" else "cpu"

    @property
    def python_label(self) -> str:
        return PYTHON_ABI_LABELS[self.python_abi]


def _normalize_platform(value: object) -> str:
    raw = str(value or "").strip().casefold()
    if raw.startswith("linux"):
        return "linux"
    if raw in {"win32", "windows"}:
        return "win32"
    return raw


def _normalize_arch(value: object) -> str:
    raw = str(value or "").strip().casefold().replace("-", "_")
    if raw in {"amd64", "x86_64", "x64"}:
        return "x64"
    if raw in {"arm64", "aarch64"}:
        return "arm64"
    return raw


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise DependencyError("DEPENDENCY_METADATA_INVALID", f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise DependencyError("DEPENDENCY_METADATA_INVALID", f"{name} must be an integer") from exc
    if result < 0 or result > 999:
        raise DependencyError("DEPENDENCY_METADATA_INVALID", f"{name} is outside its valid range")
    return result


def python_abi_from_version(version: object) -> str:
    if isinstance(version, str):
        raw = version.strip().casefold()
        if raw in SUPPORTED_PYTHON_ABIS:
            return raw
        raw = raw.replace("cpython", "").replace("python", "").strip()
        if raw.startswith("cp"):
            raw = raw[2:]
        raw = raw.replace("-", ".").replace("_", ".").replace(" ", "")
        parts = [raw[0], raw[1:]] if re.fullmatch(r"\d{3}", raw) else raw.split(".")
    elif isinstance(version, Sequence) and not isinstance(version, (bytes, bytearray)):
        parts = [str(item) for item in version]
    else:
        raise DependencyError(
            "PYTHON_ABI_UNSUPPORTED",
            "AnyTop setup supports only 64-bit CPython 3.11 or 3.12",
        )
    try:
        major = int(parts[0])
        minor = int(parts[1])
    except (IndexError, TypeError, ValueError) as exc:
        raise DependencyError(
            "PYTHON_ABI_UNSUPPORTED",
            "AnyTop setup supports only 64-bit CPython 3.11 or 3.12",
        ) from exc
    for abi, expected in SUPPORTED_PYTHON_ABIS.items():
        if (major, minor) == expected:
            return abi
    raise DependencyError(
        "PYTHON_ABI_UNSUPPORTED",
        f"AnyTop setup supports only 64-bit CPython 3.11 or 3.12; got Python {major}.{minor}",
    )


def python_abi_from_fingerprint(fingerprint: Mapping[str, object]) -> str:
    if fingerprint.get("implementation") != "cpython" or fingerprint.get("pointer_bits") != 64:
        raise DependencyError(
            "PYTHON_ABI_UNSUPPORTED",
            "AnyTop setup supports only 64-bit CPython 3.11 or 3.12",
        )
    return python_abi_from_version(fingerprint.get("version"))


def _python_abi(payload: Mapping[str, object]) -> str:
    candidates: list[tuple[str, str]] = []
    raw = payload.get("python_abi")
    if raw is not None:
        if isinstance(raw, str):
            normalized = raw.strip().casefold()
            if normalized in SUPPORTED_PYTHON_ABIS:
                candidates.append(("python_abi", normalized))
            else:
                candidates.append(("python_abi", python_abi_from_version(raw)))
        else:
            candidates.append(("python_abi", python_abi_from_version(raw)))
    host = payload.get("host_python")
    if host is not None:
        if not isinstance(host, Mapping):
            raise DependencyError(
                "PYTHON_ABI_UNSUPPORTED",
                "host_python must contain the probed 64-bit CPython ABI fingerprint",
            )
        candidates.append(("host_python", python_abi_from_fingerprint(host)))
    version = payload.get("python_version")
    if version is not None:
        candidates.append(("python_version", python_abi_from_version(version)))
    if candidates:
        authoritative = next(
            (abi for source, abi in candidates if source == "host_python"),
            candidates[0][1],
        )
        conflicts = [source for source, abi in candidates if abi != authoritative]
        if conflicts:
            sources = ", ".join(source for source, _abi in candidates)
            raise DependencyError(
                "PYTHON_ABI_CONFLICT",
                "Modly supplied conflicting Python ABI metadata; run Repair with one "
                f"64-bit CPython 3.11/3.12 runtime ({sources})",
            )
        return authoritative
    raise DependencyError(
        "PYTHON_ABI_MISSING",
        "setup must pass the probed Modly Python ABI; refusing to infer from the launcher",
    )


def tegra_evidence(payload: Mapping[str, object]) -> str | None:
    for key in (
        "platform_variant",
        "platformVariant",
        "device_name",
        "deviceName",
        "gpu_name",
        "gpuName",
        "soc",
    ):
        value = payload.get(key)
        if isinstance(value, str) and any(
            marker in value.casefold() for marker in ("tegra", "jetson", "orin", "thor")
        ):
            return f"Modly metadata {key}={value}"
    for marker in (Path("/etc/nv_tegra_release"), Path("/proc/device-tree/compatible")):
        try:
            if marker.is_file():
                if marker.name == "nv_tegra_release":
                    return str(marker)
                raw = marker.read_bytes()[:4096].lower()
                if b"tegra" in raw or b"jetson" in raw:
                    return str(marker)
        except OSError:
            continue
    return None


def select_dependency_plan(payload: Mapping[str, object]) -> DependencyPlan:
    python_abi = _python_abi(payload)
    system = _normalize_platform(payload.get("platform") or platform.system())
    arch = _normalize_arch(payload.get("arch") or platform.machine())
    if system not in {"linux", "win32"}:
        raise DependencyError("PLATFORM_UNSUPPORTED", "AnyTop setup supports Windows and Linux only")
    if arch not in {"x64", "arm64"} or (system == "win32" and arch != "x64"):
        raise DependencyError(
            "ARCH_UNSUPPORTED",
            "AnyTop supports Windows x64 plus Linux x64 and Linux ARM64 (SBSA)",
        )
    if system == "linux" and arch == "arm64":
        evidence = tegra_evidence(payload)
        if evidence:
            raise DependencyError(
                "TEGRA_UNSUPPORTED",
                "stock Jetson/Tegra Python is not an NVIDIA SBSA environment; "
                "use a platform-specific container or an SBSA host "
                f"({evidence})",
            )
    gpu_sm = _integer(payload.get("gpu_sm", 0), "gpu_sm")
    cuda_version = _integer(payload.get("cuda_version", 0), "cuda_version")
    accelerator = str(
        payload.get("accelerator") or ("cuda" if gpu_sm else "cpu")
    ).strip().casefold()
    if accelerator in {"none", "host"}:
        accelerator = "cpu"
    if accelerator not in {"cpu", "cuda"}:
        raise DependencyError("ACCELERATOR_UNSUPPORTED", "accelerator must be cpu or cuda")
    if accelerator == "cpu":
        if gpu_sm != 0:
            raise DependencyError(
                "ACCELERATOR_CONFLICT", "CPU setup requires Modly to report gpu_sm=0"
            )
        lane = "cpu"
        support = "extension-owned CPU compatibility"
    else:
        if gpu_sm == 0:
            raise DependencyError(
                "CUDA_METADATA_MISSING", "CUDA setup requires Modly to report the GPU compute capability"
            )
        if gpu_sm in BLACKWELL_LANES:
            lane = BLACKWELL_LANES[gpu_sm]
            if gpu_sm == 121 and not (system == "linux" and arch == "arm64"):
                raise DependencyError(
                    "GPU_PLATFORM_UNSUPPORTED",
                    "SM121 is supported only by the official Linux ARM64 CUDA 13.0 lane",
                )
        elif 50 <= gpu_sm < 100:
            # The official cu124 index exposes only the CPU-built raw
            # torch-2.4.1 wheel for Linux aarch64. cu126 is the first pinned
            # lane here with an official CUDA-enabled SBSA wheel.
            lane = "cu126" if system == "linux" and arch == "arm64" else "cu124"
        else:
            raise DependencyError(
                "GPU_SM_UNSUPPORTED",
                f"GPU SM{gpu_sm} has no validated AnyTop Torch lane; setup will not fall back to CPU",
            )
        support = "official PyTorch CUDA wheel"
        # Fail before downloading several GiB when Modly can already prove the
        # driver is too old.  Modly 0.4.2 currently caps its reported value at
        # 128 on some CUDA 13 ARM64 hosts, so cu130 is intentionally validated
        # by the mandatory real-device smoke instead of this metadata field.
        if lane == "cu124" and cuda_version and cuda_version < 124:
            raise DependencyError(
                "CUDA_DRIVER_TOO_OLD",
                "the selected cu124 Torch lane requires a CUDA 12.4-compatible NVIDIA driver",
            )
        if lane == "cu126" and cuda_version and cuda_version < 126:
            raise DependencyError(
                "CUDA_DRIVER_TOO_OLD",
                "the Linux ARM64 CUDA lane requires a CUDA 12.6-compatible NVIDIA driver",
            )
        if lane == "cu128" and cuda_version and cuda_version < 128:
            raise DependencyError(
                "CUDA_DRIVER_TOO_OLD",
                "this Blackwell GPU requires a CUDA 12.8-compatible NVIDIA driver",
            )
    return DependencyPlan(
        platform=system,
        arch=arch,
        accelerator=accelerator,
        gpu_sm=gpu_sm,
        cuda_version=cuda_version,
        torch_lane=lane,
        torch_version=TORCH_VERSIONS[lane],
        torch_index=PYTORCH_INDEXES[lane],
        support_level=support,
        python_abi=python_abi,
    )


def sanitize_subprocess_environment(
    source: Mapping[str, str] | None = None, *, for_pip: bool = False
) -> dict[str, str]:
    values = os.environ if source is None else source
    result: dict[str, str] = {}
    for key, value in values.items():
        upper = key.upper()
        if _SENSITIVE_ENV.search(upper):
            continue
        if upper in _SAFE_EXACT or upper in _NETWORK_ENV or upper.startswith(_SAFE_PREFIXES):
            result[key] = value
    result["PYTHONNOUSERSITE"] = "1"
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    result["TOKENIZERS_PARALLELISM"] = "false"
    if for_pip:
        result["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        result["PIP_NO_INPUT"] = "1"
    return result


def _version_tuple(raw: str) -> tuple[int, ...]:
    match = re.match(r"\s*(\d+(?:\.\d+)*)", raw)
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def compatible_arch_token(architectures: Sequence[str], gpu_sm: int) -> str | None:
    """Return an exact or same-major forward-compatible CUDA target token."""

    exact = {f"sm_{gpu_sm}", f"compute_{gpu_sm}"}
    for token in architectures:
        if token in exact:
            return token
    expected_major = gpu_sm // 10
    compatible: list[tuple[int, str]] = []
    for token in architectures:
        match = re.fullmatch(r"(?:sm|compute)_(\d+)[a-z]?", token)
        if not match:
            continue
        target = int(match.group(1))
        if target // 10 == expected_major and target <= gpu_sm:
            compatible.append((target, token))
    return max(compatible)[1] if compatible else None


def validate_host_runtime(
    plan: DependencyPlan,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
) -> dict[str, object]:
    """Fail before multi-GiB downloads when host ABI/driver evidence is conclusive."""

    report: dict[str, object] = {}
    if plan.platform == "linux":
        libc_name, libc_version = platform.libc_ver()
        parsed_libc = _version_tuple(libc_version)
        minimum = (2, 28) if plan.torch_lane in {"cu126", "cu128", "cu130"} else (2, 17)
        if libc_name.casefold() != "glibc" or not parsed_libc or parsed_libc < minimum:
            minimum_text = ".".join(str(part) for part in minimum)
            raise DependencyError(
                "GLIBC_UNSUPPORTED",
                f"the selected official Torch wheels require glibc {minimum_text} or newer",
            )
        report["glibc"] = libc_version

    nvidia_smi = which("nvidia-smi") if plan.accelerator == "cuda" else None
    if nvidia_smi:
        try:
            completed = runner(
                [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=15,
                env=sanitize_subprocess_environment(),
            )
            versions = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        except (OSError, subprocess.SubprocessError):
            versions = []
        if versions:
            report["nvidia_driver"] = versions[0]
            # CUDA 13 Linux wheels cannot initialize on an older branch.  Do
            # not infer this from Modly 0.4.2's cuda_version field: that field
            # is currently capped at 128 on some CUDA 13 ARM64 systems.
            if (
                plan.platform == "linux"
                and plan.torch_lane == "cu130"
                and _version_tuple(versions[0]) < (580, 65, 6)
            ):
                raise DependencyError(
                    "NVIDIA_DRIVER_TOO_OLD",
                    "the CUDA 13 Torch lane requires NVIDIA Linux driver 580.65.06 or newer",
                )
    return report


def inference_requirements(plan: DependencyPlan) -> tuple[str, ...]:
    """Return the exact dependency closure compatible with ``plan``'s Torch."""

    try:
        python_requirements = INFERENCE_REQUIREMENTS_BY_ABI[plan.python_abi]
        torch_requirements = TORCH_LANE_REQUIREMENTS[plan.torch_lane]
    except KeyError as exc:
        raise DependencyError(
            "DEPENDENCY_LANE_INVALID",
            f"unknown dependency lane: python={plan.python_abi} torch={plan.torch_lane}",
        ) from exc
    return (*python_requirements, *torch_requirements)


def requirements_digest(plan: DependencyPlan) -> str:
    payload = {
        "python_abi": plan.python_abi,
        "bootstrap": BOOTSTRAP_REQUIREMENTS,
        "inference": inference_requirements(plan),
        "torch": plan.torch_requirement,
        "torch_index": plan.torch_index,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def locked_distribution_versions(plan: DependencyPlan) -> dict[str, str]:
    locked: dict[str, str] = {}
    for requirement in (*BOOTSTRAP_REQUIREMENTS, *inference_requirements(plan)):
        name, separator, version = requirement.partition("==")
        if separator != "==" or not name or not version:
            raise DependencyError(
                "DEPENDENCY_LOCK_INVALID", "every AnyTop Python dependency must be exactly pinned"
            )
        locked[name] = version
    return dict(sorted(locked.items()))


def dependency_state_payload(
    plan: DependencyPlan, host_fingerprint: Mapping[str, object]
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "extension_id": EXTENSION_ID,
        "revision_id": REVISION_ID,
        "plan": asdict(plan),
        "requirements_digest": requirements_digest(plan),
        "source_patchset": SOURCE_PATCHSET,
        "anytop_source_tree": ANYTOP_SOURCE_TREE_SHA256,
        "motion_source_tree": MOTION_SOURCE_TREE_SHA256,
        "host_python": dict(host_fingerprint),
    }


def state_matches(path: Path, expected: Mapping[str, object]) -> bool:
    return dependency_state_matches(path, expected)


def write_state(path: Path, payload: Mapping[str, object]) -> Path:
    return write_dependency_state(path, payload)


def _run_checked(command: Sequence[str], *, stage: str, timeout: int = COMMAND_TIMEOUT_SECONDS) -> None:
    try:
        subprocess.run(
            list(command),
            check=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            env=sanitize_subprocess_environment(for_pip="pip" in command),
        )
    except subprocess.TimeoutExpired as exc:
        raise DependencyError("DEPENDENCY_TIMEOUT", f"{stage} exceeded its safe time limit") from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DependencyError(
            "DEPENDENCY_INSTALL_FAILED", f"{stage} failed; review pip output and run Repair"
        ) from exc


def pip_install_commands(python: Path, plan: DependencyPlan, motion_source: Path) -> tuple[tuple[str, ...], ...]:
    """Return the exact command sequence, useful for tests and transparent logs."""

    # Motion is intentionally imported from its immutable, verified source
    # path.  Installing it in place would let setuptools create egg-info/build
    # files inside models_dir and invalidate the release tree digest.
    del motion_source
    base = (str(python), "-I", "-m", "pip")
    return (
        (*base, "install", "--upgrade", *BOOTSTRAP_REQUIREMENTS),
        (*base, "install", "--index-url", PYPI_INDEX, *inference_requirements(plan)),
        (
            *base,
            "install",
            "--index-url",
            plan.torch_index,
            plan.torch_requirement,
        ),
        (*base, "check"),
    )


def install_dependencies(
    python: Path,
    plan: DependencyPlan,
    motion_source: Path,
    *,
    log: Callable[[str], None] = print,
    runner: Callable[..., None] | None = None,
) -> None:
    stages = (
        "Bootstrapping the isolated AnyTop environment",
        "Installing the complete AnyTop inference/preprocessing dependency closure",
        f"Installing pinned Torch {plan.torch_version} ({plan.torch_lane})",
        "Checking the installed dependency graph",
    )
    execute = _run_checked if runner is None else runner
    for command, stage in zip(pip_install_commands(python, plan, motion_source), stages):
        log(stage)
        execute(command, stage=stage)


SMOKE_SCRIPT = r"""
import json
import pathlib
import re
import sys
from types import SimpleNamespace
from importlib import metadata

revision = pathlib.Path(sys.argv[1])
anytop = pathlib.Path(sys.argv[2])
motion = pathlib.Path(sys.argv[3])
t5_path = pathlib.Path(sys.argv[4])
checkpoints = pathlib.Path(sys.argv[5])
expected_torch = sys.argv[6]
expected_device = sys.argv[7]
expected_cuda_runtime = sys.argv[8] or None
expected_sm = int(sys.argv[9])
locked_versions = json.loads(sys.argv[10])
sys.path.insert(0, str(motion))
sys.path.insert(0, str(anytop))

for distribution, expected_version in locked_versions.items():
    actual_version = metadata.version(distribution)
    if actual_version != expected_version:
        raise RuntimeError(
            f"{distribution} version {actual_version} does not match {expected_version}"
        )

import numpy
import scipy
import matplotlib
import moviepy.editor
import spacy
import blobfile
import num2words
import torch
import transformers
import BVH
import Animation
from model.anytop import AnyTop
from utils import model_util

if torch.__version__.split('+', 1)[0] != expected_torch:
    raise RuntimeError(f"Torch version {torch.__version__} does not match {expected_torch}")
cuda_matmul = False
if expected_device == "cuda":
    actual_cuda_runtime = torch.version.cuda
    if actual_cuda_runtime is None or not actual_cuda_runtime.startswith(expected_cuda_runtime):
        raise RuntimeError(
            f"Torch CUDA runtime {actual_cuda_runtime!r} does not match {expected_cuda_runtime}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("the selected CUDA Torch lane cannot access the GPU")
    capability = torch.cuda.get_device_capability()
    actual_sm = capability[0] * 10 + capability[1]
    torch_arch_list = torch.cuda.get_arch_list()
    if not torch_arch_list:
        raise RuntimeError("CUDA Torch wheel reports no compiled GPU architectures")
    expected_tokens = {f"sm_{expected_sm}", f"compute_{expected_sm}"}
    exact_tokens = sorted(expected_tokens.intersection(torch_arch_list))
    if exact_tokens:
        torch_arch_match = exact_tokens[0]
    else:
        compatible_tokens = []
        for token in torch_arch_list:
            match = re.fullmatch(r"(?:sm|compute)_(\d+)[a-z]?", token)
            if match:
                target = int(match.group(1))
                if target // 10 == expected_sm // 10 and target <= expected_sm:
                    compatible_tokens.append((target, token))
        if not compatible_tokens:
            raise RuntimeError(
                f"Torch wheel architectures {torch_arch_list} do not support SM{expected_sm}"
            )
        torch_arch_match = max(compatible_tokens)[1]
    if actual_sm != expected_sm:
        raise RuntimeError(f"visible GPU is SM{actual_sm}; Modly selected SM{expected_sm}")
    left = torch.ones((16, 16), device="cuda")
    result = (left @ left).sum().item()
    if result != 4096.0:
        raise RuntimeError("CUDA matmul produced an invalid result")
    torch.cuda.synchronize()
    cuda_matmul = True
else:
    actual_cuda_runtime = torch.version.cuda
    actual_sm = 0
    torch_arch_list = []
    torch_arch_match = "cpu"
    if actual_cuda_runtime is not None:
        raise RuntimeError(f"CPU lane installed a CUDA Torch wheel ({actual_cuda_runtime})")
    value = (torch.ones((4, 4)) @ torch.ones((4, 4))).sum().item()
    if value != 64.0:
        raise RuntimeError("CPU Torch smoke failed")

from transformers import T5EncoderModel, T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained(str(t5_path), local_files_only=True)
model = T5EncoderModel.from_pretrained(str(t5_path), local_files_only=True)
if model.config.model_type != "t5" or tokenizer.vocab_size <= 0:
    raise RuntimeError("offline T5 smoke failed")
del model

checkpoint_count = 0
for checkpoint in sorted(checkpoints.glob("*/model*.pt")):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError(f"invalid checkpoint payload: {checkpoint}")
    args = checkpoint.parent / "args.json"
    parsed = json.loads(args.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"invalid checkpoint arguments: {args}")
    checkpoint_model, _diffusion = model_util.create_model_and_diffusion_general_skeleton(
        SimpleNamespace(**parsed)
    )
    model_util.load_model(checkpoint_model, payload)
    del checkpoint_model, _diffusion, payload
    checkpoint_count += 1
if checkpoint_count != 5:
    raise RuntimeError(f"expected five AnyTop checkpoints, found {checkpoint_count}")

print(json.dumps({
    "torch": torch.__version__,
    "torch_lane_device": expected_device,
    "cuda_matmul": cuda_matmul,
    "torch_cuda_runtime": actual_cuda_runtime,
    "gpu_sm": actual_sm,
    "torch_arch_list": torch_arch_list,
    "torch_arch_match": torch_arch_match,
    "numpy": numpy.__version__,
    "matplotlib": matplotlib.__version__,
    "spacy": spacy.__version__,
    "transformers": transformers.__version__,
    "checkpoints": checkpoint_count,
    "t5_offline": True,
    "source_imports": True,
    "locked_distributions": len(locked_versions),
}, sort_keys=True))
"""


def verify_dependencies(
    python: Path,
    plan: DependencyPlan,
    revision_root: Path,
    anytop_source: Path,
    motion_source: Path,
    t5_path: Path,
    checkpoints_root: Path,
) -> dict[str, object]:
    command = [
        str(python),
        "-I",
        "-B",
        "-c",
        SMOKE_SCRIPT,
        str(revision_root),
        str(anytop_source),
        str(motion_source),
        str(t5_path),
        str(checkpoints_root),
        plan.torch_version,
        plan.default_device,
        plan.torch_cuda_runtime or "",
        str(plan.gpu_sm),
        json.dumps(locked_distribution_versions(plan), sort_keys=True, separators=(",", ":")),
    ]
    smoke_environment = sanitize_subprocess_environment()
    smoke_environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=30 * 60,
            env=smoke_environment,
        )
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except subprocess.CalledProcessError as exc:
        diagnostic_source = (exc.stderr or exc.stdout or "").strip().splitlines()
        diagnostic = diagnostic_source[-1][:500] if diagnostic_source else "no worker diagnostic"
        raise DependencyError(
            "DEPENDENCY_SMOKE_FAILED",
            "the installed Torch/T5/source/checkpoint smoke test failed: " + diagnostic,
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise DependencyError(
            "DEPENDENCY_SMOKE_TIMEOUT",
            "the Torch/T5/source/checkpoint smoke exceeded 30 minutes; close other GPU jobs and run Repair",
        ) from exc
    except (OSError, json.JSONDecodeError, IndexError) as exc:
        raise DependencyError(
            "DEPENDENCY_SMOKE_FAILED",
            "the installed Torch/T5/source/checkpoint smoke test failed; review output and run Repair",
        ) from exc
    if not isinstance(payload, dict) or payload.get("checkpoints") != 5:
        raise DependencyError("DEPENDENCY_SMOKE_INVALID", "dependency smoke returned invalid results")
    if plan.accelerator == "cuda" and payload.get("cuda_matmul") is not True:
        raise DependencyError("CUDA_SMOKE_INVALID", "CUDA setup did not complete a real GPU matmul")
    if payload.get("torch_cuda_runtime") != plan.torch_cuda_runtime:
        raise DependencyError("TORCH_FLAVOR_INVALID", "the installed Torch wheel has the wrong CUDA runtime")
    if payload.get("gpu_sm") != plan.gpu_sm:
        raise DependencyError("CUDA_DEVICE_MISMATCH", "the visible GPU does not match Modly setup metadata")
    if plan.accelerator == "cuda":
        raw_architectures = payload.get("torch_arch_list")
        if not isinstance(raw_architectures, list) or not all(
            isinstance(item, str) for item in raw_architectures
        ):
            raise DependencyError("TORCH_ARCH_INVALID", "Torch returned an invalid architecture list")
        matched = compatible_arch_token(raw_architectures, plan.gpu_sm)
        if matched is None or payload.get("torch_arch_match") != matched:
            raise DependencyError(
                "TORCH_ARCH_INVALID", "the installed Torch wheel has no compatible compiled architecture"
            )
    return payload

from __future__ import annotations

import json
from pathlib import Path
import platform
import sys

import pytest

import setup
from anytop_modly import assets
from anytop_modly import dependencies as deps
from anytop_modly.constants import EXTENSION_ID


def test_parse_current_json_and_legacy_contract() -> None:
    payload = {"python_exe": "/python", "ext_dir": "/extension", "gpu_sm": 89}
    assert setup.parse_args(["setup.py", json.dumps(payload)]) == payload
    legacy = setup.parse_args(["setup.py", "/python", "/extension", "89", "124"])
    assert legacy["gpu_sm"] == 89
    assert legacy["cuda_version"] == 124
    assert legacy["accelerator"] == "cuda"
    with pytest.raises(setup.SetupFailure, match="SETUP_JSON_INVALID"):
        setup.parse_args(["setup.py", "[]"])


@pytest.mark.parametrize(
    ("payload", "lane", "version", "index", "cuda_runtime"),
    (
        (
            {"platform": "linux", "arch": "x86_64", "accelerator": "cpu", "gpu_sm": 0},
            "cpu",
            "2.4.1",
            "https://download.pytorch.org/whl/cpu",
            None,
        ),
        (
            {
                "platform": "win32",
                "arch": "amd64",
                "accelerator": "cuda",
                "gpu_sm": 89,
                "cuda_version": 124,
            },
            "cu124",
            "2.4.1",
            "https://download.pytorch.org/whl/cu124",
            "12.4",
        ),
        (
            {
                "platform": "linux",
                "arch": "x64",
                "accelerator": "cuda",
                "gpu_sm": 89,
                "cuda_version": 124,
            },
            "cu124",
            "2.4.1",
            "https://download.pytorch.org/whl/cu124",
            "12.4",
        ),
        (
            {
                "platform": "linux",
                "arch": "aarch64",
                "accelerator": "cuda",
                "gpu_sm": 89,
                "cuda_version": 126,
            },
            "cu126",
            "2.6.0",
            "https://download.pytorch.org/whl/cu126",
            "12.6",
        ),
        (
            {
                "platform": "linux",
                "arch": "x64",
                "accelerator": "cuda",
                "gpu_sm": 120,
                "cuda_version": 128,
            },
            "cu128",
            "2.7.1",
            "https://download.pytorch.org/whl/cu128",
            "12.8",
        ),
        (
            {
                "platform": "linux",
                "arch": "arm64",
                "accelerator": "cuda",
                "gpu_sm": 120,
                "cuda_version": 128,
            },
            "cu128",
            "2.7.1",
            "https://download.pytorch.org/whl/cu128",
            "12.8",
        ),
        (
            {
                "platform": "linux",
                "arch": "x64",
                "accelerator": "cuda",
                "gpu_sm": 103,
                "cuda_version": 130,
            },
            "cu130",
            "2.9.1",
            "https://download.pytorch.org/whl/cu130",
            "13.0",
        ),
        (
            {
                "platform": "linux",
                "arch": "arm64",
                "accelerator": "cuda",
                "gpu_sm": 121,
                # Modly 0.4.2 can still report 128 on CUDA 13 hosts.
                "cuda_version": 128,
            },
            "cu130",
            "2.9.1",
            "https://download.pytorch.org/whl/cu130",
            "13.0",
        ),
    ),
)
def test_dependency_lane_matrix(payload, lane, version, index, cuda_runtime, monkeypatch) -> None:
    monkeypatch.setattr(deps, "tegra_evidence", lambda _payload: None)
    plan = deps.select_dependency_plan(payload)
    assert plan.torch_lane == lane
    assert plan.torch_version == version
    assert plan.torch_index == index
    assert plan.torch_cuda_runtime == cuda_runtime


def test_cuda_architecture_matching_is_exact_or_same_major_compatible() -> None:
    assert deps.compatible_arch_token(["sm_80", "sm_89"], 89) == "sm_89"
    assert deps.compatible_arch_token(["sm_60", "sm_70"], 61) == "sm_60"
    assert deps.compatible_arch_token(["compute_86", "sm_75"], 89) == "compute_86"
    assert deps.compatible_arch_token(["sm_90"], 89) is None
    assert deps.compatible_arch_token([], 120) is None


def test_dependency_selection_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(deps, "tegra_evidence", lambda _payload: None)
    with pytest.raises(deps.DependencyError, match="ACCELERATOR_CONFLICT"):
        deps.select_dependency_plan(
            {"platform": "linux", "arch": "x64", "accelerator": "cpu", "gpu_sm": 89}
        )
    with pytest.raises(deps.DependencyError, match="GPU_SM_UNSUPPORTED"):
        deps.select_dependency_plan(
            {
                "platform": "linux",
                "arch": "x64",
                "accelerator": "cuda",
                "gpu_sm": 37,
                "cuda_version": 124,
            }
        )
    with pytest.raises(deps.DependencyError, match="GPU_SM_UNSUPPORTED"):
        deps.select_dependency_plan(
            {
                "platform": "linux",
                "arch": "x64",
                "accelerator": "cuda",
                "gpu_sm": 101,
                "cuda_version": 130,
            }
        )
    with pytest.raises(deps.DependencyError, match="CUDA_DRIVER_TOO_OLD"):
        deps.select_dependency_plan(
            {
                "platform": "linux",
                "arch": "x64",
                "accelerator": "cuda",
                "gpu_sm": 120,
                "cuda_version": 124,
            }
        )
    with pytest.raises(deps.DependencyError, match="CUDA_DRIVER_TOO_OLD"):
        deps.select_dependency_plan(
            {
                "platform": "linux",
                "arch": "arm64",
                "accelerator": "cuda",
                "gpu_sm": 89,
                "cuda_version": 124,
            }
        )
    with pytest.raises(deps.DependencyError, match="GPU_PLATFORM_UNSUPPORTED"):
        deps.select_dependency_plan(
            {
                "platform": "win32",
                "arch": "x64",
                "accelerator": "cuda",
                "gpu_sm": 121,
                "cuda_version": 130,
            }
        )


def test_tegra_stock_setup_is_actionably_rejected(monkeypatch) -> None:
    monkeypatch.setattr(deps, "tegra_evidence", lambda _payload: "/etc/nv_tegra_release")
    with pytest.raises(deps.DependencyError, match="TEGRA_UNSUPPORTED"):
        deps.select_dependency_plan(
            {"platform": "linux", "arch": "arm64", "accelerator": "cpu", "gpu_sm": 0}
        )


def test_host_preflight_checks_glibc_and_cuda13_driver(monkeypatch) -> None:
    monkeypatch.setattr(deps, "tegra_evidence", lambda _payload: None)
    plan = deps.select_dependency_plan(
        {
            "platform": "linux",
            "arch": "arm64",
            "accelerator": "cuda",
            "gpu_sm": 121,
            "cuda_version": 128,
        }
    )
    monkeypatch.setattr(deps.platform, "libc_ver", lambda: ("glibc", "2.39"))

    def old_driver(*_args, **_kwargs):
        return deps.subprocess.CompletedProcess([], 0, stdout="575.64.03\n", stderr="")

    with pytest.raises(deps.DependencyError, match="NVIDIA_DRIVER_TOO_OLD"):
        deps.validate_host_runtime(plan, runner=old_driver, which=lambda _name: "/nvidia-smi")

    def new_driver(*_args, **_kwargs):
        return deps.subprocess.CompletedProcess([], 0, stdout="580.65.06\n", stderr="")

    report = deps.validate_host_runtime(
        plan, runner=new_driver, which=lambda _name: "/nvidia-smi"
    )
    assert report == {"glibc": "2.39", "nvidia_driver": "580.65.06"}


def test_blackwell_manylinux_preflight_requires_glibc_228(monkeypatch) -> None:
    plan = deps.select_dependency_plan(
        {
            "platform": "linux",
            "arch": "x64",
            "accelerator": "cuda",
            "gpu_sm": 120,
            "cuda_version": 128,
        }
    )
    monkeypatch.setattr(deps.platform, "libc_ver", lambda: ("glibc", "2.27"))
    with pytest.raises(deps.DependencyError, match="GLIBC_UNSUPPORTED"):
        deps.validate_host_runtime(plan, which=lambda _name: None)


def test_arm64_cu126_preflight_requires_glibc_228(monkeypatch) -> None:
    monkeypatch.setattr(deps, "tegra_evidence", lambda _payload: None)
    plan = deps.select_dependency_plan(
        {
            "platform": "linux",
            "arch": "arm64",
            "accelerator": "cuda",
            "gpu_sm": 89,
            "cuda_version": 126,
        }
    )
    monkeypatch.setattr(deps.platform, "libc_ver", lambda: ("glibc", "2.27"))
    with pytest.raises(deps.DependencyError, match="GLIBC_UNSUPPORTED"):
        deps.validate_host_runtime(plan, which=lambda _name: None)


def test_host_preflight_runs_before_models_or_install(tmp_path: Path, monkeypatch) -> None:
    context = _context(tmp_path)
    plan = deps.select_dependency_plan(context.payload)
    events: list[str] = []
    monkeypatch.setattr(deps, "select_dependency_plan", lambda _payload: plan)
    monkeypatch.setattr(
        deps, "validate_host_runtime", lambda _plan: events.append("host-preflight") or {}
    )

    def stop_before_storage(*_args, **_kwargs):
        events.append("models-dir")
        raise setup.SetupFailure("TEST_STOP", "stop")

    monkeypatch.setattr(setup, "resolve_models_root", stop_before_storage)
    with pytest.raises(setup.SetupFailure, match="TEST_STOP"):
        setup._run_setup_locked(context)
    assert events == ["host-preflight", "models-dir"]


def test_dependency_commands_are_pinned_and_omit_incompatible_packages(tmp_path: Path) -> None:
    plan = deps.select_dependency_plan(
        {"platform": "linux", "arch": "x64", "accelerator": "cpu", "gpu_sm": 0}
    )
    commands = deps.pip_install_commands(tmp_path / "python", plan, tmp_path / "Motion")
    flattened = "\n".join(" ".join(command).casefold() for command in commands)
    assert "torch==2.4.1" in flattened
    assert "matplotlib==3.7.5" in flattened
    assert "moviepy==1.0.3" in flattened
    assert "pydantic==1.10.26" in flattened
    assert "spacy==3.7.2" in flattened
    assert str(tmp_path / "Motion").casefold() not in flattened
    assert "pymel" not in flattened
    assert "triton==" not in flattened
    assert "nvidia-" not in flattened


@pytest.mark.parametrize(
    ("payload", "lane", "version", "index", "sympy", "lock_digest"),
    (
        (
            {"platform": "linux", "arch": "x64", "accelerator": "cpu", "gpu_sm": 0},
            "cpu",
            "2.4.1",
            "https://download.pytorch.org/whl/cpu",
            "sympy==1.13.3",
            "ef088a666241cf26b91482a46421d53400d0e56db5b08758a20c0d37b2fd7122",
        ),
        (
            {
                "platform": "linux",
                "arch": "x64",
                "accelerator": "cuda",
                "gpu_sm": 89,
                "cuda_version": 124,
            },
            "cu124",
            "2.4.1",
            "https://download.pytorch.org/whl/cu124",
            "sympy==1.13.3",
            "049e3153835071350d579c3d1360ffdd703cf565692dfe71f6e0e27dacd490fb",
        ),
        (
            {
                "platform": "linux",
                "arch": "arm64",
                "accelerator": "cuda",
                "gpu_sm": 89,
                "cuda_version": 126,
            },
            "cu126",
            "2.6.0",
            "https://download.pytorch.org/whl/cu126",
            "sympy==1.13.1",
            "99859f9598ead2453b0eb692734078a5cd2a977c192906030c1ad9249e6973aa",
        ),
        (
            {
                "platform": "linux",
                "arch": "x64",
                "accelerator": "cuda",
                "gpu_sm": 120,
                "cuda_version": 128,
            },
            "cu128",
            "2.7.1",
            "https://download.pytorch.org/whl/cu128",
            "sympy==1.13.3",
            "911bbaa47e21aa7fedc99920f8d67ea6b34836b72cf03daa2d16599b92b2bba3",
        ),
        (
            {
                "platform": "linux",
                "arch": "arm64",
                "accelerator": "cuda",
                "gpu_sm": 121,
                "cuda_version": 128,
            },
            "cu130",
            "2.9.1",
            "https://download.pytorch.org/whl/cu130",
            "sympy==1.13.3",
            "ecd37d0564a23e7816770436c9d6bdb42a7bd9ee2f126ae5835c22aa61fc287e",
        ),
    ),
)
def test_pip_commands_and_locks_follow_exact_torch_lane(
    tmp_path: Path, payload, lane, version, index, sympy, lock_digest, monkeypatch
) -> None:
    monkeypatch.setattr(deps, "tegra_evidence", lambda _payload: None)
    plan = deps.select_dependency_plan(payload)
    commands = deps.pip_install_commands(tmp_path / "python", plan, tmp_path / "Motion")

    assert plan.torch_lane == lane
    assert commands[2][-3:] == ("--index-url", index, f"torch=={version}")
    assert sympy in commands[1]
    assert ({"sympy==1.13.1", "sympy==1.13.3"} - {sympy}).isdisjoint(commands[1])
    assert deps.locked_distribution_versions(plan)["sympy"] == sympy.partition("==")[2]
    assert deps.requirements_digest(plan) == lock_digest


def test_requirement_digests_distinguish_all_torch_lanes(monkeypatch) -> None:
    monkeypatch.setattr(deps, "tegra_evidence", lambda _payload: None)
    payloads = (
        {"platform": "linux", "arch": "x64", "accelerator": "cpu", "gpu_sm": 0},
        {
            "platform": "linux", "arch": "x64", "accelerator": "cuda",
            "gpu_sm": 89, "cuda_version": 124,
        },
        {
            "platform": "linux", "arch": "arm64", "accelerator": "cuda",
            "gpu_sm": 89, "cuda_version": 126,
        },
        {
            "platform": "linux", "arch": "x64", "accelerator": "cuda",
            "gpu_sm": 120, "cuda_version": 128,
        },
        {
            "platform": "linux", "arch": "arm64", "accelerator": "cuda",
            "gpu_sm": 121, "cuda_version": 128,
        },
    )
    digests = {deps.requirements_digest(deps.select_dependency_plan(payload)) for payload in payloads}
    assert len(digests) == 5


def test_install_dependencies_is_fully_mockable(tmp_path: Path) -> None:
    plan = deps.select_dependency_plan(
        {"platform": "linux", "arch": "x64", "accelerator": "cpu", "gpu_sm": 0}
    )
    calls = []
    motion = tmp_path / "Motion"
    motion.mkdir()
    source = motion / "BVH.py"
    source.write_bytes(b"immutable-motion-source")
    before = source.read_bytes()

    def runner(command, *, stage):
        calls.append((tuple(command), stage))

    deps.install_dependencies(
        tmp_path / "python",
        plan,
        motion,
        log=lambda _message: None,
        runner=runner,
    )
    assert len(calls) == 4
    assert calls[-1][0][-1] == "check"
    assert source.read_bytes() == before


def test_sanitized_environment_preserves_nvidia_loader_paths() -> None:
    environment = deps.sanitize_subprocess_environment(
        {
            "PATH": "/bin",
            "LD_LIBRARY_PATH": "/usr/local/nvidia/lib64",
            "LIBRARY_PATH": "/usr/local/nvidia/lib64",
            "HF_TOKEN": "must-not-leak",
            "PYTHONPATH": "/untrusted/site-packages",
        }
    )
    assert environment["LD_LIBRARY_PATH"] == "/usr/local/nvidia/lib64"
    assert environment["LIBRARY_PATH"] == "/usr/local/nvidia/lib64"
    assert "HF_TOKEN" not in environment
    assert "PYTHONPATH" not in environment


def test_dependency_smoke_rejects_wrong_torch_flavor(tmp_path: Path, monkeypatch) -> None:
    plan = deps.select_dependency_plan(
        {
            "platform": "linux",
            "arch": "x64",
            "accelerator": "cuda",
            "gpu_sm": 89,
            "cuda_version": 124,
        }
    )

    def fake_run(*_args, **_kwargs):
        payload = {
            "checkpoints": 5,
            "cuda_matmul": True,
            "torch_cuda_runtime": "11.8",
            "gpu_sm": 89,
            "torch_arch_match": "sm_89",
        }
        return deps.subprocess.CompletedProcess([], 0, stdout=json.dumps(payload) + "\n", stderr="")

    monkeypatch.setattr(deps.subprocess, "run", fake_run)
    with pytest.raises(deps.DependencyError, match="TORCH_FLAVOR_INVALID"):
        deps.verify_dependencies(
            tmp_path / "python",
            plan,
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
        )


def test_dependency_smoke_requires_compatible_compiled_arch(tmp_path: Path, monkeypatch) -> None:
    plan = deps.select_dependency_plan(
        {
            "platform": "linux",
            "arch": "x64",
            "accelerator": "cuda",
            "gpu_sm": 89,
            "cuda_version": 124,
        }
    )

    def result(architectures, match):
        payload = {
            "checkpoints": 5,
            "cuda_matmul": True,
            "torch_cuda_runtime": "12.4",
            "gpu_sm": 89,
            "torch_arch_list": architectures,
            "torch_arch_match": match,
        }
        return deps.subprocess.CompletedProcess([], 0, stdout=json.dumps(payload) + "\n", stderr="")

    monkeypatch.setattr(deps.subprocess, "run", lambda *_a, **_k: result(["sm_90"], "sm_90"))
    with pytest.raises(deps.DependencyError, match="TORCH_ARCH_INVALID"):
        deps.verify_dependencies(
            tmp_path / "python", plan, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path
        )
    monkeypatch.setattr(
        deps.subprocess, "run", lambda *_a, **_k: result(["sm_80", "compute_86"], "compute_86")
    )
    verified = deps.verify_dependencies(
        tmp_path / "python", plan, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path
    )
    assert verified["torch_arch_match"] == "compute_86"


def _context(tmp_path: Path) -> setup.SetupContext:
    extension = tmp_path / "extension"
    extension.mkdir()
    fingerprint = {
        "implementation": "cpython",
        "version": [3, 11],
        "pointer_bits": 64,
        "machine": platform.machine().lower(),
    }
    payload = {
        "python_exe": sys.executable,
        "ext_dir": str(extension),
        "platform": "linux",
        "arch": "x64",
        "accelerator": "cpu",
        "gpu_sm": 0,
        "cuda_version": 0,
    }
    return setup.SetupContext(
        Path(sys.executable), extension, 0, 0, "cpu", "linux", "x64", payload, fingerprint
    )


def test_repair_reuses_valid_environment_without_install(tmp_path: Path, monkeypatch) -> None:
    context = _context(tmp_path)
    venv = context.ext_dir / setup.VENV_NAME
    python = setup.venv_python(venv, "linux")
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    state = context.ext_dir / setup.SETUP_STATE_FILENAME
    state.write_text("{}")
    revision = tmp_path / "revision"
    revision.mkdir()
    plan = deps.select_dependency_plan(context.payload)
    expected = deps.dependency_state_payload(plan, context.host_fingerprint)
    monkeypatch.setattr(deps, "state_matches", lambda *_args: True)
    monkeypatch.setattr(setup, "interpreter_fingerprint", lambda _python: dict(context.host_fingerprint))
    monkeypatch.setattr(setup, "_environment_smoke", lambda *_args: {"checkpoints": 5})
    result = setup._reusable_environment(context, plan, revision, expected)
    assert result is not None and result.reused is True


def test_install_promotes_only_after_smoke_and_state(tmp_path: Path, monkeypatch) -> None:
    context = _context(tmp_path)
    revision = tmp_path / "revision"
    (revision / "source" / "Motion").mkdir(parents=True)
    plan = deps.select_dependency_plan(context.payload)
    events: list[str] = []

    def fake_create(_context, staging):
        python = setup.venv_python(staging, "linux")
        python.parent.mkdir(parents=True)
        python.write_bytes(b"python")
        events.append("venv")
        return python

    def fake_install(*_args, **_kwargs):
        events.append("install")

    def fake_smoke(*_args):
        events.append("smoke")
        return {"checkpoints": 5}

    def fake_state(path, _payload):
        events.append("state")
        path.write_text("{}")
        return path

    monkeypatch.setattr(setup, "_create_venv", fake_create)
    monkeypatch.setattr(deps, "install_dependencies", fake_install)
    monkeypatch.setattr(setup, "_environment_smoke", fake_smoke)
    monkeypatch.setattr(deps, "write_state", fake_state)
    monkeypatch.setattr(setup, "interpreter_fingerprint", lambda _python: dict(context.host_fingerprint))
    monkeypatch.setattr(setup, "_available_bytes", lambda _path: 100 * setup.GIB)
    expected = deps.dependency_state_payload(plan, context.host_fingerprint)
    result = setup._install_environment(context, plan, revision, expected)
    assert result.python == setup.venv_python(context.ext_dir / "venv", "linux")
    assert events == ["venv", "install", "smoke", "state", "smoke"]


def _write_fake_venv(path: Path, generation: str) -> None:
    path.mkdir(parents=True)
    (path / "generation.txt").write_text(generation, encoding="utf-8")


def test_recovery_restores_config_moved_before_old_venv_backup(tmp_path: Path) -> None:
    """Crash boundary: config was backed up, but the old venv is still active."""

    context = _context(tmp_path)
    extension = context.ext_dir
    _write_fake_venv(extension / setup.VENV_NAME, "old")
    _write_fake_venv(extension / setup.VENV_STAGING_NAME, "new-staging")
    (extension / setup.SETUP_STATE_FILENAME).write_text("old-state", encoding="utf-8")
    (extension / setup.STATE_STAGING_FILENAME).write_text("new-state", encoding="utf-8")
    (extension / setup.CONFIG_BACKUP_FILENAME).write_text("old-config", encoding="utf-8")

    setup._recover_transaction(context)

    assert (extension / setup.VENV_NAME / "generation.txt").read_text() == "old"
    assert (extension / setup.SETUP_STATE_FILENAME).read_text() == "old-state"
    assert (extension / setup.RUNTIME_CONFIG_FILENAME).read_text() == "old-config"
    assert not (extension / setup.CONFIG_BACKUP_FILENAME).exists()
    assert not (extension / setup.VENV_STAGING_NAME).exists()
    assert not (extension / setup.STATE_STAGING_FILENAME).exists()


@pytest.mark.parametrize("new_venv_promoted", (False, True))
def test_recovery_preserves_old_state_around_venv_promotion_boundary(
    tmp_path: Path, new_venv_promoted: bool
) -> None:
    """The state is still old until its own backup step has happened."""

    context = _context(tmp_path)
    extension = context.ext_dir
    _write_fake_venv(extension / setup.VENV_BACKUP_NAME, "old")
    if new_venv_promoted:
        _write_fake_venv(extension / setup.VENV_NAME, "new")
    else:
        _write_fake_venv(extension / setup.VENV_STAGING_NAME, "new-staging")
    (extension / setup.SETUP_STATE_FILENAME).write_text("old-state", encoding="utf-8")
    (extension / setup.STATE_STAGING_FILENAME).write_text("new-state", encoding="utf-8")
    (extension / setup.CONFIG_BACKUP_FILENAME).write_text("old-config", encoding="utf-8")

    setup._recover_transaction(context)

    assert (extension / setup.VENV_NAME / "generation.txt").read_text() == "old"
    assert (extension / setup.SETUP_STATE_FILENAME).read_text() == "old-state"
    assert (extension / setup.RUNTIME_CONFIG_FILENAME).read_text() == "old-config"
    assert not (extension / setup.VENV_STAGING_NAME).exists()
    assert not (extension / setup.STATE_STAGING_FILENAME).exists()


@pytest.mark.parametrize("new_state_promoted", (False, True))
def test_recovery_restores_state_after_its_backup_boundary(
    tmp_path: Path, new_state_promoted: bool
) -> None:
    context = _context(tmp_path)
    extension = context.ext_dir
    _write_fake_venv(extension / setup.VENV_BACKUP_NAME, "old")
    _write_fake_venv(extension / setup.VENV_NAME, "new")
    (extension / setup.STATE_BACKUP_FILENAME).write_text("old-state", encoding="utf-8")
    if new_state_promoted:
        (extension / setup.SETUP_STATE_FILENAME).write_text("new-state", encoding="utf-8")
    (extension / setup.CONFIG_BACKUP_FILENAME).write_text("old-config", encoding="utf-8")

    setup._recover_transaction(context)

    assert (extension / setup.VENV_NAME / "generation.txt").read_text() == "old"
    assert (extension / setup.SETUP_STATE_FILENAME).read_text() == "old-state"
    assert (extension / setup.RUNTIME_CONFIG_FILENAME).read_text() == "old-config"


def test_recovery_keeps_committed_generation_after_venv_backup_cleanup(tmp_path: Path) -> None:
    """Removing the venv backup is the durable commit point."""

    context = _context(tmp_path)
    extension = context.ext_dir
    _write_fake_venv(extension / setup.VENV_NAME, "new")
    (extension / setup.SETUP_STATE_FILENAME).write_text("new-state", encoding="utf-8")
    (extension / setup.RUNTIME_CONFIG_FILENAME).write_text("new-config", encoding="utf-8")
    (extension / setup.STATE_BACKUP_FILENAME).write_text("old-state", encoding="utf-8")
    (extension / setup.CONFIG_BACKUP_FILENAME).write_text("old-config", encoding="utf-8")

    setup._recover_transaction(context)

    assert (extension / setup.VENV_NAME / "generation.txt").read_text() == "new"
    assert (extension / setup.SETUP_STATE_FILENAME).read_text() == "new-state"
    assert (extension / setup.RUNTIME_CONFIG_FILENAME).read_text() == "new-config"
    assert not (extension / setup.STATE_BACKUP_FILENAME).exists()
    assert not (extension / setup.CONFIG_BACKUP_FILENAME).exists()


def test_failed_fresh_config_publication_removes_new_generation(tmp_path: Path) -> None:
    context = _context(tmp_path)
    venv = context.ext_dir / setup.VENV_NAME
    setup.venv_python(venv, "linux").parent.mkdir(parents=True)
    setup.venv_python(venv, "linux").write_bytes(b"python")
    state = context.ext_dir / setup.SETUP_STATE_FILENAME
    state.write_text("new")
    config = context.ext_dir / setup.RUNTIME_CONFIG_FILENAME
    config.write_text("partial")
    setup._rollback_promotion(
        context,
        setup.EnvironmentPromotion(False, False, False),
    )
    assert not venv.exists()
    assert not state.exists()
    assert not config.exists()


def test_failed_publication_restores_config_backup_without_old_venv(tmp_path: Path) -> None:
    context = _context(tmp_path)
    venv = context.ext_dir / setup.VENV_NAME
    setup.venv_python(venv, "linux").parent.mkdir(parents=True)
    setup.venv_python(venv, "linux").write_bytes(b"new")
    state = context.ext_dir / setup.SETUP_STATE_FILENAME
    state.write_text("new-state")
    config = context.ext_dir / setup.RUNTIME_CONFIG_FILENAME
    config.write_text("partial")
    config_backup = context.ext_dir / setup.CONFIG_BACKUP_FILENAME
    config_backup.write_text("old-config")
    setup._rollback_promotion(
        context,
        setup.EnvironmentPromotion(False, False, True),
    )
    assert not venv.exists()
    assert not state.exists()
    assert config.read_text() == "old-config"


def test_failed_wrapper_update_keeps_published_snapshot_untouched(
    tmp_path: Path, monkeypatch
) -> None:
    context = _context(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    revision = setup.owned_snapshot_directory(models, create=True)
    source = revision / "source" / "AnyTop" / "sentinel.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source-used-by-previous-wrapper")
    marker = revision / assets.READY_MARKER_FILENAME
    marker.write_text(json.dumps(assets.ready_payload()), encoding="utf-8")
    marker_before = marker.read_bytes()
    source_before = source.read_bytes()

    monkeypatch.setattr(deps, "validate_host_runtime", lambda _plan: {})
    monkeypatch.setattr(setup, "resolve_models_root", lambda *_a, **_k: models)
    monkeypatch.setattr(setup, "_preflight_storage", lambda *_a, **_k: None)
    monkeypatch.setattr(assets, "verify_snapshot", lambda *_a, **_k: [])
    monkeypatch.setattr(setup, "verify_snapshot", lambda *_a, **_k: [])

    def fail_environment(*_args, **_kwargs):
        raise setup.SetupFailure("TEST_UPDATE_FAILED", "simulated wrapper update failure")

    monkeypatch.setattr(setup, "install_or_reuse_environment", fail_environment)
    with pytest.raises(setup.SetupFailure, match="TEST_UPDATE_FAILED"):
        setup._run_setup_locked(context)

    assert marker.read_bytes() == marker_before
    assert source.read_bytes() == source_before


def test_validate_context_uses_cpython311_contract(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "extension"
    root.mkdir()
    monkeypatch.setattr(setup, "current_platform_name", lambda: "linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        setup,
        "interpreter_fingerprint",
        lambda _python: {
            "implementation": "cpython",
            "version": [3, 11],
            "pointer_bits": 64,
        },
    )
    context = setup.validate_context(
        {
            "python_exe": sys.executable,
            "ext_dir": str(root),
            "platform": "linux",
            "arch": "x64",
            "accelerator": "cpu",
            "gpu_sm": 0,
        },
        root,
    )
    assert context.arch == "x64"
    assert context.host_fingerprint["version"] == [3, 11]


def test_setup_main_reports_known_errors_without_subprocess(monkeypatch, capsys) -> None:
    def fail(_payload, root=setup.ROOT):
        raise deps.DependencyError("GPU_SM_UNSUPPORTED", "no validated lane")

    monkeypatch.setattr(setup, "run_setup", fail)
    result = setup.main(["setup.py", json.dumps({"python_exe": "x"})])
    assert result == 1
    assert "GPU_SM_UNSUPPORTED" in capsys.readouterr().err

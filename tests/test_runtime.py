from __future__ import annotations

import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from anytop_modly import runtime
from anytop_modly import assets
from anytop_modly.bundles import BundleError, verify_bundle


class ProtocolTests(unittest.TestCase):
    def test_success_is_monotonic_and_has_one_terminal_record(self) -> None:
        output = io.StringIO()
        sentinel = object()
        result_path = Path(tempfile.gettempdir()) / "result.glb"

        def fake_process(request: object, emitter: runtime.ProtocolEmitter):
            self.assertIs(request, sentinel)
            emitter.progress(60, "heavy work")
            emitter.progress(40, "cannot regress")
            return "file", result_path

        with mock.patch.object(runtime, "validate_request", return_value=sentinel), mock.patch.object(
            runtime, "process", side_effect=fake_process
        ):
            code = runtime.run_protocol(io.StringIO('{"input":{},"params":{}}\n'), output)

        self.assertEqual(code, 0)
        import json

        records = [json.loads(line) for line in output.getvalue().splitlines()]
        progress = [record["percent"] for record in records if record["type"] == "progress"]
        self.assertEqual(progress, sorted(progress))
        self.assertEqual(progress[-1], 100)
        terminal = [record for record in records if record["type"] in {"done", "error"}]
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["result"]["filePath"], str(result_path))

    def test_multiple_request_lines_fail_with_one_public_error(self) -> None:
        output = io.StringIO()
        code = runtime.run_protocol(io.StringIO("{}\n{}\n"), output)
        self.assertEqual(code, 1)
        import json

        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(records, [{
            "type": "error",
            "message": "[REQUEST_INVALID] AnyTop received an invalid Modly process request. Recreate the node and try again.",
        }])

    def test_emitter_clamps_regressions_and_closes_after_terminal(self) -> None:
        output = io.StringIO()
        emitter = runtime.ProtocolEmitter(output)
        emitter.progress(20, "one")
        emitter.progress(10, "two")
        emitter.done_text("ok")
        with self.assertRaises(RuntimeError):
            emitter.error("late")
        import json

        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([record["percent"] for record in records[:2]], [20, 20])
        self.assertEqual(records[-1], {"type": "done", "result": {"text": "ok"}})


class BundlePackagingTests(unittest.TestCase):
    def test_output_run_rejects_linked_workflows_before_external_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            outside = root / "outside"
            workspace.mkdir()
            outside.mkdir()
            workflows = workspace / "Workflows"
            try:
                workflows.symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("directory symlinks are not permitted")

            with self.assertRaises(runtime.ProcessFailure):
                runtime.OutputRun.create(workspace, "generate-custom")
            self.assertFalse((outside / "AnyTop").exists())

    def test_packaging_is_chainable_and_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            run = runtime.OutputRun.create(workspace, "generate-custom")
            upstream = run.staging / "upstream"
            upstream.mkdir()
            motion = upstream / "Custom_rep_0.npy"
            bvh = upstream / "Custom_rep_0.bvh"
            video = upstream / "Custom_rep_0.mp4"
            condition = root / "source-condition.npy"
            motion.write_bytes(b"npy-motion")
            bvh.write_text("HIERARCHY\nMOTION\n", encoding="utf-8")
            video.write_bytes(b"mp4")
            condition.write_bytes(b"trusted-condition")
            authentication_key = b"K" * 32

            fake_glb = types.ModuleType("anytop_modly.glb")

            def bvh_to_glb(
                bvh_path: Path,
                output_path: Path,
                *,
                extras: object,
                motion_source: Path,
            ) -> Path:
                self.assertEqual(bvh_path.name, "motion-00.bvh")
                self.assertEqual(bvh_path.parent, run.staging)
                self.assertEqual(motion_source, root)
                self.assertIsInstance(extras, dict)
                output_path.write_bytes(b"glTF-preview")
                return output_path

            fake_glb.bvh_to_glb = bvh_to_glb  # type: ignore[attr-defined]
            with mock.patch.dict(sys.modules, {"anytop_modly.glb": fake_glb}):
                result = runtime._package_motion_bundle(
                    run=run,
                    result={
                        "object_name": "Custom",
                        "items": [{
                            "motion": str(motion.resolve()),
                            "bvh": str(bvh.resolve()),
                            "video": str(video.resolve()),
                        }],
                    },
                    operation="generate-custom",
                    condition_kind="custom",
                    condition_source=condition,
                    parameters={"seed": 10},
                    provenance={"modelFamily": "unified"},
                    motion_source=root,
                    authentication_key=authentication_key,
                )

            bundle = verify_bundle(result, authentication_key=authentication_key)
            self.assertEqual(bundle.object_name, "Custom")
            self.assertIsNotNone(bundle.motion)
            self.assertIsNotNone(bundle.condition)
            self.assertEqual(bundle.motion.read_bytes(), b"npy-motion")
            self.assertEqual(bundle.condition.read_bytes(), b"trusted-condition")
            self.assertFalse(any(result.parent.rglob("model*.pt")))

            bundle.motion.write_bytes(b"tampered")
            with self.assertRaises(BundleError):
                verify_bundle(result, authentication_key=authentication_key)

    def test_upper_body_roots_are_bounded_unique_integers(self) -> None:
        self.assertEqual(runtime._upper_body_roots({"upper_body_root": "0, 4 9"}), [0, 4, 9])
        for value in ("", "1,1", "-1", "a", [True]):
            with self.assertRaises(runtime.ProcessFailure):
                runtime._upper_body_roots({"upper_body_root": value})

    def test_edit_ratios_must_be_ordered(self) -> None:
        self.assertEqual(runtime._number({"value": 0.25}, "value", 0.0, 0.0, 1.0), 0.25)
        with self.assertRaises(runtime.ProcessFailure):
            runtime._number({"value": float("nan")}, "value", 0.0, 0.0, 1.0)

    def test_hidden_edit_controls_do_not_block_the_other_mode(self) -> None:
        self.assertEqual(
            runtime._edit_controls({
                "edit_mode": "upper_body",
                "prefix_end": 500,
                "suffix_start": -100,
                "upper_body_root": "2,5",
            }),
            ("upper_body", 0.25, 0.75, [2, 5]),
        )
        self.assertEqual(
            runtime._edit_controls({
                "edit_mode": "in_between",
                "prefix_end": 0.2,
                "suffix_start": 0.8,
                "upper_body_root": "not-an-index",
            }),
            ("in_between", 0.2, 0.8, [0]),
        )
        with self.assertRaises(runtime.ProcessFailure):
            runtime._edit_controls({
                "edit_mode": "in_between",
                "prefix_end": 0.8,
                "suffix_start": 0.2,
            })


class WorkerLaunchTests(unittest.TestCase):
    def test_worker_environment_preserves_gpu_loader_paths_only(self) -> None:
        source = {
            "PATH": "/usr/bin",
            "LD_LIBRARY_PATH": "/usr/local/nvidia/lib64:/opt/cuda/lib64",
            "LIBRARY_PATH": "/opt/cuda/lib64",
            "NVIDIA_VISIBLE_DEVICES": "all",
            "CUDA_HOME": "/opt/cuda",
            "PYTHONPATH": "/private/injected-python",
            "PYTHONHOME": "/private/injected-home",
            "HF_TOKEN": "secret-token",
            "API_KEY": "secret-key",
            "LD_PRELOAD": "/private/injected.so",
        }
        with mock.patch.dict(runtime.os.environ, source, clear=True):
            environment = runtime._worker_environment()
        self.assertEqual(environment["LD_LIBRARY_PATH"], source["LD_LIBRARY_PATH"])
        self.assertEqual(environment["LIBRARY_PATH"], source["LIBRARY_PATH"])
        self.assertEqual(environment["NVIDIA_VISIBLE_DEVICES"], "all")
        self.assertEqual(environment["CUDA_HOME"], "/opt/cuda")
        for blocked in ("PYTHONPATH", "PYTHONHOME", "HF_TOKEN", "API_KEY", "LD_PRELOAD"):
            self.assertNotIn(blocked, environment)

    def test_child_disables_bytecode_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            temp = root / "temp"
            staging.mkdir()
            temp.mkdir()
            observed: dict[str, object] = {}

            class FakeProcess:
                returncode = 0
                pid = 123

                def communicate(self, payload: bytes) -> None:
                    request = __import__("json").loads(payload)
                    Path(request["result_path"]).write_text(
                        '{"ok":true,"result":{"value":1}}\n', encoding="utf-8"
                    )

            def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
                observed["command"] = command
                observed["environment"] = kwargs["env"]
                return FakeProcess()

            with mock.patch.object(runtime.subprocess, "Popen", side_effect=fake_popen):
                result = runtime.run_worker({"operation": "fake"}, staging, temp)
            self.assertEqual(result, {"value": 1})
            self.assertEqual(observed["command"][1:3], ["-B", "-m"])
            self.assertEqual(observed["environment"]["PYTHONDONTWRITEBYTECODE"], "1")

    def test_worker_diagnostic_is_bounded_and_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "stderr.log"
            private_path = "/private/user/project/skeleton.bvh"
            secret = "do-not-leak-this-value"
            log.write_text(
                ("noise" * 5000)
                + f'\nFile "{private_path}", line 5\nTOKEN={secret}\nRuntimeError: bad skeleton\n',
                encoding="utf-8",
            )
            diagnostic = runtime._sanitized_worker_tail(log)
            self.assertIsNotNone(diagnostic)
            self.assertLessEqual(len(diagnostic), len("AnyTop worker diagnostic: ") + runtime.MAX_WORKER_DIAGNOSTIC_CHARS)
            self.assertNotIn(private_path, diagnostic)
            self.assertNotIn(secret, diagnostic)
            self.assertIn("<path>", diagnostic)
            self.assertIn("<redacted>", diagnostic)


class SnapshotVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        with runtime._SNAPSHOT_VERIFICATION_LOCK:
            runtime._VERIFIED_SNAPSHOT_IDENTITIES.clear()

    def tearDown(self) -> None:
        with runtime._SNAPSHOT_VERIFICATION_LOCK:
            runtime._VERIFIED_SNAPSHOT_IDENTITIES.clear()

    def test_full_snapshot_hashing_runs_once_per_marker_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            revision = Path(temporary).resolve()
            marker = revision / "ready.json"
            marker.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(assets, "verify_snapshot", return_value=[]) as verify:
                runtime._verify_snapshot_once(revision, marker)
                runtime._verify_snapshot_once(revision, marker)
            self.assertEqual(verify.call_count, 1)

    def test_failed_full_snapshot_hash_is_not_cached(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            revision = Path(temporary).resolve()
            marker = revision / "ready.json"
            marker.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(
                assets,
                "verify_snapshot",
                return_value=["t5-base/model.safetensors: SHA-256 mismatch"],
            ) as verify:
                for _attempt in range(2):
                    with self.assertRaises(runtime.ProcessFailure) as raised:
                        runtime._verify_snapshot_once(revision, marker)
                    self.assertEqual(raised.exception.code, "ASSET_INVALID")
            self.assertEqual(verify.call_count, 2)


class RuntimeStateRoutingTests(unittest.TestCase):
    def _snapshot(self, root: Path, revision: Path) -> dict[str, object]:
        paths = runtime.snapshot_paths(revision)
        for directory in (
            paths.anytop_source,
            paths.motion_source,
            paths.checkpoints,
            paths.builtin_cond.parent,
            paths.t5,
            revision / "runtime-cache",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        paths.builtin_cond.write_bytes(b"cond")
        ready = assets.ready_payload()
        paths.ready_marker.write_text(
            __import__("json").dumps(ready, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "schema_version": 1,
            "extension_id": runtime.EXTENSION_ID,
            "extension_version": runtime.EXTENSION_VERSION,
            "revision_id": runtime.REVISION_ID,
            "models_dir": str(root.resolve()),
            "revision_root": str(revision.resolve()),
            "source_root": str(paths.anytop_source.resolve()),
            "motion_source": str(paths.motion_source.resolve()),
            "checkpoints_root": str(paths.checkpoints.resolve()),
            "builtin_cond": str(paths.builtin_cond.resolve()),
            "t5_path": str(paths.t5.resolve()),
            "ready_marker": str(paths.ready_marker.resolve()),
            "runtime_cache_dir": str((revision / "runtime-cache").resolve()),
            "available_devices": ["cpu"],
            "default_device": "cpu",
        }

    def _write_config(self, path: Path, payload: dict[str, object]) -> None:
        path.write_text(
            __import__("json").dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_load_state_accepts_only_the_canonical_snapshot_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = root / "models"
            models.mkdir()
            revision = (
                models
                / runtime.EXTENSION_ID
                / "anytop"
                / "revisions"
                / runtime.REVISION_ID
            )
            payload = self._snapshot(models, revision)
            config = root / "runtime_config.json"
            self._write_config(config, payload)
            with mock.patch.object(runtime, "_verify_snapshot_once"), mock.patch.object(
                runtime, "load_bundle_auth_key", return_value=b"K" * 32
            ):
                state = runtime.load_state(config)
            self.assertEqual(state.revision_root, revision.resolve())
            self.assertEqual(
                state.source_root,
                runtime.snapshot_paths(state.revision_root).anytop_source,
            )

    def test_rejects_anytop_motion_and_t5_redirection_inside_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = root / "models"
            models.mkdir()
            revision = (
                models
                / runtime.EXTENSION_ID
                / "anytop"
                / "revisions"
                / runtime.REVISION_ID
            )
            original = self._snapshot(models, revision)
            redirects = revision / "runtime-cache" / "redirects"
            for name in ("anytop", "motion", "t5"):
                (redirects / name).mkdir(parents=True)
            for key, target in (
                ("source_root", redirects / "anytop"),
                ("motion_source", redirects / "motion"),
                ("t5_path", redirects / "t5"),
            ):
                with self.subTest(key=key):
                    payload = dict(original)
                    payload[key] = str(target.resolve())
                    config = root / f"{key}.json"
                    self._write_config(config, payload)
                    with mock.patch.object(runtime, "_verify_snapshot_once"), mock.patch.object(
                        runtime, "load_bundle_auth_key", return_value=b"K" * 32
                    ):
                        with self.assertRaises(runtime.ProcessFailure) as raised:
                            runtime.load_state(config)
                    self.assertEqual(raised.exception.code, "SETUP_REQUIRED")

    def test_rejects_revision_root_that_is_not_the_owned_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = root / "models"
            models.mkdir()
            canonical = (
                models
                / runtime.EXTENSION_ID
                / "anytop"
                / "revisions"
                / runtime.REVISION_ID
            )
            self._snapshot(models, canonical)
            redirected = models / "redirected-revision"
            payload = self._snapshot(models, redirected)
            config = root / "runtime_config.json"
            self._write_config(config, payload)
            with mock.patch.object(runtime, "_verify_snapshot_once"), mock.patch.object(
                runtime, "load_bundle_auth_key", return_value=b"K" * 32
            ):
                with self.assertRaises(runtime.ProcessFailure) as raised:
                    runtime.load_state(config)
            self.assertEqual(raised.exception.code, "SETUP_REQUIRED")

    def test_rejects_alias_in_canonical_snapshot_components(self) -> None:
        if not hasattr(Path, "symlink_to"):
            self.skipTest("symlinks are unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = root / "models"
            models.mkdir()
            revision = (
                models
                / runtime.EXTENSION_ID
                / "anytop"
                / "revisions"
                / runtime.REVISION_ID
            )
            payload = self._snapshot(models, revision)
            source = revision / "source"
            outside = revision / "runtime-cache" / "aliased-source"
            source.rename(outside)
            try:
                source.symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("symlinks are not permitted")
            config = root / "runtime_config.json"
            self._write_config(config, payload)
            with mock.patch.object(runtime, "_verify_snapshot_once"), mock.patch.object(
                runtime, "load_bundle_auth_key", return_value=b"K" * 32
            ):
                with self.assertRaises(runtime.ProcessFailure) as raised:
                    runtime.load_state(config)
            self.assertEqual(raised.exception.code, "SETUP_REQUIRED")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from anytop_modly import worker


class LocalT5Tests(unittest.TestCase):
    def test_local_path_is_registered_before_original_constructor(self) -> None:
        class FakeT5:
            MODELS = ["t5-base"]
            MODELS_DIMS = {"t5-base": 768}

            def __init__(self, name: str, finetune: bool, device: str, **kwargs: object) -> None:
                assert name in self.MODELS
                assert self.MODELS_DIMS[name] == 768
                self.name = name
                self.device = device
                self.finetune = finetune

        module = types.SimpleNamespace(T5Conditioner=FakeT5)
        local = Path("/models/AnyTop/t5-base")
        worker._patch_t5(module, local, "cpu")
        instance = module.T5Conditioner(
            name="t5-base",
            finetune=False,
            device="cuda",
            word_dropout=0.0,
        )
        self.assertEqual(instance.name, str(local))
        self.assertEqual(instance.device, "cpu")
        self.assertIn(str(local), FakeT5.MODELS)
        self.assertEqual(FakeT5.MODELS_DIMS[str(local)], 768)


class CorrespondenceWorkerTests(unittest.TestCase):
    def test_dift_stages_explicit_names_and_removes_checkpoint_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = root / "models"
            checkpoint_dir = models / "all"
            checkpoint_dir.mkdir(parents=True)
            checkpoint = checkpoint_dir / "model000459999.pt"
            checkpoint.write_bytes(b"pinned-checkpoint")
            (checkpoint_dir / "args.json").write_text("{}", encoding="utf-8")
            original_inventory = {
                path.relative_to(models).as_posix(): path.read_bytes()
                for path in models.rglob("*") if path.is_file()
            }

            reference_motion = root / "random-user-name.npy"
            target_motion = root / "another-name.npy"
            np.save(reference_motion, np.zeros((3, 2, 13), dtype=np.float32), allow_pickle=False)
            np.save(target_motion, np.ones((3, 2, 13), dtype=np.float32), allow_pickle=False)
            reference_cond = root / "reference-cond.npy"
            target_cond = root / "target-cond.npy"
            np.save(reference_cond, {"Monkey": {"object_type": "Monkey"}}, allow_pickle=True)
            np.save(target_cond, {"Spider": {"object_type": "Spider"}}, allow_pickle=True)
            output = root / "run" / "upstream"
            t5 = root / "t5"
            t5.mkdir()

            class FakeT5:
                MODELS = ["t5-base"]
                MODELS_DIMS = {"t5-base": 768}

                def __init__(self, *args: object, **kwargs: object) -> None:
                    pass

            fake_dift = types.ModuleType("sample.dift_correspondence")
            fake_dift.T5Conditioner = FakeT5  # type: ignore[attr-defined]

            def run_dift(*, args: object, cond_dict: object) -> None:
                self.assertEqual(Path(args.sample_ref).name, "Monkey_reference.npy")
                self.assertEqual(Path(args.sample_tgt[0]).name, "Spider_target.npy")
                self.assertTrue(
                    Path(args.model_path).resolve().is_relative_to(output.resolve())
                )
                self.assertEqual(set(cond_dict), {"Monkey", "Spider"})
                destination = Path(args.model_path).parent / Path(args.model_path).stem / "dift_out"
                destination.mkdir(parents=True)
                np.save(destination / "mapping.npy", {"ref": {}, "tgt": {}}, allow_pickle=True)
                (destination / "mapping.mp4").write_bytes(b"video")

            fake_dift.run_dift = run_dift  # type: ignore[attr-defined]
            sample_package = types.ModuleType("sample")
            sample_package.__path__ = []  # type: ignore[attr-defined]
            sample_package.dift_correspondence = fake_dift  # type: ignore[attr-defined]

            request = {
                "checkpoint": str(checkpoint.resolve()),
                "output_dir": str(output.resolve()),
                "device_mode": "cpu",
                "cuda_device": 0,
                "seed": 10,
                "dift_type": "spatial",
                "layer": 0,
                "timestep": 90,
                "reference": {
                    "motion": str(reference_motion.resolve()),
                    "condition": str(reference_cond.resolve()),
                    "object_name": "Monkey",
                },
                "target": {
                    "motion": str(target_motion.resolve()),
                    "condition": str(target_cond.resolve()),
                    "object_name": "Spider",
                },
            }
            with mock.patch.dict(
                sys.modules,
                {"sample": sample_package, "sample.dift_correspondence": fake_dift},
            ), mock.patch.object(worker, "_device", return_value=(-1, "cpu")):
                result = worker._run_correspondence(request, t5)

            self.assertFalse((output / "checkpoint").exists())
            self.assertTrue(Path(result["mappings"][0]).is_file())
            self.assertTrue(Path(result["videos"][0]).is_file())
            self.assertFalse(any(output.rglob("*.pt")))
            final_inventory = {
                path.relative_to(models).as_posix(): path.read_bytes()
                for path in models.rglob("*") if path.is_file()
            }
            self.assertEqual(final_inventory, original_inventory)


class WorkerProtocolTests(unittest.TestCase):
    def test_execute_rejects_unknown_operation_without_importing_upstream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            motion = root / "motion"
            t5 = root / "t5"
            for path in (source, motion, t5):
                path.mkdir()
            with self.assertRaises(worker.WorkerFailure):
                worker.execute({
                    "source_root": str(source.resolve()),
                    "motion_source": str(motion.resolve()),
                    "t5_path": str(t5.resolve()),
                    "operation": "not-real",
                })


if __name__ == "__main__":
    unittest.main()

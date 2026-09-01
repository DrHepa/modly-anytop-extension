from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))


class ManifestContractTests(unittest.TestCase):
    def test_extension_identity_and_process_entry_are_frozen(self) -> None:
        self.assertEqual(MANIFEST["id"], "modly-anytop-extension")
        self.assertEqual(MANIFEST["type"], "process")
        self.assertEqual(MANIFEST["version"], "1.0.0")
        self.assertEqual(MANIFEST["author"], "DrHepa")
        self.assertEqual(MANIFEST["entry"], "processor.py")
        self.assertEqual(
            MANIFEST["source"],
            "https://github.com/DrHepa/modly-anytop-extension",
        )

    def test_node_inventory_and_ports_are_exact(self) -> None:
        expected = {
            "anytop-preprocess": ("text", "mesh"),
            "anytop-generate": ("text", "mesh"),
            "anytop-generate-custom": ("mesh", "mesh"),
            "anytop-edit": ("mesh", "mesh"),
            "anytop-correspondence": ("mesh", "text"),
        }
        self.assertEqual(
            {node["id"]: (node["input"], node["output"]) for node in MANIFEST["nodes"]},
            expected,
        )

    def test_public_parameter_inventory_is_exact(self) -> None:
        expected = {
            "anytop-preprocess": {
                "bvh_directory",
                "right_hip",
                "left_hip",
                "right_shoulder",
                "left_shoulder",
                "tpos_bvh",
            },
            "anytop-generate": {
                "model_family",
                "motion_length",
                "num_repetitions",
                "seed",
                "device_mode",
                "cuda_device",
            },
            "anytop-generate-custom": {
                "model_family",
                "motion_length",
                "num_repetitions",
                "seed",
                "device_mode",
                "cuda_device",
            },
            "anytop-edit": {
                "edit_mode",
                "prefix_end",
                "suffix_start",
                "upper_body_root",
                "model_family",
                "num_repetitions",
                "seed",
                "device_mode",
                "cuda_device",
            },
            "anytop-correspondence": {
                "reference_bundle",
                "dift_type",
                "layer",
                "timestep",
                "model_family",
                "seed",
                "device_mode",
                "cuda_device",
            },
        }
        actual = {
            node["id"]: {param["id"] for param in node["params_schema"]}
            for node in MANIFEST["nodes"]
        }
        self.assertEqual(actual, expected)

    def test_only_modly_042_parameter_types_and_fields_are_used(self) -> None:
        allowed_types = {"select", "int", "float", "string", "file-select"}
        allowed_fields = {
            "id",
            "label",
            "type",
            "default",
            "options",
            "min",
            "max",
            "step",
            "tooltip",
            "show_if",
            "picker_intent",
            "dir_from",
            "extensions",
        }
        for node in MANIFEST["nodes"]:
            params = node["params_schema"]
            self.assertEqual(len(params), len({item["id"] for item in params}))
            by_id = {item["id"]: item for item in params}
            for param in params:
                self.assertIn(param["type"], allowed_types)
                self.assertLessEqual(set(param), allowed_fields)
                self.assertIn("default", param)
                if param["type"] == "select":
                    values = [option["value"] for option in param["options"]]
                    self.assertEqual(len(values), len(set(values)))
                    self.assertIn(param["default"], values)
                if param["type"] in {"int", "float"}:
                    if "min" in param:
                        self.assertGreaterEqual(param["default"], param["min"])
                    if "max" in param:
                        self.assertLessEqual(param["default"], param["max"])
                for controller, expected in param.get("show_if", {}).items():
                    self.assertIn(controller, by_id)
                    values = expected if isinstance(expected, list) else [expected]
                    controlling = by_id[controller]
                    if controlling["type"] == "select":
                        valid = {option["value"] for option in controlling["options"]}
                        self.assertLessEqual(set(values), valid)

    def test_file_select_and_native_pickers_are_wired_to_real_inputs(self) -> None:
        nodes = {node["id"]: node for node in MANIFEST["nodes"]}
        preprocess = {
            param["id"]: param
            for param in nodes["anytop-preprocess"]["params_schema"]
        }
        self.assertEqual(preprocess["bvh_directory"]["picker_intent"], "folder")
        self.assertEqual(preprocess["tpos_bvh"]["type"], "file-select")
        self.assertEqual(preprocess["tpos_bvh"]["dir_from"], "bvh_directory")
        self.assertEqual(preprocess["tpos_bvh"]["extensions"], ["bvh"])

        correspondence = {
            param["id"]: param
            for param in nodes["anytop-correspondence"]["params_schema"]
        }
        self.assertEqual(correspondence["reference_bundle"]["type"], "string")
        self.assertEqual(correspondence["reference_bundle"]["picker_intent"], "mesh")

    def test_upstream_defaults_and_published_limits_are_preserved(self) -> None:
        nodes = {
            node["id"]: {param["id"]: param for param in node["params_schema"]}
            for node in MANIFEST["nodes"]
        }
        generated = nodes["anytop-generate"]
        self.assertEqual(generated["motion_length"]["default"], 6.0)
        self.assertEqual(generated["motion_length"]["max"], 9.8)
        self.assertEqual(generated["num_repetitions"]["default"], 3)
        self.assertEqual(generated["seed"]["default"], 10)
        self.assertEqual(generated["cuda_device"]["default"], 0)

        edited = nodes["anytop-edit"]
        self.assertEqual(edited["edit_mode"]["default"], "in_between")
        self.assertEqual(edited["prefix_end"]["default"], 0.25)
        self.assertEqual(edited["suffix_start"]["default"], 0.75)
        self.assertEqual(edited["upper_body_root"]["default"], "0")

        dift = nodes["anytop-correspondence"]
        self.assertEqual(dift["dift_type"]["default"], "spatial")
        self.assertEqual(dift["layer"]["default"], 0)
        self.assertEqual(dift["timestep"]["default"], 90)

    def test_checkpoint_and_device_choices_match_runtime_routes(self) -> None:
        nodes = {
            node["id"]: {param["id"]: param for param in node["params_schema"]}
            for node in MANIFEST["nodes"]
        }
        expected_families = [
            "auto",
            "unified",
            "bipeds",
            "flying",
            "millipeds_snakes",
            "quadropeds",
        ]
        for node_id in (
            "anytop-generate",
            "anytop-generate-custom",
            "anytop-edit",
            "anytop-correspondence",
        ):
            params = nodes[node_id]
            self.assertEqual(
                [option["value"] for option in params["model_family"]["options"]],
                expected_families,
            )
            self.assertEqual(
                [option["value"] for option in params["device_mode"]["options"]],
                ["auto", "cuda", "cpu"],
            )
            self.assertEqual(
                params["cuda_device"]["show_if"],
                {"device_mode": ["auto", "cuda"]},
            )


if __name__ == "__main__":
    unittest.main()

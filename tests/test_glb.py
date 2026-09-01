from __future__ import annotations

import json
import math
import struct
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from anytop_modly.glb import build_animated_skeleton_glb, bvh_to_glb
import anytop_modly.glb as glb_module


def _axis_angle_wxyz(axis: tuple[float, float, float], angles: list[float]) -> np.ndarray:
    axis_array = np.asarray(axis, dtype=np.float64)
    axis_array /= np.linalg.norm(axis_array)
    half = np.asarray(angles, dtype=np.float64) / 2.0
    return np.column_stack((np.cos(half), np.sin(half)[:, None] * axis_array))


def _chicken_motion() -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, float]:
    """Small generated chicken-like fixture; no Truebones asset is copied."""

    parents = np.asarray([-1, 0, 1, 2, 1], dtype=np.int64)
    names = ["Chicken_Hips", "Chicken_Spine", "Chicken_Neck", "Chicken_Beak", "Chicken_Wíng"]
    frames = 4
    positions = np.zeros((frames, len(parents), 3), dtype=np.float64)
    positions[:, 0] = np.asarray(
        ((0.0, 0.0, 0.0), (0.05, 0.0, 0.02), (0.1, 0.0, 0.05), (0.16, 0.0, 0.09))
    )
    positions[:, 1] = (0.0, 0.45, 0.0)
    positions[:, 2] = (0.0, 0.35, 0.0)
    positions[:, 3] = (0.0, 0.15, 0.25)
    positions[:, 4] = np.asarray(
        ((0.30, 0.20, 0.0), (0.34, 0.20, 0.02), (0.27, 0.21, -0.01), (0.31, 0.20, 0.0))
    )

    rotations = np.zeros((frames, len(parents), 4), dtype=np.float64)
    rotations[..., 0] = 1.0
    rotations[:, 0] = _axis_angle_wxyz((0.0, 1.0, 0.0), [0.0, 0.1, 0.2, 0.3])
    rotations[:, 2] = _axis_angle_wxyz((1.0, 0.0, 0.0), [0.0, 0.04, -0.03, 0.02])
    rotations[:, 4] = _axis_angle_wxyz((0.0, 0.0, 1.0), [0.0, 0.5, -0.4, 0.1])
    # Deliberately alternate equivalent quaternion signs; the writer must make
    # keys hemisphere-continuous before glTF interpolation.
    rotations[1, 1] *= -1.0
    rotations[3, 1] *= -1.0
    return parents, names, positions, rotations, 0.05


def _read_accessor(document: dict, binary: memoryview, index: int) -> np.ndarray:
    return glb_module._accessor_array(document, binary, index).copy()


def _rotation_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


def _global_positions(
    parents: np.ndarray, translations: np.ndarray, rotations_xyzw: np.ndarray
) -> np.ndarray:
    frames, joints = translations.shape[:2]
    local = np.zeros((frames, joints, 4, 4), dtype=np.float64)
    local[..., 3, 3] = 1.0
    for frame in range(frames):
        for joint in range(joints):
            local[frame, joint, :3, :3] = _rotation_matrix_xyzw(rotations_xyzw[frame, joint])
            local[frame, joint, :3, 3] = translations[frame, joint]
    world = np.zeros_like(local)
    world[:, 0] = local[:, 0]
    for joint in range(1, joints):
        world[:, joint] = world[:, parents[joint]] @ local[:, joint]
    return world[..., :3, 3]


def _animation_joint_transforms(
    document: dict, binary: memoryview, joint_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    animation = document["animations"][0]
    first_sampler = animation["samplers"][0]
    times = _read_accessor(document, binary, first_sampler["input"])
    translations = np.repeat(
        np.asarray([document["nodes"][joint]["translation"] for joint in range(joint_count)])[None],
        len(times),
        axis=0,
    )
    rotations = np.repeat(
        np.asarray([document["nodes"][joint]["rotation"] for joint in range(joint_count)])[None],
        len(times),
        axis=0,
    )
    for channel in animation["channels"]:
        target = channel["target"]
        if target["node"] >= joint_count:
            continue
        sampler = animation["samplers"][channel["sampler"]]
        values = _read_accessor(document, binary, sampler["output"])
        if target["path"] == "translation":
            translations[:, target["node"]] = values
        elif target["path"] == "rotation":
            rotations[:, target["node"]] = values
    return times, translations, rotations


def test_build_is_deterministic_embedded_and_semantically_exact() -> None:
    parents, names, positions, rotations_wxyz, frame_time = _chicken_motion()
    extras_a = {
        "z": 2,
        "anytop": {"schemaVersion": 1, "canonicalMotion": "chicken.bvh"},
        "a": [True, None, 1.25],
    }
    extras_b = {
        "a": [True, None, 1.25],
        "anytop": {"canonicalMotion": "chicken.bvh", "schemaVersion": 1},
        "z": 2,
    }

    first = build_animated_skeleton_glb(
        parents, names, positions, rotations_wxyz, frame_time, extras=extras_a
    )
    second = build_animated_skeleton_glb(
        parents, names, positions, rotations_wxyz, frame_time, extras=extras_b
    )
    assert first == second

    document, binary = glb_module._split_glb(first)
    assert glb_module._validate_glb_structure(first) == document
    assert document["asset"]["extras"] == extras_a
    assert document["buffers"] == [{"byteLength": len(binary)}]
    assert all("uri" not in buffer for buffer in document["buffers"])
    assert len(document["meshes"]) == 2
    assert document["nodes"][4]["name"] == "Chicken_Wíng"

    times, actual_translations, actual_rotations = _animation_joint_transforms(
        document, binary, len(parents)
    )
    np.testing.assert_allclose(times, np.arange(4) * frame_time, atol=1e-7)
    np.testing.assert_allclose(actual_translations, positions, atol=1e-6)

    expected_xyzw = rotations_wxyz[..., [1, 2, 3, 0]].copy()
    expected_xyzw /= np.linalg.norm(expected_xyzw, axis=-1, keepdims=True)
    expected_world = _global_positions(parents, positions, expected_xyzw)
    actual_world = _global_positions(parents, actual_translations, actual_rotations)
    np.testing.assert_allclose(actual_world, expected_world, rtol=1e-6, atol=1e-6)

    for joint in range(len(parents)):
        dots = np.sum(actual_rotations[:-1, joint] * actual_rotations[1:, joint], axis=-1)
        assert np.all(dots >= -1e-6)

    # The moving non-root wing receives animated visual-link scale/rotation.
    visual_channels = [
        channel
        for channel in document["animations"][0]["channels"]
        if channel["target"]["node"] >= 2 * len(parents)
    ]
    assert {channel["target"]["path"] for channel in visual_channels} >= {"rotation", "scale"}


def test_build_supports_a_single_static_frame_and_zero_length_end_site() -> None:
    parents = np.asarray([-1, 0], dtype=np.int64)
    positions = np.zeros((1, 2, 3), dtype=np.float64)
    rotations = np.zeros((1, 2, 4), dtype=np.float64)
    rotations[..., 0] = 1.0
    glb = build_animated_skeleton_glb(
        parents, ["Chicken", "Chicken_End"], positions, rotations, 1 / 20, extras={}
    )
    document, binary = glb_module._split_glb(glb)
    sampler = document["animations"][0]["samplers"][0]
    np.testing.assert_array_equal(_read_accessor(document, binary, sampler["input"]), [0.0])
    assert len(document["nodes"]) == 4  # two joints and two markers; no zero-length bone


def test_build_rejects_invalid_inputs_and_metadata() -> None:
    parents, names, positions, rotations, frame_time = _chicken_motion()

    with pytest.raises(TypeError, match="integer array"):
        build_animated_skeleton_glb(parents.astype(float), names, positions, rotations, frame_time)
    with pytest.raises(ValueError, match="parent-before-child"):
        build_animated_skeleton_glb(np.asarray([-1, 2, 1, 2, 1]), names, positions, rotations, frame_time)
    with pytest.raises(ValueError, match="one root"):
        build_animated_skeleton_glb(np.asarray([-1, -1, 1, 2, 1]), names, positions, rotations, frame_time)
    with pytest.raises(ValueError, match="non-empty"):
        build_animated_skeleton_glb(parents, [*names[:-1], ""], positions, rotations, frame_time)
    with pytest.raises(ValueError, match="positions"):
        build_animated_skeleton_glb(parents, names, positions[..., :2], rotations, frame_time)
    with pytest.raises(ValueError, match="rotations_wxyz"):
        build_animated_skeleton_glb(parents, names, positions, rotations[..., :3], frame_time)

    bad_positions = positions.copy()
    bad_positions[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_animated_skeleton_glb(parents, names, bad_positions, rotations, frame_time)
    bad_rotations = rotations.copy()
    bad_rotations[0, 0] = 0
    with pytest.raises(ValueError, match="zero-length"):
        build_animated_skeleton_glb(parents, names, positions, bad_rotations, frame_time)
    with pytest.raises(ValueError, match="positive"):
        build_animated_skeleton_glb(parents, names, positions, rotations, 0)
    with pytest.raises(ValueError, match="non-finite"):
        build_animated_skeleton_glb(
            parents, names, positions, rotations, frame_time, extras={"bad": math.inf}
        )
    with pytest.raises(TypeError, match="unsupported"):
        build_animated_skeleton_glb(
            parents, names, positions, rotations, frame_time, extras={"bad": {1, 2}}
        )
    with pytest.raises(TypeError, match="keys"):
        build_animated_skeleton_glb(
            parents, names, positions, rotations, frame_time, extras={1: "bad"}  # type: ignore[dict-item]
        )


_CHICKEN_BVH = """HIERARCHY
ROOT Chicken_Hips
{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Chicken_Spine
  {
    OFFSET 0.0 0.45 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT Chicken_Head
    {
      OFFSET 0.0 0.35 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET 0.0 0.15 0.20
      }
    }
  }
}
MOTION
Frames: 3
Frame Time: 0.050000
0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.05 0.00 0.02 4.0 0.0 0.0 2.0 0.0 0.0 -1.0 0.0 0.0
0.11 0.00 0.05 8.0 0.0 0.0 3.0 0.0 0.0 -2.0 0.0 0.0
"""


_FAKE_MOTION_MODULE = '''from pathlib import Path
import math
import numpy as np

class _Rotations:
    def __init__(self, values):
        self.qs = values

class _Animation:
    def __init__(self, parents, positions, rotations):
        self.parents = parents
        self.positions = positions
        self.rotations = _Rotations(rotations)

def _z_rotation(degrees):
    half = math.radians(degrees) / 2.0
    return [math.cos(half), 0.0, 0.0, math.sin(half)]

def load(path):
    text = Path(path).read_text(encoding="utf-8")
    assert "ROOT Chicken_Hips" in text
    assert "JOINT Chicken_Spine" in text
    lines = [line.strip() for line in text.splitlines()]
    frame_line = next(line for line in lines if line.startswith("Frame Time:"))
    frame_time = float(frame_line.split(":", 1)[1])
    start = lines.index(frame_line) + 1
    rows = np.asarray([[float(value) for value in line.split()] for line in lines[start:] if line], dtype=float)
    frames = len(rows)
    positions = np.zeros((frames, 3, 3), dtype=float)
    positions[:, 0] = rows[:, :3]
    positions[:, 1] = [0.0, 0.45, 0.0]
    positions[:, 2] = [0.0, 0.35, 0.0]
    rotations = np.zeros((frames, 3, 4), dtype=float)
    rotations[..., 0] = 1.0
    rotations[:, 0] = np.asarray([_z_rotation(value) for value in rows[:, 3]])
    rotations[:, 1] = np.asarray([_z_rotation(value) for value in rows[:, 6]])
    rotations[:, 2] = np.asarray([_z_rotation(value) for value in rows[:, 9]])
    return _Animation(np.asarray([-1, 0, 1]), positions, rotations), ["Chicken_Hips", "Chicken_Spine", "Chicken_Head"], frame_time
'''


def _write_chicken_fixture(root: Path, *, invalid_rotations: bool = False) -> tuple[Path, Path]:
    bvh = root / "fixtures" / "chicken_generated.bvh"
    bvh.parent.mkdir(parents=True)
    bvh.write_text(_CHICKEN_BVH, encoding="utf-8")
    motion = root / "Motion"
    motion.mkdir()
    source = _FAKE_MOTION_MODULE
    if invalid_rotations:
        source = source.replace(
            "    return _Animation(",
            "    rotations[0, 2] = 0.0\n    return _Animation(",
        )
    (motion / "BVH.py").write_text(source, encoding="utf-8")
    return bvh, motion


def test_bvh_to_glb_uses_exact_motion_source_and_atomically_replaces(tmp_path: Path) -> None:
    bvh, motion_source = _write_chicken_fixture(tmp_path)
    output = tmp_path / "workspace" / "Workflows" / "AnyTop" / "chicken.glb"
    sentinel = types.ModuleType("BVH")
    previous = sys.modules.get("BVH")
    sys.modules["BVH"] = sentinel
    try:
        result = bvh_to_glb(
            bvh,
            output,
            extras={"anytop": {"canonicalMotion": bvh.name, "schemaVersion": 1}},
            motion_source=motion_source,
        )
        assert sys.modules["BVH"] is sentinel
    finally:
        if previous is None:
            sys.modules.pop("BVH", None)
        else:
            sys.modules["BVH"] = previous

    assert result == output
    assert output.is_file()
    first = output.read_bytes()
    document = glb_module._validate_glb_structure(first)
    assert document["asset"]["extras"]["anytop"]["canonicalMotion"] == bvh.name
    assert document["nodes"][0]["name"] == "Chicken_Hips"
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))

    output.write_bytes(b"old incomplete output")
    assert bvh_to_glb(bvh, output, extras={}, motion_source=motion_source) == output
    assert output.read_bytes() != b"old incomplete output"
    assert output.read_bytes() == bvh_to_glb(
        bvh, output, extras={}, motion_source=motion_source
    ).read_bytes()
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_bvh_to_glb_uses_an_already_configured_sys_path(tmp_path: Path) -> None:
    bvh, motion_source = _write_chicken_fixture(tmp_path)
    output = tmp_path / "configured.glb"
    previous = sys.modules.pop("BVH", None)
    sys.path.insert(0, str(motion_source))
    try:
        assert bvh_to_glb(bvh, output, extras={}) == output
        assert glb_module._validate_glb_structure(output.read_bytes())
    finally:
        sys.modules.pop("BVH", None)
        sys.path.remove(str(motion_source))
        if previous is not None:
            sys.modules["BVH"] = previous


def test_bvh_to_glb_preserves_existing_output_on_conversion_failure(tmp_path: Path) -> None:
    bvh, motion_source = _write_chicken_fixture(tmp_path, invalid_rotations=True)
    output = tmp_path / "chicken.glb"
    output.write_bytes(b"known-good-placeholder")
    with pytest.raises(ValueError, match="zero-length"):
        bvh_to_glb(bvh, output, extras={}, motion_source=motion_source)
    assert output.read_bytes() == b"known-good-placeholder"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_bvh_to_glb_rejects_missing_input_and_non_glb_output(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        bvh_to_glb(tmp_path / "missing.bvh", tmp_path / "out.glb", extras={})

    bvh, motion_source = _write_chicken_fixture(tmp_path)
    with pytest.raises(ValueError, match=".glb"):
        bvh_to_glb(
            bvh, tmp_path / "out.gltf", extras={}, motion_source=motion_source
        )


def test_binary_header_and_chunk_alignment() -> None:
    parents, names, positions, rotations, frame_time = _chicken_motion()
    glb = build_animated_skeleton_glb(parents, names, positions, rotations, frame_time)
    magic, version, declared_length = struct.unpack_from("<4sII", glb, 0)
    assert (magic, version, declared_length) == (b"glTF", 2, len(glb))
    json_length, json_type = struct.unpack_from("<I4s", glb, 12)
    assert json_type == b"JSON"
    assert json_length % 4 == 0
    json.loads(glb[20 : 20 + json_length])
    binary_length, binary_type = struct.unpack_from("<I4s", glb, 20 + json_length)
    assert binary_type == b"BIN\0"
    assert binary_length % 4 == 0

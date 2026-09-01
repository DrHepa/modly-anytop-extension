"""Portable AnyTop BVH to animated GLB conversion.

The generated GLB is a visible skeleton preview: joint markers and bone segments
are attached to the BVH hierarchy and animated with its local transforms.  The
canonical motion remains the sibling BVH file; this module does not retarget or
skin a character mesh.

No geometry/runtime dependency is used beyond NumPy.  In particular, conversion
does not require Blender, pygltflib, trimesh, SciPy, or a native extension.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import struct
import sys
import tempfile
import threading
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import numpy as np


__all__ = ["build_animated_skeleton_glb", "bvh_to_glb"]


_FLOAT = 5126
_UNSIGNED_SHORT = 5123
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_JSON_CHUNK = b"JSON"
_BIN_CHUNK = b"BIN\0"
_GLB_MAGIC = b"glTF"
_GLB_VERSION = 2
_EPSILON = 1.0e-7
_MOTION_MODULE_NAMES = (
    "BVH",
    "Animation",
    "AnimationStructure",
    "Quaternions",
    "quaternion",
    "Pivots",
)
_MOTION_IMPORT_LOCK = threading.RLock()
_MISSING = object()


class _BinaryBuilder:
    def __init__(self) -> None:
        self.data = bytearray()
        self.buffer_views: list[dict[str, Any]] = []
        self.accessors: list[dict[str, Any]] = []

    def add_accessor(
        self,
        values: np.ndarray,
        *,
        accessor_type: str,
        component_type: int,
        target: int | None = None,
        include_bounds: bool = False,
    ) -> int:
        array = np.ascontiguousarray(values)
        expected = {
            _FLOAT: np.dtype("<f4"),
            _UNSIGNED_SHORT: np.dtype("<u2"),
        }.get(component_type)
        if expected is None or array.dtype != expected:
            raise TypeError(
                f"Accessor component {component_type} requires {expected}, got {array.dtype}"
            )
        if array.ndim < 1 or array.shape[0] < 1:
            raise ValueError("Accessor arrays must contain at least one element")

        while len(self.data) % 4:
            self.data.append(0)
        offset = len(self.data)
        raw = array.tobytes(order="C")
        self.data.extend(raw)

        view: dict[str, Any] = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(raw),
        }
        if target is not None:
            view["target"] = target
        view_index = len(self.buffer_views)
        self.buffer_views.append(view)

        count = int(array.shape[0])
        accessor: dict[str, Any] = {
            "bufferView": view_index,
            "componentType": component_type,
            "count": count,
            "type": accessor_type,
        }
        if include_bounds:
            flat = array.reshape(count, -1)
            accessor["min"] = [float(item) for item in flat.min(axis=0)]
            accessor["max"] = [float(item) for item in flat.max(axis=0)]
        accessor_index = len(self.accessors)
        self.accessors.append(accessor)
        return accessor_index


def _as_json_value(value: object, path: str = "extras") -> object:
    """Copy an object into a deterministic, strict JSON-compatible value."""

    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            copied[key] = _as_json_value(item, f"{path}.{key}")
        return copied
    if isinstance(value, (list, tuple)):
        return [_as_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{path} contains unsupported value {type(value).__name__}")


def _coerce_inputs(
    parents: Sequence[int] | np.ndarray,
    names: Sequence[str],
    positions: np.ndarray,
    rotations_wxyz: np.ndarray,
    frame_time: float,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, float]:
    parent_array = np.asarray(parents)
    if parent_array.ndim != 1 or parent_array.dtype.kind not in "iu":
        raise TypeError("parents must be a one-dimensional integer array")
    parent_array = np.ascontiguousarray(parent_array, dtype=np.int64)

    name_list = list(names)
    if any(not isinstance(name, str) or not name.strip() for name in name_list):
        raise ValueError("every joint name must be a non-empty string")

    position_array = np.asarray(positions)
    rotation_array = np.asarray(rotations_wxyz)
    if position_array.ndim != 3 or position_array.shape[-1] != 3:
        raise ValueError("positions must have shape [frames, joints, 3]")
    if rotation_array.shape != position_array.shape[:2] + (4,):
        raise ValueError("rotations_wxyz must have shape [frames, joints, 4]")
    frames, joints = position_array.shape[:2]
    if frames < 1 or joints < 1:
        raise ValueError("the motion must contain at least one frame and one joint")
    if parent_array.shape != (joints,) or len(name_list) != joints:
        raise ValueError("parents and names must match the joint count")
    if int(parent_array[0]) != -1 or int(np.count_nonzero(parent_array == -1)) != 1:
        raise ValueError("the skeleton must contain one root at joint index 0")
    for joint in range(1, joints):
        parent = int(parent_array[joint])
        if parent < 0 or parent >= joint:
            raise ValueError(
                "parents must form an acyclic parent-before-child hierarchy"
            )

    try:
        frame_time_value = float(frame_time)
    except (TypeError, ValueError) as error:
        raise TypeError("frame_time must be a real number") from error
    if not math.isfinite(frame_time_value) or frame_time_value <= 0:
        raise ValueError("frame_time must be finite and positive")

    try:
        position_array = np.ascontiguousarray(position_array, dtype="<f4")
        rotation_array = np.ascontiguousarray(rotation_array, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError("positions and rotations must be numeric arrays") from error
    if not np.isfinite(position_array).all() or not np.isfinite(rotation_array).all():
        raise ValueError("motion arrays must contain only finite values")

    lengths = np.linalg.norm(rotation_array, axis=-1, keepdims=True)
    if np.any(lengths < 1.0e-12):
        raise ValueError("joint rotations must not contain zero-length quaternions")
    rotation_array = rotation_array / lengths

    # q and -q encode the same rotation.  Adjacent keys must use the same
    # hemisphere so glTF LINEAR quaternion interpolation follows the short arc.
    for joint in range(joints):
        for frame in range(1, frames):
            if np.dot(rotation_array[frame - 1, joint], rotation_array[frame, joint]) < 0:
                rotation_array[frame, joint] *= -1

    return parent_array, name_list, position_array, rotation_array, frame_time_value


def _quat_y_to(vector: np.ndarray) -> np.ndarray:
    """Return an XYZW quaternion rotating +Y onto ``vector``."""

    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length < 1.0e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    target = vector / length
    dot = float(target[1])
    if dot < -0.999999:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    xyz = np.cross(np.asarray([0.0, 1.0, 0.0]), target)
    quaternion = np.concatenate((xyz, np.asarray([1.0 + dot])))
    return quaternion / np.linalg.norm(quaternion)


def _continuous_xyzw(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.float64).copy()
    lengths = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(lengths < 1.0e-12):
        raise ValueError("visual rotations contain a zero-length quaternion")
    result /= lengths
    for frame in range(1, len(result)):
        if np.dot(result[frame - 1], result[frame]) < 0:
            result[frame] *= -1
    return np.ascontiguousarray(result, dtype="<f4")


def _joint_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(
        (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ),
        dtype="<f4",
    )
    normals = positions.copy()
    indices = np.asarray(
        (
            2, 0, 4, 2, 4, 1, 2, 1, 5, 2, 5, 0,
            3, 4, 0, 3, 1, 4, 3, 5, 1, 3, 0, 5,
        ),
        dtype="<u2",
    )
    return positions, normals, indices


def _bone_geometry(segments: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a unit-radius open prism extending from y=0 to y=1."""

    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    indices: list[int] = []
    for y in (0.0, 1.0):
        for segment in range(segments):
            angle = 2.0 * math.pi * segment / segments
            x, z = math.cos(angle), math.sin(angle)
            positions.append((x, y, z))
            normals.append((x, 0.0, z))
    for segment in range(segments):
        a0 = segment
        b0 = (segment + 1) % segments
        a1 = segment + segments
        b1 = (segment + 1) % segments + segments
        indices.extend((a0, a1, b1, a0, b1, b0))
    return (
        np.asarray(positions, dtype="<f4"),
        np.asarray(normals, dtype="<f4"),
        np.asarray(indices, dtype="<u2"),
    )


def build_animated_skeleton_glb(
    parents: Sequence[int] | np.ndarray,
    names: Sequence[str],
    positions: np.ndarray,
    rotations_wxyz: np.ndarray,
    frame_time: float,
    *,
    extras: Mapping[str, object] | None = None,
) -> bytes:
    """Build a deterministic, self-contained GLB 2.0 skeleton animation.

    ``positions`` and ``rotations_wxyz`` are local transforms with shapes
    ``[frames, joints, 3]`` and ``[frames, joints, 4]`` respectively.  Rotation
    input uses Motion's WXYZ convention; glTF output uses XYZW.
    """

    (
        parent_array,
        name_list,
        position_array,
        rotation_array_wxyz,
        frame_time_value,
    ) = _coerce_inputs(parents, names, positions, rotations_wxyz, frame_time)
    extras_value = _as_json_value(extras or {})
    if not isinstance(extras_value, dict):  # Defensive: Mapping always copies to dict.
        raise TypeError("extras must be a mapping")

    rotations_xyzw = np.ascontiguousarray(
        rotation_array_wxyz[..., [1, 2, 3, 0]], dtype="<f4"
    )
    frames, joints = position_array.shape[:2]
    builder = _BinaryBuilder()

    joint_positions, joint_normals, joint_indices = _joint_geometry()
    bone_positions, bone_normals, bone_indices = _bone_geometry()
    joint_position_accessor = builder.add_accessor(
        joint_positions,
        accessor_type="VEC3",
        component_type=_FLOAT,
        target=_ARRAY_BUFFER,
        include_bounds=True,
    )
    joint_normal_accessor = builder.add_accessor(
        joint_normals,
        accessor_type="VEC3",
        component_type=_FLOAT,
        target=_ARRAY_BUFFER,
    )
    joint_index_accessor = builder.add_accessor(
        joint_indices,
        accessor_type="SCALAR",
        component_type=_UNSIGNED_SHORT,
        target=_ELEMENT_ARRAY_BUFFER,
    )
    bone_position_accessor = builder.add_accessor(
        bone_positions,
        accessor_type="VEC3",
        component_type=_FLOAT,
        target=_ARRAY_BUFFER,
        include_bounds=True,
    )
    bone_normal_accessor = builder.add_accessor(
        bone_normals,
        accessor_type="VEC3",
        component_type=_FLOAT,
        target=_ARRAY_BUFFER,
    )
    bone_index_accessor = builder.add_accessor(
        bone_indices,
        accessor_type="SCALAR",
        component_type=_UNSIGNED_SHORT,
        target=_ELEMENT_ARRAY_BUFFER,
    )

    times = np.arange(frames, dtype="<f4") * np.float32(frame_time_value)
    if frames > 1 and not np.all(np.diff(times) > 0):
        raise ValueError("frame_time cannot be represented by strictly increasing float32 keys")
    time_accessor = builder.add_accessor(
        times,
        accessor_type="SCALAR",
        component_type=_FLOAT,
        include_bounds=True,
    )

    all_bone_lengths = np.linalg.norm(position_array[:, 1:], axis=-1).reshape(-1)
    nonzero_bone_lengths = all_bone_lengths[all_bone_lengths > _EPSILON]
    median_length = (
        float(np.median(nonzero_bone_lengths)) if len(nonzero_bone_lengths) else 0.02
    )
    bone_radius = float(np.clip(median_length * 0.07, 0.004, 0.04))
    joint_radius = bone_radius * 1.4

    nodes: list[dict[str, Any]] = []
    for joint in range(joints):
        nodes.append(
            {
                "name": name_list[joint],
                "translation": [float(value) for value in position_array[0, joint]],
                "rotation": [float(value) for value in rotations_xyzw[0, joint]],
                "children": [],
            }
        )

    # Scaling a joint node would also scale its kinematic descendants.  Marker
    # geometry therefore lives in a dedicated visual child node.
    for joint in range(joints):
        marker_index = len(nodes)
        nodes.append(
            {
                "name": f"{name_list[joint]}__joint_visual",
                "mesh": 0,
                "scale": [joint_radius, joint_radius, joint_radius],
            }
        )
        nodes[joint]["children"].append(marker_index)

    bone_nodes: dict[int, int] = {}
    for child in range(1, joints):
        parent = int(parent_array[child])
        vectors = position_array[:, child]
        lengths = np.linalg.norm(vectors, axis=-1)
        if float(lengths.max()) <= _EPSILON:
            continue
        vector = vectors[0]
        length = max(float(lengths[0]), _EPSILON)
        bone_index = len(nodes)
        bone_nodes[child] = bone_index
        nodes.append(
            {
                "name": f"{name_list[parent]}__to__{name_list[child]}",
                "mesh": 1,
                "rotation": [float(value) for value in _quat_y_to(vector)],
                "scale": [bone_radius, length, bone_radius],
            }
        )
        nodes[parent]["children"].append(bone_index)

    for child in range(1, joints):
        nodes[int(parent_array[child])]["children"].append(child)
    for node in nodes:
        if not node.get("children"):
            node.pop("children", None)

    samplers: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []

    def animate(node: int, path: str, output_accessor: int) -> None:
        sampler_index = len(samplers)
        samplers.append(
            {
                "input": time_accessor,
                "output": output_accessor,
                "interpolation": "LINEAR",
            }
        )
        channels.append(
            {"sampler": sampler_index, "target": {"node": node, "path": path}}
        )

    for joint in range(joints):
        rotation_accessor = builder.add_accessor(
            rotations_xyzw[:, joint],
            accessor_type="VEC4",
            component_type=_FLOAT,
        )
        animate(joint, "rotation", rotation_accessor)

        translations = position_array[:, joint]
        if float(np.max(np.abs(translations - translations[0]))) > _EPSILON:
            translation_accessor = builder.add_accessor(
                translations,
                accessor_type="VEC3",
                component_type=_FLOAT,
            )
            animate(joint, "translation", translation_accessor)

    # AnyTop-generated BVHs normally translate only the root.  Animating these
    # visual links additionally keeps valid six-channel non-root BVHs connected
    # at every source keyframe.
    for child, node_index in bone_nodes.items():
        vectors = position_array[:, child]
        if float(np.max(np.abs(vectors - vectors[0]))) <= _EPSILON:
            continue
        lengths = np.linalg.norm(vectors, axis=-1)
        scales = np.column_stack(
            (
                np.full(frames, bone_radius),
                np.maximum(lengths, _EPSILON),
                np.full(frames, bone_radius),
            )
        ).astype("<f4")
        visual_rotations = _continuous_xyzw(
            np.asarray([_quat_y_to(vector) for vector in vectors], dtype=np.float64)
        )
        scale_accessor = builder.add_accessor(
            scales,
            accessor_type="VEC3",
            component_type=_FLOAT,
        )
        rotation_accessor = builder.add_accessor(
            visual_rotations,
            accessor_type="VEC4",
            component_type=_FLOAT,
        )
        animate(node_index, "scale", scale_accessor)
        animate(node_index, "rotation", rotation_accessor)

    document: dict[str, Any] = {
        "asset": {
            "version": "2.0",
            "generator": "Modly AnyTop extension",
            "extras": extras_value,
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "meshes": [
            {
                "name": "AnyTop joint marker",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": joint_position_accessor,
                            "NORMAL": joint_normal_accessor,
                        },
                        "indices": joint_index_accessor,
                        "material": 0,
                    }
                ],
            },
            {
                "name": "AnyTop bone segment",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": bone_position_accessor,
                            "NORMAL": bone_normal_accessor,
                        },
                        "indices": bone_index_accessor,
                        "material": 0,
                    }
                ],
            },
        ],
        "materials": [
            {
                "name": "AnyTop skeleton",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 0.18, 0.06, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.65,
                },
                "doubleSided": True,
            }
        ],
        "buffers": [{"byteLength": len(builder.data)}],
        "bufferViews": builder.buffer_views,
        "accessors": builder.accessors,
        "animations": [
            {"name": "AnyTop Motion", "samplers": samplers, "channels": channels}
        ],
    }

    try:
        json_chunk = json.dumps(
            document,
            separators=(",", ":"),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"GLB metadata is not valid JSON: {error}") from error
    json_chunk += b" " * (-len(json_chunk) % 4)
    binary_chunk = bytes(builder.data)
    binary_chunk += b"\0" * (-len(binary_chunk) % 4)
    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary_chunk)
    glb = b"".join(
        (
            struct.pack("<4sII", _GLB_MAGIC, _GLB_VERSION, total_length),
            struct.pack("<I4s", len(json_chunk), _JSON_CHUNK),
            json_chunk,
            struct.pack("<I4s", len(binary_chunk), _BIN_CHUNK),
            binary_chunk,
        )
    )
    _validate_glb_structure(glb)
    return glb


def _split_glb(glb: bytes) -> tuple[dict[str, Any], memoryview]:
    if len(glb) < 28:
        raise ValueError("truncated GLB")
    magic, version, declared_length = struct.unpack_from("<4sII", glb, 0)
    if magic != _GLB_MAGIC or version != _GLB_VERSION or declared_length != len(glb):
        raise ValueError("invalid GLB header")

    json_length, json_type = struct.unpack_from("<I4s", glb, 12)
    if json_type != _JSON_CHUNK or json_length % 4:
        raise ValueError("invalid GLB JSON chunk")
    json_start, json_end = 20, 20 + json_length
    if json_end + 8 > len(glb):
        raise ValueError("truncated GLB JSON chunk")
    try:
        document = json.loads(glb[json_start:json_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid GLB JSON document") from error
    if not isinstance(document, dict):
        raise ValueError("GLB JSON root must be an object")

    binary_length, binary_type = struct.unpack_from("<I4s", glb, json_end)
    if binary_type != _BIN_CHUNK or binary_length % 4:
        raise ValueError("invalid GLB BIN chunk")
    binary_start, binary_end = json_end + 8, json_end + 8 + binary_length
    if binary_end != len(glb):
        raise ValueError("trailing or truncated GLB binary data")
    return document, memoryview(glb)[binary_start:binary_end]


def _accessor_array(
    document: Mapping[str, Any], binary: memoryview, accessor_index: int
) -> np.ndarray:
    accessors = document.get("accessors")
    views = document.get("bufferViews")
    if not isinstance(accessors, list) or not isinstance(views, list):
        raise ValueError("GLB accessors and bufferViews must be arrays")
    if not 0 <= accessor_index < len(accessors):
        raise ValueError("GLB accessor index is out of range")
    accessor = accessors[accessor_index]
    if not isinstance(accessor, dict):
        raise ValueError("GLB accessor must be an object")
    view_index = accessor.get("bufferView")
    if not isinstance(view_index, int) or not 0 <= view_index < len(views):
        raise ValueError("GLB accessor bufferView is invalid")
    view = views[view_index]
    if not isinstance(view, dict):
        raise ValueError("GLB bufferView must be an object")

    component_type = accessor.get("componentType")
    dtype = {_FLOAT: np.dtype("<f4"), _UNSIGNED_SHORT: np.dtype("<u2")}.get(
        component_type
    )
    component_count = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT4": 16,
    }.get(accessor.get("type"))
    count = accessor.get("count")
    if dtype is None or component_count is None or not isinstance(count, int) or count < 1:
        raise ValueError("GLB accessor description is invalid")
    view_offset = view.get("byteOffset", 0)
    accessor_offset = accessor.get("byteOffset", 0)
    view_length = view.get("byteLength")
    if (
        not isinstance(view_offset, int)
        or not isinstance(accessor_offset, int)
        or not isinstance(view_length, int)
        or min(view_offset, accessor_offset, view_length) < 0
    ):
        raise ValueError("GLB accessor offsets are invalid")
    if view_offset % 4 or (view_offset + accessor_offset) % dtype.itemsize:
        raise ValueError("GLB accessor is not correctly aligned")
    required = count * component_count * dtype.itemsize
    if accessor_offset + required > view_length or view_offset + view_length > len(binary):
        raise ValueError("GLB accessor exceeds its binary bufferView")
    array = np.frombuffer(
        binary,
        dtype=dtype,
        count=count * component_count,
        offset=view_offset + accessor_offset,
    )
    return array.reshape((count,) if component_count == 1 else (count, component_count))


def _validate_glb_structure(glb: bytes) -> dict[str, Any]:
    """Validate the exact self-contained GLB subset emitted by this module."""

    document, binary = _split_glb(glb)
    asset = document.get("asset")
    if not isinstance(asset, dict) or asset.get("version") != "2.0":
        raise ValueError("GLB asset must target glTF 2.0")
    buffers = document.get("buffers")
    if not isinstance(buffers, list) or len(buffers) != 1:
        raise ValueError("GLB must contain exactly one embedded buffer")
    buffer = buffers[0]
    if (
        not isinstance(buffer, dict)
        or "uri" in buffer
        or buffer.get("byteLength") != len(binary)
    ):
        raise ValueError("GLB embedded buffer description is invalid")

    accessors = document.get("accessors")
    if not isinstance(accessors, list) or not accessors:
        raise ValueError("GLB must contain accessors")
    for index in range(len(accessors)):
        values = _accessor_array(document, binary, index)
        if values.dtype.kind == "f" and not np.isfinite(values).all():
            raise ValueError("GLB accessor contains a non-finite float")

    nodes = document.get("nodes")
    scenes = document.get("scenes")
    scene_index = document.get("scene")
    if (
        not isinstance(nodes, list)
        or not nodes
        or not isinstance(scenes, list)
        or not isinstance(scene_index, int)
        or not 0 <= scene_index < len(scenes)
    ):
        raise ValueError("GLB scene graph is invalid")
    parents: dict[int, int] = {}
    for parent_index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError("GLB node must be an object")
        for child in node.get("children", []):
            if not isinstance(child, int) or not 0 <= child < len(nodes):
                raise ValueError("GLB child node index is invalid")
            if child in parents:
                raise ValueError("GLB node has more than one parent")
            parents[child] = parent_index
    roots = scenes[scene_index].get("nodes") if isinstance(scenes[scene_index], dict) else None
    if not isinstance(roots, list) or roots != [0]:
        raise ValueError("GLB preview scene must use joint 0 as its sole root")
    for node_index in range(len(nodes)):
        seen: set[int] = set()
        current = node_index
        while current in parents:
            if current in seen:
                raise ValueError("GLB scene graph contains a cycle")
            seen.add(current)
            current = parents[current]

    meshes = document.get("meshes")
    if not isinstance(meshes, list) or len(meshes) != 2:
        raise ValueError("GLB preview must contain shared joint and bone meshes")
    for mesh in meshes:
        primitives = mesh.get("primitives") if isinstance(mesh, dict) else None
        if not isinstance(primitives, list) or len(primitives) != 1:
            raise ValueError("GLB preview mesh primitive is invalid")
        primitive = primitives[0]
        attributes = primitive.get("attributes") if isinstance(primitive, dict) else None
        if not isinstance(attributes, dict) or not {
            "POSITION",
            "NORMAL",
        }.issubset(attributes):
            raise ValueError("GLB preview mesh attributes are incomplete")
        for accessor_index in (
            attributes["POSITION"],
            attributes["NORMAL"],
            primitive.get("indices"),
        ):
            if not isinstance(accessor_index, int):
                raise ValueError("GLB preview mesh accessor is invalid")
            _accessor_array(document, binary, accessor_index)
        position_accessor = accessors[attributes["POSITION"]]
        if "min" not in position_accessor or "max" not in position_accessor:
            raise ValueError("GLB POSITION accessor requires min/max bounds")

    animations = document.get("animations")
    if not isinstance(animations, list) or len(animations) != 1:
        raise ValueError("GLB must contain exactly one animation")
    animation = animations[0]
    samplers = animation.get("samplers") if isinstance(animation, dict) else None
    channels = animation.get("channels") if isinstance(animation, dict) else None
    if not isinstance(samplers, list) or not samplers or not isinstance(channels, list):
        raise ValueError("GLB animation is incomplete")
    for channel in channels:
        if not isinstance(channel, dict):
            raise ValueError("GLB animation channel must be an object")
        sampler_index = channel.get("sampler")
        target = channel.get("target")
        if (
            not isinstance(sampler_index, int)
            or not 0 <= sampler_index < len(samplers)
            or not isinstance(target, dict)
            or not isinstance(target.get("node"), int)
            or not 0 <= target["node"] < len(nodes)
            or target.get("path") not in {"rotation", "translation", "scale"}
        ):
            raise ValueError("GLB animation channel target is invalid")
        sampler = samplers[sampler_index]
        if not isinstance(sampler, dict) or sampler.get("interpolation") != "LINEAR":
            raise ValueError("GLB animation sampler must use LINEAR interpolation")
        input_index = sampler.get("input")
        output_index = sampler.get("output")
        if not isinstance(input_index, int) or not isinstance(output_index, int):
            raise ValueError("GLB animation sampler accessors are invalid")
        times = _accessor_array(document, binary, input_index)
        values = _accessor_array(document, binary, output_index)
        if (
            accessors[input_index].get("componentType") != _FLOAT
            or accessors[input_index].get("type") != "SCALAR"
            or "min" not in accessors[input_index]
            or "max" not in accessors[input_index]
            or float(times[0]) < 0
            or (len(times) > 1 and not np.all(np.diff(times) > 0))
            or len(values) != len(times)
        ):
            raise ValueError("GLB animation key accessors are invalid")
        expected_type = "VEC4" if target["path"] == "rotation" else "VEC3"
        if accessors[output_index].get("type") != expected_type:
            raise ValueError("GLB animation value accessor has the wrong type")
        if target["path"] == "rotation":
            norms = np.linalg.norm(values, axis=-1)
            if not np.allclose(norms, 1.0, rtol=1.0e-5, atol=1.0e-6):
                raise ValueError("GLB rotation keys must be unit quaternions")
            if len(values) > 1 and np.any(np.sum(values[:-1] * values[1:], axis=-1) < -1e-6):
                raise ValueError("GLB rotation keys cross quaternion hemispheres")
    return document


@contextmanager
def _provided_motion_module(source: Path) -> Iterator[ModuleType]:
    source = source.resolve()
    bvh_module_path = source / "BVH.py"
    if not source.is_dir() or not bvh_module_path.is_file():
        raise FileNotFoundError(f"Motion source does not contain BVH.py: {source}")

    source_text = str(source)
    saved_modules = {
        name: sys.modules.get(name, _MISSING) for name in _MOTION_MODULE_NAMES
    }
    with _MOTION_IMPORT_LOCK:
        for name in _MOTION_MODULE_NAMES:
            sys.modules.pop(name, None)
        sys.path.insert(0, source_text)
        importlib.invalidate_caches()
        try:
            module = importlib.import_module("BVH")
            module_file = Path(getattr(module, "__file__", "")).resolve()
            if module_file != bvh_module_path.resolve():
                raise ImportError(f"Imported BVH from unexpected location: {module_file}")
            yield module
        finally:
            try:
                sys.path.remove(source_text)
            except ValueError:
                pass
            for name in _MOTION_MODULE_NAMES:
                sys.modules.pop(name, None)
            for name, previous in saved_modules.items():
                if previous is not _MISSING:
                    sys.modules[name] = previous  # type: ignore[assignment]
            importlib.invalidate_caches()


@contextmanager
def _motion_bvh_module(motion_source: Path | None) -> Iterator[ModuleType]:
    if motion_source is not None:
        with _provided_motion_module(Path(motion_source)) as module:
            yield module
        return
    try:
        module = importlib.import_module("BVH")
    except ImportError as error:
        raise RuntimeError(
            "Motion BVH module is unavailable; pass motion_source or configure sys.path"
        ) from error
    yield module


def _load_bvh(
    bvh_path: Path, motion_source: Path | None
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, float]:
    with _motion_bvh_module(motion_source) as bvh_module:
        load = getattr(bvh_module, "load", None)
        if not callable(load):
            raise RuntimeError("Motion BVH module does not expose callable load()")
        try:
            animation, names, frame_time = load(str(bvh_path))
            rotations = animation.rotations.qs
            return (
                np.asarray(animation.parents),
                [str(name) for name in names],
                np.asarray(animation.positions),
                np.asarray(rotations),
                float(frame_time),
            )
        except Exception as error:
            raise RuntimeError(f"Failed to load BVH {bvh_path}: {error}") from error


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def bvh_to_glb(
    bvh_path: Path,
    output_path: Path,
    *,
    extras: Mapping[str, object],
    motion_source: Path | None = None,
) -> Path:
    """Load a BVH with pinned Motion code and atomically write its GLB preview."""

    bvh_path = Path(bvh_path)
    output_path = Path(output_path)
    if not bvh_path.is_file():
        raise FileNotFoundError(f"BVH file does not exist: {bvh_path}")
    if output_path.suffix.lower() != ".glb":
        raise ValueError("output_path must use the .glb extension")
    if output_path.exists() and output_path.is_dir():
        raise IsADirectoryError(output_path)
    try:
        same_path = bvh_path.resolve() == output_path.resolve()
    except OSError:
        same_path = False
    if same_path:
        raise ValueError("BVH input and GLB output paths must be different")

    parents, names, positions, rotations, frame_time = _load_bvh(
        bvh_path, Path(motion_source) if motion_source is not None else None
    )
    glb = build_animated_skeleton_glb(
        parents,
        names,
        positions,
        rotations,
        frame_time,
        extras=extras,
    )
    _atomic_write(output_path, glb)
    return output_path

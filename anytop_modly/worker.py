"""Isolated AnyTop execution worker.

This module is never Modly's protocol endpoint.  ``runtime.py`` launches it in
a child interpreter with stdout/stderr captured, so progress bars and native
library banners cannot become fake NDJSON messages.
"""

from __future__ import annotations

from argparse import Namespace
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any, Iterator, Mapping


OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "MPLBACKEND": "Agg",
}
for _name, _value in OFFLINE_ENVIRONMENT.items():
    os.environ[_name] = _value


class WorkerFailure(RuntimeError):
    """The isolated upstream invocation did not satisfy its output contract."""


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _path(value: object, *, directory: bool = False) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise WorkerFailure("invalid worker path")
    path = Path(value)
    if not path.is_absolute():
        raise WorkerFailure("worker paths must be absolute")
    if directory:
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise WorkerFailure("worker directory is unavailable")
    elif not path.is_file():
        raise WorkerFailure("worker file is unavailable")
    return path.resolve()


@contextmanager
def _upstream_imports(source_root: Path, motion_source: Path) -> Iterator[None]:
    old_cwd = Path.cwd()
    additions = [str(motion_source), str(source_root)]
    for addition in reversed(additions):
        if addition not in sys.path:
            sys.path.insert(0, addition)
    os.chdir(source_root)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _device(request: Mapping[str, Any]) -> tuple[int, str]:
    import torch

    mode = request.get("device_mode", "auto")
    index = int(request.get("cuda_device", 0))
    use_cuda = mode == "cuda" or (mode == "auto" and torch.cuda.is_available())
    if use_cuda:
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise WorkerFailure("requested CUDA device is unavailable")
        torch.cuda.set_device(index)
        # Upstream passes this value to ``torch.autocast(device_type=...)``,
        # which accepts ``cuda`` but not ``cuda:0``.  The selected index is
        # already made current above and is also passed through args.device.
        return index, "cuda"
    return -1, "cpu"


def _patch_t5(module: Any, t5_path: Path, device: str) -> None:
    """Force upstream T5 construction to use the pinned offline directory."""

    original = module.T5Conditioner
    local_name = str(t5_path)
    # Upstream validates names against class-level registries before calling
    # Transformers.  Register the pinned local t5-base under its absolute path
    # without altering the checkout on disk.
    if local_name not in original.MODELS:
        original.MODELS = [*original.MODELS, local_name]
    original.MODELS_DIMS = {**original.MODELS_DIMS, local_name: 768}

    def local_conditioner(*args: object, **kwargs: object) -> object:
        positional = list(args)
        if positional:
            positional[0] = local_name
        else:
            kwargs["name"] = local_name
        kwargs["device"] = device
        return original(*positional, **kwargs)

    module.T5Conditioner = local_conditioner


def _namespace(checkpoint: Path) -> Namespace:
    args_path = checkpoint.with_name("args.json")
    try:
        value = json.loads(args_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkerFailure("checkpoint arguments are invalid") from exc
    if not isinstance(value, dict):
        raise WorkerFailure("checkpoint arguments are invalid")
    return Namespace(**value)


def _apply_common(args: Namespace, request: Mapping[str, Any], checkpoint: Path, device_index: int) -> None:
    args.model_path = str(checkpoint)
    args.seed = int(request["seed"])
    args.device = device_index
    args.cuda = device_index >= 0
    args.cond_path = ""
    args.num_samples = 1
    args.batch_size = 1
    args.num_repetitions = int(request.get("num_repetitions", 1))


def _condition(path: Path) -> dict[str, Any]:
    import numpy as np

    value = np.load(path, allow_pickle=True)
    try:
        result = value.item()
    except (AttributeError, ValueError) as exc:
        raise WorkerFailure("condition asset is invalid") from exc
    if not isinstance(result, dict):
        raise WorkerFailure("condition asset is invalid")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _collect_triplets(output: Path) -> list[dict[str, str | None]]:
    motions = sorted(output.glob("*.npy"))
    if not motions:
        raise WorkerFailure("upstream produced no motion arrays")
    bvhs = {path.stem: path for path in output.glob("*.bvh")}
    videos = {path.stem: path for path in output.glob("*.mp4")}
    items: list[dict[str, str | None]] = []
    for motion in motions:
        bvh = bvhs.get(motion.stem)
        video = videos.get(motion.stem)
        if bvh is None or video is None:
            raise WorkerFailure("upstream generation outputs are incomplete")
        items.append(
            {
                "motion": str(motion.resolve()),
                "bvh": str(bvh.resolve()),
                "video": str(video.resolve()),
            }
        )
    return items


def _run_preprocess(request: Mapping[str, Any]) -> dict[str, object]:
    from data_loaders.truebones.truebones_utils.motion_process import process_skeleton

    output = _path(request["output_dir"], directory=True)
    bvh_directory = _path(request["bvh_directory"], directory=True)
    tpos_value = request.get("tpos_bvh")
    tpos = _path(tpos_value) if tpos_value else None
    process_skeleton(
        str(request["object_name"]),
        str(bvh_directory),
        list(request["face_joints"]),
        str(output),
        str(tpos) if tpos else "",
    )
    condition = output / "cond.npy"
    motions = sorted((output / "motions").glob("*.npy"))
    bvhs = sorted((output / "bvhs").glob("*.bvh"))
    videos = sorted((output / "animations").glob("*.mp4"))
    if not condition.is_file() or not motions or not bvhs or not videos:
        raise WorkerFailure("preprocessing outputs are incomplete")
    motion_by_stem = {path.stem: path for path in motions}
    bvh_by_stem = {path.stem: path for path in bvhs}
    video_by_stem = {
        path.stem.removesuffix("_from_ric"): path
        for path in videos
    }
    if set(motion_by_stem) != set(bvh_by_stem) or set(motion_by_stem) != set(video_by_stem):
        raise WorkerFailure("preprocessing sidecars do not match by motion stem")
    items = [
        {
            "motion": str(motion_by_stem[stem].resolve()),
            "bvh": str(bvh_by_stem[stem].resolve()),
            "video": str(video_by_stem[stem].resolve()),
        }
        for stem in sorted(motion_by_stem)
    ]
    return {
        "object_name": str(request["object_name"]),
        "condition": str(condition.resolve()),
        "items": items,
        "motions": [str(path.resolve()) for path in motions],
        "bvhs": [str(path.resolve()) for path in bvhs],
        "videos": [str(path.resolve()) for path in videos],
    }


def _run_generate(request: Mapping[str, Any], t5_path: Path) -> dict[str, object]:
    import sample.generate as generate

    checkpoint = _path(request["checkpoint"])
    cond_path = _path(request["condition"])
    output = _path(request["output_dir"], directory=True)
    device_index, t5_device = _device(request)
    _patch_t5(generate, t5_path, t5_device)
    args = _namespace(checkpoint)
    _apply_common(args, request, checkpoint, device_index)
    args.output_dir = str(output)
    args.object_type = [str(request["object_name"])]
    args.motion_length = float(request["motion_length"])
    generate.main(args=args, cond_dict=_condition(cond_path))
    return {
        "object_name": str(request["object_name"]),
        "items": _collect_triplets(output),
    }


def _run_edit(request: Mapping[str, Any], t5_path: Path) -> dict[str, object]:
    import numpy as np
    import sample.edit as edit

    checkpoint = _path(request["checkpoint"])
    cond_path = _path(request["condition"])
    input_motion = _path(request["input_motion"])
    output = _path(request["output_dir"], directory=True)
    device_index, t5_device = _device(request)
    _patch_t5(edit, t5_path, t5_device)
    args = _namespace(checkpoint)
    _apply_common(args, request, checkpoint, device_index)
    args.output_dir = str(output)
    args.object_type = str(request["object_name"])
    args.samples = [str(input_motion)]
    args.edit_mode = str(request["edit_mode"])
    args.prefix_end = float(request["prefix_end"])
    args.suffix_start = float(request["suffix_start"])
    args.upper_body_root = list(request["upper_body_root"])
    args.unique_str = ""
    edit.main(args=args, cond_dict=_condition(cond_path))

    dictionaries = sorted(output.glob("*.npy"))
    bvhs = {path.stem: path for path in output.glob("*.bvh")}
    videos = sorted(output.glob("*.mp4"))
    if not dictionaries:
        raise WorkerFailure("upstream edit produced no motion arrays")
    items: list[dict[str, str | None]] = []
    for index, dictionary_path in enumerate(dictionaries):
        loaded = np.load(dictionary_path, allow_pickle=True)
        try:
            value = loaded.item()
            motion = value["motion"]
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise WorkerFailure("upstream edit motion dictionary is invalid") from exc
        plain = output / f"edited_{index:02d}.motion.npy"
        np.save(plain, motion, allow_pickle=False)
        bvh = bvhs.get(dictionary_path.stem)
        if bvh is None:
            raise WorkerFailure("upstream edit produced no BVH")
        video = next((path for path in videos if dictionary_path.stem.rsplit("_", 1)[0] in path.stem), None)
        if video is None:
            raise WorkerFailure("upstream edit produced no MP4 preview")
        items.append(
            {
                "motion": str(plain.resolve()),
                "upstream_edit": str(dictionary_path.resolve()),
                "bvh": str(bvh.resolve()),
                "video": str(video.resolve()),
            }
        )
    return {"object_name": str(request["object_name"]), "items": items}


def _copy_checkpoint(checkpoint: Path, output: Path) -> Path:
    directory = output / "checkpoint"
    directory.mkdir(parents=True, exist_ok=True)
    staged = directory / checkpoint.name
    shutil.copyfile(checkpoint, staged)
    shutil.copyfile(checkpoint.with_name("args.json"), directory / "args.json")
    return staged


def _run_correspondence(request: Mapping[str, Any], t5_path: Path) -> dict[str, object]:
    import numpy as np
    import sample.dift_correspondence as dift

    checkpoint = _path(request["checkpoint"])
    output = _path(request["output_dir"], directory=True)
    staged_checkpoint = _copy_checkpoint(checkpoint, output)
    device_index, t5_device = _device(request)
    _patch_t5(dift, t5_path, t5_device)

    reference = request["reference"]
    target = request["target"]
    if not isinstance(reference, dict) or not isinstance(target, dict):
        raise WorkerFailure("correspondence inputs are invalid")
    reference_motion = _path(reference["motion"])
    target_motion = _path(target["motion"])
    reference_condition_path = _path(reference["condition"])
    target_condition_path = _path(target["condition"])
    reference_condition = _condition(reference_condition_path)
    target_condition = _condition(target_condition_path)
    reference_name = str(reference["object_name"])
    target_name = str(target["object_name"])
    if reference_name not in reference_condition or target_name not in target_condition:
        raise WorkerFailure("correspondence condition keys are missing")
    if reference_name == target_name and _sha256(reference_condition_path) != _sha256(target_condition_path):
        raise WorkerFailure("same-name correspondence inputs use different skeleton conditions")
    # Built-in cond.npy contains every published skeleton.  Keep only the two
    # explicitly staged identities so dictionaries from different sources can
    # never overwrite an unrelated same-named entry during a merge.
    cond_dict = {reference_name: reference_condition[reference_name]}
    if target_name != reference_name:
        cond_dict[target_name] = target_condition[target_name]

    # DIFT infers object identity from the filename prefix before the first '_'.
    # Explicit staging prevents a run-id or user filename from changing that key.
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    ref_staged = inputs / f"{reference_name}_reference.npy"
    tgt_staged = inputs / f"{target_name}_target.npy"
    np.save(ref_staged, np.load(reference_motion, allow_pickle=False), allow_pickle=False)
    np.save(tgt_staged, np.load(target_motion, allow_pickle=False), allow_pickle=False)

    args = _namespace(staged_checkpoint)
    _apply_common(args, request, staged_checkpoint, device_index)
    args.output_dir = str(output / "ignored_upstream_output")
    args.sample_ref = str(ref_staged)
    args.sample_tgt = [str(tgt_staged)]
    args.tmp_save_dir = str(output)
    args.suffix = ""
    args.dift_type = str(request["dift_type"])
    args.layer = int(request["layer"])
    args.timestep = int(request["timestep"])
    args.num_repetitions = 1
    dift.run_dift(args=args, cond_dict=cond_dict)

    dift_output = staged_checkpoint.parent / staged_checkpoint.stem / "dift_out"
    mappings = sorted(dift_output.glob("*.npy"))
    videos = sorted(dift_output.glob("*.mp4"))
    if not mappings or not videos:
        raise WorkerFailure("upstream DIFT outputs are incomplete")
    published = output / "correspondence"
    published.mkdir(parents=True, exist_ok=True)
    published_mappings: list[Path] = []
    published_videos: list[Path] = []
    for path in mappings:
        destination = published / path.name
        shutil.copyfile(path, destination)
        published_mappings.append(destination)
    for path in videos:
        destination = published / path.name
        shutil.copyfile(path, destination)
        published_videos.append(destination)
    # The copy was deliberately used only to redirect upstream's hard-coded
    # model-relative DIFT directory.  Never retain a checkpoint per workflow.
    shutil.rmtree(staged_checkpoint.parent)
    return {
        "reference_object": reference_name,
        "target_object": target_name,
        "mappings": [str(path.resolve()) for path in published_mappings],
        "videos": [str(path.resolve()) for path in published_videos],
    }


def execute(request: Mapping[str, Any]) -> dict[str, object]:
    source_root = _path(request["source_root"], directory=True)
    motion_source = _path(request["motion_source"], directory=True)
    t5_path = _path(request["t5_path"], directory=True)
    operation = request.get("operation")
    with _upstream_imports(source_root, motion_source):
        if operation == "preprocess":
            return _run_preprocess(request)
        if operation in {"generate", "generate_custom"}:
            return _run_generate(request, t5_path)
        if operation == "edit":
            return _run_edit(request, t5_path)
        if operation == "correspondence":
            return _run_correspondence(request, t5_path)
    raise WorkerFailure("unknown worker operation")


def main() -> int:
    result_path: Path | None = None
    try:
        line = sys.stdin.buffer.readline(4 * 1024 * 1024 + 1)
        if not line or len(line) > 4 * 1024 * 1024 or sys.stdin.buffer.read(1):
            raise WorkerFailure("worker requires exactly one bounded request line")
        request = json.loads(line.decode("utf-8"))
        if not isinstance(request, dict):
            raise WorkerFailure("worker request must be an object")
        result_path = _path(request["result_path"], directory=False) if Path(str(request["result_path"])).exists() else Path(str(request["result_path"]))
        if not result_path.is_absolute():
            raise WorkerFailure("result path must be absolute")
        result = execute(request)
        _write_json(result_path, {"ok": True, "result": result})
        return 0
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        if result_path is not None:
            try:
                _write_json(result_path, {"ok": False})
            except BaseException:
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

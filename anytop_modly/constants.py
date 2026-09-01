"""Immutable upstream revisions, model inventory, and public identifiers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Final


EXTENSION_ID: Final = "modly-anytop-extension"
EXTENSION_VERSION: Final = "1.0.0"
ANYTOP_COMMIT: Final = "e780d1575ca0121f29bb53821b309cf564156a95"
ANYTOP_HF_REVISION: Final = "a1efdbb4c1495efe6a1e54d19c2824d1a957ce36"
T5_REVISION: Final = "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
MOTION_COMMIT: Final = "ac236251f90e5ca37c444c53ad383fc85de6d833"
SOURCE_PATCHSET: Final = "matplotlib-cp311-v1"

RUNTIME_CONFIG_FILENAME: Final = "runtime_config.json"
SETUP_STATE_FILENAME: Final = "setup-state.json"
READY_MARKER_FILENAME: Final = "ready.json"
SETUP_LOCK_FILENAME: Final = ".setup.lock"
READY_SCHEMA_VERSION: Final = 1
ASSET_REVISION_SCHEMA: Final = 1

ANYTOP_SOURCE_RELATIVE: Final = "source/AnyTop"
MOTION_SOURCE_RELATIVE: Final = "source/Motion"
T5_SOURCE_RELATIVE: Final = "t5-base"
BUILTIN_COND_RELATIVE: Final = "dataset/truebones/zoo/truebones_processed/cond.npy"
SKELETON_LIST_RELATIVE: Final = (
    "dataset/truebones/zoo/truebones_processed/Truebones_skeletons.txt"
)

ANYTOP_SOURCE_TREE_SHA256: Final = (
    "48ad2e490bf2d0951abdb8e73c37d2613d9e9276da9680a84f7d6ebc5fabcd79"
)
MOTION_SOURCE_TREE_SHA256: Final = (
    "815669f254ba5f4e17fc0ed04ba82693290e03efdb07b454a6ee94e17910a90d"
)


@dataclass(frozen=True)
class AssetSpec:
    relative_path: str
    url: str
    size: int
    sha256: str
    role: str


def _anytop_hf(path: str) -> str:
    return (
        "https://huggingface.co/Inbar2344/AnyTop/resolve/"
        f"{ANYTOP_HF_REVISION}/{path}?download=true"
    )


def _t5_hf(path: str) -> str:
    return (
        "https://huggingface.co/google-t5/t5-base/resolve/"
        f"{T5_REVISION}/{path}?download=true"
    )


CHECKPOINTS: Final = {
    "unified": {
        "directory": "all_model_dataset_truebones_bs_16_latentdim_128",
        "filename": "model000459999.pt",
        "args": "args.json",
    },
    "bipeds": {
        "directory": "bipeds_model_dataset_truebones_bs_16_latentdim_128",
        "filename": "model000329999.pt",
        "args": "args.json",
    },
    "flying": {
        "directory": "flying_model_dataset_truebones_bs_16_latentdim_128",
        "filename": "model000229999.pt",
        "args": "args.json",
    },
    "millipeds_snakes": {
        "directory": "millipeds_snakes_model_dataset_truebones_bs_16_latentdim_128",
        "filename": "model000349999.pt",
        "args": "args.json",
    },
    "quadropeds": {
        "directory": "quadropeds_model_dataset_truebones_bs_16_latentdim_128",
        "filename": "model000189999.pt",
        "args": "args.json",
    },
}


ASSETS: Final = (
    AssetSpec(
        "archives/AnyTop.tar.gz",
        f"https://github.com/Anytop2025/Anytop/archive/{ANYTOP_COMMIT}.tar.gz",
        6_059_525,
        "a6aa800ad7f10b7c5fca40af987368d3c2f5336d4f8efac32d2db7d630cf898e",
        "AnyTop source archive",
    ),
    AssetSpec(
        "archives/Motion.tar.gz",
        f"https://github.com/inbar-2344/Motion/archive/{MOTION_COMMIT}.tar.gz",
        25_849,
        "b40a2cc7b4f4e52b4dc7b33452b9e1b41ea4e47a12fa1c6442d0bddb7cfc8a86",
        "Motion source archive",
    ),
    AssetSpec(
        "checkpoints/all_model_dataset_truebones_bs_16_latentdim_128/args.json",
        _anytop_hf("checkpoints/all_model_dataset_truebones_bs_16_latentdim_128/args.json"),
        1_118,
        "d30efa1d8e36789f4b5194bd12a4a7f554cc01d06b0ffdcbb7a0d1380f686ba6",
        "unified checkpoint arguments",
    ),
    AssetSpec(
        "checkpoints/all_model_dataset_truebones_bs_16_latentdim_128/model000459999.pt",
        _anytop_hf("checkpoints/all_model_dataset_truebones_bs_16_latentdim_128/model000459999.pt"),
        9_222_606,
        "7d137ac1d3cef817d9334298b087db6316f27e2dd2f9d83b67cb4fb1513576c0",
        "unified checkpoint",
    ),
    AssetSpec(
        "checkpoints/bipeds_model_dataset_truebones_bs_16_latentdim_128/args.json",
        _anytop_hf("checkpoints/bipeds_model_dataset_truebones_bs_16_latentdim_128/args.json"),
        1_125,
        "b1d6ab366ecbcd61e80afb6868caeedfcba9a1fa3b78a8a7f3892582cdb97245",
        "bipeds checkpoint arguments",
    ),
    AssetSpec(
        "checkpoints/bipeds_model_dataset_truebones_bs_16_latentdim_128/model000329999.pt",
        _anytop_hf("checkpoints/bipeds_model_dataset_truebones_bs_16_latentdim_128/model000329999.pt"),
        9_222_606,
        "206bcbffea78d14e53ef3f0a6694eb44bffc26639c0cd3253c9b75d0d2004b9a",
        "bipeds checkpoint",
    ),
    AssetSpec(
        "checkpoints/flying_model_dataset_truebones_bs_16_latentdim_128/args.json",
        _anytop_hf("checkpoints/flying_model_dataset_truebones_bs_16_latentdim_128/args.json"),
        1_127,
        "10515a7788d68776b37b929b50db369cb6c256d4727902669afaa2919547eb5a",
        "flying checkpoint arguments",
    ),
    AssetSpec(
        "checkpoints/flying_model_dataset_truebones_bs_16_latentdim_128/model000229999.pt",
        _anytop_hf("checkpoints/flying_model_dataset_truebones_bs_16_latentdim_128/model000229999.pt"),
        9_222_606,
        "073ae4793b1fcc8d234612a6bff0c00b1cd2d1f598e7391f3b8c2db99daa541b",
        "flying checkpoint",
    ),
    AssetSpec(
        "checkpoints/millipeds_snakes_model_dataset_truebones_bs_16_latentdim_128/args.json",
        _anytop_hf("checkpoints/millipeds_snakes_model_dataset_truebones_bs_16_latentdim_128/args.json"),
        1_157,
        "92060b25cceefaaa1dd9568b571edd7f0e2202e20fc22ce5bcbb5d4de315eebc",
        "millipeds and snakes checkpoint arguments",
    ),
    AssetSpec(
        "checkpoints/millipeds_snakes_model_dataset_truebones_bs_16_latentdim_128/model000349999.pt",
        _anytop_hf("checkpoints/millipeds_snakes_model_dataset_truebones_bs_16_latentdim_128/model000349999.pt"),
        9_222_606,
        "e08ecd5bfb0063bdcac4290706bf5d87c09387aa3be1de74d76791af8282c1aa",
        "millipeds and snakes checkpoint",
    ),
    AssetSpec(
        "checkpoints/quadropeds_model_dataset_truebones_bs_16_latentdim_128/args.json",
        _anytop_hf("checkpoints/quadropeds_model_dataset_truebones_bs_16_latentdim_128/args.json"),
        1_139,
        "18af775a0995e28531b45bbff264502f6b54fd200f23872dbd82c2b53dc3a988",
        "quadropeds checkpoint arguments",
    ),
    AssetSpec(
        "checkpoints/quadropeds_model_dataset_truebones_bs_16_latentdim_128/model000189999.pt",
        _anytop_hf("checkpoints/quadropeds_model_dataset_truebones_bs_16_latentdim_128/model000189999.pt"),
        9_222_606,
        "daf581e4fc8e601b32940e39fb24ef8cfdbf8fd3612ef9a75d1b4c7795b99afb",
        "quadropeds checkpoint",
    ),
    AssetSpec(
        BUILTIN_COND_RELATIVE,
        _anytop_hf(BUILTIN_COND_RELATIVE),
        4_319_132,
        "4a13aa1133fabc9730e86b573a8c00bb5568ca87fd71e4068c41fb81a9e99b1e",
        "built-in skeleton conditions",
    ),
    AssetSpec(
        SKELETON_LIST_RELATIVE,
        _anytop_hf(SKELETON_LIST_RELATIVE),
        650,
        "f2daa8c57a08df35626c19d15176e08c0e16580fcd634db1bdafd55ffa6911f2",
        "upstream skeleton index",
    ),
    AssetSpec(
        "t5-base/config.json",
        _t5_hf("config.json"),
        1_208,
        "46dd7cb62d29c81fb551e0ef1ea274c24a46ba441eeb948897706252933df033",
        "T5 configuration",
    ),
    AssetSpec(
        "t5-base/model.safetensors",
        _t5_hf("model.safetensors"),
        891_646_390,
        "a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4",
        "T5 encoder weights",
    ),
    AssetSpec(
        "t5-base/spiece.model",
        _t5_hf("spiece.model"),
        791_656,
        "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86",
        "T5 SentencePiece model",
    ),
    AssetSpec(
        "t5-base/tokenizer.json",
        _t5_hf("tokenizer.json"),
        1_389_353,
        "d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066",
        "T5 fast-tokenizer data",
    ),
)


def _asset_revision_digest(assets: tuple[AssetSpec, ...]) -> str:
    """Identify every byte and source transformation in an immutable snapshot.

    ``EXTENSION_VERSION`` is deliberately absent: wrapper-only releases must
    keep using the same published model snapshot.  Conversely, changing an
    asset, upstream revision, patched source tree, patchset or marker contract
    necessarily selects a different directory and leaves the prior release
    untouched.
    """

    payload = {
        "asset_revision_schema": ASSET_REVISION_SCHEMA,
        "ready_schema": READY_SCHEMA_VERSION,
        "extension_id": EXTENSION_ID,
        "upstream": {
            "anytop_commit": ANYTOP_COMMIT,
            "anytop_hf_revision": ANYTOP_HF_REVISION,
            "motion_commit": MOTION_COMMIT,
            "t5_revision": T5_REVISION,
        },
        "source_patchset": SOURCE_PATCHSET,
        "source_trees": {
            "anytop": ANYTOP_SOURCE_TREE_SHA256,
            "motion": MOTION_SOURCE_TREE_SHA256,
        },
        "inventory": [
            {
                "path": spec.relative_path,
                "size": spec.size,
                "sha256": spec.sha256,
            }
            for spec in sorted(assets, key=lambda item: item.relative_path)
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


ASSET_REVISION_DIGEST: Final = _asset_revision_digest(ASSETS)
# A 96-bit digest prefix keeps the path readable while the full digest is also
# recorded in ready.json and checked by every wrapper that consumes it.
REVISION_ID: Final = (
    f"anytop-r{ASSET_REVISION_SCHEMA}-{ASSET_REVISION_DIGEST[:24]}"
)


FLYING: Final = (
    "Bat", "Dragon", "Bird", "Buzzard", "Eagle", "Giantbee", "Parrot",
    "Parrot2", "Pigeon", "Pteranodon", "Tukan",
)
BIPEDS: Final = (
    "Ostrich", "Flamingo", "Raptor", "Raptor2", "Raptor3", "Trex",
    "Chicken", "Tyranno",
)
MILLIPEDS: Final = (
    "Cricket", "SpiderG", "Scorpion", "Isopetra", "FireAnt", "Crab",
    "Centipede", "Roach", "Ant", "HermitCrab", "Scorpion-2", "Spider",
)
SNAKES: Final = ("Anaconda", "KingCobra")
FISH: Final = ("Pirrana",)
QUADROPEDS: Final = (
    "Horse", "Hippopotamus", "Comodoa", "Camel", "Bear", "Buffalo", "Cat",
    "BrownBear", "Coyote", "Crocodile", "Elephant", "Deer", "Fox",
    "Gazelle", "Goat", "Jaguar", "Lynx", "Tricera", "Stego", "SandMouse",
    "Raindeer", "Puppy", "PolarBear", "Monkey", "Mammoth", "Alligator",
    "Hamster", "Hound", "Leapord", "Lion", "PolarBearB", "Rat", "Rhino",
    "SabreToothTiger", "Skunk", "Turtle",
)
BUILTIN_SKELETONS: Final = (
    FLYING + BIPEDS + MILLIPEDS + SNAKES + FISH + QUADROPEDS
)

SPECIALIZED_FAMILY: Final = {
    **{name: "flying" for name in FLYING},
    **{name: "bipeds" for name in BIPEDS},
    **{name: "millipeds_snakes" for name in MILLIPEDS + SNAKES},
    **{name: "quadropeds" for name in QUADROPEDS},
    **{name: "unified" for name in FISH},
}

NODE_IDS: Final = {
    "anytop-preprocess",
    "anytop-generate",
    "anytop-generate-custom",
    "anytop-edit",
    "anytop-correspondence",
}

# AnyTop for Modly

A Modly PROCESS integration by **DrHepa** whose wrapper code is MIT-licensed,
built for [AnyTop](https://github.com/Anytop2025/Anytop). It brings AnyTop's public
skeleton preprocessing, motion generation, motion editing, and DIFT
correspondence tools into Modly workflows.

AnyTop animates a **skeleton**. It does not rig, skin, or retarget a character
mesh. The returned GLB is a visible animated skeleton preview that keeps the
canonical AnyTop BVH and NPY results as verified sidecars.

## Installation

1. In Modly, open **Models/Extensions → Install from GitHub**.
2. Enter `https://github.com/DrHepa/modly-anytop-extension`.
3. Wait for setup to create the isolated `venv`, install the selected CPU/CUDA
   dependency lane, download the pinned upstream runtime files, and verify them.

Install and **Repair** own the model download because Modly v0.4.2 does not
provide the Models download button for PROCESS extensions. Existing files that
still match their exact size and SHA-256 are reused, so updating or repairing
the extension does not download valid weights again.

## Nodes

| Node | Input | Result |
| --- | --- | --- |
| **AnyTop Prepare Skeleton** | Text: a new object name | Prepared AnyTop bundle from one folder of BVH files |
| **AnyTop Generate Built-in** | Text: one exact built-in skeleton name | Generated motion bundle |
| **AnyTop Generate Prepared** | Prepared bundle | Generated motion bundle for that skeleton |
| **AnyTop Edit Motion** | Generated motion bundle | In-between or upper-body edited motion bundle |
| **AnyTop DIFT Correspondence** | Target motion bundle | Text result describing spatial/temporal mapping sidecars |

## Usage

- Built-in: **Text → Generate Built-in**
- Custom: **Text + BVH folder → Prepare Skeleton → Generate Prepared**
- Edit: **Generated bundle → Edit Motion**
- DIFT: connect the target bundle, then enter the reference bundle path in the
  `Reference bundle` string parameter. Modly v0.4.2 collapses multiple mesh
  inputs to one `filePath`, so the second artifact is intentionally a
  parameter.

## Parameters

All inference nodes expose the five official checkpoint families: `unified`,
`bipeds`, `flying`, `millipeds_snakes`, and `quadropeds`. For generation/edit,
`auto` chooses the matching specialized checkpoint for a known built-in name
and unified when no family can be inferred. DIFT `auto` uses unified because a
pair may cross families. Explicit selection never silently changes checkpoint.

- **Generation:** motion length `6.0` seconds by default, maximum `9.8` at the
  fixed 20 FPS; `3` repetitions; seed `10`.
- **Device:** `auto`, `cuda`, or `cpu`, plus CUDA device index. An unavailable
  explicit CUDA device fails instead of falling back.
- **Prepare Skeleton:** BVH folder path, four exact orientation-joint names in the
  order right hip, left hip, right shoulder, left shoulder, and an optional
  T/rest-pose BVH path from that folder.
- **Edit / in-between:** fixed prefix ends at `0.25` and fixed suffix begins at
  `0.75`; the prefix value must be lower than the suffix value.
- **Edit / upper body:** comma-separated subtree-root joint indices; default
  `0`, matching upstream's whole-skeleton behavior.
- **DIFT:** spatial or temporal, layer `0`, timestep `90`. The upstream temporal
  example recommends layer `3`, timestep `1`.

The manifest contains the complete ranges and tooltips shown in the Modly UI. Path fields intentionally use supported string controls; runtime validation still requires absolute paths, or `tpos_bvh` relative to its BVH folder and `Reference bundle` relative to the workspace.

## Preparing a custom skeleton

Use a short ASCII object name such as `Dog`, select a folder containing BVH
files for one consistent skeleton, and set the four orientation joints to names
that exist in those files. More motion files improve the denormalization
statistics. A natural rest pose is strongly recommended; if its string path is empty,
AnyTop chooses a pose from the folder.

Upstream expects fewer than 144 joints and consistent BVH hierarchy/channels.
All non-root joints must use three rotation channels. The folder needs a usable
rest pose and at least one motion clip after that pose is excluded. Skeletons
far outside the training distribution may still show foot sliding or weaker
motion quality.

## Built-in skeletons

**Generate Built-in** accepts exactly these 70 case-sensitive upstream names:

<details>
<summary>Show the complete list</summary>

| Family | Exact names |
| --- | --- |
| Flying (11) | `Bat`, `Dragon`, `Bird`, `Buzzard`, `Eagle`, `Giantbee`, `Parrot`, `Parrot2`, `Pigeon`, `Pteranodon`, `Tukan` |
| Bipeds (8) | `Ostrich`, `Flamingo`, `Raptor`, `Raptor2`, `Raptor3`, `Trex`, `Chicken`, `Tyranno` |
| Millipeds / snakes (14) | `Cricket`, `SpiderG`, `Scorpion`, `Isopetra`, `FireAnt`, `Crab`, `Centipede`, `Roach`, `Ant`, `HermitCrab`, `Scorpion-2`, `Spider`, `Anaconda`, `KingCobra` |
| Fish, unified (1) | `Pirrana` |
| Quadrupeds (36) | `Horse`, `Hippopotamus`, `Comodoa`, `Camel`, `Bear`, `Buffalo`, `Cat`, `BrownBear`, `Coyote`, `Crocodile`, `Elephant`, `Deer`, `Fox`, `Gazelle`, `Goat`, `Jaguar`, `Lynx`, `Tricera`, `Stego`, `SandMouse`, `Raindeer`, `Puppy`, `PolarBear`, `Monkey`, `Mammoth`, `Alligator`, `Hamster`, `Hound`, `Leapord`, `Lion`, `PolarBearB`, `Rat`, `Rhino`, `SabreToothTiger`, `Skunk`, `Turtle` |

</details>

## Outputs

Successful mesh nodes return a unique `.glb` below the active workspace's
`Workflows` directory. The GLB contains joint/bone geometry and an animation
track, not a skinned production mesh. Modly v0.4.2 can load it but its current
viewer displays the first pose only; the animation plays in GLB viewers that
support animation tracks.

Keep each output directory intact. A same-stem bundle manifest authenticates
the GLB and its canonical `.bvh`, motion `.npy`, condition `.npy`, and `.mp4`
sidecars before another AnyTop node will consume them. For multiple repetitions,
the returned GLB previews the first result while the bundle indexes every
repetition. MP4 is a preview; BVH/NPY remain the authoritative AnyTop results.
Authentication uses a private key created once under this installation's
AnyTop model snapshot; extension updates and Repair reuse it. Consequently,
workflow bundles are trusted only by the Modly installation that created them.
Windows stores the key through current-user DPAPI; Linux stores it with `0600`
permissions inside Modly-owned model storage.

DIFT returns compact JSON text because that Modly port is not a mesh. It lists
the operation, reference/target objects, DIFT type, bundle manifest, and result
files. Its NPY mapping and MP4 visualization remain under
`Workflows/AnyTop/correspondence-…`.

## Requirements and compatibility

- Modly upstream v0.4.2, commit `8d08249`, with a 64-bit CPython 3.11 or
  3.12 process host. Upstream Modly bundles CPython 3.11.9; this extension
  also preserves a separately validated CPython 3.12 dependency closure for
  local runtimes based on Python 3.12.
- Network during Install or Repair. The pinned downloads are about 0.885 GiB,
  but setup reserves enough free space for Torch, the venv, caches, extraction,
  and temporary files according to the selected lane:

| Setup lane | Isolated Torch | Required free space |
| --- | --- | --- |
| CPU | 2.4.1 CPU | 8 GiB |
| CUDA, SM50-SM99 on Windows/Linux x64 | 2.4.1 / CUDA 12.4 | 18 GiB |
| CUDA, SM50-SM99 on Linux ARM64 SBSA | 2.6.0 / CUDA 12.6 | 18 GiB |
| CUDA, SM100 or SM120 | 2.7.1 / CUDA 12.8 | 20 GiB |
| CUDA, SM103, SM110, or SM121 | 2.9.1 / CUDA 13.0 | 22 GiB |

- Windows x64, Linux x64, and Linux ARM64 **SBSA** setup routes are implemented.
  CPU and compatible official NVIDIA CUDA wheel lanes are selected from Modly's
  platform metadata. Dependency locks are ABI-specific: CPython 3.11 uses the
  original release closure, while CPython 3.12 uses the validated NumPy/SciPy/
  Matplotlib/contourpy/spaCy/thinc pins.

Both CPython 3.11 and 3.12 source tests run in CI. These platform routes have
static and mocked validation but remain **pre-hardware / runtime-unvalidated**. Upstream itself reports only Ubuntu
18.04, Python 3.8, and CUDA testing; CPU execution is an extension adaptation
and may be very slow. Stock Jetson/Tegra Python is not SBSA-compatible and is
rejected by setup; SM121 is limited to Linux ARM64. A Jetson-specific container
or manual environment is outside this stock extension's supported setup.

## Limitations

- AnyTop generates skeleton motion; it does not rig, skin, or retarget a mesh.
- Modly v0.4.2 displays the returned GLB's first pose but does not play its
  animation track in the built-in viewer.
- Training and full upstream evaluation are not included because the complete
  processed Truebones dataset is withheld/licensed and cannot be provisioned
  from the public upstream repositories.
- AnyTop's Blender renderer remains an external Blender workflow rather than a
  Modly node.
- A CPU inference lane is implemented but remains runtime-unvalidated and is
  expected to be substantially slower than CUDA.

## Model storage and Repair

The immutable snapshot is stored outside the extension checkout:

```text
<modelsDir>/modly-anytop-extension/anytop/revisions/
  anytop-r1-1fb04a4e2500dc28feb20e89/
```

It contains the five AnyTop checkpoints, their arguments, the built-in
70-skeleton condition, pinned AnyTop/Motion source, and offline T5-base assets.
The declared downloads total 950,352,459 bytes before extracted-source and
filesystem overhead.

Setup resolves Modly's configured model directory from its setup payload or
running settings service, with `MODLY_MODELS_DIR` available for custom layouts.
Valid downloads and safe partial files are reused; readiness is written only
after the complete pinned inventory and dependency smoke tests pass. Generation
runs offline and never downloads missing assets.

The revision name is derived from the full immutable inventory, source trees,
patch set, and model revisions. Updating wrapper code alone therefore reuses the
existing snapshot, virtual environment, and private bundle key.

If setup or a download is interrupted, use **Repair**. If Modly cannot expose a
custom models path while repairing, launch it first or set the same absolute
`MODLY_MODELS_DIR` for Install/Repair and runtime.

## Troubleshooting

- **Setup stops on a wheel or CUDA check:** the selected platform lane is not
  available or cannot access the reported GPU. Review setup output and run
  Repair after correcting the host; setup does not hide the failure with CPU.
- **Prepared bundle rejected:** keep its GLB, same-stem bundle manifest, and
  sidecars together and use it on the installation that created it; arbitrary
  or cross-installation GLB/BVH bundles are not valid downstream inputs.
- **Custom preprocessing fails:** verify the four joint names, rest pose,
  consistent hierarchy, rotation channels, and that at least one motion remains.
- **Viewer looks static:** this is the Modly v0.4.2 viewer limitation; inspect
  the GLB animation or canonical BVH in an animation-capable application.
- **Need an animated character mesh:** rig/skin and retarget the generated BVH
  in UniRig, Blender, Maya, or another downstream tool. AnyTop does not perform
  that stage.

## Credits

- Modly integration: **DrHepa**
- AnyTop: [official source](https://github.com/Anytop2025/Anytop),
  [project page](https://anytop2025.github.io/Anytop-page/), and
  [paper](https://arxiv.org/abs/2502.17327)
- Modly: [Lightning Pixel](https://github.com/lightningpixel/modly)

## License

The wrapper is MIT licensed. AnyTop, its weights/data, T5, Motion, and installed
dependencies retain their own terms. In particular, the pinned Motion source
has no license grant and the Truebones-derived condition has no independent
per-file license. Read [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before
redistribution.

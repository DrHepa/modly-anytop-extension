# Third-party notices

The Modly integration code in this repository is Copyright (c) 2026 DrHepa
and is distributed under the repository's MIT license. The components obtained
by `setup.py` remain under their own terms.

## AnyTop source and checkpoints

- Source: `Anytop2025/Anytop`, pinned to commit
  `e780d1575ca0121f29bb53821b309cf564156a95`.
- Checkpoints and built-in skeleton data: `Inbar2344/AnyTop`, pinned to revision
  `a1efdbb4c1495efe6a1e54d19c2824d1a957ce36`.
- The AnyTop repository root contains an MIT license carrying Copyright (c)
  2022 Guy Tevet. Its exact text is reproduced in
  `LICENSES/AnyTop-MIT.txt`.
- The Hugging Face model card is tagged `license: mit`, but that repository
  does not contain a separate license file specifically for the weights.
  `LICENSES/AnyTop-HuggingFace-NOTICE.txt` records that distinction.

AnyTop acknowledges code and ideas from MDM, GRPE, and Audiocraft. Their
licenses and notices remain applicable to the portions used upstream.

## T5-base

AnyTop uses `google-t5/t5-base`. Setup downloads the pinned public revision
`a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`. The model is identified as
Apache-2.0; the license text is reproduced in
`LICENSES/Apache-2.0.txt`.

## Motion

Setup downloads `inbar-2344/Motion` at commit
`ac236251f90e5ca37c444c53ad383fc85de6d833`. No license file or explicit
license grant was found in that pinned repository during this extension's
source audit. Motion is downloaded directly and is not vendored in this Git
repository. Absence of a license is not a permission grant; users and
distributors must assess whether their use is authorized. See
`LICENSES/Motion-NO-LICENSE-NOTICE.txt`.

## Truebones-derived data

The official `cond.npy` used for the 70 built-in skeletons is obtained from the
AnyTop Hugging Face repository and appears to be derived from the Truebones
dataset. That file has no independent license notice. The Truebones source BVH
dataset is not bundled or downloaded by this extension; upstream withholds the
processed dataset while its licensing is clarified and directs users to obtain
the dataset separately. Training and full upstream evaluation are therefore
not included. Users preprocessing BVH files must have the rights required for
those files. See `LICENSES/Truebones-Derived-Data-NOTICE.txt`.

## PyTorch3D notice and installed packages

The pinned AnyTop source contains a PyTorch3D BSD notice, reproduced in
`LICENSES/PyTorch3D-BSD-3-Clause.txt`. Python packages installed into the
extension's isolated environment are not relicensed by this repository; consult
their installed metadata and distributions for their respective terms.

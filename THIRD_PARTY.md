# Third-party components

This repository does not redistribute pretrained weights or robot meshes. The
download script points users to the original publishers so the model card,
license, revision and access terms remain attached to each artifact.

| Component | Role | Upstream terms used here |
|---|---|---|
| NVIDIA GEM-X source | Video-to-SOMA extraction | Apache-2.0 by default, with documented per-file exceptions |
| NVIDIA GEM-X weights | Body pose, ViTPose observations, temporal motion | NVIDIA Open Model License |
| NVIDIA SOMA assets/runtime | Parametric body representation | Upstream terms; obtain from the official GEM-X/SOMA instructions |
| NVIDIA SOMA Retargeter | Optional SOMA BVH to Unitree G1 adapter | Apache-2.0 for the NVIDIA source; optional assets may differ |
| NVIDIA GR00T Whole-Body Control / SONIC source | Controller and deployment implementation | Apache-2.0 |
| NVIDIA GEAR-SONIC weights | Low-latency control policy | NVIDIA Open Model License |
| Unitree G1 model assets | Robot geometry/physics | Obtain from the official upstream path and preserve its BSD-3-Clause notice |

The NVIDIA Open Model License permits model redistribution subject to its
conditions, including carrying the agreement and this attribution:

> Licensed by NVIDIA Corporation under the NVIDIA Open Model License.

It also incorporates NVIDIA Trustworthy AI and trade/export requirements. We
avoid mirroring the multi-gigabyte files because an official, revision-pinned
download is smaller, clearer and less likely to detach future users from those
terms.

## Deliberately excluded

- `yolox_humanart.onnx`: not needed because boxes are supplied explicitly;
  this also avoids relying on an unclear HumanArt-trained detector weight.
- GVHMR, SMPL and SMPL-X: none is required by this route.
- GMR: SOMA Retargeter is used for the public G1-reference route.
- BONES-SEED training data and SONIC training-time SMPL archives: not required
  for inference and not downloaded.

This inventory is engineering documentation, not legal advice. Check the exact
upstream revision and intended use before shipping a product.

# Third-party provenance

This repository is a fork of **OnlineAnySeg** (CVPR 2025) at upstream
`yjtang249/OnlineAnySeg` commit `152466e318f8220bcc6838c03e340cce2f2153b8`
(upstream HEAD at integration time, 2026-08-24; the fork base is byte-identical
to upstream). Upstream publishes **no license**, so the code defaults to
all-rights-reserved: this fork exists for research reproduction only and must
not be redistributed further.

Components and their terms:

| Component | Location | License / terms |
| --- | --- | --- |
| OnlineAnySeg | repo root | none published (all rights reserved; research use) |
| FCGF (vendored, incl. `ResUNetBN2C-16feat-3conv.pth`) | `third_party/FCGF/` | MIT (Chris Choy / Jaesik Park) |
| MinkowskiEngine | cloned into `third_party/MinkowskiEngine/` by `scripts/bootstrap_minkowski.sh` (pinned `02fc608b`) | MIT (NVIDIA) |
| Entity / CropFormer sources | cloned into `third_party/Entity/` by `scripts/bootstrap_cropformer.sh` (pinned `6e7e13ac`) | Apache-2.0 |
| CropFormer checkpoint `Mask2Former_hornet_3x_576d0b.pth` | downloaded from HF `qqlu1992/Adobe_EntitySeg` (gated) | Adobe EntitySeg research terms; not redistributed |
| OpenCLIP ViT-H-14 checkpoint | downloaded from HF `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | LAION (MIT model license); not redistributed |
| detectron2 / mmcv / pytorch3d / torch | conda-forge & PyPI binaries | Apache-2.0 / Apache-2.0 / BSD / BSD |

DEG adapter files added by this fork (`segment.py`, `segmenter.py`,
`mask_predict.py`, `scripts/*`, `config/deg_cropformer.yaml`, `pyproject.toml`)
follow the deg workspace's terms.

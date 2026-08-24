"""CLI entry point for the OnlineAnySeg external segmenter.

Usage (invoked by pixi task):
    python segment.py <transforms_path> <scene_path>

Outputs ``objects_path: <path>`` to stdout for the parent process to read.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import logging
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import tyro

from initializerdefs import Observations, SceneSetup, get_mesh_path_for_transforms, load_observations_from_transforms_path
from segmenter import (
    DEFAULT_CLIP_CHECKPOINT,
    DEFAULT_CONFIG_TEMPLATE,
    DEFAULT_CROPFORMER_CHECKPOINT,
    DEFAULT_NN_DISTANCE_BOUND,
    initialize_scene,
)


DEFAULT_SEED = 0


@dataclass
class Args:
    transforms_path: tyro.conf.Positional[Path]
    """Path to transforms.json for the dataset."""

    scene_path: tyro.conf.Positional[Path]
    """Path to the pickled SceneSetup."""

    config_template: Path = DEFAULT_CONFIG_TEMPLATE
    """OnlineAnySeg yaml config template (instantiated per run with the frame size)."""

    cropformer_checkpoint: Path = DEFAULT_CROPFORMER_CHECKPOINT
    """Path to the CropFormer checkpoint (gated on HF; download once via the
    download_cropformer_checkpoint task or place manually)."""

    clip_checkpoint: Path = DEFAULT_CLIP_CHECKPOINT
    """Path to the OpenCLIP ViT-H-14 checkpoint."""

    confidence_threshold: float = 0.6
    """Minimum score for CropFormer instance predictions (native default)."""

    min_mask_pixel_size: int = 500
    """Minimum pixel count for a valid 2D mask (native default)."""

    mask_weight_threshold: int = 1
    """Export co-observation threshold. Native default is 5 -- unreachable at
    ~5 total frames; the deg pipeline applies its own >=3-frame filter."""

    min_instance_points: int = 200
    """Minimum reconstructed points per exported instance (native default)."""

    nn_distance_bound: float = DEFAULT_NN_DISTANCE_BOUND
    """Recon-cloud -> deg-mesh nearest-neighbour bound in metres (upstream's
    own evaluation convention)."""


def run() -> None:
    logger = logging.getLogger("onlineanyseg-segmenter")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s"))
    logger.addHandler(ch)

    args = tyro.cli(Args)

    random.seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)
    torch.manual_seed(DEFAULT_SEED)
    torch.cuda.manual_seed_all(DEFAULT_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    with contextlib.redirect_stdout(sys.stderr):
        logger.info("--------------")
        logger.info("Starting OnlineAnySeg initialization")
        logger.info("params: %s", args)
        logger.info("Determinism enabled with seed=%d", DEFAULT_SEED)

        logger.info("Loading observations from %s ...", args.transforms_path)
        dataset: Observations = load_observations_from_transforms_path(args.transforms_path)
        logger.info("Observations loaded.")

        logger.info("Loading scene setup from %s ...", args.scene_path)
        scene = SceneSetup.load(args.scene_path)
        logger.info("Scene loaded.")

        if dataset.id is None:
            logger.info("Dataset has no id, using transient id")
            dataset.id = f"transient_{time.strftime('%Y%m%d-%H%M%S')}"

        project_root = Path(__file__).parent
        output_dir = project_root / "outputs" / f"{time.strftime('%Y%m%d-%H%M%S')}_{dataset.id}"

        logger.info("Initializing scene ...")
        objects = initialize_scene(
            dataset,
            scene,
            intermediate_outputs_path=output_dir,
            mesh_path=get_mesh_path_for_transforms(args.transforms_path),
            config_template=args.config_template,
            cropformer_checkpoint=args.cropformer_checkpoint,
            clip_checkpoint=args.clip_checkpoint,
            confidence_threshold=args.confidence_threshold,
            min_mask_pixel_size=args.min_mask_pixel_size,
            mask_weight_threshold=args.mask_weight_threshold,
            min_instance_points=args.min_instance_points,
            nn_distance_bound=args.nn_distance_bound,
        )

        output_path = output_dir / "objectsdef.pkl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        objects.save(output_path)

        logger.info("Objects saved to %s", output_path)

    print(f"objects_path: {output_path}")


if __name__ == "__main__":
    run()

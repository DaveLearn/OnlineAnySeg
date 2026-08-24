"""Download the OpenCLIP ViT-H-14 checkpoint OnlineAnySeg's mask stage uses.

Ungated on Hugging Face, so this is safe as a hard `depends-on` of
segment_external (unlike the gated CropFormer checkpoint). Downloads via
huggingface_hub into its cache and copies to a stable path through a
``.partial`` rename so an interrupted run never leaves a truncated file that
satisfies the pixi task's ``outputs`` check.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
REPO_FILE = "open_clip_pytorch_model.bin"


def parse_args() -> argparse.Namespace:
    argv = sys.argv[1:]
    if argv[:1] == ["--"]:
        argv = argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/clip/open_clip_pytorch_model.bin"),
        help="Local output path for the checkpoint.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    output: Path = args.output

    if output.exists() and not args.force:
        print(f"CLIP checkpoint already present at {output}", file=sys.stderr)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID}/{REPO_FILE} ...", file=sys.stderr)
    cached = hf_hub_download(repo_id=REPO_ID, filename=REPO_FILE)

    partial = output.with_suffix(output.suffix + ".partial")
    shutil.copyfile(cached, partial)
    partial.replace(output)
    print(f"CLIP checkpoint ready at {output}", file=sys.stderr)


if __name__ == "__main__":
    main()

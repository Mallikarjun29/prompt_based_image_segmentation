#!/usr/bin/env python
"""Run a fine-tuned DeepLabV3 model to generate drywall joint masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.deeplab_inference import DeepLabInferenceConfig, run_deeplab_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/deeplab_drywall_joint.pth"))
    parser.add_argument("--target-label", default="drywall_joint")
    parser.add_argument("--image-cache", type=Path, default=Path("data/processed_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/deeplab_masks"))
    parser.add_argument("--mask-suffix", default="deeplab")
    parser.add_argument("--class-labels", nargs="+", default=None, help="Specific class labels to emit masks for. Defaults to checkpoint labels or target label.")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(512, 512))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resize = tuple(args.resize) if args.resize else None
    config = DeepLabInferenceConfig(
        manifest=args.manifest,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        image_cache=args.image_cache,
        mask_suffix=args.mask_suffix,
        class_labels=args.class_labels,
        target_label=args.target_label,
        resize=resize,
        batch_size=args.batch_size,
        device=args.device,
        max_samples=args.max_samples,
        skip_existing=args.skip_existing,
    )
    summary = run_deeplab_inference(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

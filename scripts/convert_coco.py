#!/usr/bin/env python
"""Batch-convert COCO exports into processed JSON manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.coco_converter import convert_coco_annotations  # noqa: E402

DATASETS = {
    "coco_cracks": {
        "dataset_name": "cracks",
        "input_dir": ROOT / "data" / "raw" / "coco_cracks",
        "splits": ("train", "valid", "test"),
        "label_map": {
            "crack": "drywall_crack",
            "NewCracks - v2 2024-05-18 10-54pm": "drywall_crack",
        },
        "prompt": "detect drywall cracks",
    },
    "coco_joints": {
        "dataset_name": "drywall_join_detect",
        "input_dir": ROOT / "data" / "raw" / "coco_joints",
        "splits": ("train", "valid", "test"),
        "label_map": {
            "Drywall-Join": "drywall_joint",
            "drywall-join": "drywall_joint",
        },
        "prompt": "detect drywall joints",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Dataset key to convert (defaults to all).",
    )
    parser.add_argument(
        "--output-dir",
        default=ROOT / "data" / "processed",
        type=Path,
        help="Directory to store processed JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = DATASETS.keys() if args.dataset == "all" else (args.dataset,)

    summaries = []
    for dataset_key in targets:
        cfg = DATASETS[dataset_key]
        dataset_name = cfg.get("dataset_name", dataset_key)
        input_dir: Path = cfg["input_dir"]
        label_map = cfg["label_map"]
        prompt = cfg.get("prompt", "")
        output_dir = args.output_dir / dataset_name

        for split in cfg["splits"]:
            split_dir = input_dir / split
            annotations_path = split_dir / "_annotations.coco.json"
            if not annotations_path.exists():
                continue
            output_path = output_dir / f"{split}.json"
            summary = convert_coco_annotations(
                annotations_path=annotations_path,
                image_dir=split_dir,
                output_path=output_path,
                dataset_name=dataset_name,
                split=split,
                label_map=label_map,
                prompt_template=prompt,
            )
            summaries.append(summary.__dict__)

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

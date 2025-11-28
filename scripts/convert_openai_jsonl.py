#!/usr/bin/env python
"""Batch-convert Roboflow/OpenAI JSONL exports into processed JSON datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.openai_jsonl_converter import convert_jsonl_file  # noqa: E402

DATASETS = {
    "drywall_join_detect": {
        "input_dir": ROOT / "data" / "raw" / "drywall_join_detect",
        "splits": ("train", "valid"),
        "label_map": {"drywall-join": "drywall_joint"},
    },
    "cracks": {
        "input_dir": ROOT / "data" / "raw" / "cracks",
        "splits": ("train", "valid", "test"),
        "label_map": {"NewCracks - v2 2024-05-18 10-54pm": "drywall_crack"},
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
    for dataset_name in targets:
        cfg = DATASETS[dataset_name]
        input_dir: Path = cfg["input_dir"]
        label_map = cfg["label_map"]

        for split in cfg["splits"]:
            input_path = input_dir / f"_annotations.{split}.jsonl"
            if not input_path.exists():
                continue
            output_dir = args.output_dir / dataset_name
            output_path = output_dir / f"{split}.json"
            summary = convert_jsonl_file(
                input_path=input_path,
                output_path=output_path,
                dataset_name=dataset_name,
                split=split,
                label_map=label_map,
            )
            summaries.append(summary)

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

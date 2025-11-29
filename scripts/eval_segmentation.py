#!/usr/bin/env python
"""Compute IoU/Dice between predicted masks and manifest-derived ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.evaluator import EvalConfig, evaluate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--mask-dir", type=Path, required=True, help="Directory with predicted masks")
    parser.add_argument("--prompt-label", required=True)
    parser.add_argument("--prompt-text", default=None, help="Override prompt text used for mask names")
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config to resolve prompt text")
    parser.add_argument("--image-cache", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Optional path to dump JSON summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prompt_text = args.prompt_text
    if prompt_text is None:
        if args.config:
            config_data = yaml.safe_load(args.config.read_text())
            prompt_entries = {entry["label"]: entry for entry in config_data.get("prompts", [])}
            if args.prompt_label not in prompt_entries:
                raise KeyError(f"Prompt label {args.prompt_label} not found in {args.config}")
            prompt_text = prompt_entries[args.prompt_label]["text"]
        else:
            prompt_text = args.prompt_label

    config = EvalConfig(
        manifest=args.manifest,
        mask_dir=args.mask_dir,
        prompt_label=args.prompt_label,
        prompt_text=prompt_text,
        target_label=args.target_label,
        threshold=args.threshold,
        image_cache=args.image_cache,
        max_samples=args.max_samples,
    )
    summary = evaluate(config)
    output = json.dumps(summary, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)


if __name__ == "__main__":
    main()

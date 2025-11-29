#!/usr/bin/env python
"""Run prompt-driven segmentation on a single test image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.deeplab_inference import (
    SingleImageInferenceConfig,
    run_single_image_inference,
)
from src.utils.evaluator import sanitize_prompt_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to the RGB image to segment")
    parser.add_argument("--prompt-label", required=True, help="Route label describing which checkpoint to run")
    parser.add_argument("--prompt-text", required=True, help="Free-form description of what to segment")
    parser.add_argument(
        "--routes-config",
        type=Path,
        default=Path("configs/segmentation_routes.yaml"),
        help="Routing table describing prompt-label -> checkpoint mapping",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory (defaults to outputs/manual/<prompt-label>)",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional override for the saved mask base name (defaults to the image stem)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override torch device (cpu/cuda)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optional resize override",
    )
    parser.add_argument(
        "--no-label-suffix",
        action="store_true",
        help="Do not append the class label to the sanitized prompt suffix when writing masks",
    )
    return parser.parse_args()


def _load_routes(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("routes", {})


def _resolve_resize(route_entry: Dict[str, Any], override: tuple[int, int] | None):
    if override:
        return override
    if "resize" in route_entry:
        width, height = route_entry["resize"]
        return int(width), int(height)
    return None


def main() -> None:
    args = parse_args()
    routes = _load_routes(args.routes_config)
    if args.prompt_label not in routes:
        known = ", ".join(sorted(routes)) or "<none>"
        raise KeyError(f"Prompt label '{args.prompt_label}' not found (known: {known})")

    entry = routes[args.prompt_label]
    engine = entry.get("engine", "deeplab").lower()
    if engine != "deeplab":
        raise NotImplementedError(f"Engine '{engine}' is not supported yet")

    if not args.image.exists():
        raise FileNotFoundError(args.image)

    output_dir = args.output_dir or Path(entry.get("output_dir", f"outputs/manual/{args.prompt_label}"))
    resize = _resolve_resize(entry, tuple(args.resize) if args.resize else None)
    prompt_suffix = sanitize_prompt_text(args.prompt_text)

    config = SingleImageInferenceConfig(
        image_path=args.image,
        checkpoint=Path(entry["checkpoint"]),
        output_dir=output_dir,
        prompt_suffix=prompt_suffix,
        class_labels=entry.get("class_labels"),
        target_label=entry.get("target_label"),
        resize=resize,
        device=args.device,
        image_id=args.output_name,
        include_label_suffix=not args.no_label_suffix,
    )

    summary = run_single_image_inference(config)
    summary.update(
        {
            "prompt_label": args.prompt_label,
            "prompt_text": args.prompt_text,
            "engine": engine,
        }
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

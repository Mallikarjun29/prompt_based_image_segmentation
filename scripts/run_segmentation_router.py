#!/usr/bin/env python
"""Route a segmentation request to the correct inference engine for a prompt label."""

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

from src.pipeline.deeplab_inference import DeepLabInferenceConfig, run_deeplab_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Processed dataset manifest to iterate")
    parser.add_argument("--prompt-label", required=True, help="Logical prompt or defect label to run")
    parser.add_argument(
        "--routes-config",
        type=Path,
        default=Path("configs/segmentation_routes.yaml"),
        help="YAML file describing per-label routing rules",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory (defaults to outputs/routed/<prompt-label>)",
    )
    parser.add_argument(
        "--image-cache",
        type=Path,
        default=Path("data/processed_images"),
        help="Directory containing cached manifest imagery",
    )
    parser.add_argument("--device", default=None, help="Override torch device (cpu/cuda)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for inference")
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optional resize override for DeepLab inference",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap")
    parser.add_argument("--skip-existing", action="store_true", help="Skip samples when masks already exist")
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


def _resolve_batch_size(route_entry: Dict[str, Any], override: int | None) -> int:
    if override is not None:
        return override
    return int(route_entry.get("batch_size", 4))


def main() -> None:
    args = parse_args()
    routes = _load_routes(args.routes_config)
    if args.prompt_label not in routes:
        known = ", ".join(sorted(routes)) or "<none>"
        raise KeyError(f"Prompt label '{args.prompt_label}' not found in {args.routes_config} (known: {known})")

    entry = routes[args.prompt_label]
    engine = entry.get("engine", "deeplab").lower()
    if engine != "deeplab":
        raise NotImplementedError(f"Engine '{engine}' is not supported yet. Use DeepLab routes or extend the router.")

    output_dir = args.output_dir or Path(entry.get("output_dir", f"outputs/routed/{args.prompt_label}"))
    resize = _resolve_resize(entry, tuple(args.resize) if args.resize else None)
    batch_size = _resolve_batch_size(entry, args.batch_size)

    config = DeepLabInferenceConfig(
        manifest=args.manifest,
        checkpoint=Path(entry["checkpoint"]),
        output_dir=output_dir,
        image_cache=args.image_cache,
        mask_suffix=entry.get("mask_suffix", args.prompt_label),
        class_labels=entry.get("class_labels"),
        target_label=entry.get("target_label"),
        resize=resize,
        batch_size=batch_size,
        device=args.device,
        max_samples=args.max_samples,
        skip_existing=args.skip_existing,
    )
    summary = run_deeplab_inference(config)
    summary.update({"prompt_label": args.prompt_label, "engine": engine})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

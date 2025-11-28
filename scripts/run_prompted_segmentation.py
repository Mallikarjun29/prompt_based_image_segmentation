#!/usr/bin/env python
"""Run prompt-conditioned segmentation over a processed dataset manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.image_cache import ImageCache
from src.data.processed_dataset import ProcessedDataset
from src.models.grounding_dino import GroundingDINOModel
from src.models.sam_wrapper import SamSegmentor
from src.pipeline.prompted_segmentor import PromptConfig, PromptedSegmentor


def load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_prompt_config(config: Dict[str, object], label: str) -> PromptConfig:
    prompts = {item["label"]: item for item in config.get("prompts", [])}
    if label not in prompts:
        raise KeyError(f"Prompt label '{label}' not found in config")
    entry = prompts[label]
    return PromptConfig(
        text=entry["text"],
        box_threshold=float(entry.get("box_threshold", config["models"]["detector"].get("box_threshold", 0.35))),
        text_threshold=float(entry.get("text_threshold", config["models"]["detector"].get("text_threshold", 0.25))),
        mask_threshold=float(entry.get("mask_threshold", config["models"]["segmentor"].get("mask_threshold", 0.5))),
        top_k=entry.get("top_k"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to processed dataset JSON")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--prompt-label", required=True, help="Prompt label defined in the config")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/masks"))
    parser.add_argument("--image-cache", type=Path, default=Path("data/processed_images"))
    parser.add_argument("--device", default=None, help="Override torch device (cpu/cuda)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    detector_cfg = config["models"]["detector"]
    segmentor_cfg = config["models"]["segmentor"]

    detector = GroundingDINOModel(
        config_path=detector_cfg["config_path"],
        checkpoint_path=detector_cfg["checkpoint_path"],
        device=args.device,
    )
    segmentor = SamSegmentor(
        checkpoint_path=segmentor_cfg["checkpoint_path"],
        model_type=segmentor_cfg.get("model_type", "vit_h"),
        device=args.device,
    )
    image_cache = ImageCache(args.image_cache)
    prompt_cfg = build_prompt_config(config, args.prompt_label)

    dataset = ProcessedDataset(args.manifest)
    pipeline = PromptedSegmentor(detector=detector, segmentor=segmentor, image_cache=image_cache)

    results = []
    for idx, sample in enumerate(dataset):
        if args.max_samples is not None and idx >= args.max_samples:
            break
        mask_path = pipeline.segment_sample(sample, prompt_cfg, args.output_dir, overwrite=args.overwrite)
        if mask_path is not None:
            results.append({"sample_id": sample.sample_id, "mask": str(mask_path)})

    summary = {
        "manifest": str(args.manifest),
        "prompt_label": args.prompt_label,
        "num_predictions": len(results),
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

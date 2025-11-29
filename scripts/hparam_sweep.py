#!/usr/bin/env python
"""Hyper-parameter sweep for prompt-based segmentation."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable, List

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
from src.utils.evaluator import EvalConfig, evaluate


def parse_float_list(values: Iterable[str] | None) -> List[float]:
    if not values:
        return []
    return [float(v) for v in values]


def parse_int_list(values: Iterable[str] | None) -> List[int]:
    if not values:
        return []
    return [int(v) for v in values]


def load_prompt_text(config: dict, label: str) -> str:
    prompts = {entry["label"]: entry for entry in config.get("prompts", [])}
    if label not in prompts:
        raise KeyError(f"Prompt label {label} not found in config")
    return prompts[label]["text"]


def build_prompt_config(
    base_config: dict,
    label: str,
    box_threshold: float,
    text_threshold: float,
    mask_threshold: float,
    top_k: int | None,
    min_mask_area: int,
    multimask_output: bool,
) -> PromptConfig:
    return PromptConfig(
        text=load_prompt_text(base_config, label),
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        mask_threshold=mask_threshold,
        top_k=top_k,
        min_mask_area=min_mask_area,
        multimask_output=multimask_output,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--prompt-label", required=True)
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--mask-root", type=Path, default=Path("outputs/sweeps"))
    parser.add_argument("--image-cache", type=Path, default=Path("data/processed_images"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--box-thresholds", nargs="+", default=[0.25, 0.35, 0.45])
    parser.add_argument("--text-thresholds", nargs="+", default=[0.15, 0.25, 0.35])
    parser.add_argument("--mask-thresholds", nargs="+", default=[0.3, 0.45, 0.6])
    parser.add_argument("--top-ks", nargs="+", default=[1, 3])
    parser.add_argument("--min-mask-areas", nargs="+", default=[500, 2500])
    parser.add_argument("--multimask", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/hparam_sweep.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())

    box_thresholds = parse_float_list(args.box_thresholds)
    text_thresholds = parse_float_list(args.text_thresholds)
    mask_thresholds = parse_float_list(args.mask_thresholds)
    top_ks = parse_int_list(args.top_ks)
    min_mask_areas = parse_int_list(args.min_mask_areas)

    dataset = ProcessedDataset(args.manifest)
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
    pipeline = PromptedSegmentor(detector, segmentor, image_cache)

    combos = list(
        itertools.product(box_thresholds, text_thresholds, mask_thresholds, top_ks, min_mask_areas)
    )

    results = []
    prompt_text = load_prompt_text(config, args.prompt_label)
    args.mask_root.mkdir(parents=True, exist_ok=True)

    for box_th, text_th, mask_th, top_k, min_area in combos:
        combo_slug = f"bt{box_th:.2f}_tt{text_th:.2f}_mt{mask_th:.2f}_top{top_k}_mma{min_area}"
        output_dir = args.mask_root / args.prompt_label / combo_slug
        output_dir.mkdir(parents=True, exist_ok=True)

        prompt_cfg = build_prompt_config(
            config,
            args.prompt_label,
            box_threshold=box_th,
            text_threshold=text_th,
            mask_threshold=mask_th,
            top_k=top_k,
            min_mask_area=min_area,
            multimask_output=args.multimask,
        )

        for idx, sample in enumerate(dataset):
            if args.max_samples is not None and idx >= args.max_samples:
                break
            pipeline.segment_sample(sample, prompt_cfg, output_dir, overwrite=True)

        eval_summary = evaluate(
            EvalConfig(
                manifest=args.manifest,
                mask_dir=output_dir,
                prompt_label=args.prompt_label,
                prompt_text=prompt_cfg.text,
                target_label=args.target_label,
                threshold=128,
                image_cache=args.image_cache,
                max_samples=args.max_samples,
            )
        )
        eval_summary.update(
            {
                "box_threshold": box_th,
                "text_threshold": text_th,
                "mask_threshold": mask_th,
                "top_k": top_k,
                "min_mask_area": min_area,
                "multimask_output": args.multimask,
                "mask_dir": str(output_dir),
            }
        )
        results.append(eval_summary)
        print(
            json.dumps(
                {
                    "combo": combo_slug,
                    "mean_iou": eval_summary["mean_iou"],
                    "mean_dice": eval_summary["mean_dice"],
                }
            )
        )

    results.sort(key=lambda r: r["mean_iou"], reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Saved sweep results to {args.output}")


if __name__ == "__main__":
    main()

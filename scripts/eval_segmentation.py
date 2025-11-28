#!/usr/bin/env python
"""Compute IoU/Dice between predicted masks and manifest-derived ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.image_cache import ImageCache  # noqa: E402
from src.data.processed_dataset import ProcessedDataset, SampleRecord  # noqa: E402
from src.utils.mask_metrics import compute_dice, compute_iou  # noqa: E402


@dataclass
class EvalConfig:
    manifest: Path
    mask_dir: Path
    prompt_label: str
    prompt_text: str
    target_label: str
    threshold: int = 128
    image_cache: Optional[Path] = None
    max_samples: Optional[int] = None


def sanitize_prompt_text(prompt_text: str) -> str:
    safe = prompt_text.lower().replace(" ", "_")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in {"_", "-"})
    return safe


def load_mask(mask_path: Path, threshold: int) -> np.ndarray:
    arr = np.array(Image.open(mask_path).convert("L"))
    return (arr >= threshold).astype(np.uint8)


def build_gt_mask(sample: SampleRecord, width: int, height: int, target_label: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for obj in sample.objects:
        if obj.label != target_label:
            continue
        try:
            x1, y1, x2, y2 = obj.bbox_norm_xyxy
        except ValueError:
            continue
        x1_abs = max(0, min(width, int(round(x1 * width))))
        y1_abs = max(0, min(height, int(round(y1 * height))))
        x2_abs = max(0, min(width, int(round(x2 * width))))
        y2_abs = max(0, min(height, int(round(y2 * height))))
        if x2_abs <= x1_abs or y2_abs <= y1_abs:
            continue
        mask[y1_abs:y2_abs, x1_abs:x2_abs] = 1
    return mask


def evaluate(config: EvalConfig) -> Dict[str, object]:
    dataset = ProcessedDataset(config.manifest)
    cache = ImageCache(config.image_cache or ROOT / "data" / "processed_images")
    prompt_suffix = sanitize_prompt_text(config.prompt_text)

    per_sample: List[Dict[str, object]] = []
    missing = 0
    denom = 0
    iou_sum = 0.0
    dice_sum = 0.0

    for idx, sample in enumerate(dataset):
        if config.max_samples is not None and idx >= config.max_samples:
            break
        mask_name = f"{sample.sample_id}__{prompt_suffix}.png"
        mask_path = config.mask_dir / mask_name
        if not mask_path.exists():
            missing += 1
            continue

        image_path = cache.fetch(sample.image_url)
        image = Image.open(image_path)
        width, height = image.size
        gt_mask = build_gt_mask(sample, width, height, config.target_label)
        pred_mask = load_mask(mask_path, config.threshold)
        if gt_mask.sum() == 0 and pred_mask.sum() == 0:
            iou = dice = 1.0
        else:
            iou = compute_iou(pred_mask, gt_mask)
            dice = compute_dice(pred_mask, gt_mask)

        per_sample.append(
            {
                "sample_id": sample.sample_id,
                "iou": iou,
                "dice": dice,
                "gt_pixels": int(gt_mask.sum()),
                "pred_pixels": int(pred_mask.sum()),
                "mask": str(mask_path),
            }
        )
        denom += 1
        iou_sum += iou
        dice_sum += dice

    summary = {
        "dataset": dataset.dataset_name,
        "split": dataset.split,
        "prompt_label": config.prompt_label,
        "target_label": config.target_label,
        "num_evaluated": denom,
        "missing_predictions": missing,
        "mean_iou": float(iou_sum / denom) if denom else 0.0,
        "mean_dice": float(dice_sum / denom) if denom else 0.0,
        "details": per_sample,
    }
    return summary


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

"""Evaluation helpers for comparing predicted masks to processed dataset annotations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from src.data.image_cache import ImageCache
from src.data.processed_dataset import ProcessedDataset, SampleRecord
from src.utils.annotations import build_gt_mask
from src.utils.mask_metrics import compute_dice, compute_iou


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "data" / "processed_images"


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


def _resolve_cache_dir(config: EvalConfig) -> Path:
    if config.image_cache:
        return Path(config.image_cache)
    parents = list(config.manifest.parents)
    if len(parents) >= 3:
        data_root = parents[2]
    elif parents:
        data_root = parents[-1]
    else:
        data_root = ROOT / "data"
    candidate = data_root / "processed_images"
    if candidate.exists():
        return candidate
    return DEFAULT_CACHE


def evaluate(config: EvalConfig) -> Dict[str, object]:
    dataset = ProcessedDataset(config.manifest)
    cache = ImageCache(_resolve_cache_dir(config))
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
        width, height = Image.open(image_path).size
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

"""Annotation helpers shared across evaluation and training."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from src.data.processed_dataset import SampleRecord


def build_gt_mask(
    sample: SampleRecord,
    width: int,
    height: int,
    target_label: str,
    labels_of_interest: Iterable[str] | None = None,
) -> np.ndarray:
    """Rasterize normalized xyxy boxes into a binary mask."""

    if labels_of_interest is None:
        labels = {target_label}
    else:
        labels = set(labels_of_interest)

    mask = np.zeros((height, width), dtype=np.uint8)
    for obj in sample.objects:
        if obj.label not in labels:
            continue
        coords = obj.bbox_norm_xyxy
        if len(coords) != 4:
            continue
        x1, y1, x2, y2 = coords
        x1_abs = max(0, min(width, int(round(x1 * width))))
        y1_abs = max(0, min(height, int(round(y1 * height))))
        x2_abs = max(0, min(width, int(round(x2 * width))))
        y2_abs = max(0, min(height, int(round(y2 * height))))
        if x2_abs <= x1_abs or y2_abs <= y1_abs:
            continue
        mask[y1_abs:y2_abs, x1_abs:x2_abs] = 1
    return mask


def build_multiclass_mask(
    sample: SampleRecord,
    width: int,
    height: int,
    label_to_index: Mapping[str, int],
) -> np.ndarray:
    """Rasterize boxes into an integer mask with one channel per label."""

    mask = np.zeros((height, width), dtype=np.uint8)
    for obj in sample.objects:
        class_idx = label_to_index.get(obj.label)
        if class_idx is None:
            continue
        coords = obj.bbox_norm_xyxy
        if len(coords) != 4:
            continue
        x1, y1, x2, y2 = coords
        x1_abs = max(0, min(width, int(round(x1 * width))))
        y1_abs = max(0, min(height, int(round(y1 * height))))
        x2_abs = max(0, min(width, int(round(x2 * width))))
        y2_abs = max(0, min(height, int(round(y2 * height))))
        if x2_abs <= x1_abs or y2_abs <= y1_abs:
            continue
        mask[y1_abs:y2_abs, x1_abs:x2_abs] = class_idx
    return mask

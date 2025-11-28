"""Mask metric helpers (IoU, Dice, etc.)."""

from __future__ import annotations

import numpy as np


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Intersection-over-Union for binary masks (values 0/1)."""

    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Dice coefficient for binary masks (values 0/1)."""

    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2 * intersection / denom)

"""Segment Anything (SAM / HQ-SAM) helper utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("segment-anything is required for SamSegmentor") from exc


@dataclass
class SamMask:
    mask: np.ndarray
    score: float
    box_xyxy: np.ndarray

    def to_image(self) -> Image.Image:
        return Image.fromarray((self.mask.astype(np.uint8) * 255))


class SamSegmentor:
    """Runs SAM on a set of bounding boxes to obtain binary masks."""

    def __init__(
        self,
        checkpoint_path: Path | str,
        model_type: str = "vit_h",
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        registry_key = model_type
        if registry_key not in sam_model_registry:
            raise ValueError(f"Unknown SAM model_type: {model_type}")
        self.sam = sam_model_registry[registry_key](checkpoint=str(checkpoint_path))
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)

    def _prepare_boxes(self, boxes_xyxy_abs: Sequence[Sequence[float]]) -> np.ndarray:
        if not boxes_xyxy_abs:
            return np.empty((0, 4), dtype=np.float32)
        boxes = np.asarray(boxes_xyxy_abs, dtype=np.float32)
        return boxes

    def predict_masks(
        self,
        image_rgb: np.ndarray,
        boxes_xyxy_abs: Sequence[Sequence[float]],
        multimask_output: bool = False,
    ) -> List[SamMask]:
        boxes = self._prepare_boxes(boxes_xyxy_abs)
        if boxes.size == 0:
            return []
        self.predictor.set_image(image_rgb)
        masks, scores, _ = self.predictor.predict(box=boxes, multimask_output=multimask_output)
        results: List[SamMask] = []
        for idx, mask in enumerate(masks):
            score = float(scores[idx]) if scores is not None else 0.0
            results.append(SamMask(mask=mask, score=score, box_xyxy=boxes[idx]))
        return results

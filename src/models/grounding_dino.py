"""Thin wrapper around GroundingDINO for prompt-conditioned box predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch

try:
    from groundingdino.util.inference import Model as GroundingDINOInference
except ImportError as exc:  # pragma: no cover - dependency issues are surfaced to caller
    raise RuntimeError("groundingdino-py is required for GroundingDINOModel") from exc


@dataclass
class Detection:
    box_xyxy_rel: np.ndarray
    box_xyxy_abs: np.ndarray
    score: float
    phrase: str


class GroundingDINOModel:
    """Loads a GroundingDINO checkpoint and runs prompt-conditioned detection."""

    def __init__(
        self,
        config_path: Path | str,
        checkpoint_path: Path | str,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GroundingDINOInference(
            model_config_path=str(config_path),
            model_checkpoint_path=str(checkpoint_path),
            device=self.device,
        )

    @staticmethod
    def _to_relative(box: np.ndarray, width: int, height: int) -> np.ndarray:
        rel = box.astype(np.float32).copy()
        if rel.max() > 1.0:  # convert absolute to relative
            rel[0::2] /= max(width, 1)
            rel[1::2] /= max(height, 1)
        return np.clip(rel, 0.0, 1.0)

    @staticmethod
    def _to_absolute(box_rel: np.ndarray, width: int, height: int) -> np.ndarray:
        abs_box = box_rel.astype(np.float32).copy()
        abs_box[0::2] *= width
        abs_box[1::2] *= height
        return abs_box

    def predict(
        self,
        image_rgb: np.ndarray,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
        top_k: int | None = None,
    ) -> List[Detection]:
        boxes, logits, phrases = self.model.predict_with_caption(
            image=image_rgb,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        h, w = image_rgb.shape[:2]
        detections: List[Detection] = []
        for idx, box in enumerate(boxes):
            box = np.array(box, dtype=np.float32)
            rel = self._to_relative(box, w, h)
            abs_box = self._to_absolute(rel, w, h)
            detections.append(
                Detection(
                    box_xyxy_rel=rel,
                    box_xyxy_abs=abs_box,
                    score=float(logits[idx]),
                    phrase=phrases[idx],
                )
            )

        detections.sort(key=lambda det: det.score, reverse=True)
        if top_k is not None:
            detections = detections[:top_k]
        return detections

    def predict_multi_prompt(
        self,
        image_rgb: np.ndarray,
        prompts: Sequence[str],
        box_threshold: float,
        text_threshold: float,
        top_k_per_prompt: int | None = None,
    ) -> List[Detection]:
        combined: List[Detection] = []
        for prompt in prompts:
            combined.extend(
                self.predict(
                    image_rgb=image_rgb,
                    prompt=prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    top_k=top_k_per_prompt,
                )
            )
        combined.sort(key=lambda det: det.score, reverse=True)
        return combined

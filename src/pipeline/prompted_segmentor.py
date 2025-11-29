"""Prompted segmentation pipeline combining GroundingDINO + SAM."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image

from src.data.image_cache import ImageCache
from src.data.processed_dataset import SampleRecord
from src.models.grounding_dino import GroundingDINOModel
from src.models.sam_wrapper import SamSegmentor


@dataclass
class PromptConfig:
    text: str
    box_threshold: float
    text_threshold: float
    mask_threshold: float = 0.5
    top_k: int | None = None
    min_mask_area: int = 0
    multimask_output: bool = False


class PromptedSegmentor:
    """Runs prompt-conditioned segmentation for drywall QA."""

    def __init__(
        self,
        detector: GroundingDINOModel,
        segmentor: SamSegmentor,
        image_cache: ImageCache,
    ) -> None:
        self.detector = detector
        self.segmentor = segmentor
        self.image_cache = image_cache

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        with Image.open(path) as img:
            return np.array(img.convert("RGB"))

    @staticmethod
    def _sanitize_filename(text: str) -> str:
        safe = text.lower().replace(" ", "_")
        safe = "".join(ch for ch in safe if ch.isalnum() or ch in {"_", "-"})
        return safe

    def segment_sample(
        self,
        sample: SampleRecord,
        prompt_cfg: PromptConfig,
        output_dir: Path | str,
        overwrite: bool = False,
    ) -> Optional[Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_name = f"{sample.sample_id}__{self._sanitize_filename(prompt_cfg.text)}.png"
        mask_path = output_dir / mask_name
        if mask_path.exists() and not overwrite:
            return mask_path

        image_path = self.image_cache.fetch(sample.image_url)
        image_rgb = self._load_image(image_path)

        detections = self.detector.predict(
            image_rgb=image_rgb,
            prompt=prompt_cfg.text,
            box_threshold=prompt_cfg.box_threshold,
            text_threshold=prompt_cfg.text_threshold,
            top_k=prompt_cfg.top_k,
        )
        if not detections:
            return None

        boxes_abs = [det.box_xyxy_abs for det in detections]
        sam_masks = self.segmentor.predict_masks(
            image_rgb=image_rgb,
            boxes_xyxy_abs=boxes_abs,
            multimask_output=prompt_cfg.multimask_output,
        )
        if not sam_masks:
            return None

        combined = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        kept = 0
        for sam_mask in sam_masks:
            binary = (sam_mask.mask > prompt_cfg.mask_threshold).astype(np.uint8)
            if binary.sum() < prompt_cfg.min_mask_area:
                continue
            combined = np.maximum(combined, binary)
            kept += 1

        if kept == 0:
            return None

        binary_mask = combined * 255
        Image.fromarray(binary_mask).save(mask_path)
        return mask_path

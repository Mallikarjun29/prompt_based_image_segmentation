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
        res = self.model.predict_with_caption(
            image=image_rgb,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # Debugging assistance: if the returned structure is unexpected, output a short
        # summary to stderr to help diagnose shape/typing issues.
        import sys
        if isinstance(res, (tuple, list)) and len(res) <= 5:
            types = [type(x).__name__ for x in res]
            print(f"[groundingdino] predict_with_caption returned {len(res)} elements: {types}", file=sys.stderr)
            for i, x in enumerate(res):
                try:
                    length = getattr(x, 'shape', None) or (len(x) if hasattr(x, '__len__') else None)
                except Exception:
                    length = None
                print(f"  element {i}: type={type(x).__name__} info={length}", file=sys.stderr)

        # GroundingDINO's inference API may return either (boxes, logits, phrases)
        # or (boxes, logits) depending on the version. Handle both safely.
        if isinstance(res, (tuple, list)):
            if len(res) == 3:
                boxes, logits, phrases = res
            elif len(res) == 2:
                # Some GroundingDINO versions return a Detections-like object and a list
                # of phrases. Try to extract box coordinates and scores from that object.
                boxes_obj, phrases = res
                # Try common attributes used by detection outputs
                if hasattr(boxes_obj, "boxes"):
                    raw_boxes = boxes_obj.boxes
                elif hasattr(boxes_obj, "pred_boxes"):
                    raw_boxes = boxes_obj.pred_boxes
                elif hasattr(boxes_obj, "xyxy"):
                    raw_boxes = boxes_obj.xyxy
                else:
                    raw_boxes = boxes_obj

                # Convert raw_boxes to a list/array of xyxy floats
                try:
                    if hasattr(raw_boxes, "cpu") and hasattr(raw_boxes, "numpy"):
                        raw_boxes_arr = raw_boxes.cpu().numpy()
                    else:
                        raw_boxes_arr = np.asarray(raw_boxes)
                except Exception:
                    raw_boxes_arr = np.asarray(list(raw_boxes))

                boxes = [rb for rb in raw_boxes_arr]

                # scores/logits
                if hasattr(boxes_obj, "scores"):
                    scores = boxes_obj.scores
                    try:
                        if hasattr(scores, "cpu") and hasattr(scores, "numpy"):
                            logits = [float(s) for s in scores.cpu().numpy()]
                        else:
                            logits = [float(s) for s in scores]
                    except Exception:
                        logits = [1.0 for _ in boxes]
                else:
                    logits = [1.0 for _ in boxes]
            else:
                raise ValueError(
                    f"Unexpected return shape from predict_with_caption: {len(res)} elements"
                )
        else:
            raise ValueError("predict_with_caption returned unsupported type")
        h, w = image_rgb.shape[:2]
        detections: List[Detection] = []
        def _box_to_xyxy(box_raw) -> np.ndarray:
            arr = np.asarray(box_raw, dtype=np.float32).ravel()
            if arr.size < 4:
                raise ValueError(f"Box has fewer than 4 values: {arr}")
            return arr[:4]

        for idx, box in enumerate(boxes):
            box = _box_to_xyxy(box)
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

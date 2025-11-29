"""Utilities for converting COCO-format annotations into processed manifests."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


@dataclass(frozen=True)
class CocoConversionSummary:
    dataset: str
    split: str
    num_samples: int
    label_frequencies: Dict[str, int]
    output: str


def _normalize_bbox(bbox: Iterable[float], width: int, height: int) -> List[float]:
    x, y, w, h = bbox
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    x1 = max(0.0, min(1.0, x / width))
    y1 = max(0.0, min(1.0, y / height))
    x2 = max(0.0, min(1.0, (x + w) / width))
    y2 = max(0.0, min(1.0, (y + h) / height))
    if x2 <= x1 or y2 <= y1:
        return []
    return [round(coord, 6) for coord in (x1, y1, x2, y2)]


def _canonical_label(source_label: str, label_map: Mapping[str, str]) -> str | None:
    if source_label in label_map:
        return label_map[source_label]
    lowered = source_label.lower()
    for key, value in label_map.items():
        if key.lower() == lowered:
            return value
    return None


def convert_coco_annotations(
    annotations_path: Path,
    image_dir: Path,
    output_path: Path,
    dataset_name: str,
    split: str,
    label_map: Mapping[str, str],
    prompt_template: str = "",
    skip_crowd: bool = True,
) -> CocoConversionSummary:
    """Convert a single COCO annotation file into the processed manifest format."""

    annotations_path = Path(annotations_path)
    image_dir = Path(image_dir)
    data = json.loads(annotations_path.read_text())

    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    images = {image["id"]: image for image in data.get("images", [])}
    image_annotations: Dict[int, List[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        if skip_crowd and ann.get("iscrowd"):
            continue
        image_annotations[ann["image_id"]].append(ann)

    samples: List[Dict[str, object]] = []
    label_counter: Counter[str] = Counter()
    prompt_value = prompt_template or ""

    for sample_idx, (image_id, anns) in enumerate(image_annotations.items()):
        image_meta = images.get(image_id)
        if not image_meta:
            continue
        file_name = image_meta.get("file_name")
        if not file_name:
            continue
        image_path = (image_dir / file_name).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file: {image_path}")
        objects: List[Dict[str, object]] = []
        width = int(image_meta.get("width", 0))
        height = int(image_meta.get("height", 0))
        if width <= 0 or height <= 0:
            continue
        for ann in anns:
            category_name = categories.get(ann.get("category_id"))
            if not category_name:
                continue
            label = _canonical_label(category_name, label_map)
            if label is None:
                continue
            bbox = _normalize_bbox(ann.get("bbox", []), width, height)
            if not bbox:
                continue
            objects.append(
                {
                    "label": label,
                    "source_label": category_name,
                    "bbox_norm_xyxy": bbox,
                    "bbox_space": "relative_0_1",
                }
            )
            label_counter[label] += 1
        if not objects:
            continue
        sample_id = f"{dataset_name}_{split}_{sample_idx:05d}"
        samples.append(
            {
                "id": sample_id,
                "image_url": image_path.as_uri(),
                "prompt": prompt_value,
                "objects": objects,
            }
        )

    manifest = {
        "dataset": dataset_name,
        "split": split,
        "num_samples": len(samples),
        "label_frequencies": dict(label_counter),
        "reference_resolution": 1024,
        "samples": samples,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2))

    return CocoConversionSummary(
        dataset=dataset_name,
        split=split,
        num_samples=len(samples),
        label_frequencies=dict(label_counter),
        output=str(output_path),
    )

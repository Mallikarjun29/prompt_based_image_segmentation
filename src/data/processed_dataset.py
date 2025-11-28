"""Dataset helpers for processed JSON manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence


@dataclass(frozen=True)
class ObjectAnnotation:
    label: str
    source_label: str
    bbox_norm_xyxy: Sequence[float]
    bbox_space: str = "relative_0_1"


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    image_url: str
    prompt: str
    objects: Sequence[ObjectAnnotation]

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.sample_id,
            "image_url": self.image_url,
            "prompt": self.prompt,
            "objects": [
                {
                    "label": obj.label,
                    "source_label": obj.source_label,
                    "bbox_norm_xyxy": list(obj.bbox_norm_xyxy),
                    "bbox_space": obj.bbox_space,
                }
                for obj in self.objects
            ],
        }


class ProcessedDataset(Sequence[SampleRecord]):
    """In-memory representation of a processed dataset manifest."""

    def __init__(self, manifest_path: Path | str):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)
        with self.manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)

        self.dataset_name: str = manifest.get("dataset", "unknown")
        self.split: str = manifest.get("split", "unknown")
        self.reference_resolution: int = manifest.get("reference_resolution", 1024)
        self._records: List[SampleRecord] = [
            SampleRecord(
                sample_id=sample["id"],
                image_url=sample["image_url"],
                prompt=sample.get("prompt", ""),
                objects=[
                    ObjectAnnotation(
                        label=obj["label"],
                        source_label=obj.get("source_label", obj["label"]),
                        bbox_norm_xyxy=tuple(obj["bbox_norm_xyxy"]),
                        bbox_space=obj.get("bbox_space", "relative_0_1"),
                    )
                    for obj in sample.get("objects", [])
                ],
            )
            for sample in manifest.get("samples", [])
        ]
        self.label_frequencies: Dict[str, int] = manifest.get(
            "label_frequencies", self._compute_label_frequencies()
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> SampleRecord:
        return self._records[idx]

    def __iter__(self) -> Iterator[SampleRecord]:
        return iter(self._records)

    def _compute_label_frequencies(self) -> Dict[str, int]:
        freqs: Dict[str, int] = {}
        for record in self._records:
            for obj in record.objects:
                freqs[obj.label] = freqs.get(obj.label, 0) + 1
        return freqs

    def filter_by_label(self, label: str) -> Iterable[SampleRecord]:
        for record in self._records:
            if any(obj.label == label for obj in record.objects):
                yield record

    def to_json(self, output_path: Path | str) -> None:
        payload = {
            "dataset": self.dataset_name,
            "split": self.split,
            "reference_resolution": self.reference_resolution,
            "label_frequencies": self.label_frequencies,
            "samples": [record.to_dict() for record in self._records],
        }
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

"""Utilities for converting OpenAI JSONL annotation exports into internal datasets."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

LOC_PATTERN = re.compile(r"<loc(\d{4})>")
MAX_LOC = 1023.0


@dataclass
class ParsedSample:
    """Represents a single annotation sample after conversion."""

    sample_id: str
    image_url: str
    prompt: str
    objects: List[Dict[str, object]]


def _slugify(label: str) -> str:
    """Normalize label names to snake_case with alnum + underscore only."""

    cleaned = label.strip().lower()
    repl_chars = "- /"
    for ch in repl_chars:
        cleaned = cleaned.replace(ch, "_")
    cleaned = re.sub(r"[^a-z0-9_]+", "", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_") or "unknown"


def _decode_loc_tokens(segment: str) -> Tuple[List[float], str]:
    """Return normalized bbox coordinates (xyxy) and remaining label text."""

    tokens = LOC_PATTERN.findall(segment)
    if len(tokens) < 4:
        raise ValueError(f"Chunk does not contain 4 location tokens: {segment}")

    coords = [min(int(token), 1023) / MAX_LOC for token in tokens[:4]]
    label_text = LOC_PATTERN.sub("", segment).strip()
    return coords, label_text


def _extract_messages(sample: Dict[str, object]) -> Tuple[str, str, str]:
    """Extract prompt, image URL, and assistant annotation text from a message list."""

    prompt: str | None = None
    image_url: str | None = None
    assistant_text: str | None = None

    for message in sample.get("messages", []):
        role = message.get("role")
        content = message.get("content")

        if role == "user":
            if isinstance(content, str):
                prompt = content.strip()
                continue

            if isinstance(content, list):
                for block in content:
                    block_type = block.get("type")
                    if block_type == "input_text":
                        prompt = block.get("text", "").strip()
                    elif block_type == "image_url":
                        image_url = block.get("image_url", {}).get("url")
        elif role == "assistant":
            if isinstance(content, str):
                assistant_text = content
            elif isinstance(content, list):
                assistant_text = " ".join(
                    block.get("text", "") for block in content if isinstance(block, dict)
                )

    if not image_url:
        raise ValueError("Missing image_url in sample")
    if assistant_text is None:
        raise ValueError("Missing assistant annotation text in sample")

    return prompt or "", image_url, assistant_text


def _parse_objects(assistant_text: str, label_map: Dict[str, str]) -> List[Dict[str, object]]:
    """Parse `<loc>` chunks out of assistant text and map to canonical labels."""

    objects: List[Dict[str, object]] = []
    for raw_chunk in assistant_text.split(";"):
        chunk = raw_chunk.strip()
        if not chunk:
            continue

        coords, label_text = _decode_loc_tokens(chunk)
        canonical_label = label_map.get(label_text, _slugify(label_text))
        objects.append(
            {
                "label": canonical_label,
                "source_label": label_text,
                "bbox_norm_xyxy": [round(v, 6) for v in coords],
                "bbox_space": "relative_0_1",
            }
        )

    return objects


def parse_jsonl_sample(
    line: str,
    dataset_name: str,
    split: str,
    index: int,
    label_map: Dict[str, str],
) -> ParsedSample:
    data = json.loads(line)
    prompt, image_url, assistant_text = _extract_messages(data)
    objects = _parse_objects(assistant_text, label_map)
    sample_id = f"{dataset_name}_{split}_{index:05d}"
    return ParsedSample(sample_id=sample_id, image_url=image_url, prompt=prompt, objects=objects)


def convert_jsonl_file(
    input_path: Path,
    output_path: Path,
    dataset_name: str,
    split: str,
    label_map: Dict[str, str],
) -> Dict[str, object]:
    """Convert an OpenAI JSONL file into the internal JSON format."""

    samples: List[Dict[str, object]] = []
    label_counter: Counter[str] = Counter()

    with input_path.open("r", encoding="utf-8") as source:
        for idx, line in enumerate(source):
            line = line.strip()
            if not line:
                continue
            parsed = parse_jsonl_sample(line, dataset_name, split, idx, label_map)
            if not parsed.objects:
                continue
            samples.append(
                {
                    "id": parsed.sample_id,
                    "image_url": parsed.image_url,
                    "prompt": parsed.prompt,
                    "objects": parsed.objects,
                }
            )
            for obj in parsed.objects:
                label_counter[obj["label"]] += 1

    manifest = {
        "dataset": dataset_name,
        "split": split,
        "num_samples": len(samples),
        "label_frequencies": dict(label_counter),
        "reference_resolution": 1024,
        "samples": samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as sink:
        json.dump(manifest, sink, indent=2)

    return {
        "dataset": dataset_name,
        "split": split,
        "num_samples": len(samples),
        "label_frequencies": dict(label_counter),
        "output": str(output_path),
    }

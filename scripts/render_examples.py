#!/usr/bin/env python
"""Render triptych examples (original | GT | prediction) for selected samples."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.image_cache import ImageCache
from src.data.processed_dataset import ProcessedDataset, SampleRecord
from src.utils.annotations import build_gt_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Processed manifest JSON")
    parser.add_argument("--mask-dir", type=Path, required=True, help="Directory containing predicted PNGs")
    parser.add_argument("--mask-suffix", required=True, help="Mask suffix used during inference (e.g., crack_latest)")
    parser.add_argument("--target-label", required=True, help="Canonical label to visualize")
    parser.add_argument("--sample-id", action="append", required=True, help="Sample ID(s) to render")
    parser.add_argument("--image-cache", type=Path, default=Path("data/processed_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/examples"))
    parser.add_argument("--pred-threshold", type=int, default=128, help="Binarization threshold for predicted masks")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha for masks")
    return parser.parse_args()


def load_sample(dataset: ProcessedDataset, sample_id: str) -> SampleRecord:
    for record in dataset:
        if record.sample_id == sample_id:
            return record
    raise KeyError(f"Sample {sample_id} not found in {dataset.manifest_path}")


def overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, color + (0,))
    mask_img = Image.fromarray((mask * int(alpha * 255)).astype(np.uint8), mode="L")
    overlay.putalpha(mask_img)
    combined = Image.alpha_composite(base, overlay)
    return combined.convert("RGB")


def add_label(image: Image.Image, text: str, bg_color: tuple[int, int, int]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    padding = 6
    text_size = draw.textbbox((0, 0), text)
    width = text_size[2] - text_size[0]
    height = text_size[3] - text_size[1]
    rect = [
        (padding, padding),
        (padding + width + 8, padding + height + 8),
    ]
    draw.rectangle(rect, fill=bg_color)
    draw.text((padding + 4, padding + 4), text, fill=(255, 255, 255))
    return image


def prepare_panels(
    original: Image.Image,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    alpha: float,
) -> List[Tuple[Image.Image, str]]:
    gt_overlay = overlay_mask(original, gt_mask, color=(0, 200, 0), alpha=alpha)
    pred_overlay = overlay_mask(original, pred_mask, color=(200, 0, 0), alpha=alpha)
    panels = [
        (original.copy(), "Original"),
        (gt_overlay, "Ground Truth"),
        (pred_overlay, "Prediction"),
    ]
    labeled: List[Tuple[Image.Image, str]] = []
    for img, label in panels:
        labeled_img = add_label(img, label, bg_color=(0, 0, 0))
        labeled.append((labeled_img, label))
    return labeled


def stitch_panels(panels: List[Tuple[Image.Image, str]]) -> Image.Image:
    images = [img for img, _ in panels]
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    canvas = Image.new("RGB", (sum(widths), max(heights)), color=(20, 20, 20))
    offset = 0
    for img in images:
        canvas.paste(img, (offset, 0))
        offset += img.width
    return canvas


def load_pred_mask(path: Path, threshold: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.array(Image.open(path).convert("L"))
    return (arr >= threshold).astype(np.uint8)


def main() -> None:
    args = parse_args()
    dataset = ProcessedDataset(args.manifest)
    cache = ImageCache(args.image_cache)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"{args.mask_suffix}_{args.target_label}"

    summaries: Dict[str, str] = {}
    for sample_id in args.sample_id:
        sample = load_sample(dataset, sample_id)
        image_path = cache.fetch(sample.image_url)
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            width, height = rgb.size
            gt_mask = build_gt_mask(sample, width, height, args.target_label)
            pred_path = args.mask_dir / f"{sample_id}__{suffix}.png"
            pred_mask = load_pred_mask(pred_path, args.pred_threshold)
            panels = prepare_panels(rgb, gt_mask, pred_mask, alpha=args.alpha)
            triptych = stitch_panels(panels)
            triptych_path = args.output_dir / f"{sample_id}_triptych.png"
            triptych.save(triptych_path)

            panel_paths: Dict[str, str] = {"triptych": str(triptych_path)}
            export_names = ["original", "gt", "prediction"]
            for (panel_img, _), name in zip(panels, export_names):
                panel_path = args.output_dir / f"{sample_id}_{name}.png"
                panel_img.save(panel_path)
                panel_paths[name] = str(panel_path)

            summaries[sample_id] = panel_paths
            print(f"Saved {triptych_path}")

    print("Rendered samples:")
    for sample_id, paths in summaries.items():
        print(f"- {sample_id}:")
        for label, path in paths.items():
            print(f"    {label}: {path}")


if __name__ == "__main__":
    main()

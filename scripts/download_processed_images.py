#!/usr/bin/env python
"""Download all images referenced in processed dataset manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.image_cache import ImageCache  # noqa: E402
from src.data.processed_dataset import ProcessedDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="+",
        help="Path(s) to processed JSON manifest(s).",
    )
    parser.add_argument(
        "--cache-dir",
        default=ROOT / "data" / "processed_images",
        type=Path,
        help="Directory where downloaded images will be cached.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a cached file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = ImageCache(args.cache_dir)
    for manifest_path in args.manifest:
        dataset = ProcessedDataset(manifest_path)
        cache.ensure_samples(dataset, force=args.force)
        print(f"Cached {len(dataset)} samples from {manifest_path}")


if __name__ == "__main__":
    main()

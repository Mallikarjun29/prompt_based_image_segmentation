"""Helpers for caching remote images referenced in processed datasets."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import requests

LOGGER = logging.getLogger(__name__)


class ImageCache:
    """Download-once cache for remote dataset imagery."""

    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_url(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.jpg"

    def fetch(self, url: str, *, force: bool = False, timeout: float = 10.0) -> Path:
        target_path = self._path_for_url(url)
        if target_path.exists() and not force:
            return target_path

        LOGGER.info("Downloading %s", url)
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        target_path.write_bytes(response.content)
        return target_path

    def ensure_samples(self, samples, *, force: bool = False) -> None:
        for sample in samples:
            try:
                self.fetch(sample.image_url, force=force)
            except requests.RequestException as exc:
                LOGGER.warning("Failed to download %s: %s", sample.image_url, exc)

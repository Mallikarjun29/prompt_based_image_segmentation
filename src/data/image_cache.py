"""Helpers for caching remote images referenced in processed datasets."""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Optional

import requests
from urllib.parse import urlparse, unquote

LOGGER = logging.getLogger(__name__)


class ImageCache:
    """Download-once cache for remote dataset imagery."""

    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_url(self, url: str) -> Path:
        local_path = self._resolve_local_path(url)
        if local_path is not None:
            return local_path
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.jpg"

    def _resolve_local_path(self, url: str) -> Path | None:
        """Return a local filesystem path if the URL already points to disk."""

        parsed = urlparse(url)
        target: Path | None = None
        if parsed.scheme == "file":
            target = Path(unquote(parsed.path))
        elif parsed.scheme == "":
            candidate = Path(url)
            if candidate.exists():
                target = candidate
        if target is None:
            return None
        return target.resolve()

    def cached_path(self, url: str) -> Path:
        """Return the expected cache path without triggering a download."""

        return self._path_for_url(url)

    def has(self, url: str) -> bool:
        """Check if a URL has already been cached locally."""

        return self.cached_path(url).exists()

    def fetch(self, url: str, *, force: bool = False, timeout: float = 10.0) -> Path:
        local_path = self._resolve_local_path(url)
        if local_path is not None:
            if not local_path.exists():
                raise FileNotFoundError(f"Missing local file for {url}")
            if force:
                # Optionally refresh the cached copy for local files.
                target_path = self._path_for_url(url)
                if target_path == local_path:
                    return local_path
                shutil.copy2(local_path, target_path)
                return target_path
            return local_path
        target_path = self.cached_path(url)
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

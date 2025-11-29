"""Dataset that rasterizes bounding boxes into segmentation masks."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.image_cache import ImageCache
from src.data.processed_dataset import ProcessedDataset, SampleRecord
from src.utils.annotations import build_gt_mask, build_multiclass_mask


DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


class BoxMaskDataset(Dataset):
    """Loads processed dataset records and returns image/mask tensors."""

    def __init__(
        self,
        manifest: Path | str,
        target_label: str | None,
        label_map: Mapping[str, int] | None = None,
        image_cache: Path | str | None = None,
        resize: Optional[Tuple[int, int]] = (512, 512),
        image_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        use_cached_only: bool = False,
    ) -> None:
        self.dataset = ProcessedDataset(Path(manifest))
        if label_map is None and target_label is None:
            raise ValueError("Either label_map or target_label must be provided")
        self.target_label = target_label
        self.label_map = label_map
        cache_root = Path(image_cache) if image_cache else Path("data/processed_images")
        self.cache = ImageCache(cache_root)
        self.resize = resize
        self.image_transform = image_transform or transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)]
        )
        self.use_cached_only = use_cached_only

        if use_cached_only:
            self.indices = [
                idx
                for idx, sample in enumerate(self.dataset)
                if self.cache.has(sample.image_url)
            ]
        else:
            self.indices = None

    def __len__(self) -> int:
        if self.indices is not None:
            return len(self.indices)
        return len(self.dataset)

    def _load_image(self, sample: SampleRecord) -> Image.Image:
        if self.use_cached_only:
            path = self.cache.cached_path(sample.image_url)
            if not path.exists():
                raise FileNotFoundError(f"Missing cached file for {sample.sample_id}: {path}")
        else:
            path = self.cache.fetch(sample.image_url)
        return Image.open(path).convert("RGB")

    def _load_mask(self, sample: SampleRecord, size: Tuple[int, int]) -> np.ndarray:
        width, height = size
        if self.label_map is not None:
            mask = build_multiclass_mask(sample, width, height, self.label_map)
        else:
            if self.target_label is None:
                raise ValueError("target_label must be provided when label_map is None")
            mask = build_gt_mask(sample, width, height, self.target_label)
        return mask

    def __getitem__(self, index: int) -> dict:
        sample_idx = self.indices[index] if self.indices is not None else index
        sample = self.dataset[sample_idx]
        image = self._load_image(sample)
        width, height = image.size
        mask = self._load_mask(sample, (width, height))

        if self.resize is not None:
            new_w, new_h = self.resize
            image = image.resize((new_w, new_h), Image.BILINEAR)
            scale_mask = mask if self.label_map is not None else mask * 255
            mask_img = Image.fromarray(scale_mask).resize((new_w, new_h), Image.NEAREST)
            mask = np.array(mask_img, dtype=np.uint8)
            if self.label_map is None:
                mask = (mask > 0).astype(np.uint8)

        image_tensor = self.image_transform(image)
        if self.label_map is None:
            mask_array = (mask > 0).astype(np.int64)
        else:
            mask_array = mask.astype(np.int64)
        mask_tensor = torch.from_numpy(mask_array)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_id": sample.sample_id,
        }

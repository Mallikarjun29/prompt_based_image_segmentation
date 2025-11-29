"""Reusable DeepLabV3 inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)

from src.data.image_cache import ImageCache
from src.data.processed_dataset import ProcessedDataset

NORMALIZE = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


@dataclass
class DeepLabInferenceConfig:
    """Configuration for running DeepLab inference over a manifest."""

    manifest: Path
    checkpoint: Path
    output_dir: Path
    image_cache: Path
    mask_suffix: str
    class_labels: Sequence[str] | None = None
    target_label: str | None = None
    resize: tuple[int, int] | None = (512, 512)
    batch_size: int = 4
    device: str | None = None
    max_samples: int | None = None
    skip_existing: bool = False


@dataclass
class SingleImageInferenceConfig:
    """Configuration for running DeepLab inference on a single image."""

    image_path: Path
    checkpoint: Path
    output_dir: Path
    prompt_suffix: str
    class_labels: Sequence[str] | None = None
    target_label: str | None = None
    resize: tuple[int, int] | None = (512, 512)
    device: str | None = None
    image_id: str | None = None
    include_label_suffix: bool = True


def load_checkpoint_metadata(checkpoint: Path, device: torch.device):
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location=device)
    state = payload.get("model_state", payload)
    labels = payload.get("labels")
    label_map = payload.get("label_map")
    if label_map is None and labels:
        label_map = {label: idx + 1 for idx, label in enumerate(labels)}
    if labels is None and label_map is not None:
        labels = sorted(label_map, key=label_map.get)
    num_classes = (len(label_map) + 1) if label_map else 2
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights)
    model.classifier[-1] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, labels, label_map


def preprocess(image: Image.Image, resize: tuple[int, int] | None) -> torch.Tensor:
    if resize:
        image = image.resize(resize, Image.BILINEAR)
    tensor = transforms.ToTensor()(image)
    tensor = NORMALIZE(tensor)
    return tensor


def save_mask(mask_tensor: torch.Tensor, size: tuple[int, int], path: Path) -> None:
    mask_np = mask_tensor.detach().cpu().numpy().astype("uint8") * 255
    mask_img = Image.fromarray(mask_np)
    mask_img = mask_img.resize(size, Image.NEAREST)
    mask_img.save(path)


def resolve_active_labels(
    requested_labels: Sequence[str] | None,
    checkpoint_labels: Sequence[str] | None,
    fallback_target: str | None,
) -> List[str]:
    if requested_labels:
        return list(requested_labels)
    if checkpoint_labels:
        return list(checkpoint_labels)
    if fallback_target:
        return [fallback_target]
    raise ValueError("Unable to resolve class labels for inference")


def build_label_indices(labels: Iterable[str], label_map: Mapping[str, int] | None) -> Dict[str, int]:
    indices: Dict[str, int] = {}
    if label_map is None:
        for label in labels:
            indices[label] = 1
        return indices
    for label in labels:
        if label not in label_map:
            raise KeyError(f"Label '{label}' not found in checkpoint metadata")
        indices[label] = label_map[label]
    return indices


def run_deeplab_inference(config: DeepLabInferenceConfig) -> Dict[str, object]:
    """Run DeepLab inference using the provided configuration."""

    dataset = ProcessedDataset(config.manifest)
    cache = ImageCache(config.image_cache)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint_labels, label_map = load_checkpoint_metadata(config.checkpoint, device)
    active_labels = resolve_active_labels(config.class_labels, checkpoint_labels, config.target_label)
    label_indices = build_label_indices(active_labels, label_map)

    resize = tuple(config.resize) if config.resize else None
    batch: List[Dict[str, object]] = []
    processed = 0
    skipped_existing = 0

    def _flush_batch(batch_items: List[Dict[str, object]]) -> int:
        if not batch_items:
            return 0
        images = torch.stack([item["tensor"] for item in batch_items]).to(device)
        with torch.no_grad():
            logits = model(images)["out"]
        predictions = torch.argmax(logits, dim=1).cpu()
        saved = 0
        for pred, item in zip(predictions, batch_items):
            saved_any = False
            for label, mask_path in item["mask_paths"].items():
                if config.skip_existing and mask_path.exists():
                    continue
                class_idx = label_indices[label]
                mask_binary = (pred == class_idx).to(torch.uint8)
                save_mask(mask_binary, item["original_size"], mask_path)
                saved_any = True
            if saved_any:
                saved += 1
        return saved

    for idx, sample in enumerate(dataset):
        if config.max_samples is not None and idx >= config.max_samples:
            break

        mask_paths = {
            label: output_dir / f"{sample.sample_id}__{config.mask_suffix}_{label}.png"
            for label in active_labels
        }
        if config.skip_existing and all(path.exists() for path in mask_paths.values()):
            skipped_existing += 1
            continue

        image_path = cache.fetch(sample.image_url)
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            original_size = image.size
            tensor = preprocess(image, resize)

        batch.append(
            {
                "tensor": tensor,
                "mask_paths": mask_paths,
                "original_size": original_size,
            }
        )

        if len(batch) >= max(config.batch_size, 1):
            processed += _flush_batch(batch)
            batch = []

    processed += _flush_batch(batch)

    return {
        "manifest": str(config.manifest),
        "checkpoint": str(config.checkpoint),
        "output_dir": str(output_dir),
        "mask_suffix": config.mask_suffix,
        "labels": active_labels,
        "num_saved": processed,
        "skipped_existing": skipped_existing,
    }


def run_single_image_inference(config: SingleImageInferenceConfig) -> Dict[str, object]:
    """Run DeepLab inference on a single RGB image."""

    if not config.image_path.exists():
        raise FileNotFoundError(config.image_path)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint_labels, label_map = load_checkpoint_metadata(config.checkpoint, device)
    active_labels = resolve_active_labels(config.class_labels, checkpoint_labels, config.target_label)
    label_indices = build_label_indices(active_labels, label_map)

    image_id = config.image_id or config.image_path.stem
    resize = tuple(config.resize) if config.resize else None

    with Image.open(config.image_path) as img:
        image = img.convert("RGB")
        original_size = image.size
        tensor = preprocess(image, resize)

    batched = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(batched)["out"]
    prediction = torch.argmax(logits, dim=1)[0].cpu()

    mask_paths: Dict[str, str] = {}
    suffix_map: Dict[str, str] = {}
    prompt_suffix = config.prompt_suffix
    saved = 0

    for label in active_labels:
        suffix = f"{prompt_suffix}_{label}" if (config.include_label_suffix or len(active_labels) > 1) else prompt_suffix
        mask_path = output_dir / f"{image_id}__{suffix}.png"
        class_idx = label_indices[label]
        mask_binary = (prediction == class_idx).to(torch.uint8)
        save_mask(mask_binary, original_size, mask_path)
        mask_paths[label] = str(mask_path)
        suffix_map[label] = suffix
        saved += 1

    return {
        "image": str(config.image_path),
        "checkpoint": str(config.checkpoint),
        "output_dir": str(output_dir),
        "prompt_suffixes": suffix_map,
        "labels": list(active_labels),
        "mask_paths": mask_paths,
        "num_saved": saved,
    }

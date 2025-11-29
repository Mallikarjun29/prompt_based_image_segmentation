#!/usr/bin/env python
"""Fine-tune a segmentation backbone on drywall joint masks derived from bounding boxes."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable, List, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.box_mask_dataset import BoxMaskDataset
from src.utils.mask_metrics import compute_dice, compute_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_manifest", type=Path)
    parser.add_argument("valid_manifest", type=Path)
    parser.add_argument("--target-label", default="drywall_joint")
    parser.add_argument("--labels", nargs="+", default=None, help="Ordered defect labels to train (excludes background)")
    parser.add_argument("--extra-train-manifest", action="append", type=Path, default=[], help="Additional manifests to include in training")
    parser.add_argument("--extra-valid-manifest", action="append", type=Path, default=[], help="Additional manifests to include in validation")
    parser.add_argument("--balance-train-manifests", action="store_true", help="Cap each train manifest to the same sample count")
    parser.add_argument("--balance-valid-manifests", action="store_true", help="Cap each valid manifest to the same sample count")
    parser.add_argument("--image-cache", type=Path, default=Path("data/processed_images"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/finetuned_deeplabv3.pth"))
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/finetune_metrics.json"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--resize", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(512, 512))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cached-only", action="store_true", help="Use only pre-downloaded images.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_subset(dataset, max_samples: int | None, seed: int = 0):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[:max_samples]
    return Subset(dataset, indices)


def collate_batch(batch):
    images = torch.stack([sample["image"] for sample in batch])
    masks = torch.stack([sample["mask"] for sample in batch])
    sample_ids = [sample["sample_id"] for sample in batch]
    return images, masks, sample_ids


def build_model(num_classes: int) -> nn.Module:
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights)
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def estimate_class_weights(dataset: Iterable, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float64)
    sample_count = 0
    for sample in dataset:
        mask = sample["mask"].numpy()
        values, value_counts = np.unique(mask, return_counts=True)
        for value, count in zip(values, value_counts):
            if 0 <= value < num_classes:
                counts[value] += count
        sample_count += 1
        if sample_count >= 256:
            break
    counts = torch.where(counts == 0, torch.ones_like(counts), counts)
    total = counts.sum()
    weights = total / (num_classes * counts)
    weights = weights / weights.min()
    return weights.to(device=device, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total = 0
    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)["out"]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        total += images.size(0)
    return running_loss / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_indices: Mapping[int, str]):
    model.eval()
    running_loss = 0.0
    total = 0
    iou_sums = {idx: 0.0 for idx in class_indices}
    dice_sums = {idx: 0.0 for idx in class_indices}
    count = 0
    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)["out"]
        loss = criterion(outputs, masks)
        running_loss += loss.item() * images.size(0)
        total += images.size(0)

        preds = torch.argmax(outputs, dim=1)
        for pred, target in zip(preds, masks):
            pred_np = pred.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            for class_idx in class_indices:
                if class_idx == 0:
                    continue
                pred_mask = (pred_np == class_idx)
                target_mask = (target_np == class_idx)
                iou_sums[class_idx] += compute_iou(pred_mask, target_mask)
                dice_sums[class_idx] += compute_dice(pred_mask, target_mask)
            count += 1
    class_metrics = {}
    valid_classes = [idx for idx in class_indices if idx != 0]
    denom = max(count, 1)
    mean_iou = 0.0
    mean_dice = 0.0
    for idx in valid_classes:
        label = class_indices[idx]
        class_iou = iou_sums[idx] / denom
        class_dice = dice_sums[idx] / denom
        class_metrics[label] = {"iou": class_iou, "dice": class_dice}
        mean_iou += class_iou
        mean_dice += class_dice
    if valid_classes:
        mean_iou /= len(valid_classes)
        mean_dice /= len(valid_classes)
    metrics = {
        "loss": running_loss / max(total, 1),
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "per_class": class_metrics,
    }
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    label_list: List[str]
    if args.labels:
        label_list = args.labels
    else:
        label_list = [args.target_label]
    label_map = {label: idx + 1 for idx, label in enumerate(label_list)}

    train_manifests = [args.train_manifest, *args.extra_train_manifest]
    valid_manifests = [args.valid_manifest, *args.extra_valid_manifest]

    def build_datasets(manifests: List[Path], balance: bool, seed: int):
        datasets = []
        for manifest in manifests:
            ds = BoxMaskDataset(
                manifest=manifest,
                target_label=None,
                label_map=label_map,
                image_cache=args.image_cache,
                resize=tuple(args.resize) if args.resize else None,
                use_cached_only=args.cached_only,
            )
            datasets.append(ds)
        if not datasets:
            raise ValueError("At least one manifest must be provided")
        if balance and len(datasets) > 1:
            min_len = min(len(ds) for ds in datasets)
            balanced = []
            for idx, ds in enumerate(datasets):
                balanced.append(maybe_subset(ds, min_len, seed + idx))
            datasets = balanced
        if len(datasets) == 1:
            return datasets[0]
        return torch.utils.data.ConcatDataset(datasets)

    train_dataset = build_datasets(train_manifests, args.balance_train_manifests, args.seed)
    valid_dataset = build_datasets(valid_manifests, args.balance_valid_manifests, args.seed + 1337)

    train_dataset = maybe_subset(train_dataset, args.max_train_samples, args.seed + 4242)
    valid_dataset = maybe_subset(valid_dataset, args.max_valid_samples, args.seed + 5252)

    pin_memory = device.type == "cuda" and args.num_workers > 0
    persistent_workers = args.num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    num_classes = len(label_map) + 1
    model = build_model(num_classes=num_classes)
    model.to(device)

    class_weights = estimate_class_weights(train_dataset, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_iou = -math.inf
    history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)

    idx_to_label = {idx: label for label, idx in label_map.items()}
    idx_to_label[0] = "background"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(
            model,
            valid_loader,
            criterion,
            device,
            class_indices=idx_to_label,
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        }
        history.append(record)
        print(json.dumps(record))

        if val_metrics["mean_iou"] > best_iou:
            best_iou = val_metrics["mean_iou"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "labels": label_list,
                    "label_map": label_map,
                    "resize": tuple(args.resize) if args.resize else None,
                },
                args.output,
            )

    args.metrics_output.write_text(json.dumps(history, indent=2))
    print(f"Saved best checkpoint to {args.output}")


if __name__ == "__main__":
    main()

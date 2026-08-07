#!/usr/bin/env python3
"""Evaluate a saved checkpoint with segmentation and presence metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.utils.data import DataLoader
from tqdm import tqdm

from lesion_segmentation.constants import ATTRIBUTES
from lesion_segmentation.data import DermoscopyDataset
from lesion_segmentation.metrics import batch_dice_iou
from lesion_segmentation.model import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", default=Path("artifacts/manifest.csv"), type=Path)
    parser.add_argument("--split", default=Path("splits/train_val.csv"), type=Path)
    parser.add_argument("--subset", default="val", choices=("train", "val"))
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hd95(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return float(np.hypot(*prediction.shape))

    prediction_boundary = prediction ^ binary_erosion(prediction)
    target_boundary = target ^ binary_erosion(target)
    distance_to_target = distance_transform_edt(~target_boundary)
    distance_to_prediction = distance_transform_edt(~prediction_boundary)
    distances = np.concatenate(
        [
            distance_to_target[prediction_boundary],
            distance_to_prediction[target_boundary],
        ]
    )
    return float(np.percentile(distances, 95))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positive_count = int(labels.sum())
    if positive_count == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    precision = true_positives / (np.arange(len(labels)) + 1)
    return float((precision * sorted_labels).sum() / positive_count)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for group in np.flatnonzero(counts > 1):
            indices = np.flatnonzero(inverse == group)
            ranks[indices] = ranks[indices].mean()
    positive_rank_sum = ranks[labels.astype(bool)].sum()
    return float(
        (
            positive_rank_sum
            - positive_count * (positive_count + 1) / 2
        )
        / (positive_count * negative_count)
    )


def positive_segmentation_metrics(
    dice_values: np.ndarray,
    iou_values: np.ndarray,
    presence_targets: np.ndarray,
    channel_names: tuple[str, ...],
) -> tuple[dict[str, dict[str, float | int | None]], float, float]:
    per_channel = {}
    positive_dice_means = []
    positive_iou_means = []
    for index, name in enumerate(channel_names):
        positive_cases = presence_targets[:, index].astype(bool)
        positive_count = int(positive_cases.sum())
        positive_dice = (
            float(dice_values[positive_cases, index].mean())
            if positive_count
            else None
        )
        positive_iou = (
            float(iou_values[positive_cases, index].mean())
            if positive_count
            else None
        )
        per_channel[name] = {
            "positive_cases": positive_count,
            "dice_positive": positive_dice,
            "iou_positive": positive_iou,
        }
        if positive_dice is not None:
            positive_dice_means.append(positive_dice)
        if positive_iou is not None:
            positive_iou_means.append(positive_iou)
    return (
        per_channel,
        float(np.mean(positive_dice_means)),
        float(np.mean(positive_iou_means)),
    )


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(
        args.checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    task = config["task"]
    image_size = int(config["image_size"])
    model = create_model(
        task,
        config["encoder"],
        encoder_weights=None,
        architecture=config.get("architecture", "unet"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    dataset = DermoscopyDataset(
        args.manifest.expanduser().resolve(),
        args.split.expanduser().resolve(),
        args.subset,
        task,
        image_size=image_size,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    channel_names = ("lesion",) if task == "task1" else ATTRIBUTES
    dice_rows = []
    iou_rows = []
    presence_rows = []
    target_presence_rows = []
    hausdorff_rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate", mininterval=5.0):
            images = batch["image"].to(device)
            target_masks = batch["mask"].to(device)
            output = model(images)
            if isinstance(output, tuple):
                mask_logits, presence_logits = output
            else:
                mask_logits, presence_logits = output, None
            dice, iou = batch_dice_iou(
                mask_logits,
                target_masks,
                threshold=args.threshold,
            )
            dice_rows.append(dice.cpu().numpy())
            iou_rows.append(iou.cpu().numpy())

            if task == "task1":
                predictions = (
                    mask_logits.sigmoid().ge(args.threshold).cpu().numpy()
                )
                targets = target_masks.ge(0.5).cpu().numpy()
                hausdorff_rows.extend(
                    hd95(prediction[0], target[0])
                    for prediction, target in zip(predictions, targets)
                )
            else:
                if presence_logits is None:
                    raise RuntimeError("Task 2 checkpoint has no presence head")
                presence_rows.append(presence_logits.sigmoid().cpu().numpy())
                target_presence_rows.append(batch["presence"].numpy())

    dice_values = np.concatenate(dice_rows, axis=0)
    iou_values = np.concatenate(iou_rows, axis=0)
    results: dict[str, object] = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "task": task,
        "subset": args.subset,
        "cases": len(dataset),
        "threshold": args.threshold,
        "mean_dice": float(dice_values.mean()),
        "mean_iou": float(iou_values.mean()),
        "per_channel": {
            name: {
                "dice": float(dice_values[:, index].mean()),
                "iou": float(iou_values[:, index].mean()),
            }
            for index, name in enumerate(channel_names)
        },
    }

    if task == "task1":
        results["hd95_pixels_at_model_resolution"] = {
            "mean": float(np.mean(hausdorff_rows)),
            "median": float(np.median(hausdorff_rows)),
            "p95": float(np.percentile(hausdorff_rows, 95)),
        }
    else:
        presence_scores = np.concatenate(presence_rows, axis=0)
        presence_targets = np.concatenate(target_presence_rows, axis=0)
        positive_metrics, mean_positive_dice, mean_positive_iou = (
            positive_segmentation_metrics(
                dice_values,
                iou_values,
                presence_targets,
                ATTRIBUTES,
            )
        )
        for attribute, metrics in positive_metrics.items():
            results["per_channel"][attribute].update(metrics)
        results["mean_dice_positive"] = mean_positive_dice
        results["mean_iou_positive"] = mean_positive_iou
        results["presence"] = {
            attribute: {
                "average_precision": average_precision(
                    presence_scores[:, index],
                    presence_targets[:, index],
                ),
                "roc_auc": roc_auc(
                    presence_scores[:, index],
                    presence_targets[:, index],
                ),
            }
            for index, attribute in enumerate(ATTRIBUTES)
        }

    output_path = (
        args.output
        if args.output is not None
        else args.checkpoint.parent / f"evaluation_{args.subset}.json"
    ).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate Task 1 at native mask resolution with threshold and subgroup analysis."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import (
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    label,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from lesion_segmentation.data import DermoscopyDataset
from lesion_segmentation.model import create_model
from lesion_segmentation.reporting import lesion_size_category


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--ensemble-checkpoint",
        action="append",
        default=[],
        type=Path,
        help="Optional additional compatible checkpoint; may be repeated.",
    )
    parser.add_argument(
        "--ensemble-weights",
        help="Comma-separated weights for the primary and additional checkpoints.",
    )
    parser.add_argument("--manifest", default=Path("artifacts/manifest.csv"), type=Path)
    parser.add_argument("--split", default=Path("splits/train_val.csv"), type=Path)
    parser.add_argument("--subset", default="val", choices=("train", "val"))
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument(
        "--thresholds",
        default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70",
    )
    parser.add_argument("--selected-threshold", default=0.5, type=float)
    parser.add_argument(
        "--postprocess",
        default="none",
        choices=("none", "largest", "largest_fill"),
    )
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--resize-mode", choices=("stretch", "letterbox"))
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Read original images even when the checkpoint records a cache.",
    )
    parser.add_argument("--tta", default="none", choices=("none", "flips", "d4"))
    parser.add_argument("--hd95-long-side", default=512, type=int)
    parser.add_argument("--bootstrap-iterations", default=2000, type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--per-case-output", type=Path)
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_thresholds(value: str, selected: float) -> tuple[float, ...]:
    thresholds = {float(item.strip()) for item in value.split(",") if item.strip()}
    thresholds.add(float(selected))
    if not thresholds or any(not 0.0 < threshold < 1.0 for threshold in thresholds):
        raise ValueError("Thresholds must be strictly between 0 and 1")
    return tuple(sorted(thresholds))


def parse_ensemble_weights(value: str | None, model_count: int) -> tuple[float, ...]:
    if value is None:
        return tuple([1.0 / model_count] * model_count)
    weights = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(weights) != model_count:
        raise ValueError(
            f"Expected {model_count} ensemble weights, received {len(weights)}"
        )
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("Ensemble weights must be non-negative with a positive sum")
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def restore_probability(
    probability: np.ndarray,
    original_width: int,
    original_height: int,
    content_left: int,
    content_top: int,
    content_width: int,
    content_height: int,
) -> np.ndarray:
    cropped = probability[
        content_top : content_top + content_height,
        content_left : content_left + content_width,
    ]
    restored = Image.fromarray(cropped.astype(np.float32), mode="F").resize(
        (original_width, original_height),
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(restored, dtype=np.float32)


def restore_binary(
    mask: np.ndarray,
    original_width: int,
    original_height: int,
    content_left: int,
    content_top: int,
    content_width: int,
    content_height: int,
) -> np.ndarray:
    cropped = np.asarray(mask, dtype=bool)[
        content_top : content_top + content_height,
        content_left : content_left + content_width,
    ]
    restored = Image.fromarray(cropped.astype(np.uint8) * 255, mode="L").resize(
        (original_width, original_height),
        resample=Image.Resampling.NEAREST,
    )
    return np.asarray(restored).astype(bool)


def binary_overlap(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = int(np.logical_and(prediction, target).sum())
    prediction_sum = int(prediction.sum())
    target_sum = int(target.sum())
    denominator = prediction_sum + target_sum
    union = denominator - intersection
    dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
    iou = 1.0 if union == 0 else intersection / union
    return float(dice), float(iou)


def postprocess_mask(mask: np.ndarray, mode: str) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if mode == "none" or not binary.any():
        return binary
    components, component_count = label(binary)
    if component_count > 1:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        binary = components == int(sizes.argmax())
    if mode == "largest_fill":
        binary = binary_fill_holes(binary)
    return np.asarray(binary, dtype=bool)


def resize_binary_long_side(mask: np.ndarray, long_side: int) -> np.ndarray:
    height, width = mask.shape
    scale = long_side / max(height, width)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
        (resized_width, resized_height),
        resample=Image.Resampling.NEAREST,
    )
    return np.asarray(image).astype(bool)


def hd95(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
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


def bootstrap_mean_interval(
    values: np.ndarray,
    iterations: int,
    seed: int = 20260728,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if iterations <= 0:
        return {}
    generator = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        means[index] = generator.choice(values, size=len(values), replace=True).mean()
    return {
        "lower_95": float(np.percentile(means, 2.5)),
        "upper_95": float(np.percentile(means, 97.5)),
    }


def predict_probabilities(
    model: torch.nn.Module,
    images: torch.Tensor,
    tta: str,
) -> torch.Tensor:
    predictions = [model(images).sigmoid()]
    if tta == "flips":
        horizontal = torch.flip(images, dims=(3,))
        vertical = torch.flip(images, dims=(2,))
        predictions.append(torch.flip(model(horizontal).sigmoid(), dims=(3,)))
        predictions.append(torch.flip(model(vertical).sigmoid(), dims=(2,)))
    if tta == "d4":
        for rotations in (1, 2, 3):
            rotated = torch.rot90(images, rotations, dims=(2, 3))
            rotated_prediction = model(rotated).sigmoid()
            predictions.append(
                torch.rot90(rotated_prediction, -rotations, dims=(2, 3))
            )
        reflected = torch.flip(images, dims=(3,))
        for rotations in (0, 1, 2, 3):
            rotated = torch.rot90(reflected, rotations, dims=(2, 3))
            rotated_prediction = model(rotated).sigmoid()
            restored = torch.rot90(
                rotated_prediction,
                -rotations,
                dims=(2, 3),
            )
            predictions.append(torch.flip(restored, dims=(3,)))
    return torch.stack(predictions).mean(dim=0)


def aggregate_cases(rows: list[dict[str, object]]) -> dict[str, object]:
    dice = np.asarray([float(row["dice"]) for row in rows])
    iou = np.asarray([float(row["iou"]) for row in rows])
    return {
        "cases": len(rows),
        "mean_dice": float(dice.mean()),
        "median_dice": float(np.median(dice)),
        "mean_iou": float(iou.mean()),
        "dice_ci": bootstrap_mean_interval(dice, 2000),
    }


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds, args.selected_threshold)
    device = select_device(args.device)
    checkpoint_paths = [
        args.checkpoint.expanduser().resolve(),
        *[
            path.expanduser().resolve()
            for path in args.ensemble_checkpoint
        ],
    ]
    ensemble_weights = parse_ensemble_weights(
        args.ensemble_weights,
        len(checkpoint_paths),
    )
    checkpoint_path = checkpoint_paths[0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["task"] != "task1":
        raise ValueError("This evaluator accepts Task 1 checkpoints only")
    image_size = args.image_size or int(config["image_size"])
    resize_mode = args.resize_mode or config.get("resize_mode", "stretch")
    cache_value = (
        None
        if args.disable_cache
        else args.cache_root or config.get("cache_root")
    )
    cache_root = Path(cache_value).expanduser().resolve() if cache_value else None
    models = []
    checkpoints = []
    for current_path in checkpoint_paths:
        current_checkpoint = torch.load(
            current_path,
            map_location="cpu",
            weights_only=False,
        )
        current_config = current_checkpoint["config"]
        for key, expected in (
            ("task", "task1"),
            ("encoder", config["encoder"]),
            ("image_size", config["image_size"]),
            ("resize_mode", config.get("resize_mode", "stretch")),
        ):
            actual = current_config.get(
                key,
                "stretch" if key == "resize_mode" else None,
            )
            if str(actual) != str(expected):
                raise ValueError(
                    f"Incompatible ensemble checkpoint {current_path}: "
                    f"{key}={actual}, expected {expected}"
                )
        current_model = create_model(
            "task1",
            current_config["encoder"],
            encoder_weights=None,
            architecture=current_config.get("architecture", "unet"),
        )
        current_model.load_state_dict(current_checkpoint["model_state_dict"])
        current_model.to(device).eval()
        models.append(current_model)
        checkpoints.append(current_checkpoint)

    dataset = DermoscopyDataset(
        args.manifest.expanduser().resolve(),
        args.split.expanduser().resolve(),
        args.subset,
        "task1",
        image_size=image_size,
        augment=False,
        resize_mode=resize_mode,
        cache_root=cache_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    threshold_metrics: dict[float, list[tuple[float, float]]] = defaultdict(list)
    per_case_rows: list[dict[str, object]] = []
    selected_hd95 = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate native", mininterval=3.0):
            images = batch["image"].to(device)
            probabilities = sum(
                weight * predict_probabilities(model, images, args.tta)
                for model, weight in zip(models, ensemble_weights)
            ).cpu().numpy()
            batch_size = probabilities.shape[0]
            for index in range(batch_size):
                original_width = int(batch["original_width"][index])
                original_height = int(batch["original_height"][index])
                content_left = int(batch["content_left"][index])
                content_top = int(batch["content_top"][index])
                content_width = int(batch["content_width"][index])
                content_height = int(batch["content_height"][index])
                restored = restore_probability(
                    probabilities[index, 0],
                    original_width,
                    original_height,
                    content_left,
                    content_top,
                    content_width,
                    content_height,
                )
                with Image.open(batch["lesion_mask_path"][index]) as source:
                    target = np.asarray(source.convert("L")).astype(np.uint8) > 127
                if restored.shape != target.shape:
                    raise ValueError(
                        f"Restored prediction shape {restored.shape} does not match "
                        f"target shape {target.shape}"
                    )
                selected_prediction = None
                selected_dice = None
                selected_iou = None
                for threshold in thresholds:
                    if args.postprocess == "none":
                        prediction = restored >= threshold
                    else:
                        canvas_prediction = postprocess_mask(
                            probabilities[index, 0] >= threshold,
                            args.postprocess,
                        )
                        prediction = restore_binary(
                            canvas_prediction,
                            original_width,
                            original_height,
                            content_left,
                            content_top,
                            content_width,
                            content_height,
                        )
                    dice, iou = binary_overlap(prediction, target)
                    threshold_metrics[threshold].append((dice, iou))
                    if threshold == args.selected_threshold:
                        selected_prediction = prediction
                        selected_dice = dice
                        selected_iou = iou
                if selected_prediction is None:
                    raise RuntimeError("Selected threshold was not evaluated")
                prediction_hd = resize_binary_long_side(
                    selected_prediction,
                    args.hd95_long_side,
                )
                target_hd = resize_binary_long_side(target, args.hd95_long_side)
                case_hd95 = hd95(prediction_hd, target_hd)
                selected_hd95.append(case_hd95)
                lesion_ratio = float(target.mean())
                per_case_rows.append(
                    {
                        "image_id": batch["image_id"][index],
                        "dice": selected_dice,
                        "iou": selected_iou,
                        "hd95_standardized_pixels": case_hd95,
                        "lesion_ratio": lesion_ratio,
                        "lesion_size": lesion_size_category(lesion_ratio),
                        "touches_border": bool(
                            target[0].any()
                            or target[-1].any()
                            or target[:, 0].any()
                            or target[:, -1].any()
                        ),
                        "original_width": target.shape[1],
                        "original_height": target.shape[0],
                    }
                )

    sweep = {}
    for threshold, values in threshold_metrics.items():
        metrics = np.asarray(values)
        sweep[f"{threshold:.2f}"] = {
            "mean_dice": float(metrics[:, 0].mean()),
            "mean_iou": float(metrics[:, 1].mean()),
        }
    best_threshold = max(
        thresholds,
        key=lambda threshold: sweep[f"{threshold:.2f}"]["mean_dice"],
    )
    selected_summary = aggregate_cases(per_case_rows)
    selected_summary["dice_ci"] = bootstrap_mean_interval(
        np.asarray([float(row["dice"]) for row in per_case_rows]),
        args.bootstrap_iterations,
    )
    subgroups = {}
    for group_name in ("small", "moderate", "large"):
        group_rows = [
            row for row in per_case_rows if row["lesion_size"] == group_name
        ]
        subgroups[f"lesion_size/{group_name}"] = aggregate_cases(group_rows)
    for touches_border in (False, True):
        group_rows = [
            row
            for row in per_case_rows
            if row["touches_border"] is touches_border
        ]
        subgroups[f"touches_border/{str(touches_border).lower()}"] = (
            aggregate_cases(group_rows)
        )

    results = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoints": [
            {
                "path": str(path),
                "epoch": int(current_checkpoint["epoch"]),
                "architecture": current_checkpoint["config"].get(
                    "architecture",
                    "unet",
                ),
                "weight": weight,
            }
            for path, current_checkpoint, weight in zip(
                checkpoint_paths,
                checkpoints,
                ensemble_weights,
            )
        ],
        "subset": args.subset,
        "cases": len(dataset),
        "evaluation_space": "native_mask_resolution",
        "model_image_size": image_size,
        "checkpoint_image_size": int(config["image_size"]),
        "resize_mode": resize_mode,
        "postprocess": args.postprocess,
        "postprocess_space": (
            None if args.postprocess == "none" else "model_canvas"
        ),
        "tta": args.tta,
        "selected_threshold": args.selected_threshold,
        "selected": selected_summary,
        "threshold_sweep": sweep,
        "best_validation_threshold": best_threshold,
        "hd95_pixels_at_standardized_long_side": {
            "long_side": args.hd95_long_side,
            "mean": float(np.mean(selected_hd95)),
            "median": float(np.median(selected_hd95)),
            "p95": float(np.percentile(selected_hd95, 95)),
        },
        "subgroups": subgroups,
        "worst_cases": sorted(
            per_case_rows,
            key=lambda row: float(row["dice"]),
        )[:20],
    }
    output_path = (
        args.output
        if args.output is not None
        else checkpoint_path.parent / f"evaluation_{args.subset}_native.json"
    ).expanduser().resolve()
    per_case_path = (
        args.per_case_output
        if args.per_case_output is not None
        else checkpoint_path.parent / f"evaluation_{args.subset}_per_case.csv"
    ).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    with per_case_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_case_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_case_rows)
    print(json.dumps(results, indent=2))
    print(f"Wrote {output_path}")
    print(f"Wrote {per_case_path}")


if __name__ == "__main__":
    main()

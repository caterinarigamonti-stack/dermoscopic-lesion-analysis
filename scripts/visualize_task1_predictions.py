#!/usr/bin/env python3
"""Create a compact qualitative QC montage for a Task 1 checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from PIL import Image, ImageDraw

from lesion_segmentation.constants import IMAGENET_MEAN, IMAGENET_STD
from lesion_segmentation.data import DermoscopyDataset
from lesion_segmentation.model import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", default=Path("artifacts/manifest.csv"), type=Path)
    parser.add_argument("--split", default=Path("splits/train_val.csv"), type=Path)
    parser.add_argument("--output", default=Path("artifacts/task1_predictions.jpg"), type=Path)
    parser.add_argument("--count", default=8, type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--threshold", default=0.5, type=float)
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    array = (tensor.cpu() * std + mean).clamp(0, 1)
    return (array.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def overlay(image: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    output = image.copy()
    output[mask] = (
        0.55 * output[mask] + 0.45 * np.asarray(colour)
    ).astype(np.uint8)
    return output


def error_map(
    image: np.ndarray, prediction: np.ndarray, target: np.ndarray
) -> np.ndarray:
    output = (image * 0.35).astype(np.uint8)
    output[prediction & target] = (40, 210, 70)
    output[prediction & ~target] = (255, 190, 0)
    output[~prediction & target] = (245, 55, 55)
    return output


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(
        args.checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    if config["task"] != "task1":
        raise ValueError("This visualizer accepts only Task 1 checkpoints")
    model = create_model(
        "task1",
        config["encoder"],
        encoder_weights=None,
        architecture=config.get("architecture", "unet"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    dataset = DermoscopyDataset(
        args.manifest.expanduser().resolve(),
        args.split.expanduser().resolve(),
        "val",
        "task1",
        image_size=int(config["image_size"]),
        augment=False,
    )
    ordered_indices = sorted(
        range(len(dataset)),
        key=lambda index: float(dataset.rows[index]["lesion_ratio"]),
    )
    sample_indices = np.linspace(
        0, len(ordered_indices) - 1, min(args.count, len(dataset)), dtype=int
    )
    selected = [ordered_indices[index] for index in sample_indices]

    tile_size = int(config["image_size"])
    label_height = 26
    columns = ("image", "ground truth", "prediction", "error: TP / FP / FN")
    canvas = Image.new(
        "RGB",
        (
            len(columns) * tile_size,
            len(selected) * (tile_size + label_height),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    with torch.no_grad():
        for row_index, dataset_index in enumerate(selected):
            sample = dataset[dataset_index]
            image_tensor = sample["image"]
            target = sample["mask"][0].numpy().astype(bool)
            logits = model(image_tensor.unsqueeze(0).to(device))
            prediction = (
                logits.sigmoid()[0, 0].ge(args.threshold).cpu().numpy()
            )
            image = denormalize(image_tensor)
            tiles = (
                image,
                overlay(image, target, (40, 210, 70)),
                overlay(image, prediction, (40, 120, 255)),
                error_map(image, prediction, target),
            )
            y = row_index * (tile_size + label_height)
            for column_index, (label, tile) in enumerate(zip(columns, tiles)):
                x = column_index * tile_size
                canvas.paste(Image.fromarray(tile), (x, y + label_height))
                header = (
                    f"{sample['image_id']} - {label}"
                    if column_index == 0
                    else label
                )
                draw.text((x + 6, y + 6), header, fill="black")

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

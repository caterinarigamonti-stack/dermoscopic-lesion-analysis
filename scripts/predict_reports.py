#!/usr/bin/env python3
"""Run both models and write masks plus evidence-anchored findings reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform
from tqdm import tqdm

from lesion_segmentation.constants import ATTRIBUTES, IMAGENET_MEAN, IMAGENET_STD
from lesion_segmentation.model import create_model
from lesion_segmentation.reporting import (
    build_findings_report,
    validate_report_consistency,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--task1-checkpoint", required=True, type=Path)
    parser.add_argument("--task2-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("outputs"), type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--mask-threshold", default=0.5, type=float)
    parser.add_argument("--split-label", default="inference")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint_model(
    path: Path,
    expected_task: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], int]:
    checkpoint = torch.load(
        path.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    if config["task"] != expected_task:
        raise ValueError(
            f"{path} is a {config['task']} checkpoint, expected {expected_task}"
        )
    model = create_model(
        expected_task,
        config["encoder"],
        encoder_weights=None,
        architecture=config.get("architecture", "unet"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, config, int(checkpoint["epoch"])


def preprocess(image: Image.Image, size: int) -> torch.Tensor:
    resized = transform.resize(
        image,
        [size, size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    tensor = transform.pil_to_tensor(resized).float().div_(255.0)
    tensor = transform.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor.unsqueeze(0)


def save_binary_mask(mask: np.ndarray, size: tuple[int, int], path: Path) -> None:
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    mask_image = mask_image.resize(size, resample=Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_image.save(path)


def discover_images(path: Path) -> list[Path]:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        if resolved.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {resolved}")
        return [resolved]
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    images = sorted(
        item
        for item in resolved.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No supported images found under {resolved}")
    return images


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task1_model, task1_config, task1_epoch = load_checkpoint_model(
        args.task1_checkpoint,
        "task1",
        device,
    )
    task2_model, task2_config, task2_epoch = load_checkpoint_model(
        args.task2_checkpoint,
        "task2",
        device,
    )
    image_paths = discover_images(args.images)
    model_version = (
        f"task1:{task1_config['encoder']}:epoch{task1_epoch};"
        f"task2:{task2_config['encoder']}:epoch{task2_epoch}"
    )
    report_payloads = []
    report_lines = []

    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="predict"):
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            original_size = image.size
            task1_input = preprocess(image, int(task1_config["image_size"])).to(device)
            task2_input = preprocess(image, int(task2_config["image_size"])).to(device)

            lesion_logits = task1_model(task1_input)
            task2_output = task2_model(task2_input)
            if not isinstance(task2_output, tuple):
                raise RuntimeError("Task 2 checkpoint has no presence head")
            attribute_logits, presence_logits = task2_output
            lesion_mask = (
                lesion_logits.sigmoid()[0, 0]
                .ge(args.mask_threshold)
                .cpu()
                .numpy()
            )
            attribute_masks = (
                attribute_logits.sigmoid()[0]
                .ge(args.mask_threshold)
                .cpu()
                .numpy()
            )
            presence_probabilities = presence_logits.sigmoid()[0].cpu().numpy()

            image_id = image_path.stem
            save_binary_mask(
                lesion_mask,
                original_size,
                output_dir / "task1_masks" / f"{image_id}.png",
            )
            for index, attribute in enumerate(ATTRIBUTES):
                save_binary_mask(
                    attribute_masks[index],
                    original_size,
                    output_dir / "task2_masks" / attribute / f"{image_id}.png",
                )

            probabilities = {
                attribute: float(presence_probabilities[index])
                for index, attribute in enumerate(ATTRIBUTES)
            }
            payload, report_text = build_findings_report(
                image_id=image_id,
                split=args.split_label,
                model_version=model_version,
                lesion_mask=lesion_mask,
                attribute_probabilities=probabilities,
            )
            consistency_errors = validate_report_consistency(payload, report_text)
            if consistency_errors:
                raise RuntimeError(
                    f"Inconsistent report for {image_id}: {consistency_errors}"
                )
            payload["source_image"] = str(image_path)
            payload["report_text"] = report_text
            report_payloads.append(payload)
            report_lines.append(f"{image_id}\t{report_text}")

    jsonl_path = output_dir / "reports.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(payload) + "\n" for payload in report_payloads),
        encoding="utf-8",
    )
    text_path = output_dir / "reports.txt"
    text_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(report_payloads)} reports to {jsonl_path}")
    print(f"Wrote masks under {output_dir}")


if __name__ == "__main__":
    main()

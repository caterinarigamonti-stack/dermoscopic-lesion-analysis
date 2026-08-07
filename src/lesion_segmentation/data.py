"""Dataset and paired image-mask transforms."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Literal

import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform

from .constants import ATTRIBUTES, IMAGENET_MEAN, IMAGENET_STD
from .preprocessing import (
    IMAGE_FILL,
    ResizeMode,
    ResizeSpec,
    make_resize_spec,
    resize_to_canvas,
)


Task = Literal["task1", "task2"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _augment_colour(image: Image.Image) -> Image.Image:
    brightness = random.uniform(0.85, 1.15)
    contrast = random.uniform(0.85, 1.15)
    saturation = random.uniform(0.85, 1.15)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    return ImageEnhance.Color(image).enhance(saturation)


def _paired_spatial_transform(
    image: Image.Image, masks: list[Image.Image]
) -> tuple[Image.Image, list[Image.Image]]:
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        masks = [mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for mask in masks]
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        masks = [mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM) for mask in masks]

    rotation = random.randrange(4)
    if rotation:
        operation = (
            Image.Transpose.ROTATE_90,
            Image.Transpose.ROTATE_180,
            Image.Transpose.ROTATE_270,
        )[rotation - 1]
        image = image.transpose(operation)
        masks = [mask.transpose(operation) for mask in masks]
    if random.random() < 0.75:
        angle = random.uniform(-25.0, 25.0)
        translate = (
            round(random.uniform(-0.05, 0.05) * image.width),
            round(random.uniform(-0.05, 0.05) * image.height),
        )
        scale = random.uniform(0.90, 1.10)
        image = transform.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=IMAGE_FILL,
        )
        masks = [
            transform.affine(
                mask,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )
            for mask in masks
        ]
    return image, masks


class DermoscopyDataset(Dataset):
    """Aligned RGB images and binary masks for either project task."""

    def __init__(
        self,
        manifest_path: Path,
        split_path: Path,
        subset: Literal["train", "val"],
        task: Task,
        image_size: int = 384,
        augment: bool = False,
        resize_mode: ResizeMode = "stretch",
        cache_root: Path | None = None,
    ) -> None:
        manifest_rows = _read_csv(manifest_path)
        split_by_id = {
            row["image_id"]: row["split"] for row in _read_csv(split_path)
        }
        self.rows = [
            row
            for row in manifest_rows
            if split_by_id.get(row["image_id"]) == subset
        ]
        if not self.rows:
            raise ValueError(f"No rows found for subset={subset!r}")
        if task not in ("task1", "task2"):
            raise ValueError(f"Unknown task: {task}")
        self.task = task
        self.image_size = image_size
        self.augment = augment
        self.resize_mode = resize_mode
        self.cache_root = cache_root.expanduser().resolve() if cache_root else None
        self.cache_metadata: dict[str, dict[str, str]] = {}
        if self.cache_root is not None:
            if task != "task1":
                raise ValueError("The current cache format supports Task 1 only")
            config_path = self.cache_root / "config.json"
            metadata_path = self.cache_root / "metadata.csv"
            if not config_path.exists() or not metadata_path.exists():
                raise FileNotFoundError(
                    f"Cache is incomplete under {self.cache_root}"
                )
            cache_config = json.loads(config_path.read_text(encoding="utf-8"))
            if int(cache_config["image_size"]) != image_size:
                raise ValueError(
                    "Cache image size does not match requested image size"
                )
            if cache_config["resize_mode"] != resize_mode:
                raise ValueError(
                    "Cache resize mode does not match requested resize mode"
                )
            self.cache_metadata = {
                row["image_id"]: row for row in _read_csv(metadata_path)
            }
        if task == "task2":
            positives = torch.tensor(
                [
                    sum(float(row[f"{attribute}_present"]) for row in self.rows)
                    for attribute in ATTRIBUTES
                ],
                dtype=torch.float32,
            )
            negatives = len(self.rows) - positives
            self.presence_pos_weight = (
                negatives.div(positives.clamp_min(1.0)).clamp_(1.0, 20.0)
            )
        else:
            self.presence_pos_weight = torch.ones(1, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        row = self.rows[index]
        if self.cache_root is None:
            with Image.open(row["image_path"]) as source:
                image = source.convert("RGB")
            with Image.open(row["lesion_mask_path"]) as source:
                lesion_mask = source.convert("L")
            spec = make_resize_spec(
                image.width,
                image.height,
                self.image_size,
                self.resize_mode,
            )
        else:
            image_id = row["image_id"]
            metadata = self.cache_metadata[image_id]
            with Image.open(self.cache_root / "images" / f"{image_id}.jpg") as source:
                image = source.convert("RGB")
            with Image.open(
                self.cache_root / "task1_gt" / f"{image_id}.png"
            ) as source:
                lesion_mask = source.convert("L")
            spec = ResizeSpec(
                original_width=int(metadata["original_width"]),
                original_height=int(metadata["original_height"]),
                canvas_size=int(metadata["canvas_size"]),
                content_left=int(metadata["content_left"]),
                content_top=int(metadata["content_top"]),
                content_width=int(metadata["content_width"]),
                content_height=int(metadata["content_height"]),
                resize_mode=metadata["resize_mode"],  # type: ignore[arg-type]
            )

        if self.task == "task1":
            masks = [lesion_mask]
            presence = torch.ones(1, dtype=torch.float32)
        else:
            masks = []
            for attribute in ATTRIBUTES:
                with Image.open(row[f"{attribute}_mask_path"]) as source:
                    masks.append(source.convert("L"))
            presence = torch.tensor(
                [float(row[f"{attribute}_present"]) for attribute in ATTRIBUTES],
                dtype=torch.float32,
            )

        if self.cache_root is None:
            image = resize_to_canvas(
                image,
                spec,
                Image.Resampling.BILINEAR,
                IMAGE_FILL,
            )
            lesion_mask = resize_to_canvas(
                lesion_mask,
                spec,
                Image.Resampling.NEAREST,
                0,
            )
            masks = [
                resize_to_canvas(
                    mask,
                    spec,
                    Image.Resampling.NEAREST,
                    0,
                )
                for mask in masks
            ]
        elif image.size != (self.image_size, self.image_size):
            raise ValueError(
                f"Cached image {row['image_id']} has unexpected size {image.size}"
            )

        if self.augment:
            image, masks_and_roi = _paired_spatial_transform(
                image, [*masks, lesion_mask]
            )
            masks = masks_and_roi[:-1]
            lesion_mask = masks_and_roi[-1]
            image = _augment_colour(image)

        image_tensor = transform.pil_to_tensor(image).float().div_(255.0)
        image_tensor = transform.normalize(
            image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD
        )
        mask_tensor = torch.cat(
            [
                transform.pil_to_tensor(mask).float().div_(255.0)
                for mask in masks
            ],
            dim=0,
        ).gt_(0.5).float()
        lesion_tensor = (
            transform.pil_to_tensor(lesion_mask)
            .float()
            .div_(255.0)
            .gt_(0.5)
            .float()
        )

        return {
            "image_id": row["image_id"],
            "image": image_tensor,
            "mask": mask_tensor,
            "lesion_roi": lesion_tensor,
            "presence": presence,
            "lesion_mask_path": row["lesion_mask_path"],
            "original_width": spec.original_width,
            "original_height": spec.original_height,
            "content_left": spec.content_left,
            "content_top": spec.content_top,
            "content_width": spec.content_width,
            "content_height": spec.content_height,
        }

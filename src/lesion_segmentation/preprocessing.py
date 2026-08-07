"""Aspect-ratio-preserving image and mask preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PIL import Image

from .constants import IMAGENET_MEAN


ResizeMode = Literal["stretch", "letterbox"]
IMAGE_FILL = tuple(round(value * 255) for value in IMAGENET_MEAN)


@dataclass(frozen=True)
class ResizeSpec:
    original_width: int
    original_height: int
    canvas_size: int
    content_left: int
    content_top: int
    content_width: int
    content_height: int
    resize_mode: ResizeMode

    @property
    def content_box(self) -> tuple[int, int, int, int]:
        return (
            self.content_left,
            self.content_top,
            self.content_left + self.content_width,
            self.content_top + self.content_height,
        )


def make_resize_spec(
    width: int,
    height: int,
    canvas_size: int,
    resize_mode: ResizeMode,
) -> ResizeSpec:
    if width <= 0 or height <= 0 or canvas_size <= 0:
        raise ValueError("Image dimensions and canvas size must be positive")
    if resize_mode == "stretch":
        content_width = canvas_size
        content_height = canvas_size
    elif resize_mode == "letterbox":
        scale = min(canvas_size / width, canvas_size / height)
        content_width = max(1, min(canvas_size, round(width * scale)))
        content_height = max(1, min(canvas_size, round(height * scale)))
    else:
        raise ValueError(f"Unknown resize mode: {resize_mode}")
    return ResizeSpec(
        original_width=width,
        original_height=height,
        canvas_size=canvas_size,
        content_left=(canvas_size - content_width) // 2,
        content_top=(canvas_size - content_height) // 2,
        content_width=content_width,
        content_height=content_height,
        resize_mode=resize_mode,
    )


def resize_to_canvas(
    image: Image.Image,
    spec: ResizeSpec,
    interpolation: Image.Resampling,
    fill: int | tuple[int, int, int],
) -> Image.Image:
    resized = image.resize(
        (spec.content_width, spec.content_height),
        resample=interpolation,
    )
    if spec.resize_mode == "stretch":
        return resized
    canvas = Image.new(image.mode, (spec.canvas_size, spec.canvas_size), color=fill)
    canvas.paste(resized, (spec.content_left, spec.content_top))
    return canvas

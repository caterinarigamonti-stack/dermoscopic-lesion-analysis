#!/usr/bin/env python3
"""Render image/mask overlays for rapid visual quality control."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ATTRIBUTES = (
    "pigment_network",
    "negative_network",
    "streaks",
    "milia_like_cysts",
    "globules",
)
COLOURS = {
    "lesion": (60, 220, 80),
    "pigment_network": (255, 184, 0),
    "negative_network": (0, 180, 255),
    "streaks": (255, 50, 50),
    "milia_like_cysts": (180, 80, 255),
    "globules": (255, 80, 180),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-ids", required=True, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", default=260, type=int)
    parser.add_argument("--height", default=190, type=int)
    return parser.parse_args()


def overlay(
    image: Image.Image,
    mask_path: str,
    colour: tuple[int, int, int],
    width: int,
    height: int,
) -> Image.Image:
    base = image.resize((width, height), Image.Resampling.BILINEAR).convert("RGB")
    with Image.open(mask_path) as source:
        mask = source.convert("L").resize((width, height), Image.Resampling.NEAREST)
    mask_array = np.asarray(mask) > 0
    output = np.asarray(base).copy()
    colour_array = np.asarray(colour)
    output[mask_array] = (
        0.55 * output[mask_array] + 0.45 * colour_array
    ).astype(np.uint8)
    return Image.fromarray(output)


def main() -> None:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = {row["image_id"]: row for row in csv.DictReader(handle)}

    columns = ("image", "lesion", *ATTRIBUTES)
    label_height = 28
    canvas = Image.new(
        "RGB",
        (
            len(columns) * args.width,
            len(args.image_ids) * (args.height + label_height),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    for row_index, image_id in enumerate(args.image_ids):
        row = rows[image_id]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        y = row_index * (args.height + label_height)
        for column_index, column in enumerate(columns):
            x = column_index * args.width
            if column == "image":
                tile = image.resize(
                    (args.width, args.height), Image.Resampling.BILINEAR
                )
                label = image_id
            else:
                mask_path = (
                    row["lesion_mask_path"]
                    if column == "lesion"
                    else row[f"{column}_mask_path"]
                )
                tile = overlay(
                    image,
                    mask_path,
                    COLOURS[column],
                    args.width,
                    args.height,
                )
                label = column
            canvas.paste(tile, (x, y + label_height))
            draw.text((x + 6, y + 7), label, fill="black")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=92)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

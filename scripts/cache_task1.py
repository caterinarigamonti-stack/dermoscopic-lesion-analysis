#!/usr/bin/env python3
"""Cache Task 1 images and masks on a fixed, aspect-preserving canvas."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PIL import Image
from tqdm import tqdm

from lesion_segmentation.preprocessing import (
    IMAGE_FILL,
    make_resize_spec,
    resize_to_canvas,
)


METADATA_FIELDS = (
    "image_id",
    "original_width",
    "original_height",
    "canvas_size",
    "content_left",
    "content_top",
    "content_width",
    "content_height",
    "resize_mode",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=Path("artifacts/manifest.csv"), type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-size", default=384, type=int)
    parser.add_argument(
        "--resize-mode",
        default="letterbox",
        choices=("stretch", "letterbox"),
    )
    parser.add_argument("--jpeg-quality", default=95, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    image_dir = output_dir / "images"
    mask_dir = output_dir / "task1_gt"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(manifest_path)
    metadata_rows = []

    for row in tqdm(rows, desc="cache task1", mininterval=2.0):
        image_id = row["image_id"]
        output_image = image_dir / f"{image_id}.jpg"
        output_mask = mask_dir / f"{image_id}.png"
        with Image.open(row["image_path"]) as source:
            original_size = source.size
            image = source.convert("RGB") if args.overwrite or not output_image.exists() else None
        spec = make_resize_spec(
            original_size[0],
            original_size[1],
            args.image_size,
            args.resize_mode,
        )
        if image is not None:
            cached_image = resize_to_canvas(
                image,
                spec,
                Image.Resampling.BILINEAR,
                IMAGE_FILL,
            )
            cached_image.save(
                output_image,
                format="JPEG",
                quality=args.jpeg_quality,
                subsampling=0,
            )
        if args.overwrite or not output_mask.exists():
            with Image.open(row["lesion_mask_path"]) as source:
                mask = source.convert("L")
            cached_mask = resize_to_canvas(
                mask,
                spec,
                Image.Resampling.NEAREST,
                0,
            )
            cached_mask.save(output_mask, format="PNG", optimize=False)
        metadata_rows.append(
            {
                "image_id": image_id,
                "original_width": spec.original_width,
                "original_height": spec.original_height,
                "canvas_size": spec.canvas_size,
                "content_left": spec.content_left,
                "content_top": spec.content_top,
                "content_width": spec.content_width,
                "content_height": spec.content_height,
                "resize_mode": spec.resize_mode,
            }
        )

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(metadata_rows)
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "cases": len(rows),
                "image_size": args.image_size,
                "resize_mode": args.resize_mode,
                "jpeg_quality": args.jpeg_quality,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Cached {len(rows)} Task 1 cases under {output_dir}")


if __name__ == "__main__":
    main()

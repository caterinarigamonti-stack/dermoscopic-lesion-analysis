#!/usr/bin/env python3
"""Audit image/mask alignment and produce a reusable dataset manifest."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_ID_PATTERN = re.compile(r"^\d{6}$")
TASK1_PATTERN = re.compile(r"^(?P<id>\d{6})_segmentation\.png$")
ATTRIBUTE_SUFFIXES = {
    "pigment_network": "attribute_pigment_network",
    "negative_network": "attribute_negative_network",
    "streaks": "attribute_streaks",
    "milia_like_cysts": "attribute_milia_like_cyst",
    "globules": "attribute_globules",
}


@dataclass(frozen=True)
class DatasetPaths:
    root: Path

    @property
    def images(self) -> Path:
        return self.root / "images"

    @property
    def task1(self) -> Path:
        return self.root / "task1_gt"

    @property
    def task2(self) -> Path:
        return self.root / "task2_gt"

    def image(self, image_id: str) -> Path:
        return self.images / f"{image_id}.jpg"

    def lesion_mask(self, image_id: str) -> Path:
        return self.task1 / f"{image_id}_segmentation.png"

    def attribute_mask(self, image_id: str, attribute: str) -> Path:
        return self.task2 / f"{image_id}_{ATTRIBUTE_SUFFIXES[attribute]}.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("artifacts"), type=Path)
    parser.add_argument(
        "--workers",
        default=min(8, (int(__import__("os").cpu_count() or 2))),
        type=int,
    )
    parser.add_argument("--image-stat-sample", default=256, type=int)
    parser.add_argument("--shards", default=1, type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--partial-output", type=Path)
    parser.add_argument("--merge-partials", nargs="+", type=Path)
    return parser.parse_args()


def quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def read_mask(path: Path) -> tuple[np.ndarray, str, set[int], int]:
    with Image.open(path) as image:
        mode = image.mode
        array = np.asarray(image)

    if array.ndim == 3:
        if array.shape[2] in (3, 4) and np.all(array[..., 0] == array[..., 1]):
            if array.shape[2] == 3 or np.all(array[..., 0] == array[..., 2]):
                array = array[..., 0]
            else:
                raise ValueError(f"mask has unequal colour channels: {array.shape}")
        else:
            raise ValueError(f"mask is not single-channel: {array.shape}")

    minimum = int(np.min(array))
    maximum = int(np.max(array))
    non_binary_pixels = int(np.count_nonzero((array != 0) & (array != 255)))
    if non_binary_pixels:
        unique = {int(value) for value in np.unique(array)}
    else:
        unique = {minimum, maximum}
    return array > 0, mode, unique, non_binary_pixels


def image_sample_stats(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((128, 128), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float64) / 255.0
    pixels = array.reshape(-1, 3)
    return {
        "sum": pixels.sum(axis=0).tolist(),
        "sum_squares": np.square(pixels).sum(axis=0).tolist(),
        "pixels": int(pixels.shape[0]),
    }


def audit_case(
    paths: DatasetPaths, image_id: str, calculate_image_stats: bool
) -> dict[str, Any]:
    image_path = paths.image(image_id)
    lesion_path = paths.lesion_mask(image_id)
    result: dict[str, Any] = {
        "image_id": image_id,
        "image_path": str(image_path.resolve()),
        "lesion_mask_path": str(lesion_path.resolve()),
        "errors": [],
        "attributes": {},
    }

    try:
        with Image.open(image_path) as image:
            result["image_size"] = list(image.size)
            result["image_mode"] = image.mode
            image.verify()
        if calculate_image_stats:
            result["image_sample_stats"] = image_sample_stats(image_path)
    except Exception as exc:  # Pillow raises several format-specific exceptions.
        result["errors"].append(f"image: {type(exc).__name__}: {exc}")
        return result

    try:
        lesion, lesion_mode, lesion_values, lesion_non_binary = read_mask(lesion_path)
        result["lesion_mask_mode"] = lesion_mode
        result["lesion_mask_values"] = sorted(lesion_values)
        result["lesion_non_binary_pixels"] = lesion_non_binary
        result["lesion_size"] = [int(lesion.shape[1]), int(lesion.shape[0])]
        result["lesion_pixels"] = int(np.count_nonzero(lesion))
        result["lesion_ratio"] = float(np.mean(lesion))
        if result["lesion_size"] != result["image_size"]:
            result["errors"].append(
                f"lesion size {result['lesion_size']} != image size {result['image_size']}"
            )
    except Exception as exc:
        result["errors"].append(f"lesion_mask: {type(exc).__name__}: {exc}")
        return result

    for attribute in ATTRIBUTE_SUFFIXES:
        mask_path = paths.attribute_mask(image_id, attribute)
        attribute_result: dict[str, Any] = {
            "path": str(mask_path.resolve()),
        }
        result["attributes"][attribute] = attribute_result
        try:
            mask, mode, values, non_binary = read_mask(mask_path)
            positive_pixels = int(np.count_nonzero(mask))
            outside_pixels = (
                int(np.count_nonzero(mask & ~lesion))
                if mask.shape == lesion.shape
                else None
            )
            attribute_result.update(
                {
                    "mode": mode,
                    "values": sorted(values),
                    "non_binary_pixels": non_binary,
                    "size": [int(mask.shape[1]), int(mask.shape[0])],
                    "present": int(positive_pixels > 0),
                    "positive_pixels": positive_pixels,
                    "positive_ratio": float(np.mean(mask)),
                    "outside_lesion_pixels": outside_pixels,
                }
            )
            if attribute_result["size"] != result["image_size"]:
                result["errors"].append(
                    f"{attribute} size {attribute_result['size']} "
                    f"!= image size {result['image_size']}"
                )
        except Exception as exc:
            result["errors"].append(
                f"{attribute}: {type(exc).__name__}: {exc}"
            )

    return result


def filename_alignment(paths: DatasetPaths) -> dict[str, Any]:
    image_ids = {
        path.stem for path in paths.images.glob("*.jpg") if path.is_file()
    }
    task1_ids = set()
    invalid_task1_names = []
    for path in paths.task1.glob("*.png"):
        match = TASK1_PATTERN.match(path.name)
        if match:
            task1_ids.add(match.group("id"))
        else:
            invalid_task1_names.append(path.name)

    attribute_ids: dict[str, set[str]] = {}
    invalid_task2_names: set[str] = set()
    for attribute, suffix in ATTRIBUTE_SUFFIXES.items():
        pattern = re.compile(rf"^(?P<id>\d{{6}})_{re.escape(suffix)}\.png$")
        ids = set()
        for path in paths.task2.glob(f"*_{suffix}.png"):
            match = pattern.match(path.name)
            if match:
                ids.add(match.group("id"))
            else:
                invalid_task2_names.add(path.name)
        attribute_ids[attribute] = ids

    valid_image_ids = {image_id for image_id in image_ids if IMAGE_ID_PATTERN.match(image_id)}
    invalid_image_names = sorted(image_ids - valid_image_ids)
    all_recognised_task2 = {
        f"{image_id}_{ATTRIBUTE_SUFFIXES[attribute]}.png"
        for attribute, ids in attribute_ids.items()
        for image_id in ids
    }
    all_task2_names = {path.name for path in paths.task2.glob("*.png")}
    invalid_task2_names.update(all_task2_names - all_recognised_task2)

    return {
        "image_ids": sorted(valid_image_ids),
        "counts": {
            "images": len(valid_image_ids),
            "task1_masks": len(task1_ids),
            **{
                f"task2_{attribute}_masks": len(ids)
                for attribute, ids in attribute_ids.items()
            },
        },
        "invalid_names": {
            "images": invalid_image_names,
            "task1": sorted(invalid_task1_names),
            "task2": sorted(invalid_task2_names),
        },
        "missing_for_images": {
            "task1": sorted(valid_image_ids - task1_ids),
            **{
                attribute: sorted(valid_image_ids - ids)
                for attribute, ids in attribute_ids.items()
            },
        },
        "orphan_masks": {
            "task1": sorted(task1_ids - valid_image_ids),
            **{
                attribute: sorted(ids - valid_image_ids)
                for attribute, ids in attribute_ids.items()
            },
        },
    }


def aggregate(
    paths: DatasetPaths,
    alignment: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    image_sizes = Counter()
    image_modes = Counter()
    lesion_modes = Counter()
    lesion_value_sets = Counter()
    lesion_ratios: list[float] = []
    lesion_empty_ids: list[str] = []
    error_cases: list[dict[str, Any]] = []
    image_sum = np.zeros(3, dtype=np.float64)
    image_sum_squares = np.zeros(3, dtype=np.float64)
    image_pixel_count = 0

    attribute_stats = {
        attribute: {
            "present_ids": [],
            "positive_ratios": [],
            "positive_ratios_when_present": [],
            "positive_pixels": 0,
            "outside_lesion_pixels": 0,
            "non_binary_pixels": 0,
            "modes": Counter(),
            "value_sets": Counter(),
        }
        for attribute in ATTRIBUTE_SUFFIXES
    }

    for row in rows:
        if row["errors"]:
            error_cases.append(
                {"image_id": row["image_id"], "errors": row["errors"]}
            )
        if "image_size" in row:
            image_sizes[f"{row['image_size'][0]}x{row['image_size'][1]}"] += 1
            image_modes[row["image_mode"]] += 1
        if "lesion_ratio" in row:
            lesion_ratios.append(row["lesion_ratio"])
            lesion_modes[row["lesion_mask_mode"]] += 1
            lesion_value_sets[str(row["lesion_mask_values"])] += 1
            if row["lesion_pixels"] == 0:
                lesion_empty_ids.append(row["image_id"])

        sample = row.get("image_sample_stats")
        if sample:
            image_sum += np.asarray(sample["sum"])
            image_sum_squares += np.asarray(sample["sum_squares"])
            image_pixel_count += sample["pixels"]

        for attribute, stats in attribute_stats.items():
            current = row["attributes"].get(attribute, {})
            if "present" not in current:
                continue
            if current["present"]:
                stats["present_ids"].append(row["image_id"])
                stats["positive_ratios_when_present"].append(
                    current["positive_ratio"]
                )
            stats["positive_ratios"].append(current["positive_ratio"])
            stats["positive_pixels"] += current["positive_pixels"]
            stats["outside_lesion_pixels"] += current["outside_lesion_pixels"] or 0
            stats["non_binary_pixels"] += current["non_binary_pixels"]
            stats["modes"][current["mode"]] += 1
            stats["value_sets"][str(current["values"])] += 1

    image_stats = None
    if image_pixel_count:
        mean = image_sum / image_pixel_count
        variance = np.maximum(
            (image_sum_squares / image_pixel_count) - np.square(mean), 0.0
        )
        image_stats = {
            "sampled_pixel_count": image_pixel_count,
            "rgb_mean_0_1": mean.tolist(),
            "rgb_std_0_1": np.sqrt(variance).tolist(),
        }

    total_cases = len(rows)
    attributes_summary = {}
    for attribute, stats in attribute_stats.items():
        positive_pixels = stats["positive_pixels"]
        attributes_summary[attribute] = {
            "present_count": len(stats["present_ids"]),
            "absent_count": total_cases - len(stats["present_ids"]),
            "prevalence": (
                len(stats["present_ids"]) / total_cases if total_cases else None
            ),
            "present_ids": stats["present_ids"],
            "positive_ratio_all_quantiles": quantiles(stats["positive_ratios"]),
            "positive_ratio_when_present_quantiles": quantiles(
                stats["positive_ratios_when_present"]
            ),
            "outside_lesion_pixel_fraction": (
                stats["outside_lesion_pixels"] / positive_pixels
                if positive_pixels
                else 0.0
            ),
            "total_non_binary_pixels": stats["non_binary_pixels"],
            "mask_modes": dict(sorted(stats["modes"].items())),
            "mask_value_sets": dict(sorted(stats["value_sets"].items())),
        }

    return {
        "data_root": str(paths.root.resolve()),
        "case_count": total_cases,
        "alignment": {key: value for key, value in alignment.items() if key != "image_ids"},
        "integrity": {
            "cases_with_errors": len(error_cases),
            "error_cases": error_cases,
            "empty_lesion_masks": len(lesion_empty_ids),
            "empty_lesion_mask_ids": lesion_empty_ids,
            "image_sizes": dict(sorted(image_sizes.items())),
            "image_modes": dict(sorted(image_modes.items())),
            "lesion_mask_modes": dict(sorted(lesion_modes.items())),
            "lesion_mask_value_sets": dict(sorted(lesion_value_sets.items())),
        },
        "image_statistics": image_stats,
        "lesion_area_ratio_quantiles": quantiles(lesion_ratios),
        "attributes": attributes_summary,
    }


def write_manifest(output_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["image_id", "image_path", "lesion_mask_path"]
    for attribute in ATTRIBUTE_SUFFIXES:
        fieldnames.extend(
            [
                f"{attribute}_mask_path",
                f"{attribute}_present",
                f"{attribute}_positive_ratio",
            ]
        )
    fieldnames.extend(["lesion_ratio", "qc_errors"])

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = {
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "lesion_mask_path": row["lesion_mask_path"],
                "lesion_ratio": row.get("lesion_ratio", ""),
                "qc_errors": " | ".join(row["errors"]),
            }
            for attribute in ATTRIBUTE_SUFFIXES:
                current = row["attributes"].get(attribute, {})
                output[f"{attribute}_mask_path"] = current.get("path", "")
                output[f"{attribute}_present"] = current.get("present", "")
                output[f"{attribute}_positive_ratio"] = current.get(
                    "positive_ratio", ""
                )
            writer.writerow(output)


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def write_markdown(output_path: Path, audit: dict[str, Any]) -> None:
    integrity = audit["integrity"]
    image_sizes = Counter(integrity["image_sizes"])
    common_sizes = ", ".join(
        f"{size}: {count}" for size, count in image_sizes.most_common(10)
    )
    lines = [
        "# Dataset audit",
        "",
        f"- Dataset: `{audit['data_root']}`",
        f"- Aligned image cases: {audit['case_count']}",
        f"- Cases with QC errors: {integrity['cases_with_errors']}",
        f"- Empty lesion masks: {integrity['empty_lesion_masks']}",
        f"- Unique image resolutions: {len(image_sizes)}",
        f"- Ten most common resolutions: {common_sizes}",
        f"- Lesion-mask values: {integrity['lesion_mask_value_sets']}",
        "",
        "## Attribute prevalence",
        "",
        "| Attribute | Present | Absent | Prevalence | Pixels outside lesion |",
        "|---|---:|---:|---:|---:|",
    ]
    for attribute, stats in audit["attributes"].items():
        lines.append(
            f"| {attribute} | {stats['present_count']} | {stats['absent_count']} "
            f"| {percent(stats['prevalence'])} "
            f"| {percent(stats['outside_lesion_pixel_fraction'])} |"
        )

    lines.extend(
        [
            "",
            "## Lesion area ratio",
            "",
            f"`{audit['lesion_area_ratio_quantiles']}`",
            "",
            "## Approximate image normalization statistics",
            "",
            f"`{audit['image_statistics']}`",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def validate_directories(paths: DatasetPaths) -> None:
    missing = [
        str(path)
        for path in (paths.images, paths.task1, paths.task2)
        if not path.is_dir()
    ]
    if missing:
        raise SystemExit(f"Missing dataset directories: {missing}")


def main() -> None:
    args = parse_args()
    paths = DatasetPaths(args.data_root.expanduser().resolve())
    validate_directories(paths)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    alignment = filename_alignment(paths)
    image_ids = alignment["image_ids"]
    random_generator = random.Random(20260727)
    sampled_ids = set(
        random_generator.sample(
            image_ids, min(args.image_stat_sample, len(image_ids))
        )
    )

    if args.merge_partials:
        rows = []
        for partial_path in args.merge_partials:
            rows.extend(
                json.loads(partial_path.expanduser().resolve().read_text(encoding="utf-8"))
            )
        rows.sort(key=lambda item: item["image_id"])
        if len(rows) != len(image_ids):
            raise SystemExit(
                f"Merged {len(rows)} rows but expected {len(image_ids)} cases"
            )
        if len({row["image_id"] for row in rows}) != len(rows):
            raise SystemExit("Duplicate image IDs found while merging partial audits")
    elif args.shard_index is not None:
        if args.shards < 1 or not 0 <= args.shard_index < args.shards:
            raise SystemExit("--shard-index must be between 0 and --shards - 1")
        if args.partial_output is None:
            raise SystemExit("--partial-output is required with --shard-index")
        shard_ids = image_ids[args.shard_index :: args.shards]
        print(
            f"Auditing shard {args.shard_index + 1}/{args.shards}: "
            f"{len(shard_ids)} cases...",
            flush=True,
        )
        rows = []
        for completed, image_id in enumerate(shard_ids, start=1):
            rows.append(audit_case(paths, image_id, image_id in sampled_ids))
            if completed % 50 == 0 or completed == len(shard_ids):
                print(f"  completed {completed}/{len(shard_ids)}", flush=True)
        partial_path = args.partial_output.expanduser().resolve()
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(json.dumps(rows), encoding="utf-8")
        print(f"Wrote {partial_path}", flush=True)
        return
    else:
        print(
            f"Auditing {len(image_ids)} cases with {args.workers} workers...",
            flush=True,
        )
        rows = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    audit_case, paths, image_id, image_id in sampled_ids
                ): image_id
                for image_id in image_ids
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if completed % 100 == 0 or completed == len(futures):
                    print(f"  completed {completed}/{len(futures)}", flush=True)
        rows.sort(key=lambda item: item["image_id"])

    audit = aggregate(paths, alignment, rows)

    manifest_path = output_dir / "manifest.csv"
    json_path = output_dir / "dataset_audit.json"
    markdown_path = output_dir / "DATASET_AUDIT.md"
    write_manifest(manifest_path, rows)
    json_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_markdown(markdown_path, audit)

    print(f"Wrote {manifest_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {markdown_path}", flush=True)


if __name__ == "__main__":
    main()

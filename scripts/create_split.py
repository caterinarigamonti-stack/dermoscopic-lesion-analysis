#!/usr/bin/env python3
"""Create a deterministic train/validation split from the audited manifest."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ATTRIBUTES = (
    "pigment_network",
    "negative_network",
    "streaks",
    "milia_like_cysts",
    "globules",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--output", default=Path("splits/train_val.csv"), type=Path
    )
    parser.add_argument("--report", default=Path("artifacts/SPLIT_REPORT.md"), type=Path)
    parser.add_argument("--val-fraction", default=0.2, type=float)
    parser.add_argument("--seed", default=20260727, type=int)
    return parser.parse_args()


def lesion_size_category(ratio: float) -> str:
    if ratio < 0.08:
        return "small"
    if ratio <= 0.25:
        return "moderate"
    return "large"


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("qc_errors"):
                raise ValueError(
                    f"Cannot split a manifest with QC errors ({row['image_id']})"
                )
            labels = tuple(
                int(row[f"{attribute}_present"]) for attribute in ATTRIBUTES
            )
            lesion_ratio = float(row["lesion_ratio"])
            rows.append(
                {
                    "image_id": row["image_id"],
                    "labels": labels,
                    "lesion_ratio": lesion_ratio,
                    "lesion_size": lesion_size_category(lesion_ratio),
                }
            )
    if not rows:
        raise ValueError("Manifest is empty")
    return rows


def allocate_validation_counts(
    stratum_sizes: dict[tuple[Any, ...], int],
    target: int,
    val_fraction: float,
) -> dict[tuple[Any, ...], int]:
    allocation: dict[tuple[Any, ...], int] = {}
    candidates: list[tuple[float, int, tuple[Any, ...]]] = []

    for key, size in stratum_sizes.items():
        exact = size * val_fraction
        initial = math.floor(exact)
        if size > 1:
            initial = min(initial, size - 1)
        else:
            initial = 0
        allocation[key] = initial
        candidates.append((exact - initial, size, key))

    remaining = target - sum(allocation.values())
    if remaining < 0:
        raise ValueError("Initial stratum allocation exceeds validation target")

    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    while remaining:
        changed = False
        for _, size, key in candidates:
            maximum = size - 1 if size > 1 else 0
            if allocation[key] < maximum:
                allocation[key] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            raise ValueError("Cannot reach requested validation size")

    return allocation


def create_split(
    rows: list[dict[str, Any]], val_fraction: float, seed: int
) -> dict[str, str]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    strata: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for row in rows:
        key = (*row["labels"], row["lesion_size"])
        strata[key].append(row["image_id"])

    target = round(len(rows) * val_fraction)
    allocation = allocate_validation_counts(
        {key: len(ids) for key, ids in strata.items()},
        target,
        val_fraction,
    )
    generator = random.Random(seed)
    split: dict[str, str] = {}
    for key in sorted(strata):
        ids = sorted(strata[key])
        generator.shuffle(ids)
        validation_count = allocation[key]
        for image_id in ids[:validation_count]:
            split[image_id] = "val"
        for image_id in ids[validation_count:]:
            split[image_id] = "train"

    if Counter(split.values())["val"] != target:
        raise AssertionError("Validation split has an unexpected size")
    return split


def prevalence(
    rows: list[dict[str, Any]], split: dict[str, str], subset: str
) -> dict[str, float]:
    selected = [row for row in rows if split[row["image_id"]] == subset]
    return {
        attribute: sum(row["labels"][index] for row in selected) / len(selected)
        for index, attribute in enumerate(ATTRIBUTES)
    }


def write_split(path: Path, split: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "split"])
        writer.writeheader()
        for image_id in sorted(split):
            writer.writerow({"image_id": image_id, "split": split[image_id]})


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    split: dict[str, str],
    seed: int,
    val_fraction: float,
) -> None:
    counts = Counter(split.values())
    train_prevalence = prevalence(rows, split, "train")
    val_prevalence = prevalence(rows, split, "val")
    size_counts = {
        subset: Counter(
            row["lesion_size"]
            for row in rows
            if split[row["image_id"]] == subset
        )
        for subset in ("train", "val")
    }
    lines = [
        "# Train/validation split",
        "",
        f"- Seed: `{seed}`",
        f"- Requested validation fraction: `{val_fraction}`",
        f"- Train cases: {counts['train']}",
        f"- Validation cases: {counts['val']}",
        "",
        "## Attribute prevalence",
        "",
        "| Attribute | Train | Validation | Absolute difference |",
        "|---|---:|---:|---:|",
    ]
    for attribute in ATTRIBUTES:
        train_value = train_prevalence[attribute]
        val_value = val_prevalence[attribute]
        lines.append(
            f"| {attribute} | {train_value:.3%} | {val_value:.3%} "
            f"| {abs(train_value - val_value):.3%} |"
        )
    lines.extend(
        [
            "",
            "## Lesion size categories",
            "",
            f"- Train: `{dict(size_counts['train'])}`",
            f"- Validation: `{dict(size_counts['val'])}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest.expanduser().resolve())
    split = create_split(rows, args.val_fraction, args.seed)
    write_split(args.output.expanduser().resolve(), split)
    write_report(
        args.report.expanduser().resolve(),
        rows,
        split,
        args.seed,
        args.val_fraction,
    )
    counts = Counter(split.values())
    print(f"Wrote {args.output}: {dict(counts)}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()

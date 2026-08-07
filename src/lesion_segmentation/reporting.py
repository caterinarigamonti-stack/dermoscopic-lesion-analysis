"""Controlled, evidence-anchored findings report generation."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .constants import ATTRIBUTES


DISPLAY_NAMES = {
    "pigment_network": "pigment network",
    "negative_network": "negative network",
    "streaks": "streaks",
    "milia_like_cysts": "milia-like cysts",
    "globules": "globules",
}
DISPLAY_VERBS = {
    "pigment_network": "is",
    "negative_network": "is",
    "streaks": "are",
    "milia_like_cysts": "are",
    "globules": "are",
}


def attribute_status(probability: float) -> str:
    if probability >= 0.60:
        return "present"
    if probability <= 0.40:
        return "absent"
    return "uncertain"


def lesion_area_ratio(mask: np.ndarray) -> float:
    binary = np.asarray(mask, dtype=bool)
    return float(binary.mean()) if binary.size else 0.0


def lesion_size_category(area_ratio: float) -> str:
    if area_ratio < 0.08:
        return "small"
    if area_ratio <= 0.25:
        return "moderate"
    return "large"


def border_irregularity(mask: np.ndarray) -> float:
    binary = np.asarray(mask, dtype=bool)
    area = int(binary.sum())
    if area == 0:
        return 0.0
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    vertical_edges = np.count_nonzero(padded[1:, :] != padded[:-1, :])
    horizontal_edges = np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    perimeter = vertical_edges + horizontal_edges
    return float((perimeter**2) / (4.0 * math.pi * area))


def border_category(irregularity: float) -> str:
    return "irregular" if irregularity >= 1.60 else "regular"


def build_findings_report(
    image_id: str,
    split: str,
    model_version: str,
    lesion_mask: np.ndarray,
    attribute_probabilities: Mapping[str, float],
) -> tuple[dict[str, object], str]:
    missing = set(ATTRIBUTES) - set(attribute_probabilities)
    if missing:
        raise ValueError(f"Missing attribute probabilities: {sorted(missing)}")

    area_ratio = lesion_area_ratio(lesion_mask)
    irregularity = border_irregularity(lesion_mask)
    presence = {}
    for attribute in ATTRIBUTES:
        probability = float(attribute_probabilities[attribute])
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{attribute} probability is outside [0, 1]")
        presence[attribute] = {
            "prob": probability,
            "status": attribute_status(probability),
        }

    payload: dict[str, object] = {
        "image_id": image_id,
        "split": split,
        "model_version": model_version,
        "attributes_order": list(ATTRIBUTES),
        "outputs": {
            "lesion": {
                "area_ratio": area_ratio,
                "size_category": lesion_size_category(area_ratio),
                "border_irregularity": irregularity,
                "border_category": border_category(irregularity),
            },
            "presence": presence,
        },
    }

    lesion_fields = payload["outputs"]["lesion"]  # type: ignore[index]
    first_sentence = (
        f"The lesion is {lesion_fields['size_category']} with "
        f"{lesion_fields['border_category']} borders."
    )
    findings = [
        f"{DISPLAY_NAMES[attribute]} {DISPLAY_VERBS[attribute]} "
        f"{presence[attribute]['status']}"
        for attribute in ATTRIBUTES
    ]
    findings[0] = findings[0].capitalize()
    report_text = f"{first_sentence} {'; '.join(findings)}."
    return payload, report_text


def validate_report_consistency(payload: dict[str, object], text: str) -> list[str]:
    errors = []
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        return ["outputs must be an object"]
    presence = outputs.get("presence")
    if not isinstance(presence, dict):
        return ["outputs.presence must be an object"]

    lower_text = text.lower()
    for attribute in ATTRIBUTES:
        display_name = DISPLAY_NAMES[attribute]
        if display_name not in lower_text:
            errors.append(f"missing required term: {display_name}")
            continue
        fields = presence.get(attribute)
        if not isinstance(fields, dict):
            errors.append(f"missing presence object: {attribute}")
            continue
        expected_phrase = (
            f"{display_name} {DISPLAY_VERBS[attribute]} {fields.get('status')}"
        )
        if expected_phrase not in lower_text:
            errors.append(f"status mismatch for {attribute}")
    return errors

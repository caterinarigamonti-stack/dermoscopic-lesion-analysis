"""Segmentation metrics with explicit empty-mask behaviour."""

from __future__ import annotations

import torch
from torch import Tensor


def batch_dice_iou(
    logits: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    prediction = logits.sigmoid().ge(threshold)
    target_binary = target.ge(0.5)
    if valid_mask is not None:
        valid = valid_mask.ge(0.5)
        if valid.shape[1] == 1 and prediction.shape[1] > 1:
            valid = valid.expand(-1, prediction.shape[1], -1, -1)
        prediction = prediction & valid
        target_binary = target_binary & valid

    dimensions = (2, 3)
    intersection = (prediction & target_binary).sum(dim=dimensions).float()
    prediction_sum = prediction.sum(dim=dimensions).float()
    target_sum = target_binary.sum(dim=dimensions).float()
    union = prediction_sum + target_sum - intersection
    dice = torch.where(
        prediction_sum + target_sum == 0,
        torch.ones_like(intersection),
        (2.0 * intersection) / (prediction_sum + target_sum).clamp_min(1.0),
    )
    iou = torch.where(
        union == 0,
        torch.ones_like(intersection),
        intersection / union.clamp_min(1.0),
    )
    return dice, iou

"""Losses for dense masks and auxiliary attribute-presence labels."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor


def _expand_valid_mask(valid_mask: Tensor | None, target: Tensor) -> Tensor:
    if valid_mask is None:
        return torch.ones_like(target)
    if valid_mask.shape[1] == 1 and target.shape[1] > 1:
        return valid_mask.expand(-1, target.shape[1], -1, -1)
    return valid_mask


def dice_loss(
    logits: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
    positive_only: bool = False,
) -> Tensor:
    probabilities = logits.sigmoid()
    valid = _expand_valid_mask(valid_mask, target)
    probabilities = probabilities * valid
    target = target * valid
    dimensions = (2, 3)
    intersection = (probabilities * target).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + target.sum(dim=dimensions)
    score = (2.0 * intersection + 1.0) / (denominator + 1.0)
    losses = 1.0 - score
    if positive_only:
        positive_channels = target.sum(dim=dimensions).gt(0)
        if not positive_channels.any():
            return losses.sum() * 0.0
        per_channel_losses = []
        for channel in range(target.shape[1]):
            channel_positive = positive_channels[:, channel]
            if channel_positive.any():
                per_channel_losses.append(
                    losses[channel_positive, channel].mean()
                )
        return torch.stack(per_channel_losses).mean()
    return losses.mean()


def focal_loss(
    logits: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> Tensor:
    valid = _expand_valid_mask(valid_mask, target)
    binary_cross_entropy = functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    probabilities = logits.sigmoid()
    probability_of_target = probabilities * target + (1 - probabilities) * (1 - target)
    alpha_factor = alpha * target + (1 - alpha) * (1 - target)
    loss = alpha_factor * (1 - probability_of_target).pow(gamma)
    loss = loss * binary_cross_entropy * valid
    return loss.sum() / valid.sum().clamp_min(1.0)


def lovasz_gradient(sorted_target: Tensor) -> Tensor:
    """Gradient of the Lovasz extension for a sorted binary target."""
    target_sum = sorted_target.sum()
    intersection = target_sum - sorted_target.cumsum(dim=0)
    union = target_sum + (1.0 - sorted_target).cumsum(dim=0)
    gradient = 1.0 - intersection / union.clamp_min(1.0)
    if sorted_target.numel() > 1:
        gradient[1:] = gradient[1:] - gradient[:-1]
    return gradient


def lovasz_hinge_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Binary Lovasz hinge, averaged per image and output channel."""
    if logits.shape != target.shape:
        raise ValueError(
            f"Lovasz inputs must have the same shape: {logits.shape} != {target.shape}"
        )
    losses = []
    for image_logits, image_target in zip(logits, target):
        for channel_logits, channel_target in zip(image_logits, image_target):
            flat_logits = channel_logits.reshape(-1)
            flat_target = channel_target.reshape(-1)
            signs = 2.0 * flat_target - 1.0
            errors = 1.0 - flat_logits * signs
            sorted_errors, permutation = torch.sort(errors, descending=True)
            sorted_target = flat_target[permutation]
            losses.append(
                torch.dot(
                    functional.relu(sorted_errors),
                    lovasz_gradient(sorted_target),
                )
            )
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def task_loss(
    task: str,
    mask_logits: Tensor,
    target_masks: Tensor,
    lesion_roi: Tensor,
    presence_logits: Tensor | None,
    target_presence: Tensor,
    presence_pos_weight: Tensor | None = None,
    task1_loss_name: str = "focal_dice",
) -> tuple[Tensor, dict[str, float]]:
    if task == "task1":
        dice = dice_loss(mask_logits, target_masks)
        if task1_loss_name == "focal_dice":
            overlap_partner = focal_loss(mask_logits, target_masks)
        elif task1_loss_name == "bce_dice":
            overlap_partner = functional.binary_cross_entropy_with_logits(
                mask_logits, target_masks
            )
        elif task1_loss_name == "lovasz_dice":
            overlap_partner = lovasz_hinge_loss(mask_logits, target_masks)
        else:
            raise ValueError(f"Unsupported Task 1 loss: {task1_loss_name}")
        dense_loss = 0.5 * overlap_partner + 0.5 * dice
        return dense_loss, {
            "dense_loss": float(dense_loss.detach()),
            "dice_loss": float(dice.detach()),
            f"{task1_loss_name.removesuffix('_dice')}_loss": float(
                overlap_partner.detach()
            ),
        }

    del lesion_roi  # Kept in the interface for later ROI-aware experiments.
    focal = focal_loss(mask_logits, target_masks)
    dice = dice_loss(mask_logits, target_masks, positive_only=True)
    if presence_logits is None:
        raise ValueError("Task 2 requires auxiliary presence logits")
    presence = functional.binary_cross_entropy_with_logits(
        presence_logits,
        target_presence,
        pos_weight=presence_pos_weight,
    )
    total = 0.5 * focal + 0.5 * dice + 0.5 * presence
    return total, {
        "focal_loss": float(focal.detach()),
        "dice_loss": float(dice.detach()),
        "presence_loss": float(presence.detach()),
    }

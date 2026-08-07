"""Model factory for lesion and dermoscopic-attribute segmentation."""

from __future__ import annotations

from typing import Literal

import segmentation_models_pytorch as smp
from torch import nn

from .constants import ATTRIBUTES


Architecture = Literal["unet", "unetplusplus", "deeplabv3plus"]


def create_model(
    task: Literal["task1", "task2"],
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    architecture: Architecture = "unet",
) -> nn.Module:
    if task == "task1":
        classes = 1
        auxiliary_head = None
    elif task == "task2":
        classes = len(ATTRIBUTES)
        auxiliary_head = {
            "classes": len(ATTRIBUTES),
            "pooling": "avg",
            "dropout": 0.2,
            "activation": None,
        }
    else:
        raise ValueError(f"Unknown task: {task}")

    architectures = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "deeplabv3plus": smp.DeepLabV3Plus,
    }
    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}")
    return architectures[architecture](
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=classes,
        aux_params=auxiliary_head,
    )

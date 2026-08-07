#!/usr/bin/env python3
"""Train a reproducible U-Net baseline for Task 1 or Task 2."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from lesion_segmentation.constants import ATTRIBUTES
from lesion_segmentation.data import DermoscopyDataset
from lesion_segmentation.losses import task_loss
from lesion_segmentation.metrics import batch_dice_iou
from lesion_segmentation.model import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("task1", "task2"))
    parser.add_argument("--manifest", default=Path("artifacts/manifest.csv"), type=Path)
    parser.add_argument(
        "--split", default=Path("splits/train_val.csv"), type=Path
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--architecture",
        default="unet",
        choices=("unet", "unetplusplus", "deeplabv3plus"),
    )
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--image-size", default=384, type=int)
    parser.add_argument(
        "--resize-mode",
        default="letterbox",
        choices=("stretch", "letterbox"),
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--gradient-accumulation", default=1, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--learning-rate", default=3e-4, type=float)
    parser.add_argument("--encoder-learning-rate", type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--patience", default=7, type=int)
    parser.add_argument(
        "--freeze-encoder-epochs",
        default=0,
        type=int,
        help="Keep the pretrained encoder frozen for the first N epochs.",
    )
    parser.add_argument("--lr-scheduler-patience", default=2, type=int)
    parser.add_argument("--lr-scheduler-factor", default=0.5, type=float)
    parser.add_argument("--min-learning-rate", default=1e-6, type=float)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--seed", default=20260727, type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument(
        "--init-encoder-checkpoint",
        type=Path,
        help="Initialize only the encoder, allowing a different decoder architecture.",
    )
    parser.add_argument("--freeze-batchnorm", action="store_true")
    parser.add_argument(
        "--task1-loss",
        default="focal_dice",
        choices=("focal_dice", "bce_dice", "lovasz_dice"),
        help="Dense loss used for Task 1; ignored for Task 2.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def unpack_output(output: Tensor | tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor | None]:
    if isinstance(output, tuple):
        return output[0], output[1]
    return output, None


def freeze_batchnorm_running_statistics(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            module.eval()


def set_encoder_trainable(model: torch.nn.Module, trainable: bool) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(trainable)


class MetricAccumulator:
    def __init__(self, channels: int) -> None:
        self.channels = channels
        self.loss_sum = 0.0
        self.samples = 0
        self.dice_sum = torch.zeros(channels, dtype=torch.float64)
        self.iou_sum = torch.zeros(channels, dtype=torch.float64)
        self.all_count = torch.zeros(channels, dtype=torch.float64)
        self.positive_dice_sum = torch.zeros(channels, dtype=torch.float64)
        self.positive_iou_sum = torch.zeros(channels, dtype=torch.float64)
        self.positive_count = torch.zeros(channels, dtype=torch.float64)
        self.loss_components: dict[str, float] = defaultdict(float)

    def update(
        self,
        loss: Tensor,
        batch_size: int,
        dice: Tensor,
        iou: Tensor,
        target_presence: Tensor,
        loss_components: dict[str, float],
    ) -> None:
        self.loss_sum += float(loss.detach()) * batch_size
        self.samples += batch_size
        dice_cpu = dice.detach().cpu().double()
        iou_cpu = iou.detach().cpu().double()
        self.dice_sum += dice_cpu.sum(dim=0)
        self.iou_sum += iou_cpu.sum(dim=0)
        self.all_count += batch_size
        positive = target_presence.detach().bool().cpu()
        self.positive_dice_sum += (dice_cpu * positive).sum(dim=0)
        self.positive_iou_sum += (iou_cpu * positive).sum(dim=0)
        self.positive_count += positive.sum(dim=0)
        for name, value in loss_components.items():
            self.loss_components[name] += value * batch_size

    def compute(self, channel_names: tuple[str, ...]) -> dict[str, float]:
        metrics = {"loss": self.loss_sum / max(self.samples, 1)}
        all_dice = self.dice_sum / self.all_count.clamp_min(1)
        all_iou = self.iou_sum / self.all_count.clamp_min(1)
        positive_dice = self.positive_dice_sum / self.positive_count.clamp_min(1)
        positive_iou = self.positive_iou_sum / self.positive_count.clamp_min(1)
        for index, name in enumerate(channel_names):
            metrics[f"dice/{name}"] = float(all_dice[index])
            metrics[f"iou/{name}"] = float(all_iou[index])
            metrics[f"dice_positive/{name}"] = float(positive_dice[index])
            metrics[f"iou_positive/{name}"] = float(positive_iou[index])
        metrics["mean_dice"] = float(all_dice.mean())
        metrics["mean_dice_positive"] = float(positive_dice.mean())
        for name, value in self.loss_components.items():
            metrics[name] = value / max(self.samples, 1)
        return metrics


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    task: str,
    device: torch.device,
    optimizer: AdamW | None,
    presence_pos_weight: Tensor,
    smoke_test: bool,
    gradient_accumulation: int = 1,
    freeze_batchnorm: bool = False,
    freeze_encoder: bool = False,
    task1_loss_name: str = "focal_dice",
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if training and freeze_encoder:
        model.encoder.eval()
    if training and freeze_batchnorm:
        freeze_batchnorm_running_statistics(model)
    channel_names = ("lesion",) if task == "task1" else ATTRIBUTES
    accumulator = MetricAccumulator(len(channel_names))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    progress = tqdm(
        loader,
        desc="train" if training else "val",
        leave=False,
        mininterval=5.0,
    )
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(progress):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        lesion_roi = batch["lesion_roi"].to(device)
        target_presence = batch["presence"].to(device)
        gradient_context = torch.enable_grad() if training else torch.no_grad()
        autocast_context = (
            torch.amp.autocast("cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with gradient_context, autocast_context:
            mask_logits, presence_logits = unpack_output(model(images))
            loss, components = task_loss(
                task,
                mask_logits,
                masks,
                lesion_roi,
                presence_logits,
                target_presence,
                presence_pos_weight=presence_pos_weight,
                task1_loss_name=task1_loss_name,
            )

        if training:
            scaler.scale(loss / gradient_accumulation).backward()
            stopping_after_batch = smoke_test and batch_index >= 1
            update_step = (
                (batch_index + 1) % gradient_accumulation == 0
                or batch_index + 1 == len(loader)
                or stopping_after_batch
            )
            if update_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        dice, iou = batch_dice_iou(mask_logits, masks)
        accumulator.update(
            loss,
            images.shape[0],
            dice,
            iou,
            target_presence,
            components,
        )
        progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        if smoke_test and batch_index >= 1:
            break

    return accumulator.compute(channel_names)


def main() -> None:
    args = parse_args()
    if args.gradient_accumulation < 1:
        raise ValueError("--gradient-accumulation must be at least 1")
    if args.freeze_encoder_epochs < 0:
        raise ValueError("--freeze-encoder-epochs cannot be negative")
    if args.lr_scheduler_patience < 0:
        raise ValueError("--lr-scheduler-patience cannot be negative")
    if not 0.0 < args.lr_scheduler_factor < 1.0:
        raise ValueError("--lr-scheduler-factor must be between 0 and 1")
    if args.min_learning_rate <= 0.0:
        raise ValueError("--min-learning-rate must be positive")
    initialization_options = (
        args.resume,
        args.init_checkpoint,
        args.init_encoder_checkpoint,
    )
    if sum(option is not None for option in initialization_options) > 1:
        raise ValueError(
            "--resume, --init-checkpoint, and --init-encoder-checkpoint "
            "are mutually exclusive"
        )
    seed_everything(args.seed)
    device = select_device(args.device)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.resume.parent
        if args.resume is not None
        else Path("checkpoints") / args.task
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = DermoscopyDataset(
        args.manifest.expanduser().resolve(),
        args.split.expanduser().resolve(),
        subset="train",
        task=args.task,
        image_size=args.image_size,
        augment=True,
        resize_mode=args.resize_mode,
        cache_root=args.cache_root,
    )
    validation_dataset = DermoscopyDataset(
        args.manifest.expanduser().resolve(),
        args.split.expanduser().resolve(),
        subset="val",
        task=args.task,
        image_size=args.image_size,
        augment=False,
        resize_mode=args.resize_mode,
        cache_root=args.cache_root,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_options: dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, **loader_options
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, drop_last=False, **loader_options
    )

    encoder_weights = (
        None
        if args.resume is not None
        or args.init_checkpoint is not None
        or args.init_encoder_checkpoint is not None
        or args.encoder_weights.lower() == "none"
        else args.encoder_weights
    )
    model = create_model(
        args.task,
        args.encoder,
        encoder_weights,
        architecture=args.architecture,
    ).to(device)
    if args.init_checkpoint is not None:
        initial_checkpoint = torch.load(
            args.init_checkpoint.expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        initial_config = initial_checkpoint["config"]
        for key in ("task", "encoder", "architecture"):
            actual = initial_config.get(
                key,
                "unet" if key == "architecture" else None,
            )
            if str(actual) != str(getattr(args, key)):
                raise ValueError(
                    f"Initialization mismatch for {key}: "
                    f"checkpoint={actual}, current={getattr(args, key)}"
                )
        model.load_state_dict(initial_checkpoint["model_state_dict"])
    if args.init_encoder_checkpoint is not None:
        encoder_checkpoint = torch.load(
            args.init_encoder_checkpoint.expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        encoder_config = encoder_checkpoint["config"]
        if str(encoder_config["encoder"]) != str(args.encoder):
            raise ValueError(
                "Encoder initialization mismatch: "
                f"checkpoint={encoder_config['encoder']}, current={args.encoder}"
            )
        encoder_state = {
            key.removeprefix("encoder."): value
            for key, value in encoder_checkpoint["model_state_dict"].items()
            if key.startswith("encoder.")
        }
        model.encoder.load_state_dict(encoder_state)

    if args.encoder_learning_rate is None:
        parameter_groups = [{"params": model.parameters(), "lr": args.learning_rate}]
    else:
        encoder_parameter_ids = {id(parameter) for parameter in model.encoder.parameters()}
        decoder_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in encoder_parameter_ids
        ]
        parameter_groups = [
            {
                "params": model.encoder.parameters(),
                "lr": args.encoder_learning_rate,
            },
            {"params": decoder_parameters, "lr": args.learning_rate},
        ]
    optimizer = AdamW(parameter_groups, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        min_lr=args.min_learning_rate,
    )
    presence_pos_weight = train_dataset.presence_pos_weight.to(device)
    start_epoch = 1
    best_metric = -1.0
    if args.resume is not None:
        resume_checkpoint = torch.load(
            args.resume.expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        resume_config = resume_checkpoint["config"]
        for key in (
            "task",
            "architecture",
            "encoder",
            "image_size",
            "resize_mode",
        ):
            expected = str(getattr(args, key.replace("image_size", "image_size")))
            actual = str(
                resume_config.get(
                    key,
                    "stretch"
                    if key == "resize_mode"
                    else "unet"
                    if key == "architecture"
                    else None,
                )
            )
            if actual != expected:
                raise ValueError(
                    f"Resume mismatch for {key}: checkpoint={actual}, current={expected}"
                )
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        previous_metrics = resume_checkpoint.get("validation_metrics", {})
        metric_name = "mean_dice" if args.task == "task1" else "mean_dice_positive"
        best_metric = float(
            resume_checkpoint.get(
                "best_metric",
                previous_metrics.get(metric_name, -1.0),
            )
        )

    run_config = vars(args).copy()
    run_config.update(
        {
            "device_resolved": str(device),
            "train_cases": len(train_dataset),
            "validation_cases": len(validation_dataset),
            "presence_pos_weight": presence_pos_weight.detach().cpu().tolist(),
        }
    )
    for key, value in list(run_config.items()):
        if isinstance(value, Path):
            run_config[key] = str(value)
    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    epochs_without_improvement = 0
    history_path = output_dir / "history.jsonl"
    print(
        f"Training {args.task} on {device}: "
        f"{len(train_dataset)} train / {len(validation_dataset)} val"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        freeze_encoder = epoch <= args.freeze_encoder_epochs
        set_encoder_trainable(model, not freeze_encoder)
        train_metrics = run_epoch(
            model,
            train_loader,
            args.task,
            device,
            optimizer,
            presence_pos_weight,
            args.smoke_test,
            gradient_accumulation=args.gradient_accumulation,
            freeze_batchnorm=args.freeze_batchnorm,
            freeze_encoder=freeze_encoder,
            task1_loss_name=args.task1_loss,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            args.task,
            device,
            None,
            presence_pos_weight,
            args.smoke_test,
            gradient_accumulation=1,
            freeze_batchnorm=False,
            task1_loss_name=args.task1_loss,
        )
        monitored_metric = (
            validation_metrics["mean_dice"]
            if args.task == "task1"
            else validation_metrics["mean_dice_positive"]
        )
        scheduler.step(monitored_metric)
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "learning_rates": [
                parameter_group["lr"] for parameter_group in optimizer.param_groups
            ],
            "encoder_frozen": freeze_encoder,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"epoch={epoch:02d} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"monitor={monitored_metric:.4f}"
        )

        if monitored_metric > best_metric:
            best_metric = monitored_metric
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_metrics": validation_metrics,
                    "config": run_config,
                    "best_metric": monitored_metric,
                },
                output_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "validation_metrics": validation_metrics,
                "config": run_config,
                "best_metric": best_metric,
            },
            output_dir / "last.pt",
        )
        if args.smoke_test or epochs_without_improvement >= args.patience:
            break

    print(f"Best monitored Dice: {best_metric:.4f}")
    print(f"Checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

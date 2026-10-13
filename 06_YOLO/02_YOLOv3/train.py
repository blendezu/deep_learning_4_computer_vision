"""Main training script for YOLOv3 on Pascal VOC.

Orchestrates:
- Phase 1: Backbone frozen (warmup + training detection heads only)
- Phase 2: Backbone transfer learning only for stage_3 and stage_4 (differential learning rates + cosine annealing)
- Multi-scale training
- Mixed precision (AMP)
- Validation loss and periodic mAP calculation
- Checkpoint management with Early Stopping
- Training curve visualization
"""

import argparse
import gc
import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from config import (
    ANCHORS,
    BATCH_SIZE,
    DEFAULT_CHECKPOINT,
    DEVICE,
    EPOCHS,
    FREEZE_BACKBONE_EPOCHS,
    IGNORE_IOU_THRESHOLD,
    IMG_SIZE,
    LR,
    MULTI_SCALES,
    NUM_CLASSES,
    PATIENCE,
    SCALE_INTERVAL,
    SEED,
    STRIDES,
    TRAINING_CURVES_PATH,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from dataset import download_dataset, get_dataloaders
from loss import YOLOv3Loss
from model import YOLOv3
from torch import nn, optim
from tqdm.auto import tqdm
from utils import EarlyStopping, compute_map, prepare_targets, set_seed


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    device: torch.device = DEVICE,
    multi_scale: bool = True,
) -> float:
    """Train the model for one epoch."""
    model.train()
    total_loss, count = 0.0, 0
    current_img_size = IMG_SIZE
    loop = tqdm(loader, desc=f"Epoch {epoch + 1:03d} Train", leave=False)

    for batch_idx, (images, gt_boxes, gt_labels) in enumerate(loop):
        images = images.to(device)

        # Multi-scale training: sample random resolution periodically
        if multi_scale and (batch_idx % SCALE_INTERVAL == 0):
            current_img_size = random.choice(MULTI_SCALES)

        if images.shape[-1] != current_img_size:
            images = F.interpolate(
                images,
                size=(current_img_size, current_img_size),
                mode="bilinear",
                align_corners=False,
            )

        targets = prepare_targets(
            gt_boxes,
            gt_labels,
            anchors=ANCHORS,
            strides=STRIDES,
            img_size=current_img_size,
            num_classes=NUM_CLASSES,
            ignore_thresh=IGNORE_IOU_THRESHOLD,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            predictions = model(images)

        preds_float = [p.float() for p in predictions]
        loss = criterion(preds_float, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        count += 1
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(count, 1)


def validate_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    epoch: int,
    device: torch.device = DEVICE,
) -> float:
    """Evaluate loss on validation set for one epoch."""
    model.eval()
    total_loss, count = 0.0, 0
    loop = tqdm(loader, desc=f"Epoch {epoch + 1:03d} Val", leave=False)

    with torch.no_grad():
        for images, gt_boxes, gt_labels in loop:
            images = images.to(device)
            targets = prepare_targets(
                gt_boxes,
                gt_labels,
                anchors=ANCHORS,
                strides=STRIDES,
                img_size=IMG_SIZE,
                num_classes=NUM_CLASSES,
                ignore_thresh=IGNORE_IOU_THRESHOLD,
                device=device,
            )

            with torch.autocast(
                device_type=device.type, enabled=(device.type == "cuda")
            ):
                predictions = model(images)

            preds_float = [p.float() for p in predictions]
            loss = criterion(preds_float, targets)

            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_maps: list[float],
    save_path: str = TRAINING_CURVES_PATH,
):
    """Plot and save training/validation loss curves and mAP evolution."""
    _, axes = plt.subplots(1, 2, figsize=(16, 6))
    epochs_range = range(1, len(train_losses) + 1)

    # 1. Loss curves
    axes[0].plot(epochs_range, train_losses, label="Train Loss", linewidth=2)
    axes[0].plot(epochs_range, val_losses, label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("YOLOv3 Loss Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. mAP curve
    map_epochs = [i + 1 for i, m in enumerate(val_maps) if m > 0]
    map_values = [m for m in val_maps if m > 0]
    if len(map_values) > 0:
        axes[1].plot(
            map_epochs, map_values, "o-", label="mAP@0.5", linewidth=2, color="green"
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP@0.5")
    axes[1].set_title("YOLOv3 Validation mAP@0.5")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Training curves saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv3 on Pascal VOC")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to dataset root containing train/ and val/ dirs",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LR, help="Base learning rate")
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=FREEZE_BACKBONE_EPOCHS,
        help="Epochs to freeze backbone",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=WARMUP_EPOCHS,
        help="Epochs for linear LR warmup",
    )
    parser.add_argument(
        "--patience", type=int, default=PATIENCE, help="Early stopping patience"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint save path",
    )
    parser.add_argument(
        "--no-multiscale",
        action="store_true",
        help="Disable multi-scale image size randomization",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="Evaluate mAP every N epochs",
    )
    args = parser.parse_args()

    set_seed(SEED)
    print(f"Training on device: {DEVICE}")

    # 1. Dataset setup
    if args.data_root is None:
        data_root = download_dataset()
    else:
        data_root = args.data_root

    print(f"Loading data from: {data_root}")
    train_dataset, val_dataset, train_loader, val_loader = get_dataloaders(
        data_root=data_root, batch_size=args.batch_size, img_size=IMG_SIZE
    )
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # 2. Model and Criterion setup
    model = YOLOv3(num_anchor_per_scale=3, num_classes=NUM_CLASSES).to(DEVICE)
    total_params, trainable_params = model.get_parameter_counts()
    print(
        f"Total parameters: {total_params:.2f}M | Initial trainable: {trainable_params:.2f}M"
    )

    criterion = YOLOv3Loss().to(DEVICE)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=(DEVICE.type == "cuda"))
    early_stop = EarlyStopping(patience=args.patience, save_path=args.checkpoint)

    # -------------------------------------------------------------------------
    # Phase 1: Freeze Backbone & Initialize Optimizer + Warmup
    # -------------------------------------------------------------------------
    model.freeze_backbone(freeze=True)
    trainable_head = [p for p in model.parameters() if p.requires_grad]
    print(
        f"Phase 1: Training {sum(p.numel() for p in trainable_head) / 1e6:.2f}M head params "
        f"(backbone frozen for {args.freeze_epochs} epochs)"
    )

    optimizer = optim.AdamW(trainable_head, lr=args.lr, weight_decay=WEIGHT_DECAY)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=args.warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs]
    )

    train_losses, val_losses, val_maps = [], [], []

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    for epoch in range(args.epochs):
        # Phase 2 Transition: Unfreeze backbone with differential learning rates
        if epoch == args.freeze_epochs:
            model.freeze_backbone(freeze=False, unfreeze_stages=(3, 4))

            backbone_trainable = [
                p for p in model.backbone.parameters() if p.requires_grad
            ]

            optimizer = optim.AdamW(
                [
                    {"params": backbone_trainable, "lr": args.lr * 0.1},
                    {"params": model.det_block_0.parameters(), "lr": args.lr},
                    {"params": model.det_block_1.parameters(), "lr": args.lr},
                    {"params": model.det_block_2.parameters(), "lr": args.lr},
                    {"params": model.merge_conv_0.parameters(), "lr": args.lr},
                    {"params": model.merge_conv_1.parameters(), "lr": args.lr},
                ],
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=(args.epochs - args.freeze_epochs), eta_min=1e-6
            )
            _, total_trainable = model.get_parameter_counts()
            print(
                f"\nPhase 2: Backbone unfrozen — {total_trainable:.2f}M trainable parameters "
                f"(backbone LR: {args.lr * 0.1:.1e}, head LR: {args.lr:.1e})"
            )

        # Train one epoch
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            device=DEVICE,
            multi_scale=not args.no_multiscale,
        )

        # Validate one epoch
        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            epoch=epoch,
            device=DEVICE,
        )

        # Periodic mAP computation
        val_map = 0.0
        if ((epoch + 1) % args.eval_interval == 0) or (epoch == args.freeze_epochs - 1):
            model.zero_grad(set_to_none=True)
            gc.collect()
            val_map = compute_map(
                model=model,
                data_loader=val_loader,
                num_classes=NUM_CLASSES,
                anchors=ANCHORS,
                strides=STRIDES,
                img_size=IMG_SIZE,
                device=DEVICE,
            )

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_maps.append(val_map)

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"mAP@0.5: {val_map:.4f}"
        )

        # Early Stopping check
        early_stop(model, val_loss)
        if early_stop.early_stop:
            print("Early stopping triggered. Halting training.")
            break

    # Save training curves
    plot_training_curves(train_losses, val_losses, val_maps)

    # Load best checkpoint
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    else:
        model.load_state_dict(
            torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        )
    print(f"Best model weights successfully loaded from '{args.checkpoint}'.")


if __name__ == "__main__":
    main()

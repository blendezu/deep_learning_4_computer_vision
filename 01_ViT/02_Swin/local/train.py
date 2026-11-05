"""
Training pipeline for Swin Transformer V2 from scratch on Food-101.
Includes AMP (Mixed Precision), OneCycleLR, CutMix/MixUp, and Early Stopping.
"""

import argparse
from datetime import datetime
from pathlib import Path

import config
import torch
from dataset import apply_mixup_cutmix, get_dataloaders
from model import build_swin_v2
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from utils import EarlyStopping, plot_training_history, set_seed


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    use_amp: bool = True,
    mixup_p: float = config.MIXUP_P,
    mixup_alpha: float = config.MIXUP_ALPHA,
    cutmix_prob: float = config.CUTMIX_PROB,
    writer: SummaryWriter | None = None,
    epoch: int = 0,
    log_interval: int = 50,
) -> tuple[float, float]:
    """Runs a single training epoch with CutMix/MixUp and AMP."""
    model.train()
    train_loss, train_acc = 0.0, 0.0
    loop = tqdm(loader, desc="Training", leave=False)

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Batch-level CutMix or MixUp
        x, y_a, y_b, lam, use_mix = apply_mixup_cutmix(
            x, y, p=mixup_p, alpha=mixup_alpha, cutmix_prob=cutmix_prob
        )

        # Forward pass with AMP autocast
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(x)
            if use_mix:
                loss = lam * criterion(out, y_a) + (1.0 - lam) * criterion(out, y_b)
            else:
                loss = criterion(out, y)

        # Backward pass with scaled gradients
        scaler.scale(loss).backward()

        # Unscale and clip gradients
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)

        # Optimizer & Scaler step
        scaler.step(optimizer)
        scaler.update()

        # Step scheduler per batch (OneCycleLR requirement)
        scheduler.step()

        # Metrics aggregation
        batch_size = x.size(0)
        train_loss += loss.detach().item() * batch_size

        preds = out.argmax(dim=1)
        if use_mix:
            train_acc += (
                (lam * (preds == y_a).float() + (1.0 - lam) * (preds == y_b).float())
                .sum()
                .item()
            )
        else:
            train_acc += (preds == y).sum().item()

        current_lr = optimizer.param_groups[0]["lr"]

        # Batch-level TensorBoard logging
        if writer is not None and (
            (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(loader)
        ):
            global_step = epoch * len(loader) + batch_idx + 1
            writer.add_scalar("Train/Batch_Loss", loss.item(), global_step)
            writer.add_scalar("Train/Batch_LR", current_lr, global_step)

        loop.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{current_lr:.2e}",
        )

    total_samples = len(loader.dataset)
    return train_loss / total_samples, train_acc / total_samples


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[float, float]:
    """Runs validation on the validation set."""
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    loop = tqdm(loader, desc="Validation", leave=False)

    with torch.no_grad():
        for x, y in loop:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)

            batch_size = x.size(0)
            val_loss += loss.detach().item() * batch_size
            val_acc += (out.argmax(dim=1) == y).sum().item()

    total_samples = len(loader.dataset)
    return val_loss / total_samples, val_acc / total_samples


def test(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Evaluates the model on the unseen test set."""
    model.eval()
    test_acc = 0.0
    loop = tqdm(loader, desc="Testing", leave=False)

    with torch.no_grad():
        for x, y in loop:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            test_acc += (out.argmax(dim=1) == y).sum().item()

    return test_acc / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(
        description="Train Swin Transformer V2 from scratch"
    )
    parser.add_argument(
        "--epochs", type=int, default=config.EPOCHS, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.LR,
        help="Maximum learning rate for OneCycleLR",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=config.WEIGHT_DECAY,
        help="Weight decay for AdamW",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=config.MODEL_NAME,
        choices=["Swin-T", "Swin-S", "Swin-B", "Swin-L"],
        help="Model variant",
    )
    parser.add_argument(
        "--img-size", type=int, default=config.IMG_SIZE, help="Image resolution"
    )
    parser.add_argument(
        "--seed", type=int, default=config.SEED, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(config.DATA_PATH),
        help="Directory to store/load dataset",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=config.NUM_WORKERS,
        help="DataLoader num_workers",
    )
    parser.add_argument(
        "--patience", type=int, default=config.PATIENCE, help="Early stopping patience"
    )
    parser.add_argument(
        "--delta", type=float, default=config.DELTA, help="Early stopping delta"
    )
    parser.add_argument(
        "--no-amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(config.BEST_CHECKPOINT_PATH),
        help="Path to save best checkpoint",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=str(config.FIGURES_DIR / "training_history.png"),
        help="Path to save learning curves",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable strict deterministic algorithms (slower, disables cuDNN autotuner)",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard tracking",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(config.BASE_DIR / "runs"),
        help="Root directory for TensorBoard logs",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run name / directory for TensorBoard logs",
    )
    parser.add_argument(
        "--tb-log-interval",
        type=int,
        default=50,
        help="Batch interval for TensorBoard batch-level logging",
    )
    args = parser.parse_args()

    # 1. Hardware & Seed Setup
    set_seed(args.seed, deterministic=args.deterministic)
    device = config.get_device()
    use_amp = (not args.no_amp) and (device.type == "cuda")
    use_tensorboard = not args.no_tensorboard

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    print(
        f"Device: {device} | Mixed Precision (AMP): {use_amp} | TensorBoard: {use_tensorboard}"
    )
    print(
        f"Model: {args.model_name} | Epochs: {args.epochs} | Batch size: {args.batch_size} | Max LR: {args.lr}"
    )

    # 2. Data Loading
    print("Loading Food-101 dataset...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        img_size=args.img_size,
    )
    print(
        f"Dataset ready. Classes: {len(classes)} | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}"
    )

    # 3. Model Initialization
    model = build_swin_v2(
        model_name=args.model_name,
        img_size=args.img_size,
        in_chans=config.IN_CHANNELS,
        num_classes=len(classes),
        drop_path_rate=config.DROP_PATH_RATE,
        window_size=config.WINDOW_SIZE,
        mlp_ratio=config.MLP_RATIO,
        qkv_bias=config.QKV_BIAS,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # 4. Criterion, Optimizer, Scheduler, Scaler
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=config.PCT_START,
        div_factor=config.DIV_FACTOR,
        final_div_factor=config.FINAL_DIV_FACTOR,
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)
    early_stopping = EarlyStopping(
        patience=args.patience,
        delta=args.delta,
        verbose=True,
        save_path=args.checkpoint_path,
    )

    # TensorBoard setup
    if use_tensorboard:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = (
            args.run_name
            or f"{args.model_name}_bs{args.batch_size}_lr{args.lr}_{timestamp}"
        )
        run_dir = Path(args.log_dir) / run_name
        writer = SummaryWriter(log_dir=str(run_dir))
        print(f"TensorBoard logging to: {run_dir}")
    else:
        writer = None

    # 5. Training Loop
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    try:
        print("\nStarting training loop...")
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                device=device,
                use_amp=use_amp,
                writer=writer,
                epoch=epoch,
                log_interval=args.tb_log_interval,
            )
            val_loss, val_acc = validate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch + 1:02d}/{args.epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
            )

            # Epoch-level TensorBoard logging
            if writer is not None:
                writer.add_scalar("Loss/Train", train_loss, epoch + 1)
                writer.add_scalar("Loss/Val", val_loss, epoch + 1)
                writer.add_scalar("Accuracy/Train", train_acc, epoch + 1)
                writer.add_scalar("Accuracy/Val", val_acc, epoch + 1)
                writer.add_scalar("Learning_Rate/Epoch", current_lr, epoch + 1)

            early_stopping(model, val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered. Training stopped.")
                break

        # 6. Evaluation of Best Checkpoint
        checkpoint_file = Path(args.checkpoint_path)
        if checkpoint_file.exists():
            print(
                f"\nLoading best checkpoint from {checkpoint_file} for final test evaluation..."
            )
            best_state = torch.load(checkpoint_file, map_location=device)
            model_to_load = model.module if hasattr(model, "module") else model
            model_to_load.load_state_dict(best_state)

        test_acc = test(model=model, loader=test_loader, device=device)
        print(f"\n>>> Final Test Accuracy: {test_acc:.4f} <<<")

        if writer is not None:
            writer.add_scalar("Accuracy/Test", test_acc, len(train_losses))

        # 7. Save Plots
        plot_training_history(
            train_losses,
            val_losses,
            train_accuracies,
            val_accuracies,
            save_path=args.save_plot,
        )

    finally:
        if writer is not None:
            writer.flush()
            writer.close()


if __name__ == "__main__":
    main()

# & ./.DLO/Scripts/python.exe 01_ViT/02_Swin/local/train.py --batch-size 64 --num-workers 4

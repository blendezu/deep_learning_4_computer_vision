"""
Utility functions: Seeding, Early Stopping, and Metric Visualization.
"""

import os
import random
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch with high-performance CUDA settings."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        # Fast kernel autotuning for static input sizes (224x224)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """Initializes DataLoader worker seed reproducibly from generator seed."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class EarlyStopping:
    """
    Monitors validation loss and stops training if no improvement after `patience` epochs.
    Also persists the best model state_dict.
    """

    def __init__(
        self,
        patience: int = 10,
        delta: float = 0.0,
        verbose: bool = True,
        save_path: str = "best_checkpoint.pth",
    ):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        self.early_stop = False
        self.counter = 0
        self.best_loss = None

    def __call__(self, model: nn.Module, val_loss: float) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, val_loss)
        elif val_loss >= self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping: {self.counter}/{self.patience} (Best: {self.best_loss:.4f}, Current: {val_loss:.4f})"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(model, val_loss)
            self.best_loss = val_loss
            self.counter = 0

    def save_checkpoint(self, model: nn.Module, val_loss: float) -> None:
        if self.verbose:
            old_loss_str = (
                f"{self.best_loss:.4f}" if self.best_loss is not None else "N/A"
            )
            print(
                f"Validation loss improved ({old_loss_str} -> {val_loss:.4f}). Saving checkpoint to {self.save_path} ..."
            )
        state_dict = (
            model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        )
        torch.save(state_dict, str(self.save_path))


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float],
    save_path: Optional[str] = None,
) -> None:
    """
    Plots training and validation loss & accuracy history curves.
    """
    plt.figure(figsize=(12, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", color="#1f77b4", lw=2)
    plt.plot(val_losses, label="Val Loss", color="#ff7f0e", lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Acc", color="#1f77b4", lw=2)
    plt.plot(val_accuracies, label="Val Acc", color="#2ca02c", lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy History")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_p), dpi=150)
        print(f"Training curves saved to {out_p}")

    plt.close()

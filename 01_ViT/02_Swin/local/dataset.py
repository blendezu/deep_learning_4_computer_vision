"""
Dataset, Data Augmentation (CutMix & MixUp), and DataLoaders for Food-101.
"""

from pathlib import Path

import config
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from utils import seed_worker


def get_transforms(
    img_size: int = config.IMG_SIZE,
    mean: tuple[float, float, float] = config.NORM_MEAN,
    std: tuple[float, float, float] = config.NORM_STD,
) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Constructs training and validation/testing data transforms.

    Training pipeline uses TrivialAugmentWide, random rotations, and crops.
    Validation pipeline uses standard resize + center crop.
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.RandomRotation(15),
            transforms.RandomCrop(img_size),
            transforms.TrivialAugmentWide(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return train_transform, test_transform


def sample_cutmix_box(
    batch_shape: torch.Size, mix_ratio: float
) -> tuple[int, int, int, int]:
    """
    Generate bounding box coordinates (y_min, y_max, x_min, x_max) for CutMix.

    Args:
        batch_shape: Tensor shape (B, C, H, W).
        mix_ratio: Sampled lambda mixing ratio.

    Returns:
        Bounding box tuple (y_min, y_max, x_min, x_max).
    """
    img_height = batch_shape[2]
    img_width = batch_shape[3]

    cutmix_scale = np.sqrt(1.0 - mix_ratio)
    cut_height = int(img_height * cutmix_scale)
    cut_width = int(img_width * cutmix_scale)

    # Sample random center
    cx = np.random.randint(0, img_width)
    cy = np.random.randint(0, img_height)

    # Box coordinates clamped to image borders
    x_min = int(np.clip(cx - cut_width // 2, 0, img_width))
    x_max = int(np.clip(cx + cut_width // 2, 0, img_width))
    y_min = int(np.clip(cy - cut_height // 2, 0, img_height))
    y_max = int(np.clip(cy + cut_height // 2, 0, img_height))

    return y_min, y_max, x_min, x_max


def apply_mixup_cutmix(
    x: torch.Tensor,
    y: torch.Tensor,
    p: float = 0.5,
    alpha: float = 1.0,
    cutmix_prob: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, bool]:
    """
    Apply CutMix or MixUp batch-level augmentation.

    Returns:
        - x_mixed: Augmented batch images.
        - y_a: Original labels of current batch.
        - y_b: Shuffled labels for paired mixing.
        - lam: Mixing coefficient lambda.
        - use_mix: Flag indicating whether mixing was performed.
    """
    use_mix = np.random.rand() < p
    if not use_mix:
        return x, y, y, 1.0, False

    # Sample lambda from Beta distribution
    lam = float(np.random.beta(alpha, alpha))

    # Permute indices for cross-sample mixing
    rand_index = torch.randperm(x.size(0), device=x.device)
    y_a, y_b = y, y[rand_index]

    if np.random.rand() < cutmix_prob:
        # CutMix
        x = x.clone()
        y_min, y_max, x_min, x_max = sample_cutmix_box(x.size(), lam)
        x[:, :, y_min:y_max, x_min:x_max] = x[rand_index, :, y_min:y_max, x_min:x_max]

        # Adjust lambda based on actual cropped area
        box_area = (y_max - y_min) * (x_max - x_min)
        total_area = x.size(-2) * x.size(-1)
        lam = 1.0 - (box_area / float(total_area))
    else:
        # MixUp
        x = lam * x + (1.0 - lam) * x[rand_index, :]

    return x, y_a, y_b, lam, True


def get_dataloaders(
    data_path: str | None = None,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    seed: int = config.SEED,
    img_size: int = config.IMG_SIZE,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Creates reproducible DataLoaders for Food-101 (Train, Val, Test).
    """
    root_path = Path(data_path) if data_path else config.DATA_PATH
    root_path.mkdir(parents=True, exist_ok=True)

    train_transform, test_transform = get_transforms(img_size=img_size)

    # 1. Download dummy data without heavy transforms to compute splits
    dummy_data = datasets.Food101(root=str(root_path), split="train", download=download)
    classes = dummy_data.classes

    train_size = int(config.TRAIN_VAL_SPLIT_RATIO * len(dummy_data))
    val_size = len(dummy_data) - train_size

    split_generator = torch.Generator().manual_seed(seed)
    train_tmp_subset, val_tmp_subset = random_split(
        dummy_data, [train_size, val_size], generator=split_generator
    )

    train_indices = train_tmp_subset.indices
    val_indices = val_tmp_subset.indices

    # 2. Attach separate training and validation transforms
    train_dataset = datasets.Food101(
        root=str(root_path), split="train", download=False, transform=train_transform
    )
    val_dataset = datasets.Food101(
        root=str(root_path), split="train", download=False, transform=test_transform
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    # 3. Test dataset
    test_dataset = datasets.Food101(
        root=str(root_path), split="test", download=download, transform=test_transform
    )

    train_generator = torch.Generator().manual_seed(seed)
    eval_generator = torch.Generator().manual_seed(seed)

    loader_kwargs = {
        "batch_size": batch_size,
        "pin_memory": config.PIN_MEMORY and torch.cuda.is_available(),
    }

    if num_workers > 0:
        loader_kwargs.update(
            {
                "num_workers": num_workers,
                "persistent_workers": config.PERSISTENT_WORKERS,
                "worker_init_fn": seed_worker,
            }
        )

    train_loader = DataLoader(
        train_subset,
        shuffle=True,
        generator=train_generator,
        prefetch_factor=config.PREFETCH_FACTOR_TRAIN if num_workers > 0 else None,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_subset,
        shuffle=False,
        generator=eval_generator,
        prefetch_factor=config.PREFETCH_FACTOR_EVAL if num_workers > 0 else None,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        generator=eval_generator,
        prefetch_factor=config.PREFETCH_FACTOR_EVAL if num_workers > 0 else None,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader, classes

"""Dataset handling and data loaders for YOLOv3 (Pascal VOC).

Supports downloading from kagglehub, parsing YOLO format labels,
Albumentations data augmentation, and variable-length bounding box batching.
"""

import glob
import os
import random

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from config import (
    BATCH_SIZE,
    IMG_SIZE,
    KAGGLE_DATASET_ID,
    NUM_WORKERS,
    SEED,
)
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def seed_worker(worker_id: int):
    """Seed function for DataLoader workers to ensure strict reproducibility."""
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def find_split_root(base_path: str) -> str:
    """Find the directory that directly contains train/ and val/ subfolders.

    Args:
        base_path: Root folder or parent folder of the dataset.

    Returns:
        Path containing 'train' and 'val' subdirectories.
    """
    if all(os.path.isdir(os.path.join(base_path, s)) for s in ["train", "val"]):
        return base_path

    for entry in os.listdir(base_path):
        candidate = os.path.join(base_path, entry)
        if os.path.isdir(candidate):
            if all(os.path.isdir(os.path.join(candidate, s)) for s in ["train", "val"]):
                return candidate

    raise FileNotFoundError(f"Could not find train/val directories under {base_path}")


def download_dataset(dataset_id: str = KAGGLE_DATASET_ID) -> str:
    """Download Pascal VOC dataset via kagglehub if needed and locate split root."""
    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "kagglehub is required to download dataset. Install via 'pip install kagglehub'."
        )

    print(f"Downloading dataset '{dataset_id}' via kagglehub...")
    dataset_path = kagglehub.dataset_download(dataset_id)
    split_root = find_split_root(dataset_path)
    print(f"Dataset ready at: {split_root}")
    return split_root


class VOCDataset(Dataset):
    """Pascal VOC Dataset with YOLO-format annotations.

    Each item returns:
        image: (3, H, W) normalized float tensor
        boxes: (N, 4) in fractional [xmin, ymin, xmax, ymax] format (0.0 to 1.0)
        labels: (N,) tensor of 0-indexed integer class IDs
    """

    def __init__(self, data_root: str, split: str = "train", img_size: int = IMG_SIZE):
        self.img_size = img_size
        self.split = split
        self.img_dir = os.path.join(data_root, split, "images")
        self.lbl_dir = os.path.join(data_root, split, "labels")

        # Collect all image files
        self.image_paths = sorted(
            glob.glob(os.path.join(self.img_dir, "*jpg"))
            + glob.glob(os.path.join(self.img_dir, "*jpeg"))
            + glob.glob(os.path.join(self.img_dir, "*png"))
        )
        assert len(self.image_paths) > 0, f"No images found in {self.img_dir}"

        # Albumentations transforms
        if split == "train":
            transforms = [
                A.ColorJitter(
                    brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05, p=0.5
                ),
                A.BBoxSafeRandomCrop(erosion_rate=0.0, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Resize(height=img_size, width=img_size, p=1.0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        else:
            transforms = [
                A.Resize(height=img_size, width=img_size, p=1.0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]

        self.transform = A.Compose(
            transforms=transforms,
            bbox_params=A.BboxParams(
                format="albumentations",
                label_fields=["class_labels"],
                min_visibility=0.1,
            ),
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _parse_yolo_label(self, label_path: str) -> tuple[list[list[float]], list[int]]:
        """Parse YOLO txt annotation into Albumentations [xmin, ymin, xmax, ymax]."""
        boxes = []
        labels = []

        if not os.path.exists(label_path):
            return [], []

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

                xmin = max(0.0, min(1.0, x_center - w / 2.0))
                ymin = max(0.0, min(1.0, y_center - h / 2.0))
                xmax = max(0.0, min(1.0, x_center + w / 2.0))
                ymax = max(0.0, min(1.0, y_center + h / 2.0))

                if xmax <= xmin or ymax <= ymin:
                    continue

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(cls_id)

        return boxes, labels

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        img_basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.lbl_dir, f"{img_basename}.txt")
        boxes, labels = self._parse_yolo_label(label_path)

        try:
            transformed = self.transform(
                image=image_np, bboxes=boxes, class_labels=labels
            )
            image_tensor = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]
        except Exception:
            # Fallback transform if crop removes all valid boxes
            fallback_transforms = [
                A.Resize(height=self.img_size, width=self.img_size, p=1.0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
            fallback = A.Compose(
                transforms=fallback_transforms,
                bbox_params=A.BboxParams(
                    format="albumentations", label_fields=["class_labels"]
                ),
            )
            transformed = fallback(image=image_np, bboxes=boxes, class_labels=labels)
            image_tensor = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]

        if len(boxes) == 0:
            box_tensor = torch.zeros((0, 4), dtype=torch.float32)
            label_tensor = torch.zeros((0,), dtype=torch.long)
        else:
            box_tensor = torch.tensor(boxes, dtype=torch.float32)
            label_tensor = torch.tensor(labels, dtype=torch.long)

        return image_tensor, box_tensor, label_tensor


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Custom collator handling variable numbers of bounding boxes per image."""
    images = []
    boxes = []
    labels = []

    for img, bx, lbl in batch:
        images.append(img)
        boxes.append(bx)
        labels.append(lbl)

    images_batch = torch.stack(images, dim=0)
    return images_batch, boxes, labels


def get_dataloaders(
    data_root: str,
    batch_size: int = BATCH_SIZE,
    img_size: int = IMG_SIZE,
    num_workers: int = NUM_WORKERS,
    seed: int = SEED,
) -> tuple[VOCDataset, VOCDataset, DataLoader, DataLoader]:
    """Create training and validation datasets and DataLoaders."""
    train_dataset = VOCDataset(data_root=data_root, split="train", img_size=img_size)
    val_dataset = VOCDataset(data_root=data_root, split="val", img_size=img_size)

    train_generator = torch.Generator().manual_seed(seed)
    eval_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=train_generator,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=eval_generator,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    return train_dataset, val_dataset, train_loader, val_loader

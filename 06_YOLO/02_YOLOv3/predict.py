"""Inference and detection visualization script for YOLOv3.

Draws side-by-side ground truth vs predictions with colored bounding boxes,
class names, confidence scores, and per-image / average FPS timing.
"""

import argparse
import random
import time

import matplotlib.pyplot as plt
import torch
from config import (
    ANCHORS,
    CLASS_COLORS,
    CLASS_NAMES,
    CONF_THRESHOLD,
    DEFAULT_CHECKPOINT,
    DEVICE,
    IMG_SIZE,
    NMS_THRESHOLD,
    NUM_CLASSES,
    STRIDES,
)
from dataset import download_dataset, get_dataloaders
from matplotlib import patches
from model import YOLOv3
from utils import denormalize, detect_objects


def visualize_detections(
    model: torch.nn.Module,
    dataset,
    n_samples: int = 6,
    conf_threshold: float = CONF_THRESHOLD,
    nms_threshold: float = NMS_THRESHOLD,
    fixed_indices: list[int] | None = None,
    save_path: str = "predictions.png",
    show: bool = False,
    device: torch.device = DEVICE,
):
    """Visualize object detections on sample images with FPS calculation."""
    model.eval()

    if fixed_indices is not None:
        indices = fixed_indices[:n_samples]
    else:
        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    _, axes = plt.subplots(len(indices), 2, figsize=(16, 4 * len(indices)))
    if len(indices) == 1:
        axes = axes.reshape(1, 2)

    total_fps = 0.0

    for row, idx in enumerate(indices):
        image, gt_boxes, gt_labels = dataset[idx]
        image_batch = image.unsqueeze(0).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            detections = detect_objects(
                model=model,
                images=image_batch,
                anchors=ANCHORS,
                strides=STRIDES,
                img_size=IMG_SIZE,
                num_classes=NUM_CLASSES,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
            )[0]

        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()

        inference_time = end_time - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0.0
        total_fps += fps

        img_display = denormalize(image).permute(1, 2, 0).numpy()

        # --- Left column: Ground Truth ---
        axes[row, 0].imshow(img_display)
        axes[row, 0].set_title(f"Ground Truth (Image {idx})", fontsize=12)
        axes[row, 0].axis("off")

        for j in range(gt_boxes.size(0)):
            box = gt_boxes[j].numpy() * IMG_SIZE
            label = int(gt_labels[j].item())
            color = CLASS_COLORS[min(label, len(CLASS_COLORS) - 1)]
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            axes[row, 0].add_patch(rect)
            axes[row, 0].text(
                box[0],
                box[1] - 4,
                CLASS_NAMES[label],
                color="white",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
            )

        # --- Right column: Predictions ---
        axes[row, 1].imshow(img_display)
        axes[row, 1].set_title(
            f"Predictions (Image {idx}) | FPS: {fps:.1f}", fontsize=12
        )
        axes[row, 1].axis("off")

        det_boxes = detections["boxes"].cpu()
        det_labels = detections["labels"].cpu()
        det_scores = detections["scores"].cpu()

        for j in range(det_boxes.size(0)):
            box = det_boxes[j].numpy() * IMG_SIZE
            label = int(det_labels[j].item())
            score = float(det_scores[j].item())
            color = CLASS_COLORS[min(label, len(CLASS_COLORS) - 1)]
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            axes[row, 1].add_patch(rect)
            axes[row, 1].text(
                box[0],
                box[1] - 4,
                f"{CLASS_NAMES[label]} {score:.2f}",
                color="white",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
            )

    avg_fps = total_fps / len(indices)
    plt.suptitle(
        f"Average Inference FPS: {avg_fps:.1f} ({device.type.upper()})",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(
        f"Predictions visualized and saved to '{save_path}' (Average FPS: {avg_fps:.1f})"
    )

    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLOv3 detections")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Model weights path",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Dataset root directory",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=6,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.3,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="predictions.png",
        help="Output image path",
    )
    args = parser.parse_args()

    data_root = args.data_root or download_dataset()
    _, val_dataset, _, _ = get_dataloaders(
        data_root=data_root, batch_size=1, img_size=IMG_SIZE
    )

    model = YOLOv3(num_anchor_per_scale=3, num_classes=NUM_CLASSES).to(DEVICE)
    print(f"Loading checkpoint '{args.checkpoint}'...")
    map_location = None if torch.cuda.is_available() else "cpu"
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=map_location, weights_only=True)
    )

    common_eval_indices = [97, 2, 3, 4, 5, 6]
    visualize_detections(
        model=model,
        dataset=val_dataset,
        n_samples=args.n_samples,
        conf_threshold=args.conf_threshold,
        fixed_indices=common_eval_indices,
        save_path=args.save_path,
        device=DEVICE,
    )


if __name__ == "__main__":
    main()

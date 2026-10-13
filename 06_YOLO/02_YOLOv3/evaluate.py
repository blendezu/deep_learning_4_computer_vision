"""Evaluation script for YOLOv3 on Pascal VOC.

Computes and displays:
- Per-class AP (11-point interpolated)
- Number of ground-truth objects and detections per class
- Overall mean Average Precision (mAP@0.5)
"""

import argparse
from collections.abc import Sequence

import numpy as np
import torch
from config import (
    ANCHORS,
    BATCH_SIZE,
    CLASS_NAMES,
    DEFAULT_CHECKPOINT,
    DEVICE,
    IMG_SIZE,
    NMS_THRESHOLD,
    NUM_CLASSES,
    STRIDES,
)
from dataset import download_dataset, get_dataloaders
from model import YOLOv3
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils import compute_iou, detect_objects


def evaluate_dataset(
    model: torch.nn.Module,
    val_loader: DataLoader,
    num_classes: int = NUM_CLASSES,
    class_names: list[str] = CLASS_NAMES,
    anchors: Sequence[Sequence[tuple[float, float]]] = ANCHORS,
    strides: list[int] = STRIDES,
    img_size: int = IMG_SIZE,
    conf_threshold: float = 0.01,
    nms_threshold: float = NMS_THRESHOLD,
    device: torch.device = DEVICE,
) -> float:
    """Run full evaluation over the dataset and print per-class AP breakdown."""
    model.eval()

    all_det_scores: dict[int, list[float]] = {c: [] for c in range(num_classes)}
    all_det_is_tp: dict[int, list[bool]] = {c: [] for c in range(num_classes)}
    n_gt_per_class: dict[int, int] = {c: 0 for c in range(num_classes)}

    eval_loop = tqdm(val_loader, desc="Evaluating on validation set", leave=False)

    with torch.no_grad():
        for images, gt_boxes_list, gt_labels_list in eval_loop:
            images = images.to(device)

            detections = detect_objects(
                model=model,
                images=images,
                anchors=anchors,
                strides=strides,
                img_size=img_size,
                num_classes=num_classes,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
            )

            for idx in range(len(detections)):
                det = detections[idx]
                gt_boxes = gt_boxes_list[idx].to(device)
                gt_labels = gt_labels_list[idx].to(device)

                for c in range(num_classes):
                    n_gt_per_class[c] += (gt_labels == c).sum().item()

                gt_matched = torch.zeros(
                    gt_boxes.size(0), dtype=torch.bool, device=device
                )

                for det_idx in range(det["boxes"].size(0)):
                    det_box = det["boxes"][det_idx].unsqueeze(0)
                    det_label = int(det["labels"][det_idx].item())
                    det_score = float(det["scores"][det_idx].item())

                    class_mask = gt_labels == det_label
                    if class_mask.sum() == 0:
                        all_det_scores[det_label].append(det_score)
                        all_det_is_tp[det_label].append(False)
                        continue

                    gt_class_boxes = gt_boxes[class_mask]
                    gt_class_indices = torch.where(class_mask)[0]
                    iou = compute_iou(det_box, gt_class_boxes).squeeze(0)
                    best_iou, best_gt_local_idx = iou.max(dim=0)
                    best_gt_global_idx = gt_class_indices[best_gt_local_idx]

                    if best_iou >= 0.5 and not gt_matched[best_gt_global_idx]:
                        all_det_scores[det_label].append(det_score)
                        all_det_is_tp[det_label].append(True)
                        gt_matched[best_gt_global_idx] = True
                    else:
                        all_det_scores[det_label].append(det_score)
                        all_det_is_tp[det_label].append(False)

    # 11-point interpolated AP calculation per class
    print("\n" + "=" * 50)
    print("           Per-Class Evaluation Results")
    print("=" * 50)
    aps = []

    for c in range(num_classes):
        n_gt = n_gt_per_class[c]
        class_name = class_names[c]

        if n_gt == 0:
            print(f"  {class_name:14s}: No Ground Truth objects")
            continue

        scores = np.array(all_det_scores[c])
        is_tp = np.array(all_det_is_tp[c], dtype=bool)

        if len(scores) == 0:
            print(f"  {class_name:14s}: AP = 0.0000 ({n_gt:4d} GT,    0 detections)")
            aps.append(0.0)
            continue

        sort_idx = np.argsort(-scores)
        is_tp = is_tp[sort_idx]

        tp_cumsum = np.cumsum(is_tp)
        fp_cumsum = np.cumsum(~is_tp)
        recall = tp_cumsum / n_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            prec_at_recall = precision[recall >= t]
            if len(prec_at_recall) > 0:
                ap += prec_at_recall.max()
        ap /= 11.0

        aps.append(ap)
        print(
            f"  {class_name:14s}: AP = {ap:.4f} ({n_gt:4d} GT, {len(scores):4d} detections)"
        )

    val_map_final = float(np.mean(aps)) if len(aps) > 0 else 0.0
    print("-" * 50)
    print(f"  Mean Average Precision (mAP@0.5): {val_map_final:.4f}")
    print("=" * 50 + "\n")

    return val_map_final


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv3 on Pascal VOC")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to dataset root containing train/ and val/ dirs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Validation batch size"
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.01,
        help="Confidence threshold for mAP calculation",
    )
    args = parser.parse_args()

    data_root = args.data_root or download_dataset()
    _, val_dataset, _, val_loader = get_dataloaders(
        data_root=data_root, batch_size=args.batch_size, img_size=IMG_SIZE
    )

    model = YOLOv3(num_anchor_per_scale=3, num_classes=NUM_CLASSES).to(DEVICE)
    print(f"Loading weights from {args.checkpoint}...")
    map_location = None if torch.cuda.is_available() else "cpu"
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=map_location, weights_only=True)
    )

    evaluate_dataset(
        model=model,
        val_loader=val_loader,
        conf_threshold=args.conf_threshold,
        device=DEVICE,
    )


if __name__ == "__main__":
    main()

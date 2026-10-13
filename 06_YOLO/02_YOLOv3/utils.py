"""Utility functions for YOLOv3.

Includes:
- Reproducibility seeding
- IoU calculations (boxes and anchors)
- Target encoding (build_yolov3_targets, prepare_targets)
- Vectorized inference and batched Non-Maximum Suppression (detect_objects)
- VOC 11-point interpolated mAP evaluation (compute_map)
- Anchor K-Means clustering (kmeans_anchors)
- Early stopping callback
- Denormalization helper for visualization
"""

import math
import os
import random
from collections.abc import Sequence

import numpy as np
import torch
from config import (
    ANCHORS,
    CONF_THRESHOLD,
    DEFAULT_CHECKPOINT,
    DEVICE,
    IGNORE_IOU_THRESHOLD,
    IMG_SIZE,
    NMS_THRESHOLD,
    NUM_CLASSES,
    PATIENCE,
    SEED,
    STRIDES,
)
from torch import nn
from torchvision.ops import batched_nms
from tqdm.auto import tqdm


def set_seed(seed: int = SEED):
    """Set all random seeds for reproducible execution."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes.

    Args:
        boxes_a: (A, 4) in [xmin, ymin, xmax, ymax]
        boxes_b: (B, 4) in [xmin, ymin, xmax, ymax]

    Returns:
        (A, B) pairwise IoU matrix.
    """
    A = boxes_a.size(0)
    B = boxes_b.size(0)

    if A == 0 or B == 0:
        return torch.zeros((A, B), device=boxes_a.device)

    max_xy = torch.min(
        boxes_a[:, 2:].unsqueeze(1).expand(A, B, 2),
        boxes_b[:, 2:].unsqueeze(0).expand(A, B, 2),
    )
    min_xy = torch.max(
        boxes_a[:, :2].unsqueeze(1).expand(A, B, 2),
        boxes_b[:, :2].unsqueeze(0).expand(A, B, 2),
    )

    inter = torch.clamp(max_xy - min_xy, min=0)
    inter_area = inter[:, :, 0] * inter[:, :, 1]

    area_a = (
        ((boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]))
        .unsqueeze(1)
        .expand(A, B)
    )
    area_b = (
        ((boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1]))
        .unsqueeze(0)
        .expand(A, B)
    )

    return inter_area / (area_a + area_b - inter_area + 1e-6)


def compute_iou_wh(wh1: torch.Tensor, wh2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between anchor dimensions and box dimensions centered at origin.

    Args:
        wh1: (N, 2) widths and heights of set 1
        wh2: (M, 2) widths and heights of set 2

    Returns:
        (N, M) IoU matrix.
    """
    inter_w = torch.min(wh1[:, 0].unsqueeze(1), wh2[:, 0].unsqueeze(0))
    inter_h = torch.min(wh1[:, 1].unsqueeze(1), wh2[:, 1].unsqueeze(0))
    inter_area = inter_w * inter_h

    area1 = (wh1[:, 0] * wh1[:, 1]).unsqueeze(1)
    area2 = (wh2[:, 0] * wh2[:, 1]).unsqueeze(0)

    return inter_area / (area1 + area2 - inter_area + 1e-6)


def build_yolov3_targets(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    anchors: Sequence[Sequence[tuple[float, float]]] = ANCHORS,
    strides: list[int] = STRIDES,
    img_size: int = IMG_SIZE,
    num_classes: int = NUM_CLASSES,
    ignore_iou_threshold: float = IGNORE_IOU_THRESHOLD,
) -> list[torch.Tensor]:
    """Encode ground-truth bounding boxes into multi-scale YOLOv3 targets for one image.

    Args:
        boxes: (N, 4) in fractional [xmin, ymin, xmax, ymax]
        labels: (N,) class indices
        anchors: 3-scale list of anchor dimensions
        strides: Stride for each scale [32, 16, 8]
        img_size: Image resolution
        num_classes: Number of target classes
        ignore_iou_threshold: Threshold to mark non-responsible high-IoU anchors as -1 (ignored)

    Returns:
        List of 3 tensors [scale_0, scale_1, scale_2], each shaped:
        (num_anchors, grid_size, grid_size, 6 + num_classes).
    """
    device = boxes.device
    num_scales = len(strides)
    num_anchor_per_scale = len(anchors[0])

    targets = []
    for s_idx in range(num_scales):
        grid_size = img_size // strides[s_idx]
        target = torch.zeros(
            num_anchor_per_scale,
            grid_size,
            grid_size,
            6 + num_classes,
            device=device,
            dtype=torch.float32,
        )
        targets.append(target)

    if boxes.size(0) == 0:
        return targets

    # Flatten anchors across all 3 scales: (9, 2)
    all_anchors_wh = []
    anchor_to_scale = []
    for s_idx in range(num_scales):
        for a_idx in range(num_anchor_per_scale):
            all_anchors_wh.append(list(anchors[s_idx][a_idx]))
            anchor_to_scale.append((s_idx, a_idx))

    all_anchors_wh = torch.tensor(all_anchors_wh, dtype=torch.float32, device=device)

    # Convert normalized GT coordinates to pixel center and size
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0 * img_size
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0 * img_size
    w = (boxes[:, 2] - boxes[:, 0]) * img_size
    h = (boxes[:, 3] - boxes[:, 1]) * img_size
    gt_wh = torch.stack([w, h], dim=1)

    # Match GT boxes to anchors based on shape IoU
    ious = compute_iou_wh(gt_wh, all_anchors_wh)
    best_anchor_idx = ious.argmax(dim=1)

    for obj_idx in range(boxes.size(0)):
        flat_best = int(best_anchor_idx[obj_idx].item())
        s_idx, a_idx = anchor_to_scale[flat_best]

        grid_size = img_size // strides[s_idx]
        anchor_w, anchor_h = anchors[s_idx][a_idx]

        gi = int(cx[obj_idx].item() / strides[s_idx])
        gj = int(cy[obj_idx].item() / strides[s_idx])
        gi = min(max(gi, 0), grid_size - 1)
        gj = min(max(gj, 0), grid_size - 1)

        # Offsets
        tx = cx[obj_idx].item() / strides[s_idx] - gi
        ty = cy[obj_idx].item() / strides[s_idx] - gj
        tw = math.log(max(w[obj_idx].item(), 1e-6) / max(anchor_w, 1e-6))
        th = math.log(max(h[obj_idx].item(), 1e-6) / max(anchor_h, 1e-6))
        cls_id = int(labels[obj_idx].item())

        # Encode target
        targets[s_idx][a_idx, gj, gi, 0] = tx
        targets[s_idx][a_idx, gj, gi, 1] = ty
        targets[s_idx][a_idx, gj, gi, 2] = tw
        targets[s_idx][a_idx, gj, gi, 3] = th
        targets[s_idx][a_idx, gj, gi, 4] = 1.0  # Positive anchor
        targets[s_idx][a_idx, gj, gi, 5] = 1.0
        targets[s_idx][a_idx, gj, gi, 6 + cls_id] = 1.0

        # Mark other high-IoU anchors for this GT as -1 (ignored)
        for flat_idx in range(len(anchor_to_scale)):
            if flat_idx == flat_best:
                continue
            if ious[obj_idx, flat_idx] > ignore_iou_threshold:
                s2, a2 = anchor_to_scale[flat_idx]
                grid_size2 = img_size // strides[s2]
                gi2 = int(cx[obj_idx].item() / strides[s2])
                gj2 = int(cy[obj_idx].item() / strides[s2])
                gi2 = min(max(gi2, 0), grid_size2 - 1)
                gj2 = min(max(gj2, 0), grid_size2 - 1)
                if targets[s2][a2, gj2, gi2, 4] != 1.0:
                    targets[s2][a2, gj2, gi2, 4] = -1.0

    return targets


def prepare_targets(
    gt_boxes: list[torch.Tensor],
    gt_labels: list[torch.Tensor],
    anchors: Sequence[Sequence[tuple[float, float]]] = ANCHORS,
    strides: list[int] = STRIDES,
    img_size: int = IMG_SIZE,
    num_classes: int = NUM_CLASSES,
    ignore_thresh: float = IGNORE_IOU_THRESHOLD,
    device: torch.device = DEVICE,
) -> list[torch.Tensor]:
    """Build and stack target tensors for a whole batch."""
    batch_targets = [[] for _ in range(len(strides))]

    for b_idx in range(len(gt_boxes)):
        img_targets = build_yolov3_targets(
            gt_boxes[b_idx].to(device),
            gt_labels[b_idx].to(device),
            anchors=anchors,
            strides=strides,
            img_size=img_size,
            num_classes=num_classes,
            ignore_iou_threshold=ignore_thresh,
        )
        for s_idx in range(len(strides)):
            batch_targets[s_idx].append(img_targets[s_idx])

    return [
        torch.stack(batch_targets[s], dim=0).to(device) for s in range(len(strides))
    ]


def detect_objects(
    model: nn.Module,
    images: torch.Tensor,
    anchors: Sequence[Sequence[tuple[float, float]]] = ANCHORS,
    strides: list[int] = STRIDES,
    img_size: int = IMG_SIZE,
    num_classes: int = NUM_CLASSES,
    conf_threshold: float = CONF_THRESHOLD,
    nms_threshold: float = NMS_THRESHOLD,
) -> list[dict[str, torch.Tensor]]:
    """Run YOLOv3 postprocessing and batched NMS on a batch of images.

    Returns:
        List of detection dictionaries, one per image, each containing:
          - "boxes": (K, 4) in [xmin, ymin, xmax, ymax] (0.0 to 1.0)
          - "labels": (K,) class indices
          - "scores": (K,) confidence scores (obj * class_prob)
    """
    model.eval()
    with torch.no_grad():
        predictions = model(images)

    batch_size = images.size(0)
    dev = images.device
    results = []

    # Precompute anchor grids
    scale_data = []
    for s_idx in range(len(strides)):
        stride = strides[s_idx]
        grid_size = img_size // stride
        A = len(anchors[s_idx])

        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_size, device=dev, dtype=torch.float32),
            torch.arange(grid_size, device=dev, dtype=torch.float32),
            indexing="ij",
        )
        grid_x = grid_x.unsqueeze(0).expand(A, -1, -1)
        grid_y = grid_y.unsqueeze(0).expand(A, -1, -1)

        anchor_wh = torch.tensor(anchors[s_idx], device=dev, dtype=torch.float32)
        aw = anchor_wh[:, 0].view(A, 1, 1)
        ah = anchor_wh[:, 1].view(A, 1, 1)

        scale_data.append((stride, grid_size, grid_x, grid_y, aw, ah))

    for i in range(batch_size):
        all_boxes, all_scores, all_labels = [], [], []

        for s_idx, (stride, grid_size, grid_x, grid_y, aw, ah) in enumerate(scale_data):
            pred = predictions[s_idx][i]
            obj_conf = pred[..., 4]
            obj_mask = obj_conf >= conf_threshold
            if obj_mask.sum() == 0:
                continue

            tx = pred[..., 0]
            ty = pred[..., 1]
            tw = pred[..., 2]
            th = pred[..., 3]

            cx = (grid_x + tx) * stride / img_size
            cy = (grid_y + ty) * stride / img_size
            bw = aw * torch.exp(tw.clamp(max=10.0)) / img_size
            bh = ah * torch.exp(th.clamp(max=10.0)) / img_size

            cx_f = cx[obj_mask]
            cy_f = cy[obj_mask]
            bw_f = bw[obj_mask]
            bh_f = bh[obj_mask]
            obj_f = obj_conf[obj_mask]
            cls_f = pred[..., 5:][obj_mask]

            class_conf, best_class = cls_f.max(dim=1)
            score = obj_f * class_conf
            score_mask = score >= conf_threshold
            if score_mask.sum() == 0:
                continue

            cx_f = cx_f[score_mask]
            cy_f = cy_f[score_mask]
            bw_f = bw_f[score_mask]
            bh_f = bh_f[score_mask]
            score = score[score_mask]
            best_class = best_class[score_mask]

            xmin = (cx_f - bw_f / 2).clamp(min=0.0)
            ymin = (cy_f - bh_f / 2).clamp(min=0.0)
            xmax = (cx_f + bw_f / 2).clamp(max=1.0)
            ymax = (cy_f + bh_f / 2).clamp(max=1.0)
            valid = (xmax > xmin) & (ymax > ymin)
            if valid.sum() == 0:
                continue

            boxes = torch.stack(
                [xmin[valid], ymin[valid], xmax[valid], ymax[valid]], dim=1
            )
            all_boxes.append(boxes)
            all_scores.append(score[valid])
            all_labels.append(best_class[valid])

        if len(all_boxes) == 0:
            results.append(
                {
                    "boxes": torch.zeros((0, 4), device=dev),
                    "labels": torch.zeros((0,), dtype=torch.long, device=dev),
                    "scores": torch.zeros((0,), device=dev),
                }
            )
            continue

        all_boxes = torch.cat(all_boxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        keep_idx = batched_nms(all_boxes, all_scores, all_labels, nms_threshold)
        all_boxes = all_boxes[keep_idx]
        all_scores = all_scores[keep_idx]
        all_labels = all_labels[keep_idx]

        _, sort_idx = all_scores.sort(descending=True)
        results.append(
            {
                "boxes": all_boxes[sort_idx],
                "labels": all_labels[sort_idx],
                "scores": all_scores[sort_idx],
            }
        )

    return results


def compute_map(
    model: nn.Module,
    data_loader,
    num_classes: int = NUM_CLASSES,
    anchors: Sequence[Sequence[tuple[float, float]]] = ANCHORS,
    strides: list[int] = STRIDES,
    img_size: int = IMG_SIZE,
    conf_threshold: float = 0.01,
    nms_threshold: float = NMS_THRESHOLD,
    max_batches: int | None = 40,
    device: torch.device = DEVICE,
) -> float:
    """Compute mAP@0.5 using VOC 11-point interpolation with vectorized matching."""
    model.eval()

    all_det_scores = {c: [] for c in range(num_classes)}
    all_det_is_tp = {c: [] for c in range(num_classes)}
    n_gt_per_class = {c: 0 for c in range(num_classes)}

    loop = tqdm(data_loader, desc="Computing mAP", leave=False)

    with torch.no_grad():
        for b_idx, (images, gt_boxes_list, gt_labels_list) in enumerate(loop):
            if max_batches is not None and b_idx >= max_batches:
                break
            images = images.to(device)

            detections = detect_objects(
                model,
                images,
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

                n_det = det["boxes"].size(0)
                if n_det == 0:
                    continue

                n_gt = gt_boxes.size(0)
                if n_gt == 0:
                    for d in range(n_det):
                        c = int(det["labels"][d].item())
                        all_det_scores[c].append(float(det["scores"][d].item()))
                        all_det_is_tp[c].append(False)
                    continue

                iou_matrix = compute_iou(det["boxes"], gt_boxes)
                gt_matched = torch.zeros(n_gt, dtype=torch.bool, device=device)

                for d in range(n_det):
                    det_label = int(det["labels"][d].item())
                    det_score = float(det["scores"][d].item())

                    class_mask = gt_labels == det_label
                    if class_mask.sum() == 0:
                        all_det_scores[det_label].append(det_score)
                        all_det_is_tp[det_label].append(False)
                        continue

                    available = class_mask & ~gt_matched
                    if available.sum() == 0:
                        all_det_scores[det_label].append(det_score)
                        all_det_is_tp[det_label].append(False)
                        continue

                    ious = iou_matrix[d].clone()
                    ious[~available] = -1.0
                    best_iou, best_gt = ious.max(dim=0)

                    if best_iou >= 0.5:
                        all_det_scores[det_label].append(det_score)
                        all_det_is_tp[det_label].append(True)
                        gt_matched[best_gt] = True
                    else:
                        all_det_scores[det_label].append(det_score)
                        all_det_is_tp[det_label].append(False)

    # 11-point interpolated AP per class
    aps = []
    for c in range(num_classes):
        n_gt = n_gt_per_class[c]
        if n_gt == 0:
            continue

        scores = np.array(all_det_scores[c])
        is_tp = np.array(all_det_is_tp[c], dtype=bool)

        if len(scores) == 0:
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
            precisions_at_recall = precision[recall >= t]
            if len(precisions_at_recall) > 0:
                ap += precisions_at_recall.max()
        ap /= 11.0
        aps.append(ap)

    return float(np.mean(aps)) if len(aps) > 0 else 0.0


def iou_wh(boxes: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Compute IoU between boxes and centroids for K-Means clustering."""
    boxes = boxes[:, np.newaxis, :]
    centroids = centroids[np.newaxis, :, :]

    inter_w = np.minimum(boxes[..., 0], centroids[..., 0])
    inter_h = np.minimum(boxes[..., 1], centroids[..., 1])
    inter_area = inter_w * inter_h

    box_area = boxes[..., 0] * boxes[..., 1]
    cent_area = centroids[..., 0] * centroids[..., 1]
    union_area = box_area + cent_area - inter_area

    return inter_area / np.clip(union_area, a_min=1e-6, a_max=None)


def kmeans_anchors(
    boxes_wh: np.ndarray, k: int = 9, max_iters: int = 300
) -> np.ndarray:
    """K-Means clustering with distance metric d = 1 - IoU."""
    indices = np.random.choice(len(boxes_wh), size=k, replace=False)
    centroids = boxes_wh[indices].copy()

    for _ in range(max_iters):
        ious = iou_wh(boxes_wh, centroids)
        cluster_assignments = np.argmax(ious, axis=1)

        new_centroids = np.zeros_like(centroids)
        for cluster_idx in range(k):
            assigned = boxes_wh[cluster_assignments == cluster_idx]
            if len(assigned) > 0:
                new_centroids[cluster_idx] = np.mean(assigned, axis=0)
            else:
                new_centroids[cluster_idx] = boxes_wh[np.random.choice(len(boxes_wh))]

        if np.allclose(centroids, new_centroids, atol=1e-2):
            break
        centroids = new_centroids

    return centroids


class EarlyStopping:
    """Early Stopping handler to halt training when validation loss stops improving."""

    def __init__(
        self,
        patience: int = PATIENCE,
        delta: float = 0.0,
        save_path: str = DEFAULT_CHECKPOINT,
    ):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.early_stop = False
        self.counter = 0
        self.best_loss = None

    def __call__(self, model: nn.Module, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.save_path)
            print(
                f"Validation loss improved to {val_loss:.4f}. Model saved to {self.save_path}."
            )
        elif val_loss >= self.best_loss - self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print(
                f"Validation loss improved from {self.best_loss:.4f} to {val_loss:.4f}. Saving model."
            )
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization for plotting images."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return torch.clamp(tensor.cpu() * std + mean, 0.0, 1.0)

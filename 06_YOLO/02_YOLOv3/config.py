"""Configuration and Hyperparameters for YOLOv3 (Pascal VOC).

Contains all image dimensions, class names, anchors, loss weights,
and training parameters.
"""

import torch

# -----------------------------------------------------------------------------
# Device Configuration
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Image and Architecture Settings
# -----------------------------------------------------------------------------
IMG_SIZE = 416  # Base image resolution for YOLOv3
MULTI_SCALES = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
SCALE_INTERVAL = 10  # Change image size every N batches during training
IN_CHANNELS = 3
NUM_CLASSES = 20

# YOLOv3 multi-scale strides: 32 (13x13), 16 (26x26), 8 (52x52)
STRIDES = [32, 16, 8]
NUM_ANCHORS_PER_SCALE = 3

# Default anchor priors (width, height in pixels relative to 416x416):
# Scale 0 (13x13 grid) -> Large objects
# Scale 1 (26x26 grid) -> Medium objects
# Scale 2 (52x52 grid) -> Small objects
# This anchors are for COCO Dataset, but it still works well with VOC
ANCHORS = [
    [(116, 90), (156, 198), (373, 326)],  # Scale 0 (13x13)
    [(30, 61), (62, 45), (59, 119)],  # Scale 1 (26x26)
    [(10, 13), (16, 30), (33, 23)],  # Scale 2 (52x52)
]

# -----------------------------------------------------------------------------
# Pascal VOC Class Names and Palette
# -----------------------------------------------------------------------------
CLASS_NAMES = [
    "aeroplane",  # 0
    "bicycle",  # 1
    "bird",  # 2
    "boat",  # 3
    "bottle",  # 4
    "bus",  # 5
    "car",  # 6
    "cat",  # 7
    "chair",  # 8
    "cow",  # 9
    "diningtable",  # 10
    "dog",  # 11
    "horse",  # 12
    "motorbike",  # 13
    "person",  # 14
    "pottedplant",  # 15
    "sheep",  # 16
    "sofa",  # 17
    "train",  # 18
    "tvmonitor",  # 19
]

CLASS_COLORS = [
    "black",  # 0:  background offset if needed
    "#FF6B6B",  # 1:  aeroplane    (red)
    "#4ECDC4",  # 2:  bicycle      (teal)
    "#45B7D1",  # 3:  bird         (blue)
    "#96CEB4",  # 4:  boat         (green)
    "#FFEAA7",  # 5:  bottle       (yellow)
    "#DDA0DD",  # 6:  bus          (plum)
    "#FF8C42",  # 7:  car          (orange)
    "#A29BFE",  # 8:  cat          (lavender)
    "#FD79A8",  # 9:  chair        (pink)
    "#00CEC9",  # 10: cow          (cyan)
    "#E17055",  # 11: diningtable  (burnt orange)
    "#6C5CE7",  # 12: dog          (purple)
    "#55EFC4",  # 13: horse        (mint)
    "#FAB1A0",  # 14: motorbike    (salmon)
    "#74B9FF",  # 15: person       (sky blue)
    "#81ECEC",  # 16: pottedplant  (aqua)
    "#FDCB6E",  # 17: sheep        (gold)
    "#E84393",  # 18: sofa         (magenta)
    "#00B894",  # 19: train        (emerald)
    "#636E72",  # 20: tvmonitor    (grey)
]

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 100
SEED = 42
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 4

# Two-Phase Training Schedule
WARMUP_EPOCHS = 5
FREEZE_BACKBONE_EPOCHS = 5
PATIENCE = 10

# -----------------------------------------------------------------------------
# Loss Weighting
# -----------------------------------------------------------------------------
LAMBDA_COORD = 1.0  # Coordinate loss (tx, ty, tw, th)
LAMBDA_NOOBJ = 1.0  # Background confidence loss
LAMBDA_OBJ = 1.0  # Object confidence loss
LAMBDA_CLASS = 1.0  # Classification loss

# -----------------------------------------------------------------------------
# Inference and NMS Thresholds
# -----------------------------------------------------------------------------
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
IGNORE_IOU_THRESHOLD = 0.5  # Anchor assignment threshold for negative ignoring

# -----------------------------------------------------------------------------
# Paths and Checkpoint
# -----------------------------------------------------------------------------
KAGGLE_DATASET_ID = "banuprasadb/pascal-voc-2012"
DEFAULT_CHECKPOINT = "best_yolov3_voc.pth"
TRAINING_CURVES_PATH = "yolov3_training_curves.png"

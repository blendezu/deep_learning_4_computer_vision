"""
Configuration and Hyperparameters for Swin Transformer V2 Training.
"""

from pathlib import Path

import torch


def get_device() -> torch.device:
    """Returns CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Directories & Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
FIGURES_DIR = BASE_DIR / "figures"
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_checkpoint.pth"

# Ensure runtime directories exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset Hyperparameters
# --------------------------------------------------------------------------
DATASET_NAME = "Food101"
IMG_SIZE = 224
IN_CHANNELS = 3
NUM_CLASSES = 101
TRAIN_VAL_SPLIT_RATIO = 0.8

# ImageNet normalization statistics
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# DataLoader parameters
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR_TRAIN = 2
PREFETCH_FACTOR_EVAL = 4

# --------------------------------------------------------------------------
# Model Architecture Hyperparameters (Swin-T default)
# --------------------------------------------------------------------------
MODEL_NAME = "Swin-T"
PATCH_SIZE = 4
EMBED_DIM = 96
DEPTHS = [2, 2, 6, 2]
NUM_HEADS = [3, 6, 12, 24]
WINDOW_SIZE = 7
MLP_RATIO = 4.0
QKV_BIAS = True
DROP_RATE = 0.0
ATTN_DROP_RATE = 0.0
DROP_PATH_RATE = 0.2
APE = False
PATCH_NORM = True
USE_CHECKPOINT = False
PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]

# --------------------------------------------------------------------------
# Training Hyperparameters
# --------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
GRAD_CLIP_NORM = 1.0

# Early Stopping
PATIENCE = 10
DELTA = 0.01

# OneCycleLR Scheduler Settings
PCT_START = 0.1
DIV_FACTOR = 10.0
FINAL_DIV_FACTOR = 100.0

# Data Augmentation (CutMix / MixUp)
MIXUP_P = 0.5
MIXUP_ALPHA = 1.0
CUTMIX_PROB = 0.5

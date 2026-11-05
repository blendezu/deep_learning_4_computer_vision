"""
GradCAM Implementation for Swin Transformer V2.
Adapts 1D token representations (B, L, C) to 2D spatial feature maps (B, C, H, W)
for gradient-weighted class activation mapping.
"""

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import config
from dataset import get_dataloaders, get_transforms
from model import build_swin_v2


class GradCAM:
    """
    GradCAM implementation supporting Vision Transformers with sequence representations (B, L, C).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and full backward hooks
        self.hook_forward = target_layer.register_forward_hook(self.save_activation)
        self.hook_backward = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        activation = output
        if len(activation.shape) == 3:
            # Swin Transformer shape: (B, L, C) -> (B, C, H, W)
            B, L, C = activation.shape
            H = int(math.sqrt(L))
            activation = activation.transpose(1, 2).view(B, C, H, H)
        self.activations = activation

    def save_gradient(self, module, grad_input, grad_output):
        grad = grad_output[0]
        if len(grad.shape) == 3:
            # Swin Transformer shape: (B, L, C) -> (B, C, H, W)
            B, L, C = grad.shape
            H = int(math.sqrt(L))
            grad = grad.transpose(1, 2).view(B, C, H, H)
        self.gradients = grad

    def generate_heatmap(
        self, input_image: torch.Tensor, target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generates GradCAM heatmap for a single image tensor with shape (1, C, H, W).

        Returns:
            - cam: 2D normalized numpy array (H, W) in range [0, 1].
            - target_class: Predicted or target class index.
        """
        self.model.eval()

        # 1. Forward pass
        model_output = self.model(input_image)

        if target_class is None:
            target_class = int(model_output.argmax(dim=1).item())

        # 2. Backward pass for target class score
        self.model.zero_grad()
        target_score = model_output[0, target_class]
        target_score.backward()

        # 3. Global Average Pooling of gradients across spatial dimensions
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # 4. Weight activations and aggregate across channels
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # 5. ReLU activation (only positive contributions)
        cam = F.relu(cam)

        # 6. Bilinear interpolation to original image size
        cam = F.interpolate(
            cam,
            size=(input_image.shape[2], input_image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # 7. Normalize between 0 and 1
        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_min = np.min(cam_np)
        cam_max = np.max(cam_np)
        cam_norm = (cam_np - cam_min) / (cam_max - cam_min + 1e-8)

        return cam_norm, target_class

    def remove_hooks(self):
        """Clean up registered hooks."""
        self.hook_forward.remove()
        self.hook_backward.remove()


def plot_gradcam(
    image_tensor: torch.Tensor,
    cam_heatmap: np.ndarray,
    class_label: Optional[str] = None,
    save_path: Optional[str] = None,
    mean: Tuple[float, float, float] = config.NORM_MEAN,
    std: Tuple[float, float, float] = config.NORM_STD,
) -> None:
    """
    Visualizes original image, GradCAM heatmap, and overlay side-by-side.
    """
    # Denormalize image tensor to [0, 1] range for visualization
    img = image_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    mean_arr = np.array(mean)
    std_arr = np.array(std)
    img = std_arr * img + mean_arr
    img = np.clip(img, 0.0, 1.0)

    # Colorize heatmap with Jet colormap
    jet = plt.get_cmap("jet")
    heatmap_colored = jet(cam_heatmap)[..., :3]

    # Create overlay
    overlay = heatmap_colored * 0.4 + img * 0.6
    overlay = np.clip(overlay, 0.0, 1.0)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    title_suffix = f" ({class_label})" if class_label else ""
    axes[0].imshow(img)
    axes[0].set_title(f"Original Image{title_suffix}")
    axes[0].axis("off")

    axes[1].imshow(cam_heatmap, cmap="jet")
    axes[1].set_title("GradCAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_p), dpi=150)
        print(f"GradCAM plot saved to {out_p}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate GradCAM heatmap for Swin Transformer V2")
    parser.add_argument("--checkpoint", type=str, default=str(config.BEST_CHECKPOINT_PATH), help="Path to model checkpoint")
    parser.add_argument("--image", type=str, default=None, help="Path to single image file (optional)")
    parser.add_argument("--data-path", type=str, default=str(config.DATA_PATH), help="Data directory (if no image specified)")
    parser.add_argument("--model-name", type=str, default=config.MODEL_NAME, choices=["Swin-T", "Swin-S", "Swin-B", "Swin-L"])
    parser.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    parser.add_argument("--output", type=str, default=str(config.FIGURES_DIR / "gradcam_sample.png"), help="Output plot destination")
    args = parser.parse_args()

    device = config.get_device()

    # Load model
    model = build_swin_v2(
        model_name=args.model_name,
        img_size=args.img_size,
        num_classes=config.NUM_CLASSES,
    ).to(device)

    chk_path = Path(args.checkpoint)
    if chk_path.exists():
        print(f"Loading checkpoint from {chk_path}...")
        state_dict = torch.load(chk_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Checkpoint '{chk_path}' not found. Using untrained weights for demonstration.")

    # Target layer: last norm layer of last stage block
    target_layer = model.layers[-1].blocks[-1].norm2
    grad_cam = GradCAM(model, target_layer)

    class_name = None
    if args.image:
        print(f"Loading image from {args.image}...")
        raw_img = Image.open(args.image).convert("RGB")
        _, test_transform = get_transforms(img_size=args.img_size)
        sample_tensor = test_transform(raw_img).unsqueeze(0).to(device)
    else:
        print("Sampling an image from Food-101 test dataset...")
        _, _, test_loader, classes = get_dataloaders(
            data_path=args.data_path,
            batch_size=1,
            num_workers=0,
            img_size=args.img_size,
        )
        sample_tensor, sample_label = next(iter(test_loader))
        sample_tensor = sample_tensor.to(device)
        class_name = classes[sample_label.item()]

    heatmap, pred_class = grad_cam.generate_heatmap(sample_tensor)
    label_text = class_name if class_name else f"Class {pred_class}"

    plot_gradcam(
        image_tensor=sample_tensor,
        cam_heatmap=heatmap,
        class_label=label_text,
        save_path=args.output,
    )


if __name__ == "__main__":
    main()

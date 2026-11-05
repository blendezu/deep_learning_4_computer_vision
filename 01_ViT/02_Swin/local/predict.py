"""
Inference script for Swin Transformer V2.
Predicts top-k classes for an input image and optionally generates GradCAM heatmap.
"""

import argparse
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import torch
import torch.nn.functional as F

import config
from dataset import get_dataloaders, get_transforms
from gradcam import GradCAM, plot_gradcam
from model import build_swin_v2


def predict_image(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    class_names: List[str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Runs model inference and returns top-k class names and probabilities."""
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(class_names)))
    results = [
        (class_names[idx.item()], prob.item())
        for idx, prob in zip(top_indices, top_probs)
    ]
    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference using Swin Transformer V2")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default=str(config.BEST_CHECKPOINT_PATH), help="Path to checkpoint")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top classes to return")
    parser.add_argument("--model-name", type=str, default=config.MODEL_NAME, choices=["Swin-T", "Swin-S", "Swin-B", "Swin-L"])
    parser.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    parser.add_argument("--data-path", type=str, default=str(config.DATA_PATH), help="Food101 data path for class names")
    parser.add_argument("--gradcam", action="store_true", help="Generate and save GradCAM visualization")
    parser.add_argument("--gradcam-out", type=str, default=str(config.FIGURES_DIR / "predict_gradcam.png"))
    args = parser.parse_args()

    device = config.get_device()
    print(f"Device: {device}")

    # Load class names
    _, _, _, classes = get_dataloaders(
        data_path=args.data_path,
        batch_size=1,
        num_workers=0,
        img_size=args.img_size,
        download=False,
    )

    # Initialize model
    model = build_swin_v2(
        model_name=args.model_name,
        img_size=args.img_size,
        num_classes=len(classes),
    ).to(device)

    chk_path = Path(args.checkpoint)
    if chk_path.exists():
        print(f"Loading checkpoint weights from {chk_path}...")
        state_dict = torch.load(chk_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Checkpoint {chk_path} not found. Running with initial weights.")

    # Prepare image
    raw_img = Image.open(args.image).convert("RGB")
    _, test_transform = get_transforms(img_size=args.img_size)
    tensor_img = test_transform(raw_img).unsqueeze(0).to(device)

    # Prediction
    results = predict_image(model, tensor_img, classes, top_k=args.top_k)
    print("\n--- Predictions ---")
    for rank, (name, prob) in enumerate(results, start=1):
        print(f"{rank}. {name:25s} ({prob * 100:6.2f}%)")

    # GradCAM option
    if args.gradcam:
        target_layer = model.layers[-1].blocks[-1].norm2
        cam = GradCAM(model, target_layer)
        heatmap, top_class = cam.generate_heatmap(tensor_img)
        plot_gradcam(
            image_tensor=tensor_img,
            cam_heatmap=heatmap,
            class_label=f"{results[0][0]} ({results[0][1]*100:.1f}%)",
            save_path=args.gradcam_out,
        )


if __name__ == "__main__":
    main()

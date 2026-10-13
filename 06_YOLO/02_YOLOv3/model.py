"""YOLOv3 Architecture implementation using PyTorch and timm Darknet-53 backbone.

Consists of:
- Darknet-53 backbone (ImageNet pre-trained)
- Feature Pyramid Network (FPN) style top-down routing and lateral concatenations
- 3 Detection Heads at scales 13x13, 26x26, and 52x52
"""

import timm
import torch
from config import NUM_ANCHORS_PER_SCALE, NUM_CLASSES
from torch import nn


class ConvBnLeaky(nn.Module):
    """Basic convolutional block: Conv2d + BatchNorm2d + LeakyReLU(0.1)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaky(self.bn(self.conv(x)))


class YOLOv3DetectionBlock(nn.Module):
    """Detection block for a single scale.

    Applies a 5-layer alternating 1x1 and 3x3 conv sequence.
    Produces:
        1. Detection output tensor: Conv(3x3) -> Conv(1x1)
        2. Route tensor: Conv(1x1) for upsampling to the next finer scale.
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int = NUM_ANCHORS_PER_SCALE,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        mid_channels = in_channels // 2

        # 5 convolutional layers (alternating 1x1 and 3x3)
        self.conv1 = ConvBnLeaky(in_channels, mid_channels, kernel_size=1)
        self.conv2 = ConvBnLeaky(mid_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = ConvBnLeaky(in_channels, mid_channels, kernel_size=1)
        self.conv4 = ConvBnLeaky(mid_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = ConvBnLeaky(in_channels, mid_channels, kernel_size=1)

        # Route branch for the next finer resolution scale
        self.route_conv = ConvBnLeaky(mid_channels, mid_channels, kernel_size=1)

        # Detection branch
        self.det_conv = ConvBnLeaky(mid_channels, in_channels, kernel_size=3, padding=1)
        self.det_out = nn.Conv2d(
            in_channels,
            num_anchors * (5 + num_classes),
            kernel_size=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        route = self.route_conv(x)
        det = self.det_conv(x)
        det = self.det_out(det)

        return det, route


class YOLOv3(nn.Module):
    """YOLOv3 Object Detector with ImageNet-pretrained Darknet-53 backbone from timm.

    Produces detections at 3 scales:
      - Scale 0 (13x13): large objects
      - Scale 1 (26x26): medium objects
      - Scale 2 (52x52): small objects
    """

    def __init__(
        self,
        num_anchor_per_scale: int = NUM_ANCHORS_PER_SCALE,
        num_classes: int = NUM_CLASSES,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.num_anchors = num_anchor_per_scale
        self.num_classes = num_classes

        # 1. Darknet-53 backbone from timm
        # Extract features at stages 3 (256ch), 4 (512ch), 5 (1024ch)
        self.backbone = timm.create_model(
            "darknet53",
            pretrained=pretrained_backbone,
            features_only=True,
            out_indices=(3, 4, 5),
        )

        # 2. Detection Head Scale 0: Stage 5 features (1024ch -> 13x13 for 416 input)
        self.det_block_0 = YOLOv3DetectionBlock(1024, num_anchor_per_scale, num_classes)

        # Top-down FPN route: Scale 0 route (512ch) -> upsample -> concat with stage 4 (512ch)
        self.upsample_0 = nn.Upsample(scale_factor=2, mode="nearest")
        self.merge_conv_0 = ConvBnLeaky(512 + 512, 512, kernel_size=1)

        # 3. Detection Head Scale 1: Merged features (512ch -> 26x26 for 416 input)
        self.det_block_1 = YOLOv3DetectionBlock(512, num_anchor_per_scale, num_classes)

        # Top-down FPN route: Scale 1 route (256ch) -> upsample -> concat with stage 3 (256ch)
        self.upsample_1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.merge_conv_1 = ConvBnLeaky(256 + 256, 256, kernel_size=1)

        # 4. Detection Head Scale 2: Merged features (256ch -> 52x52 for 416 input)
        self.det_block_2 = YOLOv3DetectionBlock(256, num_anchor_per_scale, num_classes)

        # Initialize head weights
        self._init_head_weights()

    def _init_head_weights(self):
        """Initialize newly instantiated detection heads with Kaiming normal."""
        for module in [
            self.det_block_0,
            self.det_block_1,
            self.det_block_2,
            self.merge_conv_0,
            self.merge_conv_1,
        ]:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        layer.weight, mode="fan_out", nonlinearity="leaky_relu"
                    )
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1.0)
                    nn.init.constant_(layer.bias, 0.0)

    def freeze_backbone(
        self, freeze: bool = True, unfreeze_stages: tuple[int, ...] = ()
    ):
        """Freeze or unfreeze backbone parameters.

        Args:
            freeze: If True, freezes the entire backbone.
            unfreeze_stages: If freeze=False, specify which stages to unfreeze
        """
        for name, param in self.backbone.named_parameters():
            if freeze:
                param.requires_grad = False
            else:
                if not unfreeze_stages:
                    param.requires_grad = True
                else:
                    param.requires_grad = any(
                        f"stages_{s}" in name or f"stages.{s}" in name
                        for s in unfreeze_stages
                    )

    def get_parameter_counts(self) -> tuple[float, float]:
        """Return total and trainable parameter counts in millions."""
        total = sum(p.numel() for p in self.parameters()) / 1e6
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        return total, trainable

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            x: Input image tensor (B, 3, H, W).

        Returns:
            List of 3 tensors [scale_0, scale_1, scale_2], each shaped:
            (B, num_anchors, grid_h, grid_w, 5 + num_classes).
            Activations applied:
              - tx, ty (indices 0, 1): sigmoid
              - tw, th (indices 2, 3): linear log-offsets
              - objectness (index 4): sigmoid
              - class scores (indices 5:): sigmoid
        """
        feat_s3, feat_s4, feat_s5 = self.backbone(x)

        # Scale 0 (large objects, coarse grid 13x13)
        det_0, route_0 = self.det_block_0(feat_s5)

        # Scale 1 (medium objects, 26x26)
        route_0_up = self.upsample_0(route_0)
        merged_1 = torch.cat([route_0_up, feat_s4], dim=1)
        merged_1 = self.merge_conv_0(merged_1)
        det_1, route_1 = self.det_block_1(merged_1)

        # Scale 2 (small objects, 52x52)
        route_1_up = self.upsample_1(route_1)
        merged_2 = torch.cat([route_1_up, feat_s3], dim=1)
        merged_2 = self.merge_conv_1(merged_2)
        det_2, _ = self.det_block_2(merged_2)

        outputs = []
        for det in [det_0, det_1, det_2]:
            B, _, H, W = det.shape
            det = det.view(B, self.num_anchors, 5 + self.num_classes, H, W)
            det = det.permute(0, 1, 3, 4, 2).contiguous()

            # Apply sigmoid to tx, ty
            det[..., 0:2] = torch.sigmoid(det[..., 0:2])
            # Apply sigmoid to objectness
            det[..., 4] = torch.sigmoid(det[..., 4])
            # Apply sigmoid to class scores
            det[..., 5:] = torch.sigmoid(det[..., 5:])

            outputs.append(det)

        return outputs


if __name__ == "__main__":
    test_model = YOLOv3(pretrained_backbone=False)
    dummy_input = torch.randn(2, 3, 416, 416)
    outs = test_model(dummy_input)
    print("Model test forward successful:")
    for i, out in enumerate(outs):
        print(f"  Scale {i} output shape: {out.shape}")

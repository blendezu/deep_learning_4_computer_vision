"""YOLOv3 Multi-Part Loss Function.

Calculates:
1. Coordinate Loss: MSE on (tx, ty, tw, th) for positive anchors
2. Objectness Loss: BCE on positive anchors and negative anchors (excluding ignored ones)
3. Classification Loss: BCE (independent logistic classifiers) for positive anchors
"""

import torch
from config import (
    LAMBDA_CLASS,
    LAMBDA_COORD,
    LAMBDA_NOOBJ,
    LAMBDA_OBJ,
)
from torch import nn


class YOLOv3Loss(nn.Module):
    """Multi-scale loss calculation for YOLOv3."""

    def __init__(
        self,
        lambda_coord: float = LAMBDA_COORD,
        lambda_obj: float = LAMBDA_OBJ,
        lambda_noobj: float = LAMBDA_NOOBJ,
        lambda_class: float = LAMBDA_CLASS,
    ):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class

        self.bce = nn.BCELoss(reduction="sum")
        self.mse = nn.MSELoss(reduction="sum")

    def forward(
        self, predictions: list[torch.Tensor], targets: list[torch.Tensor]
    ) -> torch.Tensor:
        """Compute YOLOv3 loss for a batch across all 3 detection scales.

        Args:
            predictions: List of 3 tensors [ (B, A, H_s, W_s, 5+C), ... ]
            targets: List of 3 tensors [ (B, A, H_s, W_s, 6+C), ... ]
                     Target layout per anchor:
                       [0:2]: tx, ty
                       [2:4]: tw, th
                       [4]:   obj_flag (1.0 = positive, 0.0 = negative, -1.0 = ignore)
                       [5]:   has_obj
                       [6:]:  one-hot class vector (length C)

        Returns:
            Scalar batch average total loss.
        """
        batch_size = predictions[0].size(0)
        device = predictions[0].device
        total_loss = torch.tensor(0.0, device=device)

        with torch.autocast(device_type=device.type, enabled=False):
            # Ensure float32 calculations for numerical stability
            preds = [p.float() for p in predictions]
            tgts = [t.float() for t in targets]

            for scale_idx in range(len(preds)):
                pred = preds[scale_idx]  # (B, A, H, W, 5+C)
                tgt = tgts[scale_idx]  # (B, A, H, W, 6+C)

                obj_mask = tgt[..., 4] == 1.0  # Positive anchors
                noobj_mask = tgt[..., 4] == 0.0  # Negative anchors (ignores -1.0)

                # --- 1. Coordinate Loss ---
                if obj_mask.sum() > 0:
                    coord_loss = self.mse(
                        pred[obj_mask][..., 0:2], tgt[obj_mask][..., 0:2]
                    )
                    coord_loss += self.mse(
                        pred[obj_mask][..., 2:4], tgt[obj_mask][..., 2:4]
                    )
                    total_loss += self.lambda_coord * coord_loss

                # --- 2. Objectness Loss ---
                if obj_mask.sum() > 0:
                    pred_obj = pred[obj_mask][..., 4].clamp(1e-7, 1.0 - 1e-7)
                    obj_loss = self.bce(pred_obj, torch.ones_like(pred_obj))
                    total_loss += self.lambda_obj * obj_loss

                if noobj_mask.sum() > 0:
                    pred_noobj = pred[noobj_mask][..., 4].clamp(1e-7, 1.0 - 1e-7)
                    noobj_loss = self.bce(pred_noobj, torch.zeros_like(pred_noobj))
                    total_loss += self.lambda_noobj * noobj_loss

                # --- 3. Class Loss ---
                if obj_mask.sum() > 0:
                    pred_cls = pred[obj_mask][..., 5:].clamp(1e-7, 1.0 - 1e-7)
                    class_loss = self.bce(pred_cls, tgt[obj_mask][..., 6:])
                    total_loss += self.lambda_class * class_loss

        return total_loss / max(batch_size, 1)

"""Box decoding and suppression for detection heads."""

import torch
from torch import nn

from operators.activation import Softmax
from operators.convolution import Conv2d


class BoxIoU(nn.Module):
    """Pairwise intersection over union of two ``(n, 4)`` xyxy box sets."""

    def forward(
        self, boxes1: torch.Tensor, boxes2: torch.Tensor
    ) -> torch.Tensor:
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        low = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        high = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
        extent = (high - low).clamp(min=0)
        intersection = extent[..., 0] * extent[..., 1]
        union = area1[:, None] + area2[None, :] - intersection
        return intersection / union


class DistanceToBox(nn.Module):
    """Turn (left, top, right, bottom) distances from anchors into boxes."""

    def __init__(self, xywh: bool = True) -> None:
        super().__init__()
        self.xywh = xywh

    def forward(
        self, distance: torch.Tensor, anchor_points: torch.Tensor
    ) -> torch.Tensor:
        low = anchor_points - distance[..., :2]
        high = anchor_points + distance[..., 2:]
        if self.xywh:
            return torch.cat(((low + high) / 2, high - low), dim=-1)
        return torch.cat((low, high), dim=-1)


class DFL(nn.Module):
    """Distribution focal loss decode: the expected bin index."""

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.conv = Conv2d(reg_max, 1, 1, bias=False)
        # Fixed weights make the 1x1 conv an expectation over bin indices.
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.arange(reg_max, dtype=torch.float32).reshape(
                    1, reg_max, 1, 1
                )
            )
        self.conv.weight.requires_grad_(False)
        self.softmax = Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 4 * reg_max, anchors) -> (batch, 4, anchors)
        batch, _, anchors = x.shape
        bins = x.reshape(batch, 4, self.reg_max, anchors).transpose(2, 1)
        return self.conv(self.softmax(bins)).reshape(batch, 4, anchors)


class NMS(nn.Module):
    """Greedy non-maximum suppression, per class via the coordinate offset."""

    def __init__(self, iou_threshold: float) -> None:
        super().__init__()
        self.iou_threshold = iou_threshold
        self.iou = BoxIoU()

    def forward(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return kept indices in decreasing score order."""
        if boxes.shape[0] == 0:
            return torch.zeros(0, dtype=torch.long, device=boxes.device)
        if classes is not None:
            # Shift each class into its own region so classes never overlap.
            offset = classes.to(boxes.dtype) * (boxes.max() + 1)
            boxes = boxes + offset[:, None]
        order = scores.argsort(descending=True)
        iou = self.iou(boxes[order], boxes[order])
        suppressed = torch.zeros(
            order.shape[0], dtype=torch.bool, device=boxes.device
        )
        keep = []
        for i in range(order.shape[0]):
            if suppressed[i]:
                continue
            keep.append(order[i])
            suppressed = suppressed | (iou[i] > self.iou_threshold)
        return torch.stack(keep)

"""Box coding, decoding, suppression and pooling for detection heads."""

import math

import torch
from torch import nn

from operators.activation import Sigmoid, Softmax
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


class BoxCoder(nn.Module):
    """Faster R-CNN regression targets: centre shifts and log size ratios."""

    def __init__(
        self,
        weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        clip: float | None = math.log(1000.0 / 16),
    ) -> None:
        super().__init__()
        self.weights = weights
        # torchvision clamps log ratios so exp cannot overflow; effdet does
        # not, so the clamp is optional.
        self.clip = clip

    def centres(
        self, boxes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        width = boxes[..., 2] - boxes[..., 0]
        height = boxes[..., 3] - boxes[..., 1]
        x = boxes[..., 0] + 0.5 * width
        y = boxes[..., 1] + 0.5 * height
        return x, y, width, height

    def encode(
        self, targets: torch.Tensor, anchors: torch.Tensor
    ) -> torch.Tensor:
        wx, wy, ww, wh = self.weights
        ax, ay, aw, ah = self.centres(anchors)
        tx, ty, tw, th = self.centres(targets)
        deltas = (
            wx * (tx - ax) / aw,
            wy * (ty - ay) / ah,
            ww * torch.log(tw / aw),
            wh * torch.log(th / ah),
        )
        return torch.stack(deltas, dim=-1)

    def forward(
        self, deltas: torch.Tensor, anchors: torch.Tensor
    ) -> torch.Tensor:
        """Decode ``(..., 4)`` deltas against ``(..., 4)`` xyxy anchors."""
        wx, wy, ww, wh = self.weights
        ax, ay, aw, ah = self.centres(anchors)
        dx, dy = deltas[..., 0] / wx, deltas[..., 1] / wy
        dw, dh = deltas[..., 2] / ww, deltas[..., 3] / wh
        if self.clip is not None:
            dw, dh = dw.clamp(max=self.clip), dh.clamp(max=self.clip)
        x, y = dx * aw + ax, dy * ah + ay
        half_w, half_h = 0.5 * torch.exp(dw) * aw, 0.5 * torch.exp(dh) * ah
        corners = (x - half_w, y - half_h, x + half_w, y + half_h)
        return torch.stack(corners, dim=-1)


class AnchorDecode(nn.Module):
    """YOLOv5's anchor-based box decode from raw head logits."""

    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = Sigmoid()

    def forward(
        self,
        logits: torch.Tensor,
        grid: torch.Tensor,
        anchors: torch.Tensor,
        stride: float,
    ) -> torch.Tensor:
        # logits: (..., 4) as (x, y, w, h); grid: (..., 2) cell indices;
        # anchors: (..., 2) in pixels. Returns xywh boxes in pixels.
        activated = self.sigmoid(logits)
        xy = (activated[..., :2] * 2.0 - 0.5 + grid) * stride
        wh = (activated[..., 2:] * 2.0) ** 2 * anchors
        return torch.cat((xy, wh), dim=-1)


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


class RoIAlign(nn.Module):
    """Average bilinear samples on a regular grid inside each region."""

    def __init__(
        self,
        output_size: tuple[int, int],
        spatial_scale: float,
        sampling_ratio: int,
        aligned: bool = False,
    ) -> None:
        super().__init__()
        if sampling_ratio <= 0:
            raise ValueError("adaptive sampling_ratio is not supported")
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def bilinear(
        self,
        features: torch.Tensor,
        batch_index: torch.Tensor,
        ys: torch.Tensor,
        xs: torch.Tensor,
    ) -> torch.Tensor:
        """Sample ``(regions, points)`` coordinates per channel."""
        _, channels, height, width = features.shape
        # torchvision's rules: zero more than a pixel outside, clamp the
        # rest, and collapse the top tap onto the last row or column.
        inside = (ys >= -1) & (ys <= height) & (xs >= -1) & (xs <= width)
        ys, xs = ys.clamp(min=0), xs.clamp(min=0)
        y_low, x_low = ys.floor(), xs.floor()
        y_top, x_top = y_low >= height - 1, x_low >= width - 1
        y_low = torch.where(y_top, height - 1.0, y_low)
        x_low = torch.where(x_top, width - 1.0, x_low)
        y_high = torch.where(y_top, y_low, y_low + 1)
        x_high = torch.where(x_top, x_low, x_low + 1)
        ys, xs = torch.where(y_top, y_low, ys), torch.where(x_top, x_low, xs)
        ly, lx = ys - y_low, xs - x_low
        hy, hx = 1 - ly, 1 - lx
        flat = features.permute(1, 0, 2, 3).reshape(channels, -1)
        base = batch_index[:, None] * (height * width)
        out = torch.zeros(
            ys.shape[0],
            channels,
            ys.shape[1],
            dtype=features.dtype,
            device=features.device,
        )
        corners = (
            (y_low, x_low, hy * hx),
            (y_low, x_high, hy * lx),
            (y_high, x_low, ly * hx),
            (y_high, x_high, ly * lx),
        )
        for y, x, weight in corners:
            index = (base + y * width + x).long().reshape(-1)
            gathered = flat[:, index].reshape(channels, *ys.shape)
            gathered = gathered.permute(1, 0, 2)
            out = out + gathered * (weight * inside).to(out.dtype)[:, None, :]
        return out

    def forward(
        self, features: torch.Tensor, rois: torch.Tensor
    ) -> torch.Tensor:
        # features: (batch, channels, height, width)
        # rois: (regions, 5) rows of batch index then xyxy in image pixels
        out_h, out_w = self.output_size
        grid = self.sampling_ratio
        offset = 0.5 if self.aligned else 0.0
        boxes = rois[:, 1:].to(torch.float32) * self.spatial_scale - offset
        start_x, start_y = boxes[:, 0], boxes[:, 1]
        roi_w, roi_h = boxes[:, 2] - start_x, boxes[:, 3] - start_y
        if not self.aligned:
            roi_w, roi_h = roi_w.clamp(min=1.0), roi_h.clamp(min=1.0)
        bin_w, bin_h = roi_w / out_w, roi_h / out_h
        # Sample (iy, ix) sits at fraction (iy + 0.5) / grid inside bin ph.
        fractions = (torch.arange(grid, device=rois.device) + 0.5) / grid
        steps_y = torch.arange(out_h, device=rois.device)[:, None] + fractions
        steps_x = torch.arange(out_w, device=rois.device)[:, None] + fractions
        ys = start_y[:, None] + bin_h[:, None] * steps_y.reshape(1, -1)
        xs = start_x[:, None] + bin_w[:, None] * steps_x.reshape(1, -1)
        ys = ys[:, :, None].expand(-1, -1, xs.shape[1]).reshape(len(rois), -1)
        xs = xs[:, None, :].expand(-1, out_h * grid, -1).reshape(len(rois), -1)
        samples = self.bilinear(features, rois[:, 0].long(), ys, xs)
        samples = samples.reshape(len(rois), -1, out_h, grid, out_w, grid)
        return samples.mean(dim=(3, 5))

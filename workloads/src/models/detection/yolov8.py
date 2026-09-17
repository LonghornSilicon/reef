"""YOLOv8 and YOLOv3u detectors built from the operator library."""

import math

import torch
from torch import nn

from configs.detection.yolo11 import YOLO11Config
from configs.detection.yolov8 import YOLOV8_CONFIGS, YOLOv8Config
from operators.activation import Sigmoid, SiLU
from operators.convolution import Conv2d
from operators.detection import DFL, NMS, DistanceToBox
from operators.interpolation import Interpolate
from operators.normalization import BatchNorm2d
from operators.pooling import MaxPool2d

# ultralytics.utils.torch_utils.initialize_weights overrides torch's defaults.
BN_EPS = 1e-3
BN_MOMENTUM = 0.03

STRIDES = (8, 16, 32)


def make_divisible(x: float, divisor: int) -> int:
    return math.ceil(x / divisor) * divisor


class Conv(nn.Module):
    """Convolution, batch norm and SiLU (``ultralytics.nn.modules.Conv``)."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = Conv2d(
            c1, c2, k, s, padding=k // 2, groups=groups, bias=False
        )
        self.bn = BatchNorm2d(c2, eps=BN_EPS, momentum=BN_MOMENTUM)
        self.act = SiLU() if act else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        return x if self.act is None else self.act(x)


class Bottleneck(nn.Module):
    """Two convolutions with an optional residual."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
    ) -> None:
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, k[0])
        self.cv2 = Conv(hidden, c2, k[1])
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """CSP stage: split, chain bottlenecks on one half, concat all."""

    def __init__(
        self, c1: int, c2: int, n: int, shortcut: bool = False, e: float = 0.5
    ) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c)
        self.cv2 = Conv((2 + n) * self.c, c2)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, e=1.0) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial pyramid pooling as three chained 5x5 max pools."""

    def __init__(self, c1: int, c2: int, k: int = 5) -> None:
        super().__init__()
        hidden = c1 // 2
        self.cv1 = Conv(c1, hidden)
        self.cv2 = Conv(hidden * 4, c2)
        self.m = MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class Concat(nn.Module):
    """Channel concatenation of the routed feature maps."""

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, 1)


def make_anchors(
    feats: list[torch.Tensor], offset: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cell centres ``(anchors, 2)`` and their strides ``(anchors,)``."""
    points, strides = [], []
    for feat, stride in zip(feats, STRIDES, strict=True):
        h, w = feat.shape[-2:]
        sx = torch.arange(w, dtype=feat.dtype, device=feat.device) + offset
        sy = torch.arange(h, dtype=feat.dtype, device=feat.device) + offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        points.append(torch.stack((sx, sy), -1).reshape(-1, 2))
        strides.append(
            torch.full((h * w,), stride, dtype=feat.dtype, device=feat.device)
        )
    return torch.cat(points), torch.cat(strides)


class Detect(nn.Module):
    """Anchor-free head: DFL box branch (cv2) and class branch (cv3)."""

    def __init__(
        self, nc: int, reg_max: int, ch: list[int], legacy: bool
    ) -> None:
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.stride = STRIDES
        c2 = max(16, ch[0] // 4, reg_max * 4)
        c3 = max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), Conv(c2, c2, 3), Conv2d(c2, 4 * reg_max, 1)
            )
            for x in ch
        )
        # legacy is ultralytics' name for the v3/v5/v8 class branch; YOLO11
        # replaces its two 3x3 convolutions with depthwise-pointwise pairs.
        if legacy:
            self.cv3 = nn.ModuleList(
                nn.Sequential(
                    Conv(x, c3, 3), Conv(c3, c3, 3), Conv2d(c3, nc, 1)
                )
                for x in ch
            )
        else:
            self.cv3 = nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(Conv(x, x, 3, groups=x), Conv(x, c3)),
                    nn.Sequential(Conv(c3, c3, 3, groups=c3), Conv(c3, c3)),
                    Conv2d(c3, nc, 1),
                )
                for x in ch
            )
        self.dfl = DFL(reg_max)
        self.decode = DistanceToBox(xywh=True)
        self.sigmoid = Sigmoid()

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        batch = feats[0].shape[0]
        boxes = torch.cat(
            [
                box(feat).reshape(batch, 4 * self.reg_max, -1)
                for box, feat in zip(self.cv2, feats, strict=True)
            ],
            -1,
        )
        scores = torch.cat(
            [
                cls(feat).reshape(batch, self.nc, -1)
                for cls, feat in zip(self.cv3, feats, strict=True)
            ],
            -1,
        )
        anchors, strides = make_anchors(feats)
        distance = self.dfl(boxes).transpose(1, 2)
        decoded = self.decode(distance, anchors).transpose(1, 2) * strides
        return torch.cat((decoded, self.sigmoid(scores)), 1)


# (from, repeats, block, args) rows transcribed from ultralytics' yaml files;
# args[0] is the unscaled output width.
Row = tuple[int | list[int], int, type[nn.Module], list]

CSP_ROWS: list[Row] = [
    (-1, 1, Conv, [64, 3, 2]),
    (-1, 1, Conv, [128, 3, 2]),
    (-1, 3, C2f, [128, True]),
    (-1, 1, Conv, [256, 3, 2]),
    (-1, 6, C2f, [256, True]),
    (-1, 1, Conv, [512, 3, 2]),
    (-1, 6, C2f, [512, True]),
    (-1, 1, Conv, [1024, 3, 2]),
    (-1, 3, C2f, [1024, True]),
    (-1, 1, SPPF, [1024, 5]),
    (-1, 1, Interpolate, []),
    ([-1, 6], 1, Concat, []),
    (-1, 3, C2f, [512]),
    (-1, 1, Interpolate, []),
    ([-1, 4], 1, Concat, []),
    (-1, 3, C2f, [256]),
    (-1, 1, Conv, [256, 3, 2]),
    ([-1, 12], 1, Concat, []),
    (-1, 3, C2f, [512]),
    (-1, 1, Conv, [512, 3, 2]),
    ([-1, 9], 1, Concat, []),
    (-1, 3, C2f, [1024]),
    ([15, 18, 21], 1, Detect, []),
]

DARKNET53_ROWS: list[Row] = [
    (-1, 1, Conv, [32, 3, 1]),
    (-1, 1, Conv, [64, 3, 2]),
    (-1, 1, Bottleneck, [64]),
    (-1, 1, Conv, [128, 3, 2]),
    (-1, 2, Bottleneck, [128]),
    (-1, 1, Conv, [256, 3, 2]),
    (-1, 8, Bottleneck, [256]),
    (-1, 1, Conv, [512, 3, 2]),
    (-1, 8, Bottleneck, [512]),
    (-1, 1, Conv, [1024, 3, 2]),
    (-1, 4, Bottleneck, [1024]),
    (-1, 1, Bottleneck, [1024, False]),
    (-1, 1, Conv, [512, 1, 1]),
    (-1, 1, Conv, [1024, 3, 1]),
    (-1, 1, Conv, [512, 1, 1]),
    (-1, 1, Conv, [1024, 3, 1]),
    (-2, 1, Conv, [256, 1, 1]),
    (-1, 1, Interpolate, []),
    ([-1, 8], 1, Concat, []),
    (-1, 1, Bottleneck, [512, False]),
    (-1, 1, Bottleneck, [512, False]),
    (-1, 1, Conv, [256, 1, 1]),
    (-1, 1, Conv, [512, 3, 1]),
    (-2, 1, Conv, [128, 1, 1]),
    (-1, 1, Interpolate, []),
    ([-1, 6], 1, Concat, []),
    (-1, 1, Bottleneck, [256, False]),
    (-1, 2, Bottleneck, [256, False]),
    ([27, 22, 15], 1, Detect, []),
]

BACKBONE_ROWS = {"csp": CSP_ROWS, "darknet53": DARKNET53_ROWS}

# Blocks that take the scaled repeat count as a constructor argument; any
# other block repeats as an nn.Sequential, as parse_model does.
REPEATED = (C2f,)


class Detector(nn.Module):
    """Layer list plus routing table, evaluated like ``BaseModel``."""

    def __init__(
        self,
        rows: list[Row],
        config: YOLOv8Config | YOLO11Config,
        legacy: bool,
        repeated: tuple[type[nn.Module], ...] = REPEATED,
    ) -> None:
        super().__init__()
        self.config = config
        self.sources = [row[0] for row in rows]
        layers: list[nn.Module] = []
        channels: list[int] = []
        for source, repeats, block, args in rows:
            n = max(round(repeats * config.depth), 1) if repeats > 1 else 1
            if block is Concat:
                module = Concat()
                out = sum(channels[i] for i in source)
            elif block is Interpolate:
                module = Interpolate(scale_factor=2, mode="nearest")
                out = channels[source]
            elif block is Detect:
                module = Detect(
                    config.num_classes,
                    config.reg_max,
                    [channels[i] for i in source],
                    legacy,
                )
                out = 0
            else:
                c1 = channels[source] if channels else 3
                out = make_divisible(
                    min(args[0], config.max_channels) * config.width, 8
                )
                args = [c1, out, *args[1:]]
                if block in repeated:
                    args.insert(2, n)
                    n = 1
                module = (
                    block(*args)
                    if n == 1
                    else nn.Sequential(*(block(*args) for _ in range(n)))
                )
            layers.append(module)
            channels.append(out)
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for source, layer in zip(self.sources, self.model, strict=True):
            if isinstance(source, list):
                x = [x if i == -1 else outputs[i] for i in source]
            elif source != -1:
                x = outputs[source]
            x = layer(x)
            outputs.append(x)
        return x


class YOLOv8(Detector):
    """YOLOv8 (CSP backbone) or YOLOv3u (Darknet-53) with the v8 head."""

    def __init__(self, config: YOLOv8Config) -> None:
        super().__init__(BACKBONE_ROWS[config.backbone], config, legacy=True)


def postprocess(
    preds: torch.Tensor,
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 300,
) -> list[torch.Tensor]:
    """Return per-image ``(n, 6)`` rows of xyxy box, score and class."""
    nms = NMS(iou)
    results = []
    for pred in preds.transpose(1, 2):
        score, cls = pred[:, 4:].max(1)
        keep = score > conf
        boxes, score, cls = pred[keep, :4], score[keep], cls[keep]
        half = boxes[:, 2:] / 2
        boxes = torch.cat((boxes[:, :2] - half, boxes[:, :2] + half), 1)
        index = nms(boxes, score, cls)[:max_det]
        rows = torch.cat((boxes, score[:, None], cls[:, None].to(boxes)), 1)
        results.append(rows[index])
    return results


def yolov8(name: str) -> YOLOv8:
    if name not in YOLOV8_CONFIGS:
        known = ", ".join(YOLOV8_CONFIGS)
        raise KeyError(f"unknown YOLOv8 size {name!r}; known sizes: {known}")
    return YOLOv8(YOLOV8_CONFIGS[name])

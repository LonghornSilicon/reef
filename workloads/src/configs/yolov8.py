"""Hyperparameters for the published YOLOv8 sizes and YOLOv3u."""

from dataclasses import dataclass
from typing import Literal

Backbone = Literal["csp", "darknet53"]


@dataclass(frozen=True)
class YOLOv8Config:
    """Compound-scaling hyperparameters for one anchor-free YOLOv8 size."""

    depth: float
    width: float
    max_channels: int
    backbone: Backbone = "csp"
    reg_max: int = 16
    num_classes: int = 80


YOLOV8_N = YOLOv8Config(depth=0.33, width=0.25, max_channels=1024)

YOLOV8_S = YOLOv8Config(depth=0.33, width=0.50, max_channels=1024)

YOLOV8_M = YOLOv8Config(depth=0.67, width=0.75, max_channels=768)

YOLOV8_L = YOLOv8Config(depth=1.00, width=1.00, max_channels=512)

YOLOV8_X = YOLOv8Config(depth=1.00, width=1.25, max_channels=512)

# yolov3.yaml has no scales block, so ultralytics applies no channel cap;
# 1024 is its widest layer, so this value is a no-op.
YOLOV3_U = YOLOv8Config(
    depth=1.0, width=1.0, max_channels=1024, backbone="darknet53"
)

YOLOV8_CONFIGS = {
    "YOLOv8-n": YOLOV8_N,
    "YOLOv8-s": YOLOV8_S,
    "YOLOv8-m": YOLOV8_M,
    "YOLOv8-l": YOLOV8_L,
    "YOLOv8-x": YOLOV8_X,
    "YOLOv3u": YOLOV3_U,
}

_RELEASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"

YOLOV8_SOURCES = {
    "YOLOv8-n": f"{_RELEASE}/yolov8n.pt",
    "YOLOv8-s": f"{_RELEASE}/yolov8s.pt",
    "YOLOv8-m": f"{_RELEASE}/yolov8m.pt",
    "YOLOv8-l": f"{_RELEASE}/yolov8l.pt",
    "YOLOv8-x": f"{_RELEASE}/yolov8x.pt",
    "YOLOv3u": f"{_RELEASE}/yolov3u.pt",
}

"""Hyperparameters for the published YOLO11 sizes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class YOLO11Config:
    """Compound-scaling hyperparameters for one YOLO11 size."""

    depth: float
    width: float
    max_channels: int
    # parse_model forces c3k=True in every C3k2 block for scales m/l/x,
    # overriding the False in the yaml.
    c3k: bool
    backbone: str = "csp"
    reg_max: int = 16
    num_classes: int = 80


YOLO11_N = YOLO11Config(depth=0.50, width=0.25, max_channels=1024, c3k=False)

YOLO11_S = YOLO11Config(depth=0.50, width=0.50, max_channels=1024, c3k=False)

YOLO11_M = YOLO11Config(depth=0.50, width=1.00, max_channels=512, c3k=True)

YOLO11_L = YOLO11Config(depth=1.00, width=1.00, max_channels=512, c3k=True)

YOLO11_X = YOLO11Config(depth=1.00, width=1.50, max_channels=512, c3k=True)

YOLO11_CONFIGS = {
    "YOLO11-n": YOLO11_N,
    "YOLO11-s": YOLO11_S,
    "YOLO11-m": YOLO11_M,
    "YOLO11-l": YOLO11_L,
    "YOLO11-x": YOLO11_X,
}

_RELEASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"

YOLO11_SOURCES = {
    "YOLO11-n": f"{_RELEASE}/yolo11n.pt",
    "YOLO11-s": f"{_RELEASE}/yolo11s.pt",
    "YOLO11-m": f"{_RELEASE}/yolo11m.pt",
    "YOLO11-l": f"{_RELEASE}/yolo11l.pt",
    "YOLO11-x": f"{_RELEASE}/yolo11x.pt",
}

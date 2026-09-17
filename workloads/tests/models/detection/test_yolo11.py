"""Randomly-initialized YOLO11 against ``ultralytics``."""

import pytest
import torch
from ultralytics.nn.tasks import DetectionModel

from configs.detection.yolo11 import YOLO11_CONFIGS
from models.detection.yolo11 import YOLO11, C3k, C3k2, yolo11
from tests.common import assert_matches

pytestmark = pytest.mark.unit

SIZES = list(YOLO11_CONFIGS)
SMALLEST = "YOLO11-n"

YAMLS = {
    "YOLO11-n": "yolo11n.yaml",
    "YOLO11-s": "yolo11s.yaml",
    "YOLO11-m": "yolo11m.yaml",
    "YOLO11-l": "yolo11l.yaml",
    "YOLO11-x": "yolo11x.yaml",
}

# ultralytics' DetectionModel(...).parameters() for each packaged yaml.
PARAMETER_COUNTS = {
    "YOLO11-n": 2_624_080,
    "YOLO11-s": 9_458_752,
    "YOLO11-m": 20_114_688,
    "YOLO11-l": 25_372_160,
    "YOLO11-x": 56_966_176,
}

# ultralytics needs a multiple of 32; 64 gives 8x8, 4x4 and 2x2 grids.
INPUT_SIZE = 64
NUM_CLASSES = 80


def build_pair(name: str) -> tuple[YOLO11, DetectionModel]:
    reference = DetectionModel(YAMLS[name], nc=NUM_CLASSES, verbose=False)
    ours = yolo11(name)
    ours.load_state_dict(reference.state_dict(), strict=True)
    return ours.eval(), reference.eval()


@pytest.mark.parametrize("name", SIZES)
def test_state_dict_keys_match_reference(name: str) -> None:
    ours, reference = build_pair(name)
    assert set(ours.state_dict()) == set(reference.state_dict())


@pytest.mark.parametrize("name", SIZES)
def test_inference_matches_reference(name: str) -> None:
    ours, reference = build_pair(name)
    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        actual = ours(x)
        expected, _ = reference(x)

    assert_matches(actual, expected, torch.float32)


def test_bf16_inference_matches_reference() -> None:
    ours, reference = build_pair(SMALLEST)
    ours = ours.to(torch.bfloat16)
    reference = reference.to(torch.bfloat16)
    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.bfloat16)

    with torch.no_grad():
        actual = ours(x)
        expected, _ = reference(x)

    assert_matches(actual, expected, torch.bfloat16)


@pytest.mark.parametrize("name", SIZES)
def test_parameter_count(name: str) -> None:
    with torch.device("meta"):
        model = yolo11(name)
    assert sum(p.numel() for p in model.parameters()) == PARAMETER_COUNTS[name]


def test_strides() -> None:
    assert yolo11(SMALLEST).model[-1].stride == (8, 16, 32)


@pytest.mark.parametrize(
    ("name", "c3k_layers"),
    [("YOLO11-n", (6, 8, 22)), ("YOLO11-m", (2, 4, 6, 8, 13, 16, 19, 22))],
)
def test_c3k_blocks_follow_parse_model(
    name: str, c3k_layers: tuple[int, ...]
) -> None:
    model = yolo11(name)
    found = tuple(
        index
        for index, layer in enumerate(model.model)
        if isinstance(layer, C3k2) and isinstance(layer.m[0], C3k)
    )
    assert found == c3k_layers


def test_unknown_size_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown YOLO11 size"):
        yolo11("YOLO11-xl")

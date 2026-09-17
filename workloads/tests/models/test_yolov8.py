"""Randomly-initialized YOLOv8 and YOLOv3u against ``ultralytics``."""

import pytest
import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.nms import non_max_suppression

from configs.yolov8 import YOLOV8_CONFIGS
from models.yolov8 import YOLOv8, postprocess, yolov8
from tests.common import assert_matches

pytestmark = pytest.mark.unit

SIZES = list(YOLOV8_CONFIGS)
SMALLEST = "YOLOv8-n"

YAMLS = {
    "YOLOv8-n": "yolov8n.yaml",
    "YOLOv8-s": "yolov8s.yaml",
    "YOLOv8-m": "yolov8m.yaml",
    "YOLOv8-l": "yolov8l.yaml",
    "YOLOv8-x": "yolov8x.yaml",
    "YOLOv3u": "yolov3.yaml",
}

# ultralytics' DetectionModel(...).parameters() for each packaged yaml.
PARAMETER_COUNTS = {
    "YOLOv8-n": 3_157_200,
    "YOLOv8-s": 11_166_560,
    "YOLOv8-m": 25_902_640,
    "YOLOv8-l": 43_691_520,
    "YOLOv8-x": 68_229_648,
    "YOLOv3u": 103_754_144,
}

# ultralytics needs a multiple of 32; 64 gives 8x8, 4x4 and 2x2 grids.
INPUT_SIZE = 64
NUM_CLASSES = 80


def build_pair(name: str) -> tuple[YOLOv8, DetectionModel]:
    reference = DetectionModel(YAMLS[name], nc=NUM_CLASSES, verbose=False)
    ours = yolov8(name)
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
        model = yolov8(name)
    assert sum(p.numel() for p in model.parameters()) == PARAMETER_COUNTS[name]


def test_strides() -> None:
    assert yolov8(SMALLEST).model[-1].stride == (8, 16, 32)


def test_dfl_is_the_softmax_expected_bin_index() -> None:
    dfl = yolov8(SMALLEST).model[-1].dfl
    reg_max = dfl.reg_max
    x = torch.randn(2, 4 * reg_max, 5)
    bins = torch.arange(reg_max, dtype=torch.float32)
    probabilities = x.reshape(2, 4, reg_max, 5).softmax(2)
    expected = (probabilities * bins[None, None, :, None]).sum(2)

    torch.testing.assert_close(dfl(x), expected)


def test_postprocess_matches_non_max_suppression() -> None:
    torch.manual_seed(0)
    anchors = 400
    centres = torch.rand(2, 2, anchors) * INPUT_SIZE
    sizes = torch.rand(2, 2, anchors) * 24 + 8
    scores = torch.sigmoid(torch.randn(2, NUM_CLASSES, anchors) * 2)
    preds = torch.cat((centres, sizes, scores), 1)

    # non_max_suppression rewrites its input's boxes in place.
    expected = non_max_suppression(preds.clone())
    actual = postprocess(preds)

    assert len(actual) == len(expected) == 2
    for ours, theirs in zip(actual, expected, strict=True):
        assert ours.shape[0] > 1
        torch.testing.assert_close(ours, theirs)


def test_unknown_size_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown YOLOv8 size"):
        yolov8("YOLOv8-xl")

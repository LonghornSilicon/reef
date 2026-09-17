"""Randomly-initialized MobileNetV3s against ``torchvision``."""

import pytest
import torch
import torch.nn.functional as F
import torchvision

from configs.mobilenet_v3 import MOBILENET_V3_CONFIGS, TORCHVISION_BUILDERS
from models.mobilenet_v3 import (
    Hardsigmoid,
    Hardswish,
    MobileNetV3,
    mobilenet_v3,
)
from tests.common import assert_matches

pytestmark = pytest.mark.unit

SIZES = list(MOBILENET_V3_CONFIGS)
INPUT_SIZE = 64

PARAMETER_COUNTS = {
    "MobileNetV3-Small": 2_542_856,
    "MobileNetV3-Large": 5_483_032,
}


def build_pair(name: str) -> tuple[MobileNetV3, torch.nn.Module]:
    builder = getattr(torchvision.models, TORCHVISION_BUILDERS[name])
    reference = builder(weights=None).eval()
    ours = mobilenet_v3(name)
    ours.load_state_dict(reference.state_dict(), strict=True)
    return ours.eval(), reference


@pytest.mark.parametrize("name", SIZES)
def test_state_dict_keys_match_torchvision(name: str) -> None:
    ours, reference = build_pair(name)
    assert set(ours.state_dict()) == set(reference.state_dict())


@pytest.mark.parametrize("name", SIZES)
def test_logits_match_torchvision(name: str) -> None:
    ours, reference = build_pair(name)
    x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        assert_matches(ours(x), reference(x), torch.float32)


def test_bf16_logits_match_torchvision() -> None:
    ours, reference = build_pair("MobileNetV3-Small")
    ours, reference = ours.bfloat16(), reference.bfloat16()
    x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE).bfloat16()

    with torch.no_grad():
        assert_matches(ours(x), reference(x), torch.bfloat16)


@pytest.mark.parametrize("name", SIZES)
def test_parameter_count_matches_published(name: str) -> None:
    with torch.device("meta"):
        model = mobilenet_v3(name)
    assert sum(p.numel() for p in model.parameters()) == PARAMETER_COUNTS[name]


def test_hard_activations_match_functional() -> None:
    x = torch.linspace(-6.0, 6.0, 1001)
    torch.testing.assert_close(Hardswish()(x), F.hardswish(x))
    torch.testing.assert_close(Hardsigmoid()(x), F.hardsigmoid(x))


def test_unknown_size_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown MobileNetV3 size"):
        mobilenet_v3("MobileNetV3-Medium")

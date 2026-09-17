"""Randomly-initialized VGGs against ``torchvision``."""

import pytest
import torch
import torchvision

from configs.classification.vgg import TORCHVISION_BUILDERS, VGG_CONFIGS
from models.classification.vgg import VGG, vgg
from operators.activation import ReLU
from operators.convolution import Conv2d
from operators.normalization import BatchNorm2d
from tests.common import assert_matches

pytestmark = pytest.mark.unit

SIZES = list(VGG_CONFIGS)
INPUT_SIZE = 64

PARAMETER_COUNTS = {
    "VGG-11": 132_863_336,
    "VGG-11-BN": 132_868_840,
    "VGG-13": 133_047_848,
    "VGG-13-BN": 133_053_736,
    "VGG-16": 138_357_544,
    "VGG-16-BN": 138_365_992,
    "VGG-19": 143_667_240,
    "VGG-19-BN": 143_678_248,
}


def build_pair(name: str) -> tuple[VGG, torch.nn.Module]:
    builder = getattr(torchvision.models, TORCHVISION_BUILDERS[name])
    reference = builder(weights=None).eval()
    ours = vgg(name)
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
    ours, reference = build_pair("VGG-11-BN")
    ours, reference = ours.bfloat16(), reference.bfloat16()
    x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE).bfloat16()

    with torch.no_grad():
        assert_matches(ours(x), reference(x), torch.bfloat16)


@pytest.mark.parametrize("name", SIZES)
def test_parameter_count_matches_published(name: str) -> None:
    with torch.device("meta"):
        model = vgg(name)
    assert sum(p.numel() for p in model.parameters()) == PARAMETER_COUNTS[name]


def test_batch_norm_sits_between_every_conv_and_relu() -> None:
    features = vgg("VGG-16-BN").features
    convs = [i for i, m in enumerate(features) if isinstance(m, Conv2d)]
    assert len(convs) == 13
    for i in convs:
        assert isinstance(features[i + 1], BatchNorm2d)
        assert isinstance(features[i + 2], ReLU)


def test_unknown_size_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown VGG size"):
        vgg("VGG-21")

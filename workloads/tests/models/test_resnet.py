"""Randomly-initialized ResNets against ``torchvision``."""

import pytest
import torch
import torchvision

from configs.resnet import RESNET_CONFIGS, TORCHVISION_BUILDERS
from models.resnet import ResNet, resnet

pytestmark = pytest.mark.unit

DEPTHS = list(RESNET_CONFIGS)

# Not 224: im2col is quadratic in feature-map size, and 64 still downsamples
# through every stage.
INPUT_SIZE = 64

# Worst measured deviation across the three depths is ~8e-6 absolute on
# random-init logits spanning +/-30.
TOLERANCE = {"rtol": 1e-4, "atol": 1e-4}


def build_pair(name: str) -> tuple[ResNet, torch.nn.Module]:
    builder = getattr(torchvision.models, TORCHVISION_BUILDERS[name])
    reference = builder(weights=None).eval()
    ours = resnet(name)
    ours.load_state_dict(reference.state_dict(), strict=True)
    return ours.eval(), reference


@pytest.mark.parametrize("name", DEPTHS)
def test_state_dict_keys_match_torchvision(name: str) -> None:
    ours, reference = build_pair(name)
    assert set(ours.state_dict()) == set(reference.state_dict())


@pytest.mark.parametrize("name", DEPTHS)
def test_logits_match_torchvision(name: str) -> None:
    ours, reference = build_pair(name)
    x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        actual, expected = ours(x), reference(x)

    torch.testing.assert_close(actual, expected, **TOLERANCE)
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))


@pytest.mark.parametrize("name", DEPTHS)
def test_parameter_count_matches_torchvision(name: str) -> None:
    ours, reference = build_pair(name)
    ours_total = sum(p.numel() for p in ours.parameters())
    reference_total = sum(p.numel() for p in reference.parameters())
    assert ours_total == reference_total


@pytest.mark.parametrize(
    ("name", "expansion", "blocks"),
    [
        ("ResNet-18", 1, (2, 2, 2, 2)),
        ("ResNet-34", 1, (3, 4, 6, 3)),
        ("ResNet-50", 4, (3, 4, 6, 3)),
    ],
)
def test_stage_depths_and_expansion(
    name: str, expansion: int, blocks: tuple[int, ...]
) -> None:
    model = resnet(name)
    config = RESNET_CONFIGS[name]
    assert config.expansion == expansion
    assert config.blocks_per_stage == blocks
    stages = (model.layer1, model.layer2, model.layer3, model.layer4)
    assert tuple(len(stage) for stage in stages) == blocks


def test_bottleneck_places_stride_on_the_3x3() -> None:
    # The V1 ordering loads the same checkpoint but computes different numbers.
    block = resnet("ResNet-50").layer2[0]
    assert block.conv1.stride == 1
    assert block.conv2.stride == 2


def test_unknown_depth_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown ResNet depth"):
        resnet("ResNet-200")

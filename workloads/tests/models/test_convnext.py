"""Randomly-initialized ConvNeXts against ``torchvision``."""

import pytest
import torch
import torchvision

from configs.convnext import CONVNEXT_CONFIGS, TORCHVISION_BUILDERS
from models.convnext import ConvNeXt, convnext
from tests.common import assert_matches

pytestmark = pytest.mark.unit

SIZES = list(TORCHVISION_BUILDERS)
PARITY_SIZES = ["ConvNeXt-T", "ConvNeXt-S"]
INPUT_SIZE = 64

PARAMETER_COUNTS = {
    "ConvNeXt-T": 28_589_128,
    "ConvNeXt-S": 50_223_688,
    "ConvNeXt-B": 88_591_464,
    "ConvNeXt-L": 197_767_336,
}


def build_pair(
    name: str, device: str = "cpu"
) -> tuple[ConvNeXt, torch.nn.Module]:
    builder = getattr(torchvision.models, TORCHVISION_BUILDERS[name])
    with torch.device(device):
        reference = builder(weights=None)
        ours = convnext(name)
    ours.load_state_dict(reference.state_dict(), strict=True)
    return ours.eval(), reference.eval()


@pytest.mark.parametrize("name", SIZES)
def test_state_dict_keys_match_torchvision(name: str) -> None:
    ours, reference = build_pair(name, device="meta")
    assert set(ours.state_dict()) == set(reference.state_dict())


@pytest.mark.parametrize("name", SIZES)
def test_parameter_count_matches_torchvision(name: str) -> None:
    ours, reference = build_pair(name, device="meta")
    ours_total = sum(p.numel() for p in ours.parameters())
    reference_total = sum(p.numel() for p in reference.parameters())
    assert ours_total == reference_total == PARAMETER_COUNTS[name]


@pytest.mark.parametrize("name", PARITY_SIZES)
def test_logits_match_torchvision(name: str) -> None:
    ours, reference = build_pair(name)
    x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        actual, expected = ours(x), reference(x)

    assert_matches(actual, expected, torch.float32)
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))


def test_bfloat16_logits_match_torchvision() -> None:
    ours, reference = build_pair("ConvNeXt-T")
    ours.to(torch.bfloat16)
    reference.to(torch.bfloat16)
    x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.bfloat16)

    with torch.no_grad():
        actual, expected = ours(x), reference(x)

    assert_matches(actual, expected, torch.bfloat16)


def test_layer_scale_starts_at_config_value() -> None:
    model = convnext("ConvNeXt-T")
    block = model.features[1][0]
    assert block.layer_scale.shape == (96, 1, 1)
    assert torch.all(
        block.layer_scale == CONVNEXT_CONFIGS["ConvNeXt-T"].layer_scale
    )
    x = torch.randn(1, 96, 8, 8)
    with torch.no_grad():
        # At 1e-6 the branch is negligible, so the block is near identity.
        torch.testing.assert_close(block(x), x, rtol=0, atol=1e-4)


def test_xlarge_builds_with_the_hub_head_eps() -> None:
    with torch.device("meta"):
        model = convnext("ConvNeXt-XL")
    assert model.classifier[0].eps == 1e-12
    assert model.classifier[2].weight.shape == (21841, 2048)


def test_unknown_size_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown ConvNeXt size"):
        convnext("ConvNeXt-XXL")

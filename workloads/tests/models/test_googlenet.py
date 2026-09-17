"""Randomly-initialized GoogLeNet against ``torchvision``."""

import pytest
import torch
import torchvision

from configs.googlenet import GOOGLENET_CONFIGS, TORCHVISION_BUILDERS
from models.googlenet import GoogLeNet, googlenet
from tests.common import assert_matches

pytestmark = pytest.mark.unit

SIZES = list(GOOGLENET_CONFIGS)

# 96 puts 6x6 maps into the aux heads, so their 4x4 adaptive pool takes the
# overlapping-window path.
INPUT_SIZE = 96

PARAMETER_COUNTS = {"GoogLeNet": 13_004_888}


def build_pair(name: str) -> tuple[GoogLeNet, torch.nn.Module]:
    builder = getattr(torchvision.models, TORCHVISION_BUILDERS[name])
    config = GOOGLENET_CONFIGS[name]
    # init_weights=None only differs by emitting a FutureWarning.
    reference = builder(
        weights=None, aux_logits=config.aux_logits, init_weights=True
    ).eval()
    ours = googlenet(name)
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
    ours, reference = build_pair("GoogLeNet")
    ours, reference = ours.bfloat16(), reference.bfloat16()
    x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE).bfloat16()

    with torch.no_grad():
        assert_matches(ours(x), reference(x), torch.bfloat16)


@pytest.mark.parametrize("name", SIZES)
def test_parameter_count_matches_published(name: str) -> None:
    with torch.device("meta"):
        model = googlenet(name)
    assert sum(p.numel() for p in model.parameters()) == PARAMETER_COUNTS[name]


def test_aux_heads_run_only_in_training() -> None:
    ours, reference = build_pair("GoogLeNet")
    x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        assert isinstance(ours(x), torch.Tensor)
        ours.train()
        reference.train()
        actual, expected = ours(x), reference(x)

    assert isinstance(actual, tuple)
    assert len(actual) == 3
    assert actual[0].shape == expected.logits.shape
    assert actual[1].shape == expected.aux_logits2.shape
    assert actual[2].shape == expected.aux_logits1.shape


def test_unknown_size_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown GoogLeNet size"):
        googlenet("Inception-v2")

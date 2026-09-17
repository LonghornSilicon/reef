"""Randomly-initialized Swin Transformers against ``torchvision``."""

import pytest
import torch
from torchvision.models.swin_transformer import SwinTransformer as Reference

from configs.swin import SWIN_CONFIGS
from models.swin import SwinTransformer, shift_mask, swin
from tests.common import assert_matches

pytestmark = pytest.mark.unit

SIZES = list(SWIN_CONFIGS)
PARITY_SIZES = ["Swin-T", "Swin-S"]
# 64 is a multiple of the 32x total stride; 72 is not, so every stage pads.
INPUT_SIZES = [64, 72]

PARAMETER_COUNTS = {
    "Swin-T": 28_288_354,
    "Swin-S": 49_606_258,
    "Swin-B": 87_768_224,
    "Swin-L": 196_532_476,
}


def build_pair(
    name: str, device: str = "cpu"
) -> tuple[SwinTransformer, torch.nn.Module]:
    config = SWIN_CONFIGS[name]
    with torch.device(device):
        # torchvision ships no swin_l builder, so every size is constructed
        # from the config; swin_t etc. differ only in stochastic depth.
        reference = Reference(
            patch_size=[config.patch_size] * 2,
            embed_dim=config.embed_dim,
            depths=list(config.depths),
            num_heads=list(config.num_heads),
            window_size=[config.window_size] * 2,
            num_classes=config.num_labels,
        )
        ours = swin(name)
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


@pytest.mark.parametrize("input_size", INPUT_SIZES)
@pytest.mark.parametrize("name", PARITY_SIZES)
def test_logits_match_torchvision(name: str, input_size: int) -> None:
    ours, reference = build_pair(name)
    x = torch.randn(2, 3, input_size, input_size)

    with torch.no_grad():
        actual, expected = ours(x), reference(x)

    assert_matches(actual, expected, torch.float32)
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))


def test_bfloat16_logits_match_torchvision() -> None:
    ours, reference = build_pair("Swin-T")
    ours.to(torch.bfloat16)
    reference.to(torch.bfloat16)
    x = torch.randn(2, 3, INPUT_SIZES[0], INPUT_SIZES[0], dtype=torch.bfloat16)

    with torch.no_grad():
        actual, expected = ours(x), reference(x)

    assert_matches(actual, expected, torch.bfloat16)


def test_shifted_blocks_mask_cross_region_attention() -> None:
    stage = swin("Swin-T").features[1]
    assert stage[0].attn.shift_size == 0
    assert stage[1].attn.shift_size == 3
    mask = shift_mask(14, 14, 7, 3, 3, torch.device("cpu"))
    assert mask.shape == (4, 49, 49)
    # The top-left window lies in one region; the bottom-right one straddles
    # the rolled-around rows and columns.
    assert torch.all(mask[0] == 0)
    assert torch.any(mask[-1] == -100)
    assert torch.all((mask == 0) | (mask == -100))


def test_unknown_size_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown Swin size"):
        swin("Swin-H")

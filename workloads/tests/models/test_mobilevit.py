"""Tiny randomly-initialized MobileViT against ``transformers``."""

import pytest
import torch
import transformers

from configs.mobilevit import (
    MOBILEVIT_CONFIGS,
    MOBILEVIT_REPOS,
    MobileViTConfig,
)
from models.mobilevit import MobileViTForImageClassification, mobilevit
from tests.common import DTYPES, assert_matches

pytestmark = pytest.mark.unit

TINY = {
    "hidden_sizes": (16, 16, 16),
    "neck_hidden_sizes": (8, 8, 8, 8, 8, 8, 16),
    "expand_ratio": 2.0,
    "num_attention_heads": 2,
    "num_labels": 5,
}
BATCH = 2
# 64 keeps every feature map a multiple of the patch size; 48 leaves the last
# MobileViT layer a 3x3 map that has to be resized to 4x4 and back.
IMAGE_SIZES = [64, 48]


def build_pair(
    dtype: torch.dtype = torch.float32,
) -> tuple[
    MobileViTForImageClassification,
    transformers.MobileViTForImageClassification,
]:
    config = MobileViTConfig(**TINY)
    reference = transformers.MobileViTForImageClassification(
        transformers.MobileViTConfig(
            mlp_ratio=config.mlp_ratio,
            patch_size=config.patch_size,
            conv_kernel_size=config.conv_kernel_size,
            output_stride=config.output_stride,
            layer_norm_eps=config.layer_norm_eps,
            **TINY,
        )
    ).eval()
    ours = MobileViTForImageClassification(config)
    ours.load_state_dict(reference.state_dict(), strict=True)
    return ours.eval().to(dtype), reference.to(dtype)


def test_state_dict_keys_match_reference() -> None:
    ours, reference = build_pair()
    assert set(ours.state_dict()) == set(reference.state_dict())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("size", IMAGE_SIZES)
def test_logits_match_reference(size: int, dtype: torch.dtype) -> None:
    ours, reference = build_pair(dtype)
    x = torch.randn(BATCH, 3, size, size, dtype=dtype)

    with torch.no_grad():
        expected = reference(pixel_values=x).logits
        actual = ours(x)

    assert_matches(actual, expected, dtype)


def test_fold_inverts_unfold() -> None:
    layer = mobilevit("MobileViT-XXS").mobilevit.encoder.layer[2]
    x = torch.randn(BATCH, 8, 6, 10)
    patches = layer.unfolding(x)
    assert patches.shape == (BATCH * 4, 15, 8)
    assert torch.equal(layer.folding(patches, x.shape), x)


@pytest.mark.parametrize("key", list(MOBILEVIT_CONFIGS))
def test_published_size_matches_hub(key: str) -> None:
    try:
        hub = transformers.AutoConfig.from_pretrained(MOBILEVIT_REPOS[key])
    except OSError as exc:
        pytest.skip(f"hub config unavailable: {exc}")
    with torch.device("meta"):
        ours = mobilevit(key)
        reference = transformers.AutoModelForImageClassification.from_config(
            hub
        )
    ours_shapes = {k: tuple(v.shape) for k, v in ours.state_dict().items()}
    hub_shapes = {k: tuple(v.shape) for k, v in reference.state_dict().items()}
    assert ours_shapes == hub_shapes
    ours_total = sum(p.numel() for p in ours.parameters())
    reference_total = sum(p.numel() for p in reference.parameters())
    assert ours_total == reference_total


def test_unknown_size_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown MobileViT size"):
        mobilevit("MobileViT-L")

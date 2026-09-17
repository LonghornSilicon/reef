"""Tiny randomly-initialized SegFormer against ``transformers``."""

import pytest
import torch
import transformers

from configs.vision_transformer.segformer import (
    SEGFORMER_CONFIGS,
    SEGFORMER_REPOS,
    SegformerConfig,
)
from models.vision_transformer.segformer import (
    SegformerForSemanticSegmentation,
    segformer,
)
from tests.common import DTYPES, assert_matches

pytestmark = pytest.mark.unit

TINY = {
    "hidden_sizes": (8, 16, 32, 64),
    "depths": (1, 1, 1, 1),
    "num_attention_heads": (1, 2, 4, 8),
    "decoder_hidden_size": 16,
    "num_labels": 5,
}
BATCH = 2
IMAGE_SIZE = 64


def build_pair(
    dtype: torch.dtype = torch.float32,
) -> tuple[
    SegformerForSemanticSegmentation,
    transformers.SegformerForSemanticSegmentation,
]:
    config = SegformerConfig(**TINY)
    reference = transformers.SegformerForSemanticSegmentation(
        transformers.SegformerConfig(
            sr_ratios=config.sr_ratios,
            patch_sizes=config.patch_sizes,
            strides=config.strides,
            mlp_ratios=config.mlp_ratios,
            reshape_last_stage=config.reshape_last_stage,
            **TINY,
        )
    ).eval()
    ours = SegformerForSemanticSegmentation(config)
    ours.load_state_dict(reference.state_dict(), strict=True)
    return ours.eval().to(dtype), reference.to(dtype)


def test_state_dict_keys_match_reference() -> None:
    ours, reference = build_pair()
    assert set(ours.state_dict()) == set(reference.state_dict())


@pytest.mark.parametrize("dtype", DTYPES)
def test_logits_match_reference(dtype: torch.dtype) -> None:
    ours, reference = build_pair(dtype)
    x = torch.randn(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=dtype)

    with torch.no_grad():
        expected = reference(pixel_values=x).logits
        actual = ours(x)

    assert_matches(actual, expected, dtype)


def test_logits_are_at_quarter_resolution() -> None:
    ours, _ = build_pair()
    x = torch.randn(1, 3, 96, IMAGE_SIZE)

    with torch.no_grad():
        logits = ours(x)

    assert logits.shape == (1, TINY["num_labels"], 24, IMAGE_SIZE // 4)


@pytest.mark.parametrize("key", list(SEGFORMER_CONFIGS))
def test_published_size_matches_hub(key: str) -> None:
    try:
        hub = transformers.AutoConfig.from_pretrained(SEGFORMER_REPOS[key])
    except OSError as exc:
        pytest.skip(f"hub config unavailable: {exc}")
    with torch.device("meta"):
        ours = segformer(key)
        reference = transformers.AutoModelForSemanticSegmentation.from_config(
            hub
        )
    ours_shapes = {k: tuple(v.shape) for k, v in ours.state_dict().items()}
    hub_shapes = {k: tuple(v.shape) for k, v in reference.state_dict().items()}
    assert ours_shapes == hub_shapes
    ours_total = sum(p.numel() for p in ours.parameters())
    reference_total = sum(p.numel() for p in reference.parameters())
    assert ours_total == reference_total


def test_unknown_size_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown SegFormer size"):
        segformer("SegFormer-B6")

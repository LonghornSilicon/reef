"""Tiny randomly-initialized SigLIP against ``transformers``."""

import pytest
import torch
import transformers
from transformers import AutoConfig

from configs.vision_language.siglip import (
    SIGLIP_REPOS,
    SiglipConfig,
    SiglipTextConfig,
    SiglipVisionConfig,
)
from models.vision_language.siglip import SiglipModel, siglip
from tests.common import DTYPES, assert_matches

pytestmark = pytest.mark.unit

TINY_VISION = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "patch_size": 8,
    "image_size": 32,
}
TINY_TEXT = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "projection_size": 32,
    "max_position_embeddings": 16,
    "vocab_size": 64,
}
BATCH = 2

# Measured on the meta device from the hub configs in transformers 5.17.0.
PARAMETER_COUNTS = {
    "SigLIP-B/16": 203_155_970,
    "SigLIP-L/16": 652_150_786,
    "SigLIP-So400m/14": 877_960_498,
}


def build_pair(
    dtype: torch.dtype = torch.float32,
) -> tuple[SiglipModel, transformers.SiglipModel]:
    config = SiglipConfig(
        vision=SiglipVisionConfig(**TINY_VISION),
        text=SiglipTextConfig(**TINY_TEXT),
    )
    reference = transformers.SiglipModel(
        transformers.SiglipConfig(
            vision_config=TINY_VISION, text_config=TINY_TEXT
        )
    ).eval()
    ours = SiglipModel(config)
    ours.load_state_dict(reference.state_dict(), strict=True)
    return ours.to(dtype).eval(), reference.to(dtype)


def make_inputs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    # SigLIP pads to max length and attends over the padding, so full-length
    # random sequences stand in for tokenizer output.
    input_ids = torch.randint(
        0,
        TINY_TEXT["vocab_size"],
        (BATCH, TINY_TEXT["max_position_embeddings"]),
    )
    image_size = TINY_VISION["image_size"]
    pixel_values = torch.randn(BATCH, 3, image_size, image_size, dtype=dtype)
    return input_ids, pixel_values


def test_state_dict_keys_match_reference() -> None:
    ours, reference = build_pair()
    assert set(ours.state_dict()) == set(reference.state_dict())


@pytest.mark.parametrize("dtype", DTYPES)
def test_embeddings_and_logits_match_reference(dtype: torch.dtype) -> None:
    ours, reference = build_pair(dtype)
    input_ids, pixel_values = make_inputs(dtype)

    with torch.no_grad():
        expected = reference(input_ids=input_ids, pixel_values=pixel_values)
        actual = ours(input_ids, pixel_values)

    assert_matches(actual.image_embeds, expected.image_embeds, dtype)
    assert_matches(actual.text_embeds, expected.text_embeds, dtype)
    assert_matches(actual.logits_per_image, expected.logits_per_image, dtype)


@pytest.mark.parametrize("key", list(PARAMETER_COUNTS))
def test_parameter_count_matches_hub_config(key: str) -> None:
    try:
        hub_config = AutoConfig.from_pretrained(SIGLIP_REPOS[key])
    except OSError as error:
        pytest.skip(f"hub config unavailable: {error}")
    with torch.device("meta"):
        ours = siglip(key)
        reference = transformers.SiglipModel(hub_config)
    ours_total = sum(p.numel() for p in ours.parameters())
    reference_total = sum(p.numel() for p in reference.parameters())
    assert ours_total == reference_total == PARAMETER_COUNTS[key]


def test_pooling_head_returns_one_vector_per_image() -> None:
    ours, _ = build_pair()
    _, pixel_values = make_inputs(torch.float32)

    with torch.no_grad():
        hidden, pooled = ours.vision_model(pixel_values)

    num_patches = (TINY_VISION["image_size"] // TINY_VISION["patch_size"]) ** 2
    assert hidden.shape == (BATCH, num_patches, TINY_VISION["hidden_size"])
    assert pooled.shape == (BATCH, TINY_VISION["hidden_size"])


def test_logit_bias_shifts_logits_additively() -> None:
    ours, _ = build_pair()
    input_ids, pixel_values = make_inputs(torch.float32)
    shift = 3.0

    with torch.no_grad():
        before = ours(input_ids, pixel_values).logits_per_image
        ours.logit_bias += shift
        after = ours(input_ids, pixel_values).logits_per_image

    torch.testing.assert_close(after, before + shift)


def test_unknown_size_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown SigLIP size"):
        siglip("SigLIP-B/32")

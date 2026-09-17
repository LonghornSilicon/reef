"""Tiny randomly-initialized CLIP against ``transformers``."""

import pytest
import torch
import transformers
from transformers import AutoConfig

from configs.vision_language.clip import (
    CLIP_REPOS,
    CLIPConfig,
    CLIPTextConfig,
    CLIPVisionConfig,
)
from models.vision_language.clip import CLIPModel, QuickGELU, clip
from operators.activation import ErfGELU
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
    "max_position_embeddings": 16,
    "vocab_size": 64,
    "eos_token_id": 63,
}
PROJECTION_DIM = 16
PAD_TOKEN_ID = 0
BATCH = 2
SEQ_LEN = 12
EOS_POSITIONS = (4, 9)

# Measured on the meta device from the hub configs in transformers 5.17.0.
PARAMETER_COUNTS = {
    "CLIP-ViT-B/32": 151_277_313,
    "CLIP-ViT-B/16": 149_620_737,
    "CLIP-ViT-L/14": 427_616_513,
}


def build_pair(
    dtype: torch.dtype = torch.float32,
) -> tuple[CLIPModel, transformers.CLIPModel]:
    config = CLIPConfig(
        vision=CLIPVisionConfig(**TINY_VISION),
        text=CLIPTextConfig(**TINY_TEXT),
        projection_dim=PROJECTION_DIM,
    )
    reference = transformers.CLIPModel(
        transformers.CLIPConfig(
            vision_config=TINY_VISION,
            text_config={**TINY_TEXT, "pad_token_id": PAD_TOKEN_ID},
            projection_dim=PROJECTION_DIM,
            logit_scale_init_value=config.logit_scale_init_value,
        )
    ).eval()
    ours = CLIPModel(config)
    ours.load_state_dict(reference.state_dict(), strict=True)
    return ours.to(dtype).eval(), reference.to(dtype)


def make_inputs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    eos = TINY_TEXT["eos_token_id"]
    input_ids = torch.randint(1, eos, (BATCH, SEQ_LEN))
    for row, position in enumerate(EOS_POSITIONS):
        input_ids[row, position] = eos
        input_ids[row, position + 1 :] = PAD_TOKEN_ID
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
        hub_config = AutoConfig.from_pretrained(CLIP_REPOS[key])
    except OSError as error:
        pytest.skip(f"hub config unavailable: {error}")
    with torch.device("meta"):
        ours = clip(key)
        reference = transformers.CLIPModel(hub_config)
    ours_total = sum(p.numel() for p in ours.parameters())
    reference_total = sum(p.numel() for p in reference.parameters())
    assert ours_total == reference_total == PARAMETER_COUNTS[key]


def test_text_pooling_picks_eos_position() -> None:
    ours, _ = build_pair()
    input_ids, _ = make_inputs(torch.float32)

    with torch.no_grad():
        hidden, pooled = ours.text_model(input_ids)

    for row, position in enumerate(EOS_POSITIONS):
        torch.testing.assert_close(pooled[row], hidden[row, position])


def test_quick_gelu_differs_from_erf_gelu() -> None:
    x = torch.linspace(-4.0, 4.0, 101)
    quick, exact = QuickGELU()(x), ErfGELU()(x)
    assert not torch.allclose(quick, exact, atol=1e-2)


def test_unknown_size_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown CLIP size"):
        clip("CLIP-ViT-H/14")

"""Tiny randomly-initialized ViT against ``transformers``."""

import dataclasses

import pytest
import torch
import transformers

from configs.vision_transformer.vit import VIT_CONFIGS, VIT_REPOS, ViTConfig
from models.vision_transformer.vit import ViTForImageClassification, vit
from tests.common import DTYPES, assert_matches

pytestmark = pytest.mark.unit

TINY = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "image_size": 32,
    "patch_size": 8,
    "num_labels": 10,
}
BATCH = 2


def build_pair(
    dtype: torch.dtype = torch.float32,
) -> tuple[ViTForImageClassification, transformers.ViTForImageClassification]:
    config = ViTConfig(**TINY)
    reference = transformers.ViTForImageClassification(
        transformers.ViTConfig(
            layer_norm_eps=config.layer_norm_eps,
            qkv_bias=config.qkv_bias,
            **TINY,
        )
    ).eval()
    ours = ViTForImageClassification(config)
    ours.load_state_dict(reference.state_dict(), strict=True)
    return ours.eval().to(dtype), reference.to(dtype)


def test_state_dict_keys_match_reference() -> None:
    ours, reference = build_pair()
    assert set(ours.state_dict()) == set(reference.state_dict())


@pytest.mark.parametrize("dtype", DTYPES)
def test_logits_match_reference(dtype: torch.dtype) -> None:
    ours, reference = build_pair(dtype)
    size = TINY["image_size"]
    x = torch.randn(BATCH, 3, size, size, dtype=dtype)

    with torch.no_grad():
        expected = reference(pixel_values=x).logits
        actual = ours(x)

    assert_matches(actual, expected, dtype)


def test_interpolated_logits_match_reference() -> None:
    ours, reference = build_pair()
    x = torch.randn(BATCH, 3, 48, 64)

    with torch.no_grad():
        expected = reference(
            pixel_values=x, interpolate_pos_encoding=True
        ).logits
        actual = ours(x, interpolate_pos_encoding=True)

    torch.testing.assert_close(actual, expected)


def test_pos_encoding_interpolation_keeps_cls_token() -> None:
    embeddings = vit("ViT-B/16").vit.embeddings
    original = embeddings.position_embeddings
    interpolated = embeddings.interpolate_pos_encoding(96, 160)
    assert interpolated.shape == (1, 1 + 6 * 10, original.shape[-1])
    assert torch.equal(interpolated[:, 0], original[:, 0])
    assert embeddings.interpolate_pos_encoding(224, 224) is original


@pytest.mark.parametrize("key", list(VIT_CONFIGS))
def test_published_size_matches_hub(key: str) -> None:
    try:
        hub = transformers.AutoConfig.from_pretrained(VIT_REPOS[key])
    except OSError as exc:
        pytest.skip(f"hub config unavailable: {exc}")
    config = dataclasses.replace(VIT_CONFIGS[key], num_labels=hub.num_labels)
    with torch.device("meta"):
        ours = ViTForImageClassification(config)
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
    with pytest.raises(KeyError, match="unknown ViT size"):
        vit("ViT-G/14")

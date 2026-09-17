"""Tiny randomly-initialized DINOv2 against ``transformers``."""

import dataclasses

import pytest
import torch
import transformers

from configs.dinov2 import DINOV2_CONFIGS, DINOV2_REPOS, Dinov2Config
from models.dinov2 import Dinov2ForImageClassification, dinov2
from tests.common import DTYPES, assert_matches

pytestmark = pytest.mark.unit

TINY = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "image_size": 32,
    "patch_size": 8,
    "num_labels": 10,
}
BATCH = 2


def build_pair(
    use_swiglu_ffn: bool, dtype: torch.dtype = torch.float32
) -> tuple[
    Dinov2ForImageClassification, transformers.Dinov2ForImageClassification
]:
    config = Dinov2Config(use_swiglu_ffn=use_swiglu_ffn, **TINY)
    reference = transformers.Dinov2ForImageClassification(
        transformers.Dinov2Config(
            use_swiglu_ffn=use_swiglu_ffn,
            mlp_ratio=config.mlp_ratio,
            layerscale_value=config.layerscale_value,
            layer_norm_eps=config.layer_norm_eps,
            qkv_bias=config.qkv_bias,
            **TINY,
        )
    ).eval()
    ours = Dinov2ForImageClassification(config)
    ours.load_state_dict(reference.state_dict(), strict=True)
    return ours.eval().to(dtype), reference.to(dtype)


@pytest.mark.parametrize("use_swiglu_ffn", [False, True])
def test_state_dict_keys_match_reference(use_swiglu_ffn: bool) -> None:
    ours, reference = build_pair(use_swiglu_ffn)
    assert set(ours.state_dict()) == set(reference.state_dict())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("use_swiglu_ffn", [False, True])
def test_logits_match_reference(
    use_swiglu_ffn: bool, dtype: torch.dtype
) -> None:
    ours, reference = build_pair(use_swiglu_ffn, dtype)
    size = TINY["image_size"]
    x = torch.randn(BATCH, 3, size, size, dtype=dtype)

    with torch.no_grad():
        expected = reference(pixel_values=x).logits
        actual = ours(x)

    assert_matches(actual, expected, dtype)


def test_interpolated_logits_match_reference() -> None:
    ours, reference = build_pair(use_swiglu_ffn=False)
    x = torch.randn(BATCH, 3, 48, 64)

    with torch.no_grad():
        expected = reference(pixel_values=x).logits
        actual = ours(x)

    torch.testing.assert_close(actual, expected)


def test_layer_scale_starts_at_configured_value() -> None:
    config = Dinov2Config(use_swiglu_ffn=False, layerscale_value=0.1, **TINY)
    model = Dinov2ForImageClassification(config)
    for layer in model.dinov2.encoder.layer:
        expected = torch.full((TINY["hidden_size"],), 0.1)
        assert torch.equal(layer.layer_scale1.lambda1, expected)
        assert torch.equal(layer.layer_scale2.lambda1, expected)
    x = torch.randn(3, TINY["hidden_size"])
    assert torch.equal(model.dinov2.encoder.layer[0].layer_scale1(x), x * 0.1)


@pytest.mark.parametrize("key", list(DINOV2_CONFIGS))
def test_published_size_matches_hub(key: str) -> None:
    try:
        hub = transformers.AutoConfig.from_pretrained(DINOV2_REPOS[key])
    except OSError as exc:
        pytest.skip(f"hub config unavailable: {exc}")
    config = dataclasses.replace(DINOV2_CONFIGS[key], num_labels=hub.num_labels)
    with torch.device("meta"):
        ours = Dinov2ForImageClassification(config)
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
    with pytest.raises(KeyError, match="unknown DINOv2 size"):
        dinov2("DINOv2-XL")

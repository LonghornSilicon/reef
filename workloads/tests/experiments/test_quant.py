"""Fake quantization primitives and the storage formulas built on them."""

import pytest
import torch

from configs.gpt2 import GPT2Config
from configs.llama import LlamaConfig
from experiments import quant, tracer
from models.gpt2 import Conv1D, GPT2LMHeadModel
from models.llama import LlamaForCausalLM
from operators.convolution import Conv2d
from tests.models.test_gpt2 import TINY as TINY_GPT2
from tests.models.test_llama import TINY as TINY_LLAMA

pytestmark = pytest.mark.unit


def test_per_channel_error_is_within_half_a_step() -> None:
    weight = torch.randn(4, 37)
    quantized = quant.quantize_per_channel(weight, 8)
    step = weight.abs().amax(dim=1, keepdim=True) / 127

    assert ((quantized - weight).abs() <= step / 2 + 1e-6).all()
    assert torch.equal(quantized.abs().amax(dim=1), weight.abs().amax(dim=1))


def test_group_scales_keep_a_small_partial_group_alive() -> None:
    weight = torch.zeros(1, 300)
    weight[0, :256] = 100.0
    weight[0, 256:] = 0.01
    per_channel = quant.quantize_per_channel(weight, 4)
    grouped = quant.quantize_group(weight, 4)

    assert (per_channel[0, 256:] == 0).all()
    assert torch.allclose(grouped[0, 256:], weight[0, 256:])


def test_per_token_quantization_is_a_fixed_point() -> None:
    x = torch.randn(3, 5, 16)
    once = quant.quantize_per_token(x, 8)
    twice = quant.quantize_per_token(once, 8)

    torch.testing.assert_close(twice, once)


def test_all_zero_rows_quantize_to_zero() -> None:
    x = torch.zeros(2, 8)
    assert torch.equal(quant.quantize_per_token(x, 8), x)


def test_channel_view_puts_output_channels_first() -> None:
    assert quant.channel_view(Conv1D(8, 16)).shape == (16, 8)
    assert quant.channel_view(Conv2d(4, 6, 3)).shape == (6, 36)


def test_quantizable_skips_lm_head_and_counts_tied_weights_once() -> None:
    llama = LlamaForCausalLM(
        LlamaConfig(tie_word_embeddings=True, **TINY_LLAMA)
    )
    gpt2 = GPT2LMHeadModel(GPT2Config(**TINY_GPT2))

    for model in (llama, gpt2):
        names = [name for name, _ in quant.quantizable(model)]
        assert "lm_head" not in names
        assert len(names) == len(set(names))
    assert any(name.endswith("c_attn") for name, _ in quant.quantizable(gpt2))


def test_weight_bytes_include_one_scale_per_channel_or_group() -> None:
    assert quant.weight_bytes(1000, 10, "bf16") == 2000
    assert quant.weight_bytes(1000, 10, "int8") == 1000 + 10 * 2
    assert quant.weight_bytes(1000, 10, "int4") == 500 + 10 * 2


def test_smollm2_kv_bytes_per_token() -> None:
    model = tracer.build("llama", "SmolLM2-135M")
    # 2 * 30 layers * 3 kv heads * 64 head dim.
    assert quant.kv_bytes_per_token(model, "bf16") == 11_520 * 2
    assert quant.kv_bytes_per_token(model, "int8") == 11_520 + 2 * 30 * 3 * 2

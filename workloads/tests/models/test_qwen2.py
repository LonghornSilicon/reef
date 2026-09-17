"""Tiny randomly-initialized Qwen2.5 against ``transformers``."""

import pytest
import torch
import transformers

from configs.qwen2 import Qwen2Config
from models.qwen2 import Qwen2ForCausalLM, qwen2
from tests.common import DTYPES, assert_matches

pytestmark = pytest.mark.unit

TINY = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 64,
    "vocab_size": 1000,
}
HEAD_DIM = TINY["hidden_size"] // TINY["num_attention_heads"]
BATCH = 2
PROMPT_LEN = 6
DECODE_STEPS = 5

PUBLISHED_PARAMETERS = {
    "Qwen2.5-0.5B": 494_032_768,
    "Qwen2.5-1.5B": 1_543_714_304,
}


def build_pair(
    tie_word_embeddings: bool,
) -> tuple[Qwen2ForCausalLM, transformers.Qwen2ForCausalLM]:
    config = Qwen2Config(tie_word_embeddings=tie_word_embeddings, **TINY)
    reference = transformers.Qwen2ForCausalLM(
        transformers.Qwen2Config(
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            hidden_act=config.hidden_act,
            use_sliding_window=config.use_sliding_window,
            sliding_window=config.sliding_window,
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=0,
            **TINY,
        )
    ).eval()

    ours = Qwen2ForCausalLM(config)
    ours.load_state_dict(reference.state_dict(), strict=True)
    return ours.eval(), reference


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_state_dict_keys_match_reference(tie_word_embeddings: bool) -> None:
    ours, reference = build_pair(tie_word_embeddings)
    assert set(ours.state_dict()) == set(reference.state_dict())
    tied = ours.lm_head.weight is ours.model.embed_tokens.weight
    assert tied is tie_word_embeddings


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_prefill_logits_match_reference(
    tie_word_embeddings: bool, dtype: torch.dtype
) -> None:
    ours, reference = build_pair(tie_word_embeddings)
    ours, reference = ours.to(dtype), reference.to(dtype)
    input_ids = torch.randint(0, TINY["vocab_size"], (BATCH, PROMPT_LEN))

    with torch.no_grad():
        expected = reference(input_ids=input_ids).logits
        actual, cache = ours(input_ids)

    assert_matches(actual, expected, dtype)
    assert len(cache) == TINY["num_hidden_layers"]
    for key, value in cache:
        head_shape = (BATCH, TINY["num_key_value_heads"], PROMPT_LEN)
        assert key.shape == (*head_shape, HEAD_DIM)
        assert value.shape == (*head_shape, HEAD_DIM)


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_cached_decode_matches_reference(tie_word_embeddings: bool) -> None:
    ours, reference = build_pair(tie_word_embeddings)
    input_ids = torch.randint(0, TINY["vocab_size"], (BATCH, PROMPT_LEN))

    with torch.no_grad():
        reference_out = reference(input_ids=input_ids, use_cache=True)
        actual, cache = ours(input_ids)
        torch.testing.assert_close(actual, reference_out.logits)

        for step in range(DECODE_STEPS):
            token = torch.randint(0, TINY["vocab_size"], (BATCH, 1))
            reference_out = reference(
                input_ids=token,
                past_key_values=reference_out.past_key_values,
                use_cache=True,
            )
            actual, cache = ours(token, cache)
            torch.testing.assert_close(
                actual,
                reference_out.logits,
                msg=lambda built, step=step: f"decode step {step}: {built}",
            )

    expected_len = PROMPT_LEN + DECODE_STEPS
    assert all(key.shape[2] == expected_len for key, _ in cache)


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_greedy_generate_matches_reference(tie_word_embeddings: bool) -> None:
    ours, reference = build_pair(tie_word_embeddings)
    input_ids = torch.randint(0, TINY["vocab_size"], (BATCH, PROMPT_LEN))

    with torch.no_grad():
        expected = reference.generate(
            input_ids, max_new_tokens=DECODE_STEPS, do_sample=False
        )
        actual = ours.generate(input_ids, max_new_tokens=DECODE_STEPS)

    assert actual.shape == (BATCH, PROMPT_LEN + DECODE_STEPS)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(("name", "expected"), PUBLISHED_PARAMETERS.items())
def test_published_size_parameter_count(name: str, expected: int) -> None:
    with torch.device("meta"):
        model = qwen2(name)
    assert sum(p.numel() for p in model.parameters()) == expected
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_qkv_have_bias_and_o_proj_does_not() -> None:
    ours, _ = build_pair(tie_word_embeddings=True)
    attn = ours.model.layers[0].self_attn
    assert attn.q_proj.bias is not None
    assert attn.k_proj.bias is not None
    assert attn.v_proj.bias is not None
    assert attn.o_proj.bias is None


def test_unknown_size_is_rejected() -> None:
    with pytest.raises(KeyError, match=r"unknown Qwen2\.5 size"):
        qwen2("Qwen2.5-7B")

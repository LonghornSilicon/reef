"""Tiny randomly-initialized Llama against ``transformers``."""

import pytest
import torch
import transformers

from configs.llama import LLAMA_CONFIGS, LlamaConfig
from models.llama import LlamaForCausalLM, llama
from tests.common import DTYPES, assert_matches

pytestmark = pytest.mark.unit

TINY = {
    "vocab_size": 1000,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "rope_theta": 10000.0,
    "max_position_embeddings": 64,
}
BATCH = 2
PROMPT_LEN = 6
DECODE_STEPS = 5

PUBLISHED_PARAMETERS = {
    "SmolLM2-135M": 134_515_008,
}


def build_pair(
    tie_word_embeddings: bool,
) -> tuple[LlamaForCausalLM, transformers.LlamaForCausalLM]:
    config = LlamaConfig(tie_word_embeddings=tie_word_embeddings, **TINY)
    reference = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            tie_word_embeddings=tie_word_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            mlp_bias=config.mlp_bias,
            hidden_act=config.hidden_act,
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=0,
            **TINY,
        )
    ).eval()

    ours = LlamaForCausalLM(config)
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
        assert key.shape == (*head_shape, TINY["head_dim"])
        assert value.shape == (*head_shape, TINY["head_dim"])


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
        model = llama(name)
    assert sum(p.numel() for p in model.parameters()) == expected
    tied = model.lm_head.weight is model.model.embed_tokens.weight
    assert tied is LLAMA_CONFIGS[name].tie_word_embeddings


def test_unknown_size_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown Llama size"):
        llama("Llama-3.2-3B")

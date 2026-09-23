"""Tiny randomly-initialized GPT-Neo against ``transformers``."""

import pytest
import torch
import transformers

from tests.common import DTYPES, assert_matches
from workloads.configs.gpt_neo import GPTNeoConfig
from workloads.models.gpt_neo import GPTNeoForCausalLM, gpt_neo

pytestmark = pytest.mark.unit

TINY = {
    "hidden_size": 64,
    "num_layers": 2,
    "num_heads": 4,
    "vocab_size": 1000,
    "max_position_embeddings": 32,
}
# Narrower than PROMPT_LEN, so the local layer actually masks something.
WINDOW_SIZE = 4
BATCH = 2
PROMPT_LEN = 6
DECODE_STEPS = 5

PUBLISHED_PARAMETERS = {
    "TinyStories-Instruct-1M": 3_745_984,
    "TinyStories-Instruct-3M": 8_278_400,
    "TinyStories-Instruct-8M": 19_702_528,
    "TinyStories-Instruct-28M": 51_987_968,
    "TinyStories-Instruct-33M": 68_514_048,
}


def build_pair(
    tie_word_embeddings: bool,
) -> tuple[GPTNeoForCausalLM, transformers.GPTNeoForCausalLM]:
    config = GPTNeoConfig(
        tie_word_embeddings=tie_word_embeddings,
        window_size=WINDOW_SIZE,
        **TINY,
    )
    reference = transformers.GPTNeoForCausalLM(
        transformers.GPTNeoConfig(
            tie_word_embeddings=tie_word_embeddings,
            window_size=WINDOW_SIZE,
            attention_types=[[list(config.attention_types), 1]],
            intermediate_size=config.intermediate_size,
            activation_function=config.activation_function,
            layer_norm_epsilon=config.layer_norm_epsilon,
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=0,
            attn_implementation="eager",
            **TINY,
        )
    ).eval()

    ours = GPTNeoForCausalLM(config)
    ours.load_state_dict(reference.state_dict(), strict=True)
    return ours.eval(), reference


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_state_dict_keys_match_reference(tie_word_embeddings: bool) -> None:
    ours, reference = build_pair(tie_word_embeddings)
    assert set(ours.state_dict()) == set(reference.state_dict())
    tied = ours.lm_head.weight is ours.transformer.wte.weight
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
    assert len(cache) == TINY["num_layers"]
    head_dim = TINY["hidden_size"] // TINY["num_heads"]
    for key, value in cache:
        head_shape = (BATCH, TINY["num_heads"], PROMPT_LEN)
        assert key.shape == (*head_shape, head_dim)
        assert value.shape == (*head_shape, head_dim)


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
        model = gpt_neo(name)
    assert sum(p.numel() for p in model.parameters()) == expected
    assert model.lm_head.weight is model.transformer.wte.weight


def test_local_layers_use_a_sliding_window() -> None:
    ours, _ = build_pair(tie_word_embeddings=True)
    windows = [
        block.attn.attention.attention.sliding_window
        for block in ours.transformer.h
    ]
    assert windows == [None, WINDOW_SIZE]


def test_unknown_size_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown GPT-Neo size"):
        gpt_neo("TinyStories-Instruct-99M")

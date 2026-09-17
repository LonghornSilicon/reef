"""Tiny randomly-initialized GPT-2 against ``transformers``."""

import pytest
import torch
import transformers

from configs.gpt2 import GPT2Config
from models.gpt2 import GPT2LMHeadModel, gpt2
from tests.common import DTYPES, assert_matches

pytestmark = pytest.mark.unit

TINY = {
    "n_embd": 64,
    "n_layer": 2,
    "n_head": 4,
    "vocab_size": 1000,
    "n_positions": 32,
}
BATCH = 2
PROMPT_LEN = 6
DECODE_STEPS = 5

PUBLISHED_PARAMETERS = {
    "GPT-2": 124_439_808,
    "GPT-2-Medium": 354_823_168,
    "GPT-2-Large": 774_030_080,
    "GPT-2-XL": 1_557_611_200,
}


def build_pair(
    tie_word_embeddings: bool,
) -> tuple[GPT2LMHeadModel, transformers.GPT2LMHeadModel]:
    config = GPT2Config(tie_word_embeddings=tie_word_embeddings, **TINY)
    reference = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            tie_word_embeddings=tie_word_embeddings,
            n_inner=config.n_inner,
            activation_function=config.activation_function,
            layer_norm_epsilon=config.layer_norm_epsilon,
            scale_attn_weights=config.scale_attn_weights,
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=0,
            **TINY,
        )
    ).eval()

    ours = GPT2LMHeadModel(config)
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
    assert len(cache) == TINY["n_layer"]
    head_dim = TINY["n_embd"] // TINY["n_head"]
    for key, value in cache:
        head_shape = (BATCH, TINY["n_head"], PROMPT_LEN)
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
        model = gpt2(name)
    assert sum(p.numel() for p in model.parameters()) == expected
    assert model.lm_head.weight is model.transformer.wte.weight


def test_position_embeddings_are_learned() -> None:
    ours, _ = build_pair(tie_word_embeddings=True)
    table = ours.transformer.wpe.weight
    assert table.shape == (TINY["n_positions"], TINY["n_embd"])
    assert not torch.equal(table[0], table[1])


def test_unknown_size_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown GPT-2 size"):
        gpt2("GPT-3")

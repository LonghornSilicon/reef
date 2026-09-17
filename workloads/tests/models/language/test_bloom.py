"""Tiny randomly-initialized BLOOM against ``transformers``."""

import pytest
import torch
import transformers
from transformers.models.bloom.modeling_bloom import build_alibi_tensor

from configs.language.bloom import BloomConfig
from models.language.bloom import BloomForCausalLM, bloom, build_alibi
from tests.common import DTYPES, assert_matches

pytestmark = pytest.mark.unit

TINY = {
    "hidden_size": 64,
    "n_layer": 2,
    "n_head": 4,
    "vocab_size": 1000,
}
BATCH = 2
PROMPT_LEN = 6
DECODE_STEPS = 5

PUBLISHED_PARAMETERS = {
    "BLOOM-560M": 559_214_592,
    "BLOOM-1B1": 1_065_314_304,
}


def build_pair(
    tie_word_embeddings: bool,
) -> tuple[BloomForCausalLM, transformers.BloomForCausalLM]:
    config = BloomConfig(tie_word_embeddings=tie_word_embeddings, **TINY)
    reference = transformers.BloomForCausalLM(
        transformers.BloomConfig(
            tie_word_embeddings=tie_word_embeddings,
            layer_norm_epsilon=config.layer_norm_epsilon,
            apply_residual_connection_post_layernorm=(
                config.apply_residual_connection_post_layernorm
            ),
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=0,
            **TINY,
        )
    ).eval()

    ours = BloomForCausalLM(config)
    ours.load_state_dict(reference.state_dict(), strict=True)
    return ours.eval(), reference


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_state_dict_keys_match_reference(tie_word_embeddings: bool) -> None:
    ours, reference = build_pair(tie_word_embeddings)
    assert set(ours.state_dict()) == set(reference.state_dict())
    tied = ours.lm_head.weight is ours.transformer.word_embeddings.weight
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
    head_dim = TINY["hidden_size"] // TINY["n_head"]
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
        model = bloom(name)
    assert sum(p.numel() for p in model.parameters()) == expected
    assert model.lm_head.weight is model.transformer.word_embeddings.weight


@pytest.mark.parametrize("num_heads", [4, 16, 12])
def test_alibi_matches_reference(num_heads: int) -> None:
    key_len = 7
    alibi = build_alibi(num_heads, key_len, torch.device("cpu"))
    assert alibi.shape == (1, num_heads, 1, key_len)
    assert alibi.dtype == torch.float32

    # Linear in key distance: equal steps between consecutive key positions.
    steps = alibi[..., 1:] - alibi[..., :-1]
    torch.testing.assert_close(steps, steps[..., :1].expand_as(steps))
    assert (steps > 0).all()

    mask = torch.ones(BATCH, key_len)
    expected = build_alibi_tensor(mask, num_heads, torch.float32)
    actual = alibi.expand(BATCH, -1, -1, -1).reshape(
        BATCH * num_heads, 1, key_len
    )
    torch.testing.assert_close(actual, expected)


def test_unknown_size_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown BLOOM size"):
        bloom("BLOOM-176B")

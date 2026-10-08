"""Equivalence against the published pretrained checkpoints."""

import pytest
import torch
import transformers

from workloads.configs.gpt_neo import GPT_NEO_CONFIGS, GPT_NEO_REPOS
from workloads.models.gpt_neo import GPTNeoForCausalLM

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]

TINYSTORIES = "TinyStories-Instruct-8M"
PROMPT = "The capital of France is"

TINYSTORIES_TOLERANCE = {"rtol": 1e-5, "atol": 1e-4}


def test_tinystories_instruct_8m_config_matches_the_hub() -> None:
    ours = GPT_NEO_CONFIGS[TINYSTORIES]
    hub = transformers.AutoConfig.from_pretrained(GPT_NEO_REPOS[TINYSTORIES])

    assert hub.hidden_size == ours.hidden_size
    assert hub.num_layers == ours.num_layers
    assert hub.num_heads == ours.num_heads
    assert hub.vocab_size == ours.vocab_size
    assert hub.intermediate_size == ours.intermediate_size
    assert hub.activation_function == ours.activation_function
    assert hub.layer_norm_epsilon == ours.layer_norm_epsilon
    assert hub.window_size == ours.window_size
    assert hub.max_position_embeddings == ours.max_position_embeddings
    assert hub.attention_layers == [
        ours.attention_type(layer) for layer in range(ours.num_layers)
    ]


def test_tinystories_instruct_8m_logits_match_reference() -> None:
    repo = GPT_NEO_REPOS[TINYSTORIES]
    reference = transformers.GPTNeoForCausalLM.from_pretrained(
        repo, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    ours = GPTNeoForCausalLM(GPT_NEO_CONFIGS[TINYSTORIES])
    ours.load_state_dict(reference.state_dict(), strict=True)
    ours.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(repo)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids

    with torch.no_grad():
        expected = reference(input_ids=input_ids).logits
        actual, _ = ours(input_ids)

    torch.testing.assert_close(actual, expected, **TINYSTORIES_TOLERANCE)
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))

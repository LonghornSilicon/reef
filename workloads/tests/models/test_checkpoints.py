"""Equivalence against the published pretrained checkpoints."""

import pytest
import torch
import torchvision
import transformers

from configs.llama import LLAMA_CONFIGS, LLAMA_REPOS
from configs.resnet import TORCHVISION_BUILDERS
from models.llama import LlamaForCausalLM
from models.resnet import resnet

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]

SMOLLM = "SmolLM2-135M"
PROMPT = "The capital of France is"

SMOLLM_TOLERANCE = {"rtol": 1e-5, "atol": 1e-4}
# Trained logits are O(10), so these stay tighter than test_resnet.TOLERANCE.
# One depth per block type.
RESNET_TOLERANCE = {"rtol": 1e-5, "atol": 1e-4}
RESNET_REPOS = ["ResNet-18", "ResNet-50"]


def test_smollm2_135m_config_matches_the_hub() -> None:
    ours = LLAMA_CONFIGS[SMOLLM]
    hub = transformers.AutoConfig.from_pretrained(LLAMA_REPOS[SMOLLM])

    assert hub.hidden_size == ours.hidden_size
    assert hub.intermediate_size == ours.intermediate_size
    assert hub.num_hidden_layers == ours.num_hidden_layers
    assert hub.num_attention_heads == ours.num_attention_heads
    assert hub.num_key_value_heads == ours.num_key_value_heads
    assert hub.vocab_size == ours.vocab_size
    assert hub.head_dim == ours.head_dim
    assert hub.hidden_act == ours.hidden_act
    assert hub.rms_norm_eps == ours.rms_norm_eps
    assert hub.attention_bias == ours.attention_bias
    assert hub.mlp_bias == ours.mlp_bias
    assert hub.max_position_embeddings == ours.max_position_embeddings
    assert hub.tie_word_embeddings == ours.tie_word_embeddings
    assert hub.rope_parameters["rope_theta"] == ours.rope_theta


def test_smollm2_135m_logits_match_reference() -> None:
    repo = LLAMA_REPOS[SMOLLM]
    reference = transformers.LlamaForCausalLM.from_pretrained(
        repo, dtype=torch.float32
    ).eval()
    ours = LlamaForCausalLM(LLAMA_CONFIGS[SMOLLM])
    ours.load_state_dict(reference.state_dict(), strict=True)
    ours.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(repo)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids

    with torch.no_grad():
        expected = reference(input_ids=input_ids).logits
        actual, _ = ours(input_ids)

    torch.testing.assert_close(actual, expected, **SMOLLM_TOLERANCE)
    # Argmax agreement is what matters downstream.
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))


@pytest.mark.parametrize("name", RESNET_REPOS)
def test_pretrained_resnet_logits_match_torchvision(name: str) -> None:
    # Random weights leave the batch-norm buffers at mean 0 / variance 1, so
    # only a real checkpoint exercises loading and applying them.
    builder = getattr(torchvision.models, TORCHVISION_BUILDERS[name])
    reference = builder(weights="DEFAULT").eval()
    ours = resnet(name)
    ours.load_state_dict(reference.state_dict(), strict=True)
    ours.eval()

    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        expected = reference(x)
        actual = ours(x)

    torch.testing.assert_close(actual, expected, **RESNET_TOLERANCE)
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))

"""Equivalence against the published pretrained checkpoints."""

import pytest
import torch
import torchvision
import transformers

from configs.classification.resnet import TORCHVISION_BUILDERS
from configs.language.qwen3 import QWEN3_0_6B
from models.classification.resnet import resnet
from models.language.qwen3 import Qwen3ForCausalLM

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]

QWEN3_REPO = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"

# Measured worst case over Qwen3-0.6B's 28 layers is ~4e-5 absolute on logits
# spanning +/-19; these bounds sit about 10x above it.
QWEN3_TOLERANCE = {"rtol": 1e-5, "atol": 1e-4}
# Trained logits are O(10), so these stay tighter than test_resnet.TOLERANCE.
# One depth per block type.
RESNET_TOLERANCE = {"rtol": 1e-5, "atol": 1e-4}
RESNET_REPOS = ["ResNet-18", "ResNet-50"]


def test_qwen3_0_6b_config_matches_the_hub() -> None:
    hub = transformers.AutoConfig.from_pretrained(QWEN3_REPO)

    assert hub.hidden_size == QWEN3_0_6B.hidden_size
    assert hub.intermediate_size == QWEN3_0_6B.intermediate_size
    assert hub.num_hidden_layers == QWEN3_0_6B.num_hidden_layers
    assert hub.num_attention_heads == QWEN3_0_6B.num_attention_heads
    assert hub.num_key_value_heads == QWEN3_0_6B.num_key_value_heads
    assert hub.vocab_size == QWEN3_0_6B.vocab_size
    assert hub.head_dim == QWEN3_0_6B.head_dim
    assert hub.rms_norm_eps == QWEN3_0_6B.rms_norm_eps
    assert hub.attention_bias == QWEN3_0_6B.attention_bias
    assert hub.max_position_embeddings == QWEN3_0_6B.max_position_embeddings
    assert hub.tie_word_embeddings == QWEN3_0_6B.tie_word_embeddings
    assert hub.rope_parameters["rope_theta"] == QWEN3_0_6B.rope_theta


def test_qwen3_0_6b_logits_match_reference() -> None:
    reference = transformers.Qwen3ForCausalLM.from_pretrained(
        QWEN3_REPO, dtype=torch.float32
    ).eval()
    ours = Qwen3ForCausalLM(QWEN3_0_6B)
    ours.load_state_dict(reference.state_dict(), strict=True)
    ours.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(QWEN3_REPO)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids

    with torch.no_grad():
        expected = reference(input_ids=input_ids).logits
        actual, _ = ours(input_ids)

    torch.testing.assert_close(actual, expected, **QWEN3_TOLERANCE)
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

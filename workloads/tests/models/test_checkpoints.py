"""Equivalence against the published pretrained checkpoints."""

import pytest
import torch
import torchvision
import transformers

from configs.gemma3 import GEMMA3_CONFIGS, GEMMA3_REPOS
from configs.qwen3 import QWEN3_0_6B
from configs.resnet import TORCHVISION_BUILDERS
from models.alexnet import AlexNet
from models.gemma3 import gemma3
from models.qwen3 import Qwen3ForCausalLM
from models.resnet import resnet

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]

QWEN3_REPO = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"

# Measured worst case over Qwen3-0.6B's 28 layers is ~4e-5 absolute on logits
# spanning +/-19; these bounds sit about 10x above it.
QWEN3_TOLERANCE = {"rtol": 1e-5, "atol": 1e-4}
ALEXNET_TOLERANCE = {"rtol": 1e-5, "atol": 1e-5}
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


def test_pretrained_alexnet_logits_match_torchvision() -> None:
    weights = torchvision.models.AlexNet_Weights.DEFAULT
    reference = torchvision.models.alexnet(weights=weights).eval()
    ours = AlexNet()
    ours.load_state_dict(reference.state_dict(), strict=True)
    ours.eval()

    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        expected = reference(x)
        actual = ours(x)

    torch.testing.assert_close(actual, expected, **ALEXNET_TOLERANCE)
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


# google/gemma-3-* is gated (licence acceptance + `hf auth login`), so the
# Gemma 3 tests skip rather than fail where it is inaccessible.
GEMMA3_HUB_SIZE = "Gemma3-270M"


def load_gemma3_hub_config(name: str) -> transformers.PreTrainedConfig:
    try:
        config = transformers.AutoConfig.from_pretrained(GEMMA3_REPOS[name])
    except OSError as error:
        pytest.skip(f"{GEMMA3_REPOS[name]} is gated or unreachable: {error}")
    return getattr(config, "text_config", config)


@pytest.mark.parametrize("name", list(GEMMA3_CONFIGS))
def test_gemma3_configs_match_the_hub(name: str) -> None:
    hub = load_gemma3_hub_config(name)
    ours = GEMMA3_CONFIGS[name]

    assert hub.hidden_size == ours.hidden_size
    assert hub.intermediate_size == ours.intermediate_size
    assert hub.num_hidden_layers == ours.num_hidden_layers
    assert hub.num_attention_heads == ours.num_attention_heads
    assert hub.num_key_value_heads == ours.num_key_value_heads
    assert hub.head_dim == ours.head_dim
    assert hub.vocab_size == ours.vocab_size
    assert hub.sliding_window == ours.sliding_window
    assert hub.query_pre_attn_scalar == ours.query_pre_attn_scalar
    assert hub.max_position_embeddings == ours.max_position_embeddings
    assert hub.rms_norm_eps == ours.rms_norm_eps
    assert hub.attention_bias == ours.attention_bias
    rope = hub.rope_parameters
    assert rope["sliding_attention"]["rope_theta"] == ours.rope_local_base_freq
    assert rope["full_attention"]["rope_theta"] == ours.rope_global_base_freq
    assert rope["full_attention"].get("factor", 1.0) == ours.rope_global_scaling
    expected = [
        "sliding_attention" if ours.is_sliding(index) else "full_attention"
        for index in range(ours.num_hidden_layers)
    ]
    assert list(hub.layer_types) == expected


def test_gemma3_270m_logits_match_reference() -> None:
    repo = GEMMA3_REPOS[GEMMA3_HUB_SIZE]
    try:
        reference = transformers.Gemma3ForCausalLM.from_pretrained(
            repo, dtype=torch.float32
        ).eval()
    except OSError as error:
        pytest.skip(f"{repo} is gated or unreachable: {error}")

    ours = gemma3(GEMMA3_HUB_SIZE)
    ours.load_state_dict(reference.state_dict(), strict=True)
    ours.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(repo)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids

    with torch.no_grad():
        expected = reference(input_ids=input_ids).logits
        actual, _ = ours(input_ids)

    torch.testing.assert_close(actual, expected, **QWEN3_TOLERANCE)
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))

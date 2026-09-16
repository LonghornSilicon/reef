"""Equivalence against the published pretrained checkpoints.

These are the tests that prove the operator library can actually stand in for
the reference implementations on the weights we characterize. They download
multi-hundred-megabyte checkpoints and run full-size models on CPU, so they are
marked ``slow`` and excluded from the default ``-m "not slow"`` run.
"""

import pytest
import torch
import torchvision
import transformers

from configs.qwen3 import QWEN3_0_6B
from models.alexnet import AlexNet
from models.qwen3 import Qwen3ForCausalLM

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]

QWEN3_REPO = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"

# Our operators accumulate in a different order than the fused kernels, and
# that drift compounds over Qwen3-0.6B's 28 layers. Measured worst case is
# ~4e-5 absolute against logits spanning +/-19, so these bounds sit roughly an
# order of magnitude above the observed error rather than at the assert_close
# float32 defaults (rtol=1.3e-6, atol=1e-5), which are too tight here.
QWEN3_TOLERANCE = {"rtol": 1e-5, "atol": 1e-4}
ALEXNET_TOLERANCE = {"rtol": 1e-5, "atol": 1e-5}


def test_qwen3_0_6b_config_matches_the_hub() -> None:
    """Our checked-in QWEN3_0_6B hyperparameters match the published config."""
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
    """Real Qwen3-0.6B weights reproduce the reference logits on a prompt."""
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
    # Argmax agreement is the property that actually matters downstream, and
    # it is insensitive to the tolerance above.
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))


def test_pretrained_alexnet_logits_match_torchvision() -> None:
    """Pretrained torchvision AlexNet weights produce the reference logits."""
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

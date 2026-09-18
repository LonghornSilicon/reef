"""Quantization plumbing on tiny models, plus one perplexity check on GPT-2."""

import math

import pytest
import torch
import transformers

from configs.gpt2 import GPT2Config
from configs.llama import LlamaConfig
from experiments import quantization
from models.gpt2 import GPT2LMHeadModel
from models.llama import LlamaForCausalLM
from tests.models.test_gpt2 import TINY as TINY_GPT2
from tests.models.test_llama import TINY as TINY_LLAMA

pytestmark = pytest.mark.unit

WINDOWS = 4


def test_windows_score_every_token_once() -> None:
    total = 2500
    covered = []
    for begin, end, first in quantization.windows(total, None):
        assert end - begin <= quantization.WINDOW
        covered.extend(range(first, end))
    assert covered == list(range(1, total))


def test_windows_respect_the_limit() -> None:
    assert len(list(quantization.windows(10_000, 3))) == 3


def test_weight_quantization_leaves_the_tied_embedding_alone() -> None:
    model = GPT2LMHeadModel(GPT2Config(**TINY_GPT2)).eval()
    before = model.transformer.wte.weight.clone()
    quantization.quantize_weights(model, 8)

    assert torch.equal(model.transformer.wte.weight, before)
    assert not torch.equal(
        model.transformer.h[0].attn.c_attn.weight, before[:0]
    )


def test_kv_and_activation_hooks_change_logits_only_slightly() -> None:
    config = LlamaConfig(tie_word_embeddings=True, **TINY_LLAMA)
    ids = torch.randint(0, config.vocab_size, (1, 12))
    reference = LlamaForCausalLM(config).eval()
    state = reference.state_dict()
    expected, _ = reference(ids)
    for scheme in ((8, None, None), (None, 8, None), (None, None, 8)):
        model = LlamaForCausalLM(config).eval()
        model.load_state_dict(state)
        quantization.apply(model, scheme)
        actual, _ = model(ids)
        assert not torch.equal(actual, expected)
        torch.testing.assert_close(actual, expected, rtol=0.1, atol=0.1)


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_fp32_perplexity_matches_transformers_gpt2() -> None:
    model, tokenizer = quantization.language_model("gpt2", "GPT-2")
    reference = transformers.GPT2LMHeadModel.from_pretrained(
        "openai-community/gpt2", dtype=torch.float32
    ).eval()
    ids = quantization.wikitext_ids(tokenizer)
    ours, scored = quantization.perplexity(model, ids, "cpu", WINDOWS)

    nll = 0.0
    with torch.no_grad():
        for begin, end, first in quantization.windows(ids.shape[1], WINDOWS):
            chunk = ids[:, begin:end]
            log_probs = torch.log_softmax(reference(chunk).logits[0, :-1], -1)
            picked = log_probs.gather(1, chunk[0, 1:, None])[:, 0]
            nll -= picked[first - begin - 1 :].sum().item()
    assert ours == pytest.approx(math.exp(nll / scored), rel=1e-4)

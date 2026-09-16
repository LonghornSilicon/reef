"""Tiny randomly-initialized Gemma 3 against ``transformers``."""

import pytest
import torch
import transformers
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

from configs.gemma3 import ATTENTION_PATTERN, GEMMA3_CONFIGS, Gemma3Config
from models.gemma3 import Gemma3ForCausalLM, gemma3

pytestmark = pytest.mark.unit

TINY = {
    "hidden_size": 64,
    "intermediate_size": 128,
    # Two full local/global periods, so the pattern and the global layers'
    # separate RoPE table are each exercised twice.
    "num_hidden_layers": 12,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 100,
    "sliding_window": 4,
    "query_pre_attn_scalar": 16,
    "max_position_embeddings": 256,
}
BATCH = 2
# Comfortably longer than the 4-token sliding window, so the local layers
# really do have to mask something out.
PROMPT_LEN = 20
DECODE_STEPS = 4

# Global-layer RoPE extension factors: 270M and 1B ship without one, 4B and
# larger ship with 8.0.
SCALINGS = [1.0, 8.0]


def build_pair(
    rope_global_scaling: float = 1.0,
) -> tuple[Gemma3ForCausalLM, transformers.Gemma3ForCausalLM]:
    config = Gemma3Config(rope_global_scaling=rope_global_scaling, **TINY)
    scaling = None
    if rope_global_scaling != 1.0:
        scaling = {"factor": rope_global_scaling, "rope_type": "linear"}
    reference = transformers.Gemma3ForCausalLM(
        Gemma3TextConfig(
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_global_base_freq,
            rope_local_base_freq=config.rope_local_base_freq,
            rope_scaling=scaling,
            attention_bias=config.attention_bias,
            pad_token_id=0,
            attn_implementation="eager",
            **TINY,
        )
    ).eval()

    ours = Gemma3ForCausalLM(config)
    ours.load_state_dict(reference.state_dict(), strict=True)
    return ours.eval(), reference


def test_state_dict_keys_match_reference() -> None:
    ours, reference = build_pair()
    assert set(ours.state_dict()) == set(reference.state_dict())


@pytest.mark.parametrize("scaling", SCALINGS)
def test_prefill_logits_match_reference(scaling: float) -> None:
    ours, reference = build_pair(scaling)
    input_ids = torch.randint(0, TINY["vocab_size"], (BATCH, PROMPT_LEN))

    with torch.no_grad():
        expected = reference(input_ids=input_ids).logits
        actual, _ = ours(input_ids)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))


@pytest.mark.parametrize("scaling", SCALINGS)
def test_cached_decode_matches_reference(scaling: float) -> None:
    ours, reference = build_pair(scaling)
    total = PROMPT_LEN + DECODE_STEPS
    input_ids = torch.randint(0, TINY["vocab_size"], (BATCH, total))

    with torch.no_grad():
        expected = reference(input_ids=input_ids).logits
        actual, cache = ours(input_ids[:, :PROMPT_LEN])
        for step in range(PROMPT_LEN, total):
            actual, cache = ours(input_ids[:, step : step + 1], cache)

    torch.testing.assert_close(
        actual[:, -1], expected[:, -1], rtol=1e-4, atol=1e-5
    )


def test_cached_decode_matches_a_full_forward() -> None:
    # Catches an off-by-one in the sliding-window mask offset: a cached step
    # measures the window from a different origin than a full forward.
    ours, _ = build_pair()
    total = PROMPT_LEN + DECODE_STEPS
    input_ids = torch.randint(0, TINY["vocab_size"], (BATCH, total))

    with torch.no_grad():
        full, _ = ours(input_ids)
        stepped, cache = ours(input_ids[:, :PROMPT_LEN])
        for step in range(PROMPT_LEN, total):
            stepped, cache = ours(input_ids[:, step : step + 1], cache)

    torch.testing.assert_close(stepped[:, -1], full[:, -1])


def test_local_and_global_layers_alternate() -> None:
    ours, _ = build_pair()
    kinds = [layer.self_attn.sliding for layer in ours.model.layers]
    assert kinds[:6] == [True, True, True, True, True, False]
    assert kinds.count(False) == len(kinds) // ATTENTION_PATTERN


def test_global_layers_use_the_longer_rope_base() -> None:
    ours, _ = build_pair()
    local, glob = ours.model.rotary_local, ours.model.rotary_global
    assert not torch.equal(local.inv_freq, glob.inv_freq)
    for index, layer in enumerate(ours.model.layers):
        expected = local if (index + 1) % ATTENTION_PATTERN else glob
        assert layer.self_attn.rotary is expected


def test_sliding_layers_carry_a_window_and_global_layers_do_not() -> None:
    ours, _ = build_pair()
    for index, layer in enumerate(ours.model.layers):
        window = layer.self_attn.attention.sliding_window
        if (index + 1) % ATTENTION_PATTERN:
            assert window == TINY["sliding_window"]
        else:
            assert window is None


def test_embeddings_are_scaled_by_sqrt_hidden_size() -> None:
    # Captured from the real forward pass so this fails if the scaling is
    # removed, rather than merely recomputing it here.
    ours, _ = build_pair()
    model = ours.model
    ids = torch.randint(0, TINY["vocab_size"], (1, 3))
    captured: list[torch.Tensor] = []
    handle = model.layers[0].register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0])
    )

    try:
        with torch.no_grad():
            model(ids)
            raw = model.embed_tokens(ids)
    finally:
        handle.remove()

    expected = raw * (TINY["hidden_size"] ** 0.5)
    assert TINY["hidden_size"] ** 0.5 == pytest.approx(8.0)
    torch.testing.assert_close(captured[0], expected)


@pytest.mark.parametrize("name", list(GEMMA3_CONFIGS))
def test_published_sizes_alternate_attention(name: str) -> None:
    config = GEMMA3_CONFIGS[name]
    layers = range(config.num_hidden_layers)
    globals_ = [index for index in layers if not config.is_sliding(index)]
    expected = [
        index for index in layers if (index + 1) % ATTENTION_PATTERN == 0
    ]
    assert globals_ == expected
    # Every published depth is long enough to reach the third global layer;
    # 270M is the shortest at 18 layers, so it stops at 17.
    assert globals_[:3] == [5, 11, 17]
    assert all(config.is_sliding(index) for index in (0, 1, 2, 3, 4))


@pytest.mark.parametrize(
    ("name", "billions"),
    [
        ("Gemma3-270M", 0.268),
        ("Gemma3-1B", 1.000),
        ("Gemma3-4B", 3.880),
        ("Gemma3-12B", 11.766),
        ("Gemma3-27B", 27.009),
    ],
)
def test_published_sizes_have_the_expected_parameter_count(
    name: str, billions: float
) -> None:
    # Meta device: 27B would be 100 GB of real weights.
    with torch.device("meta"):
        model = gemma3(name)
    total = sum(p.numel() for p in model.parameters())
    assert total / 1e9 == pytest.approx(billions, abs=5e-4)


def test_27b_derives_its_attention_scale_from_the_published_scalar() -> None:
    config = GEMMA3_CONFIGS["Gemma3-27B"]
    assert config.head_dim == 128
    assert config.query_pre_attn_scalar == 168
    with torch.device("meta"):
        model = gemma3("Gemma3-27B")
    scale = model.model.layers[0].self_attn.attention.scale
    assert scale == pytest.approx(168**-0.5)
    assert scale != pytest.approx(128**-0.5)


def test_tied_lm_head_shares_the_embedding_table() -> None:
    with torch.device("meta"):
        model = gemma3("Gemma3-270M")
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_unknown_size_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown Gemma 3 size"):
        gemma3("Gemma3-2B")

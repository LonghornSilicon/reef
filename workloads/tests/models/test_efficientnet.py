"""Randomly-initialized EfficientNets against ``torchvision``."""

import pytest
import torch
import torchvision

from configs.efficientnet import EFFICIENTNET_CONFIGS, TORCHVISION_BUILDERS
from models.efficientnet import EfficientNet, efficientnet
from tests.common import BF16_TOLERANCE, assert_matches

pytestmark = pytest.mark.unit

SIZES = list(EFFICIENTNET_CONFIGS)
PARITY_SIZES = [
    "EfficientNet-B0",
    "EfficientNet-B1",
    "EfficientNetV2-S",
    "EfficientNetV2-M",
]
INPUT_SIZE = 64

PARAMETER_COUNTS = {
    "EfficientNet-B0": 5_288_548,
    "EfficientNet-B1": 7_794_184,
    "EfficientNet-B2": 9_109_994,
    "EfficientNet-B3": 12_233_232,
    "EfficientNet-B4": 19_341_616,
    "EfficientNet-B5": 30_389_784,
    "EfficientNet-B6": 43_040_704,
    "EfficientNet-B7": 66_347_960,
    "EfficientNetV2-S": 21_458_488,
    "EfficientNetV2-M": 54_139_356,
    "EfficientNetV2-L": 118_515_272,
}

# torchvision's kaiming fan_out init shrinks random-init logits to ~1e-14, so
# assert_matches' absolute tolerance alone passes anything; the error is also
# bounded relative to the largest logit. Measured float32 ratio is ~2.4e-6.
RELATIVE_TOLERANCE = 1e-5


def build_pair(
    name: str, device: str = "cpu"
) -> tuple[EfficientNet, torch.nn.Module]:
    builder = getattr(torchvision.models, TORCHVISION_BUILDERS[name])
    with torch.device(device):
        reference = builder(weights=None)
        ours = efficientnet(name)
    ours.load_state_dict(reference.state_dict(), strict=True)
    return ours.eval(), reference.eval()


def assert_matches_to_scale(
    actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype
) -> None:
    assert_matches(actual, expected, dtype)
    rtol = RELATIVE_TOLERANCE
    if dtype != torch.float32:
        rtol = BF16_TOLERANCE["rtol"]
    error = (actual.float() - expected.float()).abs().max()
    assert error <= rtol * expected.float().abs().max()


@pytest.mark.parametrize("name", SIZES)
def test_state_dict_keys_match_torchvision(name: str) -> None:
    ours, reference = build_pair(name, device="meta")
    assert set(ours.state_dict()) == set(reference.state_dict())


@pytest.mark.parametrize("name", SIZES)
def test_parameter_count_matches_torchvision(name: str) -> None:
    ours, reference = build_pair(name, device="meta")
    ours_total = sum(p.numel() for p in ours.parameters())
    reference_total = sum(p.numel() for p in reference.parameters())
    assert ours_total == reference_total == PARAMETER_COUNTS[name]


@pytest.mark.parametrize("name", PARITY_SIZES)
def test_logits_match_torchvision(name: str) -> None:
    ours, reference = build_pair(name)
    x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        actual, expected = ours(x), reference(x)

    assert_matches_to_scale(actual, expected, torch.float32)


def test_bfloat16_logits_match_torchvision() -> None:
    ours, reference = build_pair("EfficientNet-B0")
    ours.to(torch.bfloat16)
    reference.to(torch.bfloat16)
    x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.bfloat16)

    with torch.no_grad():
        actual, expected = ours(x), reference(x)

    assert_matches_to_scale(actual, expected, torch.bfloat16)


def test_mbconv_skips_the_residual_when_strided() -> None:
    stage = efficientnet("EfficientNet-B0").features[2]
    assert stage[0].use_res_connect is False
    assert stage[1].use_res_connect is True
    x = torch.randn(1, 16, 8, 8)
    with torch.no_grad():
        assert stage[0](x).shape == (1, 24, 4, 4)


def test_depth_multiplier_rounds_each_stage_up() -> None:
    with torch.device("meta"):
        model = efficientnet("EfficientNet-B1")
    stages = model.features[1:-1]
    assert [len(stage) for stage in stages] == [2, 3, 3, 4, 4, 5, 2]


def test_unknown_size_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown EfficientNet size"):
        efficientnet("EfficientNet-B8")

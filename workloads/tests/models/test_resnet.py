"""Randomly-initialized ResNets against ``torchvision``.

Weights are random here so nothing is downloaded; the pretrained checkpoints
are covered by the ``slow`` tests in :mod:`tests.models.test_checkpoints`.
"""

import pytest
import torch
import torchvision

from configs.resnet import RESNET_CONFIGS, TORCHVISION_BUILDERS
from models.resnet import ResNet, resnet

pytestmark = pytest.mark.unit

DEPTHS = list(RESNET_CONFIGS)

# 64x64 rather than 224x224: the im2col convolution is quadratic in the
# feature-map size and ResNet-101 is 101 layers deep, so the full resolution
# would dominate the unit suite. Every stage still downsamples (64 -> 32 -> 16
# -> 8 -> 4 -> 2), so the layer geometry under test is unchanged.
INPUT_SIZE = 64

# A randomly-initialized ResNet-101 is not a trained network: with Kaiming
# init the residual stream compounds across 101 layers until logits reach
# ~1.5e4, so the *absolute* deviation from torchvision scales with that even
# though the relative error stays at float32 reassociation noise (~1.5e-5,
# the same as ResNet-50's). Bound the relative term and leave the absolute
# term loose; the pretrained checkpoints in test_checkpoints.py, whose logits
# are O(10), are asserted far more tightly.
TOLERANCE = {"rtol": 1e-4, "atol": 1e-2}


def build_pair(name: str) -> tuple[ResNet, torch.nn.Module]:
    """Build our ResNet and a torchvision one sharing random weights.

    Args:
        name: A key of ``RESNET_CONFIGS``, such as ``"ResNet-50"``.

    Returns:
        Our model and the reference model, both in eval mode.
    """
    builder = getattr(torchvision.models, TORCHVISION_BUILDERS[name])
    reference = builder(weights=None).eval()
    ours = resnet(name)
    ours.load_state_dict(reference.state_dict(), strict=True)
    return ours.eval(), reference


@pytest.mark.parametrize("name", DEPTHS)
def test_state_dict_keys_match_torchvision(name: str) -> None:
    """Our module layout reproduces torchvision's parameter names.

    This covers the running statistics too, so a checkpoint's batch-norm
    buffers land where they belong rather than being silently dropped.
    """
    ours, reference = build_pair(name)
    assert set(ours.state_dict()) == set(reference.state_dict())


@pytest.mark.parametrize("name", DEPTHS)
def test_logits_match_torchvision(name: str) -> None:
    """Eval-mode logits match torchvision at every depth."""
    ours, reference = build_pair(name)
    x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        actual, expected = ours(x), reference(x)

    torch.testing.assert_close(actual, expected, **TOLERANCE)
    # Argmax agreement is the property that matters downstream and is
    # insensitive to the tolerance above.
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))


@pytest.mark.parametrize("name", DEPTHS)
def test_parameter_count_matches_torchvision(name: str) -> None:
    """We allocate exactly the parameters torchvision does, no more."""
    ours, reference = build_pair(name)
    ours_total = sum(p.numel() for p in ours.parameters())
    reference_total = sum(p.numel() for p in reference.parameters())
    assert ours_total == reference_total


@pytest.mark.parametrize(
    ("name", "expansion", "blocks"),
    [
        ("ResNet-18", 1, (2, 2, 2, 2)),
        ("ResNet-34", 1, (3, 4, 6, 3)),
        ("ResNet-50", 4, (3, 4, 6, 3)),
        ("ResNet-101", 4, (3, 4, 23, 3)),
    ],
)
def test_stage_depths_and_expansion(
    name: str, expansion: int, blocks: tuple[int, ...]
) -> None:
    """Each depth is assembled from the published block counts."""
    model = resnet(name)
    config = RESNET_CONFIGS[name]
    assert config.expansion == expansion
    assert config.blocks_per_stage == blocks
    stages = (model.layer1, model.layer2, model.layer3, model.layer4)
    assert tuple(len(stage) for stage in stages) == blocks


def test_bottleneck_places_stride_on_the_3x3() -> None:
    """ResNet-50 is the V1.5 variant, not the original V1 ordering.

    In V1.5 the downsampling stride sits on the 3x3 convolution; the original
    paper put it on the leading 1x1. Getting this backwards still loads a
    checkpoint cleanly but computes different numbers, so pin it directly.
    """
    block = resnet("ResNet-50").layer2[0]
    assert block.conv1.stride == 1
    assert block.conv2.stride == 2


def test_unknown_depth_is_rejected() -> None:
    """Asking for a depth we do not publish fails loudly."""
    with pytest.raises(KeyError, match="unknown ResNet depth"):
        resnet("ResNet-152")

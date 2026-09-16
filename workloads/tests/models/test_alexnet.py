"""Randomly-initialized AlexNet against ``torchvision``.

Weights are random here so nothing is downloaded; the pretrained checkpoint is
covered by the ``slow`` tests in :mod:`tests.models.test_checkpoints`.
"""

import pytest
import torch
import torchvision

from models.alexnet import AlexNet

pytestmark = pytest.mark.unit

# 224 leaves the feature map at 6x6, so the adaptive pool is a no-op; 256
# leaves it at 7x7, which forces the overlapping-window path.
INPUT_SIZES = [224, 256]


def build_pair() -> tuple[AlexNet, torch.nn.Module]:
    """Build our AlexNet and a torchvision one sharing random weights.

    Returns:
        Our model and the reference model, both in eval mode.
    """
    reference = torchvision.models.alexnet(weights=None).eval()
    ours = AlexNet()
    ours.load_state_dict(reference.state_dict(), strict=True)
    return ours.eval(), reference


def test_state_dict_keys_match_torchvision() -> None:
    """Our module layout reproduces torchvision's parameter names."""
    ours, reference = build_pair()
    assert set(ours.state_dict()) == set(reference.state_dict())


@pytest.mark.parametrize("size", INPUT_SIZES)
def test_logits_match_torchvision(size: int) -> None:
    """Eval-mode logits match torchvision for both input sizes."""
    ours, reference = build_pair()
    x = torch.randn(2, 3, size, size)

    with torch.no_grad():
        torch.testing.assert_close(ours(x), reference(x))


@pytest.mark.parametrize(("size", "spatial"), [(224, 6), (256, 7)])
def test_feature_map_size_before_adaptive_pool(size: int, spatial: int) -> None:
    """The 256 case really does exercise the adaptive pool's resize path."""
    ours, _ = build_pair()
    with torch.no_grad():
        features = ours.features(torch.randn(1, 3, size, size))
    assert features.shape == (1, 256, spatial, spatial)


def test_dropout_is_inert_in_eval_mode() -> None:
    """Repeated eval-mode passes agree despite the dropout layers."""
    ours, _ = build_pair()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        torch.testing.assert_close(ours(x), ours(x))

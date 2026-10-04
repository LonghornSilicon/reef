"""Check that the scaffold installs as an importable Python package."""

import pytest

import inference_engine


@pytest.mark.unit
def test_package_is_importable() -> None:
    """Catch Python packaging mistakes before adding orchestration code."""
    assert inference_engine.__file__ is not None

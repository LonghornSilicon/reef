"""Check that the package and its experiments are importable."""

import importlib

import pytest

import reef_perf


@pytest.mark.unit
def test_package_is_importable() -> None:
    """Catch Python packaging mistakes early."""
    assert reef_perf.__file__ is not None
    assert reef_perf.DEFAULT_SIM_CONFIG.is_file()


@pytest.mark.unit
@pytest.mark.parametrize(
    "module",
    [
        "reef_perf.microbenchmarks",
        "reef_perf.topdown",
        "reef_perf.experiments.sweep.main",
        "reef_perf.experiments.sweep.sweep",
        "reef_perf.experiments.sweep.plot",
        "reef_perf.experiments.correlation.main",
        "reef_perf.experiments.topdown.main",
    ],
)
def test_module_imports(module: str) -> None:
    """Every module and experiment script imports without side effects."""
    importlib.import_module(module)

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
@pytest.mark.parametrize("module", ["main", "run", "plot", "workloads"])
def test_run_experiment_modules_import(module: str) -> None:
    """Every script of the run experiment imports without side effects."""
    importlib.import_module(f"reef_perf.experiments.run.{module}")

"""Shared pytest setup: test categories, the C++ build and workload ELFs."""

import os
import subprocess
from pathlib import Path

import pytest

from reef_perf.experiments.run.workloads import build_workloads, list_workloads
from reef_perf.sim import BIN_DIR_ENV, MODULE_ROOT


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Require every test to declare a unit, integration, or experiment mark."""
    categories = ("unit", "integration", "experiment")
    missing = [
        item.nodeid
        for item in items
        if not any(item.get_closest_marker(name) for name in categories)
    ]
    if missing:
        raise pytest.UsageError(
            "Tests without a category mark:\n"
            + "\n".join(f"  {name}" for name in missing)
        )


def run_checked(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a command, failing the test with its output if it fails.

    Args:
        cmd: Command and arguments.
        cwd: Working directory.
    """
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        pytest.fail(
            f"{' '.join(cmd)} failed:\n{proc.stdout}\n{proc.stderr}",
            pytrace=False,
        )


@pytest.fixture(scope="session")
def cpp_build_dir() -> Path:
    """Configure and build all C++ targets (incremental across runs).

    The build lives in build/pytest so repeated test runs only rebuild what
    changed. Also points reef_perf.sim at the freshly built binaries.
    """
    build_dir = MODULE_ROOT / "build" / "pytest"
    run_checked(
        [
            "cmake",
            "-S",
            str(MODULE_ROOT),
            "-B",
            str(build_dir),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    )
    run_checked(["cmake", "--build", str(build_dir)])
    os.environ[BIN_DIR_ENV] = str(build_dir / "bin")
    return build_dir


@pytest.fixture(scope="session")
def workload_elfs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Assemble every workload once per session."""
    return build_workloads(
        list_workloads(), tmp_path_factory.mktemp("workloads")
    )

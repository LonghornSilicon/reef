"""Host correctness check of the standalone Coral W8A8 harness."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
def test_w8a8_harness(tmp_path: Path) -> None:
    """Compile the target-compatible kernel and validate all fixture stages."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a host C++ compiler is required")
    root = Path(__file__).resolve().parents[2]
    simulations = root / "src/csrc/src/simulations"
    executable = tmp_path / "mlp_test"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-DREEF_HOST_TEST",
            str(simulations / "kernels/mlp_w8a8.cc"),
            str(simulations / "integration_tests/mlp_main.cc"),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True
    )
    assert "status=0 failures=0" in result.stdout

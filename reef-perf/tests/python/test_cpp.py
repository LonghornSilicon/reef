"""Run the C++ Google Test suite through CTest."""

import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.slow
def test_ctest_passes(cpp_build_dir: Path) -> None:
    """Every Google Test case in tests/cpp passes."""
    result = subprocess.run(
        ["ctest", "--test-dir", str(cpp_build_dir), "--output-on-failure"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

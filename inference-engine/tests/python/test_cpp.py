"""Run the C++ Google Test suite from pytest."""

import subprocess
from pathlib import Path

import pytest


@pytest.mark.unit
def test_cpp_unit_tests(cpp_build_dir: Path) -> None:
    """Run every C++ test through CTest; failures show CTest's output."""
    subprocess.run(
        ["ctest", "--test-dir", str(cpp_build_dir), "--output-on-failure"],
        check=True,
    )

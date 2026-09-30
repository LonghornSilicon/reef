"""Build the host C++ slice once for the module's pytest run."""

import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def cpp_build_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Configure and build C++ targets in a temporary directory."""
    source_dir = Path(__file__).resolve().parents[1]
    build_dir = tmp_path_factory.mktemp("cpp-build")
    subprocess.run(
        ["cmake", "-S", str(source_dir), "-B", str(build_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    return build_dir

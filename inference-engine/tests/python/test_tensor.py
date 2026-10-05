"""Exercise C++ multiplication through its Python entry point."""

import subprocess
from pathlib import Path

import pytest

from inference_engine.tensor import matmul


@pytest.mark.unit
def test_cpp_tensor_unit_tests(cpp_build_dir: Path) -> None:
    """Run the native test target through the module's pytest command."""
    subprocess.run(
        ["ctest", "--test-dir", str(cpp_build_dir), "--output-on-failure"],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.integration
def test_python_calls_cpp_tensor(cpp_build_dir: Path) -> None:
    """Send values from Python to C++ and read back the result."""
    executable = cpp_build_dir / "bin" / "tensor_cli"
    result = matmul(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        executable,
    )
    assert result == [[58.0, 64.0], [139.0, 154.0]]

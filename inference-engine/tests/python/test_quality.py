"""Check source lint and generated C++ API documentation."""

import subprocess
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
@pytest.mark.slow
def test_lint() -> None:
    """Check Python and C++ lint without modifying source files."""
    subprocess.run(
        ["uv", "run", "--locked", "ruff", "check", "."],
        cwd=MODULE_ROOT,
        check=True,
    )
    subprocess.run(
        ["uv", "run", "--locked", "ruff", "format", "--check", "."],
        cwd=MODULE_ROOT,
        check=True,
    )
    subprocess.run(["bash", "tools/check_cpp.sh"], cwd=MODULE_ROOT, check=True)


@pytest.mark.integration
def test_doxygen_builds_without_warnings(tmp_path: Path) -> None:
    """Generate the C++ API documentation with warnings treated as errors."""
    output_dir = tmp_path / "doxygen"
    config = (MODULE_ROOT / "Doxyfile").read_text()
    config += f'\nOUTPUT_DIRECTORY = "{output_dir}"\n'
    subprocess.run(
        ["doxygen", "-"],
        input=config,
        text=True,
        cwd=MODULE_ROOT,
        check=True,
    )
    assert (output_dir / "html" / "index.html").is_file()

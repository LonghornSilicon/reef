"""Build the host C++ slice once for the module's pytest run."""

import subprocess
from pathlib import Path

import pytest


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

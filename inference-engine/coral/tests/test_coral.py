"""Check the Coral kernels against the reference engine on the simulator."""

import subprocess
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.slow
@pytest.mark.coral
@pytest.mark.timeout(1800)
def test_coral_kernels_match_engine() -> None:
    """Run every Coral golden test; any bit that differs fails it."""
    subprocess.run(["bash", "coral/tools/test.sh"], cwd=MODULE_ROOT, check=True)

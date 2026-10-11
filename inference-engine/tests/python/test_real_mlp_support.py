"""Check artifact isolation and the reviewed default-map memory proposal."""

import json
from pathlib import Path

import pytest

from inference_engine.experiments.w8a8_mlp.real_support import (
    REEF_ROOT,
    SIMULATIONS,
    external_directory,
    validate_memory_plan,
)


@pytest.mark.unit
def test_real_memory_plan() -> None:
    """Require the real buffers to fit with explicit stack/runtime reserves."""
    result = validate_memory_plan(
        SIMULATIONS / "validation/real_block_memory.json"
    )
    assert result["status"] == "PASS"
    assert result["regions"]["EXTMEM"]["reserved_bytes"] == 2197504
    assert result["regions"]["DTCM"]["reserved_bytes"] == 32760


@pytest.mark.unit
@pytest.mark.parametrize("defect", ["overlap", "bytes", "dtype", "missing"])
def test_invalid_memory_plan(tmp_path: Path, defect: str) -> None:
    """Reject corrupted size, placement, type and inventory descriptions."""
    plan = json.loads(
        (SIMULATIONS / "validation/real_block_memory.json").read_text()
    )
    if defect == "overlap":
        plan["buffers"][1]["address"] = plan["buffers"][0]["address"]
    elif defect == "bytes":
        plan["buffers"][0]["bytes"] -= 1
    elif defect == "dtype":
        plan["buffers"][0]["dtype"] = "fp32"
    else:
        plan["buffers"].pop()
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        validate_memory_plan(path)


@pytest.mark.unit
def test_generated_artifacts_external(tmp_path: Path) -> None:
    """Prevent generated arrays or libraries from entering the checkout."""
    assert external_directory(tmp_path) == tmp_path.resolve()
    with pytest.raises(ValueError, match="outside"):
        external_directory(REEF_ROOT / "generated")
    link = tmp_path / "repo-link"
    link.symlink_to(REEF_ROOT, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        external_directory(link / "generated")

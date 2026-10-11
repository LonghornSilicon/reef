"""Torch-dependent host-validation checks, run in the locked workloads env."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from compare_real_host import compare, load_fixture, run_kernel, stage_check

from inference_engine.experiments.w8a8_mlp.export_real import (
    reference_arrays,
    synthetic,
)
from inference_engine.experiments.w8a8_mlp.real_support import (
    build_host_library,
    sha256,
)


@pytest.mark.integration
def test_synthetic_bridge(tmp_path: Path) -> None:
    """Validate every stage and unchanged operands through the real host ABI."""
    torch.set_num_threads(1)
    tensors, _ = synthetic()
    arrays = {name: tensor.numpy() for name, tensor in tensors.items()}
    library = tmp_path / "kernel.so"
    build_host_library(library)
    actual = run_kernel(arrays, library)
    result = compare(arrays, actual)
    assert result["status"] == "PASS"
    actual["a1"][0, 0] += 1
    assert compare(arrays, actual)["checks"]["a1"]["status"] == "FAIL"


@pytest.mark.unit
def test_tolerance_and_nonfinite() -> None:
    """Reject integer differences and NaN without making reports unwritable."""
    integer = np.array([127, -127], np.int8)
    assert stage_check(integer, integer.copy())["status"] == "PASS"
    changed = integer.copy()
    changed[0] -= 1
    assert stage_check(changed, integer)["mismatches"] == 1
    zero = np.array([0], np.float32)
    assert stage_check(np.array([1e-6], np.float32), zero)["status"] == "PASS"
    assert stage_check(np.array([3e-6], np.float32), zero)["status"] == "FAIL"
    result = stage_check(np.array([np.nan], np.float32), zero)
    assert result["status"] == "FAIL"
    json.dumps(result, allow_nan=False)


@pytest.mark.unit
def test_zero_and_ties() -> None:
    """Exercise exact input ties and all-zero weight/activation scale rules."""
    x = torch.tensor([[0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 127.0]])
    arrays = reference_arrays(
        x, torch.zeros(2, 7), torch.zeros(2), torch.zeros(7, 2), torch.zeros(7)
    )
    assert arrays["expected_input_q"].tolist() == [[0, 2, 2, 0, -2, -2, 127]]
    assert arrays["s1"].tolist() == [1, 1]
    assert arrays["expected_hidden_scale"].tolist() == [1]
    assert torch.count_nonzero(arrays["expected_output"]) == 0


@pytest.mark.unit
@pytest.mark.parametrize("defect", ["archive", "shape", "reference"])
def test_fixture_rejection(tmp_path: Path, defect: str) -> None:
    """Reject corruption, unsafe shapes and self-consistent false references."""
    torch.set_num_threads(1)
    tensors, _ = synthetic()
    arrays = {name: tensor.numpy() for name, tensor in tensors.items()}
    if defect == "shape":
        arrays["w1_fp32"] = arrays["w1_fp32"][:, :-1]
    elif defect == "reference":
        arrays["expected_a1"][0, 0] += 1
    archive = tmp_path / "arrays.npz"
    np.savez(archive, **arrays)
    manifest = {
        "schema": 1,
        "kind": "synthetic-unit",
        "dimensions": {"tokens": 3, "width": 8, "hidden": 16},
        "archive_sha256": sha256(archive),
        "arrays": {
            name: {
                "shape": list(a.shape),
                "dtype": a.dtype.str,
                "bytes": a.nbytes,
                "value_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            }
            for name, a in arrays.items()
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    if defect == "archive":
        with archive.open("ab") as stream:
            stream.write(b"corruption")
    with pytest.raises(ValueError):
        load_fixture(tmp_path, allow_synthetic=True)

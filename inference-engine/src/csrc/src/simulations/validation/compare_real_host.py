"""Compare the scalar host kernel against immutable PyTorch W8A8 stages."""

import argparse
import ctypes
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from inference_engine.experiments.w8a8_mlp.export_real import (
    LAYER,
    MODEL,
    REVISION,
    reference_arrays,
)
from inference_engine.experiments.w8a8_mlp.real_support import (
    ATOL,
    RTOL,
    SIMULATIONS,
    build_host_library,
    external_directory,
    sha256,
    validate_memory_plan,
)


def load_fixture(path: Path, *, allow_synthetic: bool) -> tuple[dict, dict]:
    """Require pinned provenance, storage and recorded value hashes."""
    manifest = json.loads((path / "manifest.json").read_text())
    archive = path / "arrays.npz"
    if manifest["schema"] != 1 or sha256(archive) != manifest["archive_sha256"]:
        raise ValueError("Fixture archive integrity check failed")
    real = manifest["kind"] == "real"
    if not real and not (
        allow_synthetic and manifest["kind"] == "synthetic-unit"
    ):
        raise ValueError(
            "A real fixture is required unless --allow-synthetic is explicit"
        )
    dimensions = manifest["dimensions"]
    expected = {
        "tokens": 3,
        "width": 512 if real else 8,
        "hidden": 2048 if real else 16,
    }
    if dimensions != expected:
        raise ValueError("Unexpected fixture dimensions")
    if real and (
        manifest["model"],
        manifest["revision"],
        manifest["tokenizer_revision"],
        manifest["layer"],
    ) != (
        MODEL,
        REVISION,
        REVISION,
        LAYER,
    ):
        raise ValueError("Checkpoint/tokenizer/layer differs from the pin")
    arrays = {}
    with np.load(archive, allow_pickle=False) as stored:
        if set(stored.files) != set(manifest["arrays"]):
            raise ValueError("Array inventory differs from manifest")
        for name in stored.files:
            a = np.ascontiguousarray(stored[name])
            info = manifest["arrays"][name]
            if (
                list(a.shape),
                a.dtype.str,
                a.nbytes,
                hashlib.sha256(a.tobytes()).hexdigest(),
            ) != (
                info["shape"],
                info["dtype"],
                info["bytes"],
                info["value_sha256"],
            ):
                raise ValueError(f"Array integrity check failed: {name}")
            if a.dtype not in (
                np.dtype("float32"),
                np.dtype("int8"),
                np.dtype("int32"),
            ):
                raise ValueError(f"Unsupported array type: {name}")
            if not np.isfinite(a).all():
                raise ValueError(f"Nonfinite array: {name}")
            arrays[name] = a
    n, d, h = expected["tokens"], expected["width"], expected["hidden"]
    for name, shape in {
        "input_fp32": (n, d),
        "w1_fp32": (h, d),
        "b1": (h,),
        "w2_fp32": (d, h),
        "b2": (d,),
        "fp32_block_output": (n, d),
    }.items():
        if arrays[name].shape != shape or arrays[name].dtype != np.float32:
            raise ValueError(f"Invalid original operand shape/type: {name}")
    # Recompute the oracle from original FP32 operands; hashes alone do not
    # validate a self-consistent but incorrectly exported fixture.
    t = {
        k: torch.from_numpy(arrays[k])
        for k in (
            "input_fp32",
            "w1_fp32",
            "b1",
            "w2_fp32",
            "b2",
        )
    }
    reference = reference_arrays(
        t["input_fp32"],
        t["w1_fp32"],
        t["b1"],
        t["w2_fp32"],
        t["b2"],
    )
    if set(arrays) != set(reference) | {"fp32_block_output"}:
        raise ValueError("Unexpected reference stage inventory")
    for name, tensor in reference.items():
        expected_array = tensor.numpy()
        if arrays[name].dtype != expected_array.dtype or not np.array_equal(
            arrays[name],
            expected_array,
        ):
            raise ValueError(f"Reference recomputation differs: {name}")
    for name in ("s1", "s2", "expected_input_scale", "expected_hidden_scale"):
        if np.any(arrays[name] < np.finfo(np.float32).tiny):
            raise ValueError(f"Unsupported scale: {name}")
    return arrays, manifest


def run_kernel(arrays: dict, library: Path) -> dict:
    """Run through ctypes with typed, contiguous, non-overlapping buffers."""
    tokens, width = arrays["input_fp32"].shape
    hidden = arrays["w1"].shape[0]
    outputs = {
        "input_q": np.empty((tokens, width), np.int8),
        "input_scale": np.empty(tokens, np.float32),
        "a1": np.empty((tokens, hidden), np.int32),
        "pre": np.empty((tokens, hidden), np.float32),
        "hidden": np.empty((tokens, hidden), np.float32),
        "hidden_q": np.empty((tokens, hidden), np.int8),
        "hidden_scale": np.empty(tokens, np.float32),
        "a2": np.empty((tokens, width), np.int32),
        "output": np.empty((tokens, width), np.float32),
    }
    operands = [
        arrays[k] for k in ("input_fp32", "w1", "s1", "b1", "w2", "s2", "b2")
    ]
    buffers = operands + list(outputs.values())
    before = [hashlib.sha256(a.tobytes()).hexdigest() for a in operands]
    function = ctypes.CDLL(str(library)).reef_validate_mlp
    function.restype = ctypes.c_int
    function.argtypes = [ctypes.c_size_t] * 3 + [
        np.ctypeslib.ndpointer(dtype=a.dtype, ndim=a.ndim, flags="C_CONTIGUOUS")
        for a in buffers
    ]
    if function(tokens, width, hidden, *buffers) != 0:
        raise RuntimeError("Scalar bridge rejected dimensions or rounding mode")
    if before != [hashlib.sha256(a.tobytes()).hexdigest() for a in operands]:
        raise RuntimeError(
            "Scalar kernel changed input/weight/scale/bias operands"
        )
    return outputs


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    """Report errors in float64 without changing FP32 arithmetic."""
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return {
            "nonfinite_actual": int((~np.isfinite(actual)).sum()),
            "nonfinite_expected": int((~np.isfinite(expected)).sum()),
        }
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    norm = float(np.linalg.norm(expected.astype(np.float64)))
    relative = float(np.linalg.norm(difference)) / norm if norm else None
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "relative_l2": relative,
    }


def stage_check(actual: np.ndarray, expected: np.ndarray) -> dict:
    """Keep integer comparisons exact and the original FP32 tolerances."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Comparison shape/dtype mismatch")
    integer = np.issubdtype(actual.dtype, np.integer)
    if integer:
        bad = actual != expected
    else:
        bad = (
            ~np.isfinite(actual)
            | ~np.isfinite(expected)
            | (np.abs(actual - expected) > ATOL + RTOL * np.abs(expected))
        )
    result = {
        "status": "FAIL" if bad.any() else "PASS",
        "exact_integer": integer,
        "mismatches": int(bad.sum()),
    }
    if bad.any():
        index = tuple(np.argwhere(bad)[0])
        result["first"] = {
            "index": [int(i) for i in index],
            "actual": actual[index].item()
            if np.isfinite(actual[index])
            else str(actual[index]),
            "expected": expected[index].item()
            if np.isfinite(expected[index])
            else str(expected[index]),
        }
    if not integer:
        result["error"] = metrics(actual, expected)
    return result


def compare(arrays: dict, actual: dict) -> dict:
    """Separate correctness from original-FP32 quantization error."""
    checks = {
        name: stage_check(a, arrays["expected_" + name])
        for name, a in actual.items()
    }
    checks["original_fp32_block"] = stage_check(
        arrays["fp32_output"],
        arrays["fp32_block_output"],
    )
    hidden_check = checks["hidden_q"]
    if (
        hidden_check["mismatches"]
        and np.isfinite(actual["hidden"]).all()
        and np.isfinite(actual["hidden_scale"]).all()
        and np.all(actual["hidden_scale"] > 0)
    ):
        index = tuple(hidden_check["first"]["index"])
        token = index[0]
        raw = float(actual["hidden"][index] / actual["hidden_scale"][token])
        ref = float(
            arrays["expected_hidden"][index]
            / arrays["expected_hidden_scale"][token]
        )
        hidden_check["rounding_diagnostic"] = {
            "actual_hidden": float(actual["hidden"][index]),
            "reference_hidden": float(arrays["expected_hidden"][index]),
            "actual_scale": float(actual["hidden_scale"][token]),
            "reference_scale": float(arrays["expected_hidden_scale"][token]),
            "actual_unrounded_code": raw,
            "reference_unrounded_code": ref,
            "actual_half_boundary_distance": abs(raw - (np.floor(raw) + 0.5)),
            "reference_half_boundary_distance": abs(
                ref - (np.floor(ref) + 0.5)
            ),
        }
    return {
        "status": "PASS"
        if all(c["status"] == "PASS" for c in checks.values())
        else "FAIL",
        "tolerances": {"atol": ATOL, "rtol": RTOL},
        "checks": checks,
        "quantization_error_vs_original_fp32": {
            name: metrics(
                actual[name],
                arrays[
                    "fp32_block_output" if name == "output" else "fp32_" + name
                ],
            )
            for name in ("pre", "hidden", "output")
        },
        "fake_quantized_fp32_diagnostic": {
            "pre": metrics(arrays["expected_pre"], arrays["fake_pre"]),
            "output": metrics(arrays["expected_output"], arrays["fake_output"]),
            "hidden_code_differences": int(
                np.count_nonzero(
                    arrays["expected_hidden_q"] != arrays["fake_hidden_q"],
                )
            ),
            "scope": "different FP32 reduction order; diagnostic, "
            "not the integer oracle",
        },
    }


def main() -> None:
    """Build, validate the fixture, and save mismatch evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    fixture = external_directory(args.fixture)
    report_path = external_directory(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    report = {"status": "FAIL", "scope": "host scalar correctness only"}
    report["software"] = {"torch": torch.__version__, "numpy": np.__version__}
    try:
        arrays, manifest = load_fixture(
            fixture, allow_synthetic=args.allow_synthetic
        )
        library = report_path.parent / "libw8a8_host.so"
        build = build_host_library(library)
        actual = run_kernel(arrays, library)
        report.update(compare(arrays, actual))
        report["fixture_kind"] = manifest["kind"]
        report["fixture_manifest_sha256"] = sha256(fixture / "manifest.json")
        report["archive_sha256"] = manifest["archive_sha256"]
        report["build"] = build
        report["comparison_sha256"] = sha256(Path(__file__))
        report["memory_plan"] = validate_memory_plan(
            SIMULATIONS / "validation/real_block_memory.json",
        )
    except (
        ValueError,
        RuntimeError,
        OSError,
        KeyError,
        subprocess.SubprocessError,
    ) as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
        if isinstance(error, subprocess.CalledProcessError):
            report["build_stderr"] = error.stderr
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['status']}: {report_path}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

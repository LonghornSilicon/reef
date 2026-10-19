"""Build and run the host unit tests (no sim, no Docker; needs clang++).

    python run_unit_tests.py [test_matmul test_flash ...]

With no names it runs all of unit_tests/. The vector unit and the matrix
engine are emulated in C (-DRVV_EMULATE -DVME_EMULATE), so these check the
kernels' logic and indexing, not cycle counts.
"""

import subprocess
import sys
from pathlib import Path

SIMS = Path(__file__).resolve().parent
BUILD = SIMS / "out" / "build"

# unit test -> kernel sources it links
TESTS = {
    "test_matmul": ["matmul.cc", "matmul_rvv.cc", "vme_matmul.cc"],
    "test_softmax": ["online_softmax.cc", "softmax_lut.cc"],
    "test_gelu": ["gelu.cc"],
    "test_layernorm": ["layernorm.cc"],
    "test_residual": ["residual.cc"],
    "test_flash": [
        "attention_serial.cc",
        "attention_flash.cc",
        "attention_flash_chunk.cc",
        "vme_attention.cc",
        "online_softmax.cc",
        "softmax_lut.cc",
        "matmul.cc",
        "matmul_rvv.cc",
        "vme_matmul.cc",
    ],
}
FLAGS = ["-std=c++17", "-O2", "-DVME_EMULATE", "-DRVV_EMULATE"]


def run_test(name: str) -> bool:
    """Compile one unit test and run it; return whether it passed."""
    exe = BUILD / name
    sources = [f"kernels/{k}" for k in TESTS[name]] + [f"unit_tests/{name}.cc"]
    cmd = ["clang++", *FLAGS, "-Ikernels", "-Iincludes", *sources, "-o", exe]
    if subprocess.run(cmd, cwd=SIMS).returncode:
        return False
    return subprocess.run([exe]).returncode == 0


def main() -> int:
    """Run the requested unit tests and print a pass/fail summary."""
    names = sys.argv[1:] or list(TESTS)
    BUILD.mkdir(parents=True, exist_ok=True)
    status = {n: run_test(n) for n in names}
    print()
    for name, ok in status.items():
        print(f"{name:<16} {'pass' if ok else 'FAIL'}")
    return 0 if all(status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Run the sim tests on the ELFs in bin/ (Coral's VME RTL model, Verilator).

    python run_tests.py [matmul attention vme_attention pointwise ubench]

With no names it runs all five. Each test (tests/test_<name>.py) loads
bin/<its ELF>, checks every kernel call against NumPy and fails if any is
out of tolerance. Logs and per-call .npz records go to out/<name>/. Run
build_elfs.py first, or put prebuilt ELFs in bin/. Needs Docker; the first
run builds the RTL model, which takes hours under amd64 emulation.
"""

import shutil
import sys

from sim.coralnpu import BAZEL, BIN, OUT, docker, prepare

# test -> the ELF it loads (sim/BUILD TESTS)
TESTS = {
    "matmul": "bench_matmul",
    "attention": "bench_attention",
    "vme_attention": "bench_vme_attention",
    "pointwise": "bench_pointwise",
    "ubench": "bench_ubench",
}


def run_test(name: str) -> bool:
    """Run one test in the sim and copy its outputs to out/<name>/."""
    logs = f"bazel-testlogs/tests/cocotb/reef/{name}_test"
    script = (
        f"{BAZEL} test --jobs=6 --test_timeout=86400 "
        "--nozip_undeclared_test_outputs --test_output=errors "
        f"//tests/cocotb/reef:{name}_test 2>&1 | tail -20; "
        "rc=${PIPESTATUS[0]}; "
        f"rm -rf /reef/out/{name} && mkdir -p /reef/out/{name}; "
        f"cp -r {logs}/test.outputs/. {logs}/test.log /reef/out/{name}/ "
        "2>/dev/null; exit $rc"
    )
    shutil.rmtree(OUT / name, ignore_errors=True)
    return docker(script) == 0


def main() -> int:
    """Run the requested tests and print a pass/fail summary."""
    names = sys.argv[1:] or list(TESTS)
    unknown = [n for n in names if n not in TESTS]
    if unknown:
        raise SystemExit(
            f"unknown test(s) {unknown}; choose from {list(TESTS)}"
        )
    missing = [
        TESTS[n] for n in names if not (BIN / f"{TESTS[n]}.elf").exists()
    ]
    if missing:
        raise SystemExit(f"no ELF in bin/ for {missing}: run build_elfs.py")
    prepare()
    status = {n: run_test(n) for n in names}
    for name, ok in status.items():
        print(f"{name:<14} {'pass' if ok else 'FAIL'}  (out/{name}/test.log)")
    return 0 if all(status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

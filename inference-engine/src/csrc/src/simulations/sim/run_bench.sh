#!/usr/bin/env bash
# Runs one Reef cocotb bench inside the coralnpu image; called by run_all.py.
# Expects the simulations/ folder at /reef, the work volume at /work and the
# results dir at /out.  Usage: run_bench.sh <testcase>
set -uo pipefail
test=$1
pkg=/work/coralnpu/tests/cocotb/reef
logs=bazel-testlogs/tests/cocotb/reef/reef_bench_$test

# kernels, headers, config and bench sources go flat into one Bazel package
rm -rf "$pkg" && mkdir -p "$pkg"
cp /reef/kernels/* /reef/includes/* /reef/config/*.json \
    /reef/sim/BUILD /reef/sim/*.cc /reef/sim/*.py "$pkg"/

cd /work/coralnpu
bazel --output_user_root=/work/cache/bazel test --jobs=6 \
    --test_timeout=86400 --nozip_undeclared_test_outputs \
    --test_output=errors "//tests/cocotb/reef:reef_bench_$test" 2>&1 | tail -40
rc=${PIPESTATUS[0]}

mkdir -p "/out/$test"
cp -r "$logs/test.outputs/." "$logs/test.log" "/out/$test/" 2>/dev/null
exit "$rc"

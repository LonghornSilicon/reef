#!/usr/bin/env bash
# Run the traced (waves) version of one or more benches and copy each VCD out
# of the Bazel tree to $WAVES (default ~/LHS/scalar-timing-waves/<bench>.vcd).
#   bash bench/waves.sh trace b1_branch
# Uses the local test strategy so dump.vcd is left in bazel-bin's runfiles
# (the sandbox would delete it). VCDs never go into the shared folder.
set -euo pipefail
CORAL="${CORALNPU:-$HOME/LHS/coralnpu}"
WAVES="${WAVES:-$HOME/LHS/scalar-timing-waves}"
[ $# -ge 1 ] || { echo "usage: $0 <bench> [bench...]" >&2; exit 2; }
bash "$(dirname "$0")/sync.sh" >/dev/null
mkdir -p "$WAVES"
cd "$CORAL"
export CC=/usr/bin/clang CXX=/usr/bin/clang++
targets=()
for b in "$@"; do targets+=("//tests/cocotb/scalar_timing:scalar_timing_waves_cocotb_$b"); done
# A bench whose expectations fail still writes a complete VCD; keep going.
# --nocache_test_results: a cached pass would not regenerate dump.vcd.
bazel test --nocache_test_results --strategy=TestRunner=local "${targets[@]}" || true
for b in "$@"; do
  src="bazel-bin/tests/cocotb/scalar_timing/scalar_timing_waves_cocotb_${b}_cocotb_runner.sh.runfiles/coralnpu_hw/tests/cocotb/scalar_timing/RvvCoreMiniAxi_build/dump.vcd"
  if [ -f "$src" ]; then cp "$src" "$WAVES/$b.vcd"; echo "$WAVES/$b.vcd"; else echo "no VCD for $b" >&2; fi
done

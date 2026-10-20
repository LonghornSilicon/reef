#!/usr/bin/env bash
# Rebuild results/json/d4exp_*.json: the FPU timing with fpnew's 3 FP32 pipeline registers restored (D4),
# i.e. the hardware numbers that bench/collect.py puts in timing.json as `source_intent_if_different`.
#
#   CORALNPU=<separate M3 checkout> WAVES=<folder outside the repo> bash bench/d4_repro/regen_d4exp.sh
#
# It applies floatcore_piperegs_row_by_row.patch to $CORALNPU (row-by-row PipeRegs in float/FloatCore.scala),
# so use a SEPARATE checkout, not the one you run the main suite on. For example:
#   git -C <your-m3-checkout> worktree add --detach ../coralnpu-d4 M3-2026-04-27
# The patched model rebuilds from scratch the first time (~15 min). b5_fpu reports FAILED there on purpose:
# its expectations are calibrated to the unpatched sim; only its measured values are used.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SUITE="$(dirname "$(dirname "$HERE")")"
: "${CORALNPU:?set CORALNPU to a separate CoralNPU checkout at tag M3-2026-04-27}"
: "${WAVES:?set WAVES to a folder for the VCDs, outside the repo}"
PATCH="$HERE/floatcore_piperegs_row_by_row.patch"

if git -C "$CORALNPU" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "patch already applied in $CORALNPU"
else
  git -C "$CORALNPU" apply "$PATCH"
  echo "applied $(basename "$PATCH") to $CORALNPU"
fi
mkdir -p "$WAVES" "$SUITE/results/json"

bash "$SUITE/bench/waves.sh" b5_fpu trace
ZIP="$CORALNPU/bazel-testlogs/tests/cocotb/scalar_timing/scalar_timing_waves_cocotb_b5_fpu/test.outputs/outputs.zip"
unzip -p "$ZIP" b5_fpu.json > "$SUITE/results/json/d4exp_b5_fpu.json"
python3 "$SUITE/bench/trace_scan.py" trace --out "$SUITE/results/json/d4exp_trace_events.json" > /dev/null

echo "wrote results/json/d4exp_b5_fpu.json and d4exp_trace_events.json"
echo "check: git diff --stat -- results/json/   (then python3 bench/collect.py to refresh timing.json)"

#!/usr/bin/env bash
# Lint the vendored CoralNPU RTL with the two define sets we hand out.
#
#   sim    sim.f   + upstream's tested Verilator defines (what cocotb runs)
#   synth  synth.f + -DSYNTHESIS, no firtool verification layers (what PD gets)
#
# TB_SUPPORT is in BOTH sets on purpose: in M3 it is not testbench-only. It adds
# uop_pc/last_uop_valid to the RVV ROB->retire struct, and the scalar
# RetirementBuffer (which gates dispatch) matches vector retirements on them.
# Without it RvvCoreWrapper.sv does not elaborate. See docs/baseline-m3.md.
#
# Uses M3's hermetic Verilator (5.032) from the Bazel cache; override with
# VERILATOR=/path/to/verilator_bin VERILATOR_ROOT=/path/to/root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CORALNPU_RTL="$REPO_ROOT/rtl/vendor/coralnpu"
CACHE="${CACHE:-$HOME/coralnpu-bazel-cache-m3}"
VERILATOR="${VERILATOR:-$(find "$CACHE" -path '*k8-opt-exec*/external/verilator/verilator_bin' -type f 2>/dev/null | head -1)}"
export VERILATOR_ROOT="${VERILATOR_ROOT:-$(find "$CACHE" -maxdepth 4 -path '*/external/verilator' -type d 2>/dev/null | head -1)}"
[ -x "$VERILATOR" ] || { echo "No Verilator found; run scripts/verify_upstream_release.sh once or set VERILATOR" >&2; exit 1; }

DEFINES="-DUSE_GENERIC -DTB_SUPPORT -DVLEN_128 -DZVE32F_ON"
# Same waivers as upstream's vopts (hdl/chisel/src/coralnpu/BUILD, M3).
WAIVE="-Wno-WIDTH -Wno-CASEINCOMPLETE -Wno-LATCH -Wno-SIDEEFFECT -Wno-MULTIDRIVEN
       -Wno-UNOPTFLAT -Wno-BLKANDNBLK -Wno-CASEX -Wno-ASCRANGE -Wno-WIDTHEXPAND
       -Wno-WIDTHTRUNC -Wno-UNSIGNED"

status=0
for set in sim synth; do
  extra=""; [ "$set" = synth ] && extra="-DSYNTHESIS"
  # shellcheck disable=SC2086
  if "$VERILATOR" --lint-only --top-module RvvCoreMiniAxi -f "$CORALNPU_RTL/$set.f" \
       $DEFINES $extra $WAIVE; then
    echo "lint $set: PASS"
  else
    echo "lint $set: FAIL"; status=1
  fi
done
exit $status

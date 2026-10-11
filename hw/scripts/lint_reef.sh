#!/usr/bin/env bash
# Lint reef_top (our hand-written RTL) on top of the vendored core, synth define set.
# Our files get Verilator's full warning set; the vendored core keeps upstream's waivers.
#
# VERILATOR=/path/to/verilator scripts/lint_reef.sh   (defaults as in lint_vendor.sh)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CORALNPU_RTL="$REPO_ROOT/rtl/vendor/coralnpu" REEF_RTL="$REPO_ROOT/rtl/reef"
export SRAM_RTL="$REPO_ROOT/rtl/sram" CLOCK_RTL="$REPO_ROOT/rtl/clock" REEF_ROOT="$REPO_ROOT"
CACHE="${CACHE:-$HOME/coralnpu-bazel-cache-m3}"
VERILATOR="${VERILATOR:-$(find "$CACHE" -path '*k8-opt-exec*/external/verilator/verilator_bin' -type f 2>/dev/null | head -1 || true)}"
if [ -z "${VERILATOR_ROOT:-}" ] && [ -d "$CACHE" ]; then
  export VERILATOR_ROOT="$(find "$CACHE" -maxdepth 4 -path '*/external/verilator' -type d 2>/dev/null | head -1 || true)"
fi
[ -x "$VERILATOR" ] || VERILATOR="$(command -v verilator)"

DEFINES="-DUSE_GENERIC -DTB_SUPPORT -DVLEN_128 -DZVE32F_ON -DSYNTHESIS"
WAIVE="-Wno-WIDTH -Wno-CASEINCOMPLETE -Wno-LATCH -Wno-SIDEEFFECT -Wno-MULTIDRIVEN
       -Wno-UNOPTFLAT -Wno-BLKANDNBLK -Wno-CASEX -Wno-ASCRANGE -Wno-WIDTHEXPAND
       -Wno-WIDTHTRUNC -Wno-UNSIGNED"

# 1) Our modules alone, all warnings on (the core is a blackbox here).
for top in reef_spi2axi reef_axi_err_slave reef_rst_sync; do
  "$VERILATOR" --lint-only -Wall --top-module "$top" "$REEF_RTL/$top.sv"
done
echo "lint reef blocks (-Wall): PASS"

# 2) reef_top with the full core.
# shellcheck disable=SC2086
"$VERILATOR" --lint-only --top-module reef_top -f "$CORALNPU_RTL/synth.f" -f "$REEF_RTL/reef.f" \
  $DEFINES $WAIVE -Wno-PINCONNECTEMPTY -Wno-PINMISSING
echo "lint reef_top + core (synth set): PASS"

# 3) sky130 TCM wrappers, both TCM shapes, all warnings on: SRAM22 (Drop 0, with SRAM22's macro
#    model) and OpenRAM (fallback, with our cycle model of that macro).
for f in sram22_sim.f sky130_sim.f; do
  for n in 512 2048; do
    "$VERILATOR" --lint-only -Wall --top-module Sram -GNUM_ENTRIES=$n -f "$SRAM_RTL/$f"
  done
done
echo "lint Sram_sram22 + Sram_sky130 (-Wall, 512 and 2048 entries): PASS"

# 4) sky130 clock gate (wrapper + our model of sky130_fd_sc_hd__sdlclkp_4), all warnings on.
"$VERILATOR" --lint-only -Wall --top-module ClockGate -f "$CLOCK_RTL/sky130_sim.f"
echo "lint ClockGate_sky130 (-Wall): PASS"

# 5) reef_top + core for sky130 (TCMs + clock gate swapped), from the PD Drop 0 filelist and the
#    OpenRAM fallback list. Synthesis filelist plus the models Verilator needs in place of .lib cells.
"$REPO_ROOT/scripts/gen_filelists.sh" --check
tmpf="$(mktemp)"; trap 'rm -f "$tmpf"' EXIT
for entry in "reef_top_sky130.f sram22/sram22_512x128m4w8.v" \
             "reef_top_sky130_openram.f sim/sky130_sram_2kbyte_1rw1r_32x512_8_cycle.v"; do
  read -r list model <<< "$entry"
  cat "$REPO_ROOT/configs/filelists/$list" > "$tmpf"
  echo "$SRAM_RTL/$model" >> "$tmpf"
  echo "$CLOCK_RTL/sim/sky130_fd_sc_hd__sdlclkp_4.sv" >> "$tmpf"
  # shellcheck disable=SC2086
  "$VERILATOR" --lint-only --top-module reef_top -f "$tmpf" \
    $DEFINES $WAIVE -Wno-PINCONNECTEMPTY -Wno-PINMISSING
  echo "lint reef_top + core, sky130 TCMs + clock gate (configs/filelists/$list, synth set): PASS"
done

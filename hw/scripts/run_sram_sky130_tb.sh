#!/usr/bin/env bash
# Unit test of the sky130 TCM wrappers, ITCM and DTCM shapes: tests/sram_sky130/tb_sram_sky130.sv
# on each wrapper + macro model below. Every run must print SRAM_TB: PASS.
#
#   sram22   rtl/sram/Sram_sram22.sv on SRAM22's own model of sram22_512x128m4w8 (the Drop 0 TCM)
#   openram  rtl/sram/Sram_sky130.sv on OpenRAM's model of sky130_sram_2kbyte_1rw1r_32x512_8 (its delays)
#   cycle    rtl/sram/Sram_sky130.sv on our Verilator cycle model of that macro (rtl/sram/sim/)
#
#   scripts/run_sram_sky130_tb.sh                     VCS, all three (openram needs SKY130_SRAM_MACROS)
#   MODELS="sram22 cycle" scripts/run_sram_sky130_tb.sh
#   SIM=verilator MODELS=sram22 scripts/run_sram_sky130_tb.sh   Verilator --timing (no license; CI)
#
#   SKY130_SRAM_MACROS=<pdk>/sky130A/libs.ref/sky130_sram_macros   (openram only)
#
# The macro models are Verilog-2005 files, so only .sv files are parsed as SystemVerilog.
# OpenRAM's model uses `mem` (in a $display) before declaring it, which VCS rejects; the script
# compiles a copy in build/ with only that declaration line moved up (diff saved next to it).
# SIM=vcs needs vcs on PATH (ECE servers); SIM=verilator needs Verilator >= 5.032 (VERILATOR=...).
# Output: build/sram_sky130_tb/<sim>/<model>/.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIM="${SIM:-vcs}"
MODELS="${MODELS:-sram22 openram cycle}"
OUT="${OUT:-$REPO_ROOT/build/sram_sky130_tb}/$SIM"
VERILATOR="${VERILATOR:-verilator}"
case "$SIM" in
  vcs) command -v vcs >/dev/null || { echo "vcs not on PATH" >&2; exit 1; } ;;
  verilator) command -v "$VERILATOR" >/dev/null || { echo "verilator not found (set VERILATOR)" >&2; exit 1; } ;;
  *) echo "SIM must be vcs or verilator" >&2; exit 2 ;;
esac
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"

status=0
for model in $MODELS; do
  case $model in
    sram22)
      wrapper="$REPO_ROOT/rtl/sram/Sram_sram22.sv"
      mfile="$REPO_ROOT/rtl/sram/sram22/sram22_512x128m4w8.v" ;;
    openram)
      [ "$SIM" = vcs ] || { echo "openram: VCS only (the OpenRAM model's delays)" >&2; exit 2; }
      MACROS="${SKY130_SRAM_MACROS:?set SKY130_SRAM_MACROS to sky130A/libs.ref/sky130_sram_macros}"
      wrapper="$REPO_ROOT/rtl/sram/Sram_sky130.sv"
      orig="$MACROS/verilog/sky130_sram_2kbyte_1rw1r_32x512_8.v"
      [ -f "$orig" ] || { echo "missing $orig" >&2; exit 1; }
      mkdir -p "$OUT/openram"
      mfile="$OUT/openram/sky130_sram_2kbyte_1rw1r_32x512_8.v"
      decl='^reg \[DATA_WIDTH-1:0\] +mem \[0:RAM_DEPTH-1\];'
      [ "$(grep -cE "$decl" "$orig")" = 1 ] || { echo "OpenRAM model: mem declaration not found" >&2; exit 1; }
      DECL="$decl" awk '$0 ~ ENVIRON["DECL"] {next} {print} /^  reg \[DATA_WIDTH-1:0\]  dout0;/ {print "  reg [DATA_WIDTH-1:0]    mem [0:RAM_DEPTH-1];"}' \
        "$orig" > "$mfile"
      diff "$orig" "$mfile" > "$OUT/openram/openram_model.diff" || true ;;
    cycle)
      wrapper="$REPO_ROOT/rtl/sram/Sram_sky130.sv"
      mfile="$REPO_ROOT/rtl/sram/sim/sky130_sram_2kbyte_1rw1r_32x512_8_cycle.v" ;;
    *) echo "unknown model: $model" >&2; exit 2 ;;
  esac
  [ -f "$mfile" ] || { echo "missing $mfile" >&2; exit 1; }
  d="$OUT/$model"; mkdir -p "$d"
  tb="$REPO_ROOT/tests/sram_sky130/tb_sram_sky130.sv"
  if [ "$SIM" = vcs ]; then
    (cd "$d" && vcs -full64 +systemverilogext+.sv -timescale=1ns/1ps +lint=TFIPC-L -top tb_sram_sky130 \
        "$tb" "$wrapper" "$mfile" -o simv > compile.log 2>&1) || { echo "$model: vcs compile failed, see $d/compile.log"; exit 1; }
  else
    "$VERILATOR" --binary --timing -j "${JOBS:-8}" --top-module tb_sram_sky130 -Mdir "$d/obj" -o ../simv \
        -Wno-fatal "$tb" "$wrapper" "$mfile" > "$d/compile.log" 2>&1 \
      || { tail -20 "$d/compile.log"; echo "$model: verilator build failed, see $d/compile.log"; exit 1; }
  fi
  # The OpenRAM model prints every access (VERBOSE=1); keep only the testbench lines.
  (cd "$d" && ./simv > sim.log 2>&1) || true
  echo "=== $SIM, macro model: $model"
  grep -E "rdata|rvalid|Error" "$d/sim.log" | grep -v "Reading\|Writing" | head -6 || true
  grep "SRAM_TB" "$d/sim.log" || echo "SRAM_TB: no result line (simulation did not finish), see $d/sim.log"
  grep -q "SRAM_TB: PASS" "$d/sim.log" || status=1
done
[ $status = 0 ] && echo "SRAM TB ($SIM: $MODELS): ALL PASS" || echo "SRAM TB ($SIM: $MODELS): FAILED"
exit $status

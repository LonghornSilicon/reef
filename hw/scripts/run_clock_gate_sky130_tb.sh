#!/usr/bin/env bash
# Test of the sky130 clock gate (rtl/clock/ClockGate_sky130.sv) against M3's ClockGate contract,
# with VCS (only .sv files are parsed as SystemVerilog: the PDK models are Verilog-2005).
# The same testbench runs on:
#   generic      the vendored rtl/vendor/coralnpu/ClockGate.sv (reference behavior)
#   pdk_func     ClockGate_sky130 + the PDK's sky130_fd_sc_hd__sdlclkp_4 model, FUNCTIONAL
#   pdk_timing   ClockGate_sky130 + the PDK model, timing variant (specify block, zero delays)
#   cycle        ClockGate_sky130 + our Verilator model (rtl/clock/sim/)
#   mutant       ClockGate_sky130 + a model with the latch removed: must FAIL
#
#   SKY130_FD_SC_HD=<pdk>/sky130A/libs.ref/sky130_fd_sc_hd scripts/run_clock_gate_sky130_tb.sh
#
# Needs vcs on PATH (ECE servers). Output: build/clock_gate_sky130_tb/.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CELLS="${SKY130_FD_SC_HD:?set SKY130_FD_SC_HD to sky130A/libs.ref/sky130_fd_sc_hd}"
OUT="${OUT:-$REPO_ROOT/build/clock_gate_sky130_tb}"
command -v vcs >/dev/null || { echo "vcs not on PATH" >&2; exit 1; }
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
for f in sky130_fd_sc_hd.v primitives.v; do
  [ -f "$CELLS/verilog/$f" ] || { echo "missing $CELLS/verilog/$f" >&2; exit 1; }
done

TB="$REPO_ROOT/tests/clock_gate_sky130/tb_clock_gate.sv"
WRAP="$REPO_ROOT/rtl/clock/ClockGate_sky130.sv"
MODEL="$REPO_ROOT/rtl/clock/sim/sky130_fd_sc_hd__sdlclkp_4.sv"
# VCS rejects the PDK's UDP port declarations under `default_nettype none (IND errors), so the
# UDPs are compiled from a copy with only that directive changed to wire (diff saved next to it).
sed 's/^`default_nettype none/`default_nettype wire/' "$CELLS/verilog/primitives.v" > "$OUT/primitives.v"
diff "$CELLS/verilog/primitives.v" "$OUT/primitives.v" > "$OUT/primitives.diff" || true
PDK=(-v "$CELLS/verilog/sky130_fd_sc_hd.v" -v "$OUT/primitives.v")
# Mutant: GCLK = (GATE | SCE) & CLK with no latch (glitches when GATE changes while CLK is high).
sed -e '/always_latch/,/end/d' -e 's/assign GCLK = en_latch & CLK;/assign GCLK = (GATE | SCE) \& CLK;/' \
  -e '/logic en_latch/d' "$MODEL" > "$OUT/sdlclkp_4_mutant.sv"
grep -q 'GATE | SCE) & CLK' "$OUT/sdlclkp_4_mutant.sv" || { echo "mutant not created" >&2; exit 1; }

status=0
for cfg in generic pdk_func pdk_timing cycle mutant; do
  case $cfg in
    generic)    src=("$REPO_ROOT/rtl/vendor/coralnpu/ClockGate.sv") ;;
    pdk_func)   src=("$WRAP" "${PDK[@]}" +define+FUNCTIONAL) ;;
    pdk_timing) src=("$WRAP" "${PDK[@]}") ;;
    cycle)      src=("$WRAP" "$MODEL") ;;
    mutant)     src=("$WRAP" "$OUT/sdlclkp_4_mutant.sv") ;;
  esac
  d="$OUT/$cfg"; mkdir -p "$d"
  (cd "$d" && vcs -full64 +systemverilogext+.sv -timescale=1ns/1ps +lint=TFIPC-L -top tb_clock_gate \
      "$TB" "${src[@]}" -o simv > compile.log 2>&1) || { echo "$cfg: vcs compile failed, see $d/compile.log"; exit 1; }
  (cd "$d" && ./simv > sim.log 2>&1) || true
  echo "=== $cfg"
  grep "CG_TB" "$d/sim.log" | head -6 || echo "CG_TB: no result line, see $d/sim.log"
  if [ $cfg = mutant ]; then
    grep -q "CG_TB: FAIL" "$d/sim.log" && echo "(mutant fails as it should)" || { echo "mutant was not caught"; status=1; }
  else
    grep -q "CG_TB: PASS" "$d/sim.log" || status=1
  fi
done
[ $status = 0 ] && echo "CLOCK GATE TB: ALL PASS" || echo "CLOCK GATE TB: FAILED"
exit $status

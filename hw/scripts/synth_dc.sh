#!/usr/bin/env bash
# Timing-driven synthesis of the pinned core (or reef_top) with Synopsys Design Compiler.
# For area/timing estimates against a real clock target; not a PD flow.
#
#   scripts/synth_dc.sh --db /path/lib.db [--period NS] [--top RvvCoreMiniAxi|reef_top]
#                       [--out DIR] [--cores N] [-UFOO] [--exclude-list FILE]
#                       [--sram-db sram22_512x128m4w8_<corner>.db | sky130_sram_2kbyte_1rw1r_32x512_8_<corner>.db]
#                       [--node sky130]
#
# - Reads synth.f in order as one analyze (compilation unit), synth defines incl. TB_SUPPORT.
# - Sram.v is replaced by rtl/reef/synth/Sram_blackbox.v and kept as dont_touch black boxes
#   (TCMs become macros in PD), so area is std cells only.
# - With --sram-db, Sram.v is replaced by the sky130 macro wrapper that matches the .db name, linked
#   from that .db: rtl/sram/Sram_sram22.sv (sram22_*, 5 SRAM22 macros, the Drop 0 TCM) or
#   rtl/sram/Sram_sky130.sv (sky130_sram_*, 20 OpenRAM macros, fallback). TCM paths are timed and
#   the area includes the macros. Extra reports: sram_from.rpt (macro dout -> core) and
#   sram_to.rpt (core -> macro inputs).
# - --node sky130 (needs --sram-db) also replaces ClockGate.sv by rtl/clock/ClockGate_sky130.sv
#   (sky130_fd_sc_hd__sdlclkp_4 from --db), i.e. the configs/filelists/reef_top_sky130.f build.
#   Extra report: icg.rpt (the ICG cells and the timing to their GATE pins).
# - edff.sv is replaced by rtl/reef/synth/edff_dc.sv (equivalent; works around a DC VER-37 crash).
# - Stops before compile if elaborate/link report any error or unresolved reference.
# - compile_ultra -no_autoungroup keeps hierarchy for per-block area.
# - Constraints: one clock on io_aclk (clk_i for reef_top), IO delays 20% of the period.
# Needs dc_shell on PATH and a .db (e.g. sky130 tt .lib compiled with lc_shell).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CORALNPU_RTL="$REPO_ROOT/rtl/vendor/coralnpu" REEF_RTL="$REPO_ROOT/rtl/reef"
SRAM_RTL="$REPO_ROOT/rtl/sram" CLOCK_RTL="$REPO_ROOT/rtl/clock"

db=""; sram_db=""; period=10; top=RvvCoreMiniAxi; out=""; cores=16; exclude_list=""; node=generic
undefs=()
while [ $# -gt 0 ]; do
  case "$1" in
    --db) db="$2"; shift 2 ;;
    --sram-db) sram_db="$2"; shift 2 ;;
    --node) node="$2"; shift 2 ;;
    --period) period="$2"; shift 2 ;;
    --top) top="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --cores) cores="$2"; shift 2 ;;
    --exclude-list) exclude_list="$2"; shift 2 ;;
    -U*) undefs+=("${1#-U}"); shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ -f "$db" ] || { echo "need --db <lib.db>" >&2; exit 2; }
[ -z "$sram_db" ] || [ -f "$sram_db" ] || { echo "no such --sram-db: $sram_db" >&2; exit 2; }
case "$node" in
  generic) ;;
  sky130) [ -n "$sram_db" ] || { echo "--node sky130 needs --sram-db" >&2; exit 2; } ;;
  *) echo "--node must be generic or sky130" >&2; exit 2 ;;
esac
[ -z "$sram_db" ] || sram_db="$(cd "$(dirname "$sram_db")" && pwd)/$(basename "$sram_db")"
sram_wrapper=""
case "$(basename "${sram_db:-none}")" in
  none) ;;
  sram22_*) sram_wrapper="$SRAM_RTL/Sram_sram22.sv" ;;
  sky130_sram_*) sram_wrapper="$SRAM_RTL/Sram_sky130.sv" ;;
  *) echo "--sram-db: expected sram22_*.db or sky130_sram_*.db" >&2; exit 2 ;;
esac
command -v dc_shell >/dev/null || { echo "dc_shell not on PATH" >&2; exit 1; }
out="${out:-$REPO_ROOT/build/dc/${top}_${period}ns}"
mkdir -p "$out"; out="$(cd "$out" && pwd)"

defs=()
for d in USE_GENERIC TB_SUPPORT VLEN_128 ZVE32F_ON SYNTHESIS; do
  [[ " ${undefs[*]:-} " == *" $d "* ]] || defs+=("$d")
done

# File list: synth.f order, Sram.v -> black box (or the sky130 macro wrapper), optional excludes, then reef.f for reef_top.
excl=""
[ -n "$exclude_list" ] && excl="$(grep -v '^#' "$exclude_list" | sed '/^$/d')"
files=()
while IFS= read -r line; do
  case "$line" in
    '${CORALNPU_RTL}/'*)
      f="${line#'${CORALNPU_RTL}/'}"
      if [ "$f" = Sram.v ]; then
        if [ -n "$sram_db" ]; then files+=("$sram_wrapper"); else files+=("$REEF_RTL/synth/Sram_blackbox.v"); fi
        continue
      fi
      if [ "$f" = edff.sv ]; then files+=("$REEF_RTL/synth/edff_dc.sv"); continue; fi
      if [ "$f" = ClockGate.sv ] && [ "$node" = sky130 ]; then files+=("$CLOCK_RTL/ClockGate_sky130.sv"); continue; fi
      if [ -n "$excl" ] && grep -qxF "$f" <<< "$excl"; then continue; fi
      files+=("$CORALNPU_RTL/$f") ;;
  esac
done < "$CORALNPU_RTL/synth.f"
if [ "$top" = reef_top ]; then
  while IFS= read -r line; do
    case "$line" in '${REEF_RTL}/'*) files+=("$REEF_RTL/${line#'${REEF_RTL}/'}") ;; esac
  done < "$REEF_RTL/reef.f"
  clk=clk_i
else
  clk=io_aclk
fi

cat > "$out/run.tcl" <<TCL
set_host_options -max_cores $cores
set_app_var search_path [list . $CORALNPU_RTL]
set_app_var target_library [list $db]
set_app_var link_library [list * $db $sram_db]
define_design_lib WORK -path $out/work
set_app_var hdlin_sverilog_std 2012
if {![analyze -format sverilog -define {${defs[*]}} [list ${files[*]}]]} { puts "REEF: analyze failed"; exit 1 }
redirect -variable elab_log { set elab_ok [elaborate $top] }
puts \$elab_log
if {!\$elab_ok || [regexp {Error:} \$elab_log]} { puts "REEF: elaborate failed"; exit 1 }
current_design $top
redirect -variable link_log { set link_ok [link] }
puts \$link_log
if {!\$link_ok || [regexp {Unable to resolve|Error:} \$link_log]} { puts "REEF: link failed"; exit 1 }
$( [ -z "$sram_db" ] && echo 'set_dont_touch [get_cells -hier -filter "ref_name=~Sram*"]' )

create_clock -name clk -period $period [get_ports $clk]
set_clock_uncertainty [expr {0.05 * $period}] [get_clocks clk]
set_input_delay  [expr {0.2 * $period}] -clock clk [remove_from_collection [all_inputs] [get_ports $clk]]
set_output_delay [expr {0.2 * $period}] -clock clk [all_outputs]
set_load 0.01 [all_outputs]

compile_ultra -no_autoungroup
report_qor > $out/qor.rpt
report_area > $out/area.rpt
report_area -hierarchy -nosplit > $out/area_hier.rpt
report_timing -max_paths 10 -nosplit > $out/timing.rpt
report_constraint -all_violators -nosplit > $out/violators.rpt
$( [ -n "$sram_db" ] && cat <<'SRAMRPT'
set sram_macros [get_cells -hier -filter "ref_name=~sky130_sram* || ref_name=~sram22_*"]
puts "REEF: [sizeof_collection $sram_macros] SRAM macros"
report_timing -through [get_pins -of_objects $sram_macros -filter "direction==out"] -max_paths 5 -nosplit > sram_from.rpt
report_timing -to [get_pins -of_objects $sram_macros -filter "direction==in"] -max_paths 5 -nosplit > sram_to.rpt
report_reference -hierarchy -nosplit > references.rpt
SRAMRPT
)
$( [ "$node" = sky130 ] && cat <<'ICGRPT'
set icgs [get_cells -hier -filter "ref_name=~sky130_fd_sc_hd__sdlclkp*"]
puts "REEF: [sizeof_collection $icgs] ICG cells"
redirect icg.rpt { report_cell $icgs; report_timing -to [get_pins -of_objects $icgs -filter "name==GATE"] -max_paths 5 -nosplit }
ICGRPT
)
write -format verilog -hierarchy -output $out/netlist.v
exit
TCL

echo "dc: top=$top period=${period}ns defines=${defs[*]} files=${#files[@]} sram=${sram_db:-blackbox} wrapper=${sram_wrapper:-none} node=$node out=$out"
start=$(date +%s)
(cd "$out" && dc_shell -f run.tcl > dc.log 2>&1) || { echo "dc_shell failed, see $out/dc.log"; exit 1; }
echo "dc: done in $(( $(date +%s) - start ))s"
grep -E "Total cell area|Design Area" "$out/area.rpt" "$out/qor.rpt" | head -3
grep -E "slack \(" "$out/timing.rpt" | head -1

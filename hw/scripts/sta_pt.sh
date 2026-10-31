#!/usr/bin/env bash
# PrimeTime setup check of a synthesized netlist at one library corner.
# Use it to re-time a netlist that DC built at one corner (e.g. tt) against another (e.g. ss).
#
#   scripts/sta_pt.sh --netlist build/dc/<run>/netlist.v --db /path/lib.db [--period NS]
#                     [--top RvvCoreMiniAxi|reef_top] [--out DIR] [--extra-db FILE]...
#
# - Same constraints as synth_dc.sh: one clock, 5% uncertainty, IO delays 20% of the period.
# - Wire loads come from the library default (sky130: "Small", mode top), as in DC. No parasitics,
#   no CTS: ideal clock. Hold is not reported (meaningless before CTS).
# - Functional mode: the test-enable pin (io_te / test_en_i) is held at 0 with set_case_analysis.
#   PT checks async-reset recovery (DC does not by default), and io_te reaches the reset pins.
# - Sram black boxes stay unconstrained, so TCM access paths are NOT checked here. A netlist from
#   synth_dc.sh --sram-db has the sky130 macros (SRAM22 or OpenRAM) instead: pass their .db with --extra-db; the summary
#   then adds the worst slack from and to the macros (sram_from.rpt, sram_to.rpt).
# Needs pt_shell on PATH.
set -euo pipefail
netlist=""; db=""; extra_dbs=(); period=10; top=RvvCoreMiniAxi; out=""
while [ $# -gt 0 ]; do
  case "$1" in
    --netlist) netlist="$2"; shift 2 ;;
    --db) db="$2"; shift 2 ;;
    --extra-db) [ -f "$2" ] || { echo "no such --extra-db: $2" >&2; exit 2; }
                extra_dbs+=("$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"); shift 2 ;;
    --period) period="$2"; shift 2 ;;
    --top) top="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ -f "$netlist" ] || { echo "need --netlist <netlist.v>" >&2; exit 2; }
[ -f "$db" ] || { echo "need --db <lib.db>" >&2; exit 2; }
command -v pt_shell >/dev/null || { echo "pt_shell not on PATH" >&2; exit 1; }
netlist="$(cd "$(dirname "$netlist")" && pwd)/$(basename "$netlist")"
db="$(cd "$(dirname "$db")" && pwd)/$(basename "$db")"
corner="$(basename "$db" .db)"
out="${out:-$(dirname "$netlist")/pt_${corner}_${period}ns}"
mkdir -p "$out"; out="$(cd "$out" && pwd)"
if [ "$top" = reef_top ]; then clk=clk_i; te=test_en_i; else clk=io_aclk; te=io_te; fi

cat > "$out/pt.tcl" <<TCL
set_app_var link_path [list * $db ${extra_dbs[*]:-}]
read_verilog $netlist
current_design $top
link_design
create_clock -name clk -period $period [get_ports $clk]
set_clock_uncertainty [expr {0.05 * $period}] [get_clocks clk]
set_input_delay  [expr {0.2 * $period}] -clock clk [remove_from_collection [all_inputs] [get_ports $clk]]
set_output_delay [expr {0.2 * $period}] -clock clk [all_outputs]
set_load 0.01 [all_outputs]
set_case_analysis 0 [get_ports $te]
update_timing
set r2r [get_timing_paths -from [all_registers -clock_pins] -to [all_registers -data_pins] -max_paths 1]
set all [get_timing_paths -max_paths 1]
set viol [get_timing_paths -max_paths 1000000 -nworst 1 -slack_lesser_than 0]
set f [open $out/summary.txt w]
puts \$f "corner $corner period $period"
puts \$f "reg2reg_slack [get_attribute \$r2r slack]"
puts \$f "reg2reg_arrival [get_attribute \$r2r arrival]"
puts \$f "all_paths_slack [get_attribute \$all slack]"
puts \$f "violating_endpoints [sizeof_collection \$viol]"
close \$f
report_timing -from [all_registers -clock_pins] -to [all_registers -data_pins] -max_paths 20 -nosplit -path_type summary > $out/reg2reg_top20.rpt
report_timing -from [all_registers -clock_pins] -to [all_registers -data_pins] -max_paths 1 -nosplit > $out/reg2reg_worst.rpt
report_timing -max_paths 20 -nosplit -path_type summary > $out/all_top20.rpt
report_global_timing > $out/global_timing.rpt
# sky130 SRAM macros (netlists from synth_dc.sh --sram-db). SRAM22 launches read data on the rising
# edge; OpenRAM on the falling edge, so its macro -> core paths get half a period.
set sram [get_cells -hier -quiet -filter "ref_name=~sky130_sram* || ref_name=~sram22_*"]
if {[sizeof_collection \$sram] > 0} {
  set from [get_timing_paths -through [get_pins -of_objects \$sram -filter "direction==out"] -max_paths 1]
  set to [get_timing_paths -to [get_pins -of_objects \$sram -filter "direction==in"] -max_paths 1]
  set f [open $out/summary.txt a]
  puts \$f "sram_macros [sizeof_collection \$sram]"
  puts \$f "sram_from_slack [get_attribute \$from slack]"
  puts \$f "sram_to_slack [get_attribute \$to slack]"
  close \$f
  report_timing -through [get_pins -of_objects \$sram -filter "direction==out"] -max_paths 5 -slack_lesser_than 1000 -nosplit > $out/sram_from.rpt
  report_timing -to [get_pins -of_objects \$sram -filter "direction==in"] -max_paths 5 -slack_lesser_than 1000 -nosplit > $out/sram_to.rpt
}
exit
TCL

echo "pt: top=$top corner=$corner period=${period}ns out=$out"
(cd "$out" && pt_shell -f pt.tcl > pt.log 2>&1) || { echo "pt_shell failed, see $out/pt.log"; exit 1; }
cat "$out/summary.txt"

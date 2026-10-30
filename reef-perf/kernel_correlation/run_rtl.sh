#!/usr/bin/env bash
# Runs each kernel ELF in elfs/ on the M3 RTL. Writes the region of
# interest (ROI) of each ELF to rtl_m3.csv, in the format of
# microbenchmarks/rtl_m3.csv. Needs only bash and python3.
#
# Run it on a machine that has the CoralNPU simulator, built at tag
# M3-2026-04-27 (commit 72d700ef) with -c opt:
#
#   bazel build -c opt //tests/verilator_sim:rvv_core_mini_verification_axi_sim
#
# Usage:
#
#   bash reef-perf/kernel_correlation/run_rtl.sh [SIM]
#
# SIM is the simulator binary. The default is the binary under
# ~/coralnpu/bazel-bin. The ELFs are for the M3 as built (8 KB ITCM,
# 32 KB DTCM). Build them with inference-engine's
# `CORAL_MAP=m3 coral/tools/build.sh bench`.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
sim="${1:-$HOME/coralnpu/bazel-bin/tests/verilator_sim/rvv_core_mini_verification_axi_sim}"
if [[ ! -x "$sim" ]]; then
    echo "no simulator at $sim" >&2
    exit 1
fi

logs="$(mktemp -d)"
trap 'rm -rf "$logs"' EXIT
csv="$here/rtl_m3.csv"
# Replace rtl_m3.csv only once every kernel has run.
new="$logs/rtl_m3.csv"
echo "kernel,rtl_roi_cycles,rtl_retired_insts,notes" > "$new"
for elf in "$here"/elfs/*.elf; do
    name="$(basename "${elf%.elf}")"
    echo "== $name"
    "$sim" --binary "$elf" --backdoor_load --instr_trace > "$logs/$name.log"
    # rtl_trace.py needs only the Python standard library.
    python3 - "$here/../python/reef_perf/experiments/correlation" \
        "$logs/$name.log" "$name" >> "$new" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
from rtl_trace import measure_log
roi = measure_log(sys.argv[2])
note = "" if roi.reached_mpause else "no mpause"
print(f"{sys.argv[3]},{roi.cycles},{roi.retired_insts},{note}")
EOF
    tail -n 1 "$new"
done
cp "$new" "$csv"
echo "wrote $csv"

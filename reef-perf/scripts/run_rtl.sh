#!/usr/bin/env bash
# Runs ELFs on the M3 RTL and prints one rtl_m3.csv row per ELF (name, ROI
# cycles, ROI instructions, notes), for microbenchmarks/rtl_m3.csv. Needs
# only bash and python3, plus the CoralNPU simulator built at tag
# M3-2026-04-27 (commit 72d700ef) with -c opt:
#
#   bazel build -c opt //tests/verilator_sim:rvv_core_mini_verification_axi_sim
#
# Usage:
#
#   bash reef-perf/scripts/run_rtl.sh SIM ELF...
#
# Build the microbenchmark ELFs first (they land in build/microbenchmarks/):
#
#   uv run python -m reef_perf.experiments.correlation.main
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
sim="$1"
shift
if [[ ! -x "$sim" ]]; then
    echo "no simulator at $sim" >&2
    exit 1
fi

logs="$(mktemp -d)"
trap 'rm -rf "$logs"' EXIT
for elf in "$@"; do
    name="$(basename "${elf%.elf}")"
    "$sim" --binary "$elf" --backdoor_load --instr_trace > "$logs/$name.log"
    # rtl_trace.py needs only the Python standard library.
    python3 - "$here/../python/reef_perf/experiments/correlation" \
        "$logs/$name.log" "$name" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from rtl_trace import measure_log
roi = measure_log(sys.argv[2])
note = "" if roi.reached_mpause else "no mpause"
print(f"{sys.argv[3]},{roi.cycles},{roi.retired_insts},{note}")
PY
done

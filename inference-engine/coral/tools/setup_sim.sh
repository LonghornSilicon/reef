#!/usr/bin/env bash
# Builds coral_sim (coral/sim/coral_sim.cpp), the functional runner CI uses
# in place of reef_perf:
#
#   coral/tools/setup_sim.sh PREFIX     ->  PREFIX/bin/coral_sim
#
# Builds Spike into PREFIX with reef-perf's tools/build_deps.sh unless it is
# already there; that needs the reef-perf/ext/spike submodule checked out.
# Then compiles coral_sim against Spike and reef-perf's SpikeDriver. Linux
# or macOS with a C++20 compiler and make.
set -euo pipefail

engine="$(cd "$(dirname "$0")/../.." && pwd)"
repo="$(cd "$engine/.." && pwd)"
mkdir -p "${1:?usage: coral/tools/setup_sim.sh PREFIX}"
prefix="$(cd "$1" && pwd)"
jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

if [[ ! -f "$prefix/lib/libriscv.a" ]]; then
    bash "$repo/reef-perf/tools/build_deps.sh" "$prefix" "$jobs" spike
fi

mkdir -p "$prefix/bin"
"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Werror \
    -I"$repo/reef-perf/src/csrc/include" -isystem "$prefix/include" \
    "$engine/coral/sim/coral_sim.cpp" \
    "$repo/reef-perf/src/csrc/src/spike_driver.cpp" \
    "$prefix"/lib/lib{riscv,disasm,fesvr,fdt,softfloat}.a -ldl -lpthread \
    -o "$prefix/bin/coral_sim"
echo "built $prefix/bin/coral_sim"

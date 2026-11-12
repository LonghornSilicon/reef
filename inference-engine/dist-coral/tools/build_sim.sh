#!/usr/bin/env bash
# Builds multi_sim (dist-coral/sim/multi_sim.cpp), the host driver that runs
# N chips on Spike with no timing model:
#
#   dist-coral/tools/build_sim.sh PREFIX     ->  PREFIX/bin/multi_sim
#
# PREFIX must already hold Spike, as coral/tools/setup_sim.sh PREFIX leaves
# it. Compiles against reef-perf's FuncSim, the reference engine,
# dist-reference/ and coral/host/'s packer.
set -euo pipefail

engine="$(cd "$(dirname "$0")/../.." && pwd)"
repo="$(cd "$engine/.." && pwd)"
prefix="$(cd "${1:?usage: dist-coral/tools/build_sim.sh PREFIX}" && pwd)"
if [[ ! -f "$prefix/lib/libriscv.a" ]]; then
    echo "no Spike in $prefix; run coral/tools/setup_sim.sh $prefix first" >&2
    exit 1
fi

model="$repo/reef-perf/model"
mkdir -p "$prefix/bin"
"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Werror -ffp-contract=off \
    -I"$model" -isystem "$prefix/include" \
    -I"$engine/reference/cpp/include" -I"$engine/dist-reference/include" \
    -I"$engine/coral/include" -I"$engine/coral/host" \
    -I"$engine/dist-coral/include" -I"$engine/dist-coral/host" \
    "$engine/dist-coral/sim/multi_sim.cpp" \
    "$model/frontend/FuncSim.cpp" "$model/frontend/InstDecode.cpp" \
    "$model/common/Inst.cpp" \
    "$prefix"/lib/lib{riscv,disasm,fesvr,fdt,softfloat}.a -ldl -lpthread \
    -o "$prefix/bin/multi_sim"
echo "built $prefix/bin/multi_sim"

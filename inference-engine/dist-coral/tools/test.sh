#!/usr/bin/env bash
# Runs the Megatron firmware on 1, 2 and 4 simulated chips, in fp32 and
# W8A8, and checks every token's logits (multi_sim.cpp):
#   test A: bit for bit against dist_inference::TpGptNeo,
#   test C: within tolerance of one chip.
#
#   dist-coral/tools/test.sh PREFIX
#
# PREFIX holds Spike (coral/tools/setup_sim.sh PREFIX). Cross-compiles the
# firmware with coral/tools/build.sh dist. Exits non-zero if any run fails.
# Linux only: FuncSim needs <elf.h>, so on macOS run it in a container, e.g.
#   docker run --rm -v "$PWD/..":/reef -w /reef/inference-engine ubuntu:24.04
# with build-essential and device-tree-compiler installed, after building
# the firmware on the host (coral/tools/build.sh dist).
set -euo pipefail

engine="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$engine"
prefix="${1:?usage: dist-coral/tools/test.sh PREFIX}"

coral/tools/build.sh dist
dist-coral/tools/build_sim.sh "$prefix"

failed=0
for precision in fp32 w8a8; do
    for ranks in 1 2 4; do
        if ! "$prefix/bin/multi_sim" --elf build/coral/dist/tp_rank.elf \
            --tp "$ranks" --precision "$precision"; then
            failed=1
        fi
    done
done
exit "$failed"

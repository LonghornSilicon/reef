#!/usr/bin/env bash
# Measures assembly programs on the M3 RTL simulator and prints
# "<name> <ROI cycles>" for each.
#
# Programs are assembled in the reef Docker container (against the
# microbenchmarks' common.inc and link.ld), copied to the CoralNPU container
# and run there, 6 at a time. ROI cycles are s11 - s10 from the
# instruction trace, as in roadmap B01.
#
# Run on the host:
#   bash reef-perf/tools/correlation/measure_on_rtl.sh \
#       microbenchmarks/vector_units/vdiv_m8_indep.S ...
# Paths are relative to reef-perf/. Settings (environment):
#   REEF_CONTAINER      the reef container (default orhan-reef-rebase); the
#                       checkout is mounted at /app/reef
#   CORALNPU_CONTAINER  the CoralNPU container (default coralnpu-container)
#   RTL_SIM             the simulator there (default ~/simbins/sim_O2, a
#                       `-c opt` build of rvv_core_mini_verification_axi_sim)
#   RTL_SIM_FLAGS       extra flags, e.g. --trace for a waveform (one program)
set -euo pipefail

REEF=${REEF_CONTAINER:-orhan-reef-rebase}
CORAL=${CORALNPU_CONTAINER:-coralnpu-container}
SIM=${RTL_SIM:-'~/simbins/sim_O2'}
FLAGS=${RTL_SIM_FLAGS:-}
WORK=/tmp/measure_on_rtl
LOCAL=$(mktemp -d)
trap 'rm -rf "$LOCAL"' EXIT

NAMES=""
for SOURCE in "$@"; do
  NAME=$(basename "$SOURCE" .S)
  NAMES="$NAMES $NAME"
  docker exec "$REEF" bash -c "M=/app/reef/reef-perf/microbenchmarks; \
    mkdir -p $WORK && cd $WORK \
    && riscv64-linux-gnu-as -march=rv32imf_zicsr_zifencei_zve32f -mabi=ilp32f \
       -I \$M /app/reef/reef-perf/$SOURCE -o $NAME.o \
    && riscv64-linux-gnu-ld -m elf32lriscv -T \$M/link.ld $NAME.o -o $NAME.elf"
  docker cp -q "$REEF:$WORK/$NAME.elf" "$LOCAL/"
done
docker exec "$CORAL" mkdir -p "$WORK"
for NAME in $NAMES; do
  docker cp -q "$LOCAL/$NAME.elf" "$CORAL:$WORK/"
done
docker exec "$CORAL" bash -c "cd $WORK; echo $NAMES | tr ' ' '\n' | \
  xargs -P 6 -I{} sh -c '$SIM --binary {}.elf --backdoor_load --instr_trace \
    $FLAGS > {}.log 2>&1'; \
  for n in $NAMES; do \
    s10=\$(grep -m1 ',0x1a,' \$n.log | cut -d, -f4); \
    s11=\$(grep -m1 ',0x1b,' \$n.log | cut -d, -f4); \
    echo \$n \$((s11 - s10)); \
  done"

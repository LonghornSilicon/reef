#!/usr/bin/env bash
# Assembles every workloads/*.S into workloads/build/<name>.elf.
#
# Uses the Ubuntu cross binutils (riscv64-linux-gnu-as/ld, 2.38+), which
# understand RVV 1.0 and emit 32-bit ELFs with -march=rv32...
# Override the tools with AS=... LD=... if you use a different toolchain
# (for example the M3 toolchain built from coralnpu/toolchain).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
AS="${AS:-riscv64-linux-gnu-as}"
LD="${LD:-riscv64-linux-gnu-ld}"
MARCH="${MARCH:-rv32imf_zicsr_zifencei_zve32x}"
OUT="${HERE}/build"
mkdir -p "${OUT}"

for src in "${HERE}"/*.S; do
  name="$(basename "${src}" .S)"
  "${AS}" -march="${MARCH}" -mabi=ilp32f -I "${HERE}" "${src}" -o "${OUT}/${name}.o"
  "${LD}" -m elf32lriscv -T "${HERE}/link.ld" "${OUT}/${name}.o" -o "${OUT}/${name}.elf"
  echo "built ${OUT}/${name}.elf"
done

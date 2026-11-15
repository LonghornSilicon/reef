#!/usr/bin/env bash
# Builds one operator kernel into an ELF.
#   coral/ops/tools/build.sh <kernel> <out_dir> [compiler flags, e.g. -DH=56]
# <kernel> is coral/ops/kernels/<kernel>.cc. Run from inference-engine/.
# Writes <out_dir>/<kernel>.elf and prints its .text and DTCM (.bss+.data) use.
# Fails if code does not fit the 8 KB ITCM or the DTCM leaves < 1 KB of stack.
set -euo pipefail

engine="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$engine"
kernel="$1"; out="$2"; shift 2

cxx="${CORAL_TOOLCHAIN:+$CORAL_TOOLCHAIN/bin/}riscv64-unknown-elf-g++"
if ! command -v "$cxx" >/dev/null 2>&1; then
    exec /usr/local/bin/docker run --rm --platform linux/amd64 \
        -v coral-gcc:/opt/coral:ro -v "$engine":/engine -w /engine \
        -e CORAL_TOOLCHAIN=/opt/coral/toolchain_coralnpu_v2 \
        ubuntu:24.04 bash coral/ops/tools/build.sh "$kernel" "$out" "$@"
fi
bin="${CORAL_TOOLCHAIN:+$CORAL_TOOLCHAIN/bin/}"

arch=(-march=rv32im_zicsr_zifencei_zbb_zve32x -mabi=ilp32 -mcmodel=medany)
cflags=("${arch[@]}" -std=c++17 -O2 -fno-exceptions -fno-rtti
        -fno-asynchronous-unwind-tables -ffunction-sections -fdata-sections
        -fno-tree-loop-distribute-patterns -Wall -Wextra -Werror
        -Icoral/ops/include "$@")
ldflags=("${arch[@]}" -nostdlib -nostartfiles -Tcoral/runtime/link.ld -Wl,--gc-sections)

mkdir -p "$out/obj"
"$cxx" "${arch[@]}" -c coral/runtime/crt0.S -o "$out/obj/crt0.o"
"$cxx" "${cflags[@]}" -c coral/runtime/libc.cc -o "$out/obj/libc.o"
"$cxx" "${cflags[@]}" -c "coral/ops/kernels/$kernel.cc" -o "$out/obj/$kernel.o"
"$cxx" "${ldflags[@]}" "$out/obj/crt0.o" "$out/obj/libc.o" "$out/obj/$kernel.o" \
    -o "$out/$kernel.elf"
"${bin}riscv64-unknown-elf-size" -A "$out/$kernel.elf" | awk -v k="$kernel" '
    $1==".text"{t=$2} $1==".bss"||$1==".data"||$1==".rodata"||$1==".sdata"||$1==".sbss"{d+=$2}
    END{printf "%s: text %d B (ITCM 8192), DTCM data %d B (of 32768)\n", k, t, d}'

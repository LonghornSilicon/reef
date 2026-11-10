#!/usr/bin/env bash
# Builds the end-to-end firmware for the M3 as built (8 KB ITCM, 32 KB DTCM)
# into build/e2e/e2e.elf and prints its section sizes. Fails if the code does
# not fit the ITCM or the DTCM leaves under 1 KB of stack.
#
#   coral/tools/build_e2e.sh [-DE2E_PROMPT=32 -DE2E_NEW_TOKENS=8 ...]
#
# Uses Coral's GCC from $CORAL_TOOLCHAIN/bin or PATH; otherwise reruns itself
# in a container with the toolchain volume $CORAL_VOLUME (default
# coral-toolchain, filled by coral/tools/setup_toolchain.sh).
set -euo pipefail

engine="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$engine"

cxx="${CORAL_TOOLCHAIN:+$CORAL_TOOLCHAIN/bin/}riscv64-unknown-elf-g++"
if ! command -v "$cxx" >/dev/null; then
    exec docker run --rm --platform linux/amd64 \
        -v "${CORAL_VOLUME:-coral-toolchain}":/opt/coral:ro \
        -v "$engine":/engine -w /engine \
        -e CORAL_TOOLCHAIN=/opt/coral/toolchain_coralnpu_v2 \
        -e E2E_OUT="${E2E_OUT:-}" \
        ubuntu:24.04 bash coral/tools/build_e2e.sh "$@"
fi
size="${CORAL_TOOLCHAIN:+$CORAL_TOOLCHAIN/bin/}riscv64-unknown-elf-size"

out="build/e2e"
elf="$out/${E2E_OUT:-e2e}.elf"
mkdir -p "$out"
arch=(-march=rv32imf_zicsr_zifencei_zbb_zve32f -mabi=ilp32 -mcmodel=medany)
cflags=("${arch[@]}" -std=c++17 -Os -fno-exceptions -fno-rtti
        -fno-asynchronous-unwind-tables -ffunction-sections -fdata-sections
        -fno-tree-loop-distribute-patterns -Wall -Wextra -Werror "$@")

"$cxx" "${arch[@]}" -c coral/runtime/crt0.S -o "$out/crt0.o"
"$cxx" "${cflags[@]}" -c coral/runtime/libc.cc -o "$out/libc.o"
"$cxx" "${cflags[@]}" -c coral/kernels/e2e/e2e.cc -o "$out/e2e.o"
"$cxx" "${arch[@]}" -nostdlib -nostartfiles -Tcoral/runtime/link_e2e.ld \
    -Wl,--gc-sections "$out/crt0.o" "$out/libc.o" "$out/e2e.o" -o "$elf"
"$size" -A "$elf" | awk 'NR > 2 && $2 > 0'

#!/usr/bin/env bash
# Builds coral/kernels/e2e for the M3 as built (8 KB ITCM, 32 KB DTCM) into
# coral/bin/:
#   e2e.elf        the whole model (main.cc)
#   <kernel>.elf   one kernel on one 8-token block (standalone.cc)
# Extra arguments go to the compiler, e.g. -DE2E_PROMPT=16 -DE2E_NEW_TOKENS=2.
# Fails if code does not fit the ITCM or the DTCM leaves under 1 KB of stack.
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
        ubuntu:24.04 bash coral/tools/build_e2e.sh "$@"
fi
size="${CORAL_TOOLCHAIN:+$CORAL_TOOLCHAIN/bin/}riscv64-unknown-elf-size"

src=coral/kernels/e2e
obj=build/e2e
bin=coral/bin
mkdir -p "$obj" "$bin"
arch=(-march=rv32imf_zicsr_zifencei_zbb_zve32f -mabi=ilp32 -mcmodel=medany)
cflags=("${arch[@]}" -std=c++17 -Os -fno-exceptions -fno-rtti
        -fno-asynchronous-unwind-tables -ffunction-sections -fdata-sections
        -fno-tree-loop-distribute-patterns -Wall -Wextra -Werror "$@")
ldflags=("${arch[@]}" -nostdlib -nostartfiles -Tcoral/runtime/link_e2e.ld
         -Wl,--gc-sections)

"$cxx" "${arch[@]}" -c coral/runtime/crt0.S -o "$obj/crt0.o"
"$cxx" "${cflags[@]}" -c coral/runtime/libc.cc -o "$obj/libc.o"
lib=("$obj/crt0.o" "$obj/libc.o")
for kernel in embed layernorm linear attention lm_head; do
    "$cxx" "${cflags[@]}" -c "$src/$kernel.cc" -o "$obj/$kernel.o"
    lib+=("$obj/$kernel.o")
done

"$cxx" "${cflags[@]}" -c "$src/main.cc" -o "$obj/main.o"
"$cxx" "${ldflags[@]}" "${lib[@]}" "$obj/main.o" -o "$bin/e2e.elf"
for kernel in embed layernorm qkv attention attn_out fc1 fc2 lm_head; do
    define="KERNEL_$(echo "$kernel" | tr '[:lower:]' '[:upper:]')"
    "$cxx" "${cflags[@]}" "-D$define" -c "$src/standalone.cc" \
        -o "$obj/standalone_$kernel.o"
    "$cxx" "${ldflags[@]}" "${lib[@]}" "$obj/standalone_$kernel.o" \
        -o "$bin/$kernel.elf"
done
for elf in "$bin"/*.elf; do
    printf '%-16s text %6s B\n' "$(basename "$elf")" \
        "$("$size" -A "$elf" | awk '$1 == ".text" {print $2}')"
done

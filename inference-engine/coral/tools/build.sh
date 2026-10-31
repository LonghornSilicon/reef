#!/usr/bin/env bash
# Cross-compiles coral/ for the Coral M3 core into build/coral/:
#   tests/<name>.elf     golden tests (needs build/coral/golden; see test.sh)
#   bench/<name>.elf     reef-perf benchmarks
#   firmware/<name>.elf  the firmware
#   resident.elf         the firmware with every kernel kept: on the chip
#                        they all share the 8 KiB ITCM, so the build fails
#                        if this image's code and constants outgrow it
#
#   coral/tools/build.sh [tests|bench|firmware]...    (default: all three)
#
# Uses Coral's GCC (toolchain_coralnpu_v2) from $CORAL_TOOLCHAIN/bin, or
# from PATH. Without it, reruns itself in a linux/amd64 container that
# mounts the coral-toolchain Docker volume (see setup_toolchain.sh).
set -euo pipefail

engine="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$engine"

cxx="${CORAL_TOOLCHAIN:+$CORAL_TOOLCHAIN/bin/}riscv64-unknown-elf-g++"
if ! command -v "$cxx" >/dev/null; then
    exec docker run --rm --platform linux/amd64 \
        -v coral-toolchain:/opt/coral:ro -v "$engine":/engine -w /engine \
        -e CORAL_TOOLCHAIN=/opt/coral/toolchain_coralnpu_v2 \
        ubuntu:24.04 bash coral/tools/build.sh "$@"
fi

out=build/coral
# The ISA reef_perf simulates and the M3 implements; Coral's own builds add
# extensions the model does not run.
arch=(-march=rv32imf_zicsr_zifencei_zbb_zve32f -mabi=ilp32 -mcmodel=medany)
# -ffp-contract=off: every float operation rounds on its own, as in the
# host reference. No exceptions, RTTI or C library: these run bare-metal.
cflags=("${arch[@]}" -std=c++17 -O3 -ffp-contract=off -fno-math-errno
        -fno-exceptions -fno-rtti -fno-asynchronous-unwind-tables
        -ffunction-sections -fdata-sections -Wall -Wextra -Werror
        -Icoral/include -Icoral/kernels -Icoral/tests -Icoral/firmware
        -I"$out")
ldflags=("${arch[@]}" -nostdlib -nostartfiles -Tcoral/runtime/link.ld
         -Wl,--gc-sections)

mkdir -p "$out/obj"
"$cxx" "${arch[@]}" -c coral/runtime/crt0.S -o "$out/obj/crt0.o"
"$cxx" "${cflags[@]}" -fno-tree-loop-distribute-patterns \
    -c coral/runtime/libc.cc -o "$out/obj/libc.o"
objects=("$out/obj/crt0.o" "$out/obj/libc.o")
for source in coral/kernels/*.cc coral/firmware/gpt_neo.cc; do
    [[ -f "$source" ]] || continue
    object="$out/obj/$(basename "${source%.cc}").o"
    "$cxx" "${cflags[@]}" -c "$source" -o "$object"
    objects+=("$object")
done

# link SOURCE ELF: compile one program and link it with the library.
link() {
    mkdir -p "$(dirname "$2")"
    "$cxx" "${cflags[@]}" -c "$1" -o "${2%.elf}.o"
    "$cxx" "${ldflags[@]}" "${objects[@]}" "${2%.elf}.o" -lgcc -o "$2"
    echo "built $2"
}

groups=("$@")
[[ ${#groups[@]} -gt 0 ]] || groups=(tests bench firmware)
for group in "${groups[@]}"; do
    case "$group" in
    tests)
        for source in coral/tests/test_*.cc; do
            name="$(basename "${source%.cc}")"
            link "$source" "$out/tests/$name.elf"
        done
        ;;
    bench)
        for source in coral/bench/*.cc; do
            name="$(basename "${source%.cc}")"
            link "$source" "$out/bench/$name.elf"
        done
        ;;
    firmware)
        if [[ -f coral/firmware/main.cc ]]; then
            link coral/firmware/main.cc "$out/firmware/gpt_neo.elf"
        fi
        ;;
    *)
        echo "unknown group: $group" >&2
        exit 2
        ;;
    esac
done

# The chip's ITCM (CoralNPU's default map, 0x0/8 KiB) must hold every kernel
# at once. Link the firmware with all of them kept, and fail when its .text
# plus .rodata, which CoralNPU's own linker script also places in ITCM,
# exceeds that.
itcm_bytes=8192
nm="${cxx%g++}nm"
kernels=()
for object in "${objects[@]}"; do
    while read -r _ kind symbol; do
        if [[ "$kind" == T && "$symbol" == _ZN12reef_kernels* ]]; then
            kernels+=("-Wl,--undefined=$symbol")
        fi
    done < <("$nm" "$object")
done
"$cxx" "${cflags[@]}" -c coral/firmware/main.cc -o "$out/obj/main.o"
"$cxx" "${ldflags[@]}" "${kernels[@]}" "${objects[@]}" "$out/obj/main.o" \
    -lgcc -o "$out/resident.elf"
resident="$("${cxx%g++}size" -A "$out/resident.elf" |
    awk '$1 == ".text" || $1 == ".rodata" { sum += $2 } END { print sum }')"
echo "resident image: $resident of $itcm_bytes ITCM bytes"
if ((resident > itcm_bytes)); then
    echo "every kernel together no longer fits in the ITCM" >&2
    exit 1
fi

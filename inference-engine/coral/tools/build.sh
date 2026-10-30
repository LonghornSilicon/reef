#!/usr/bin/env bash
# Cross-compiles coral/ for the Coral M3 core into build/coral/:
#   tests/<name>.elf     golden tests (needs build/coral/golden; see test.sh)
#   bench/<name>.elf     reef-perf benchmarks
#   firmware/<name>.elf  the firmware
#
#   coral/tools/build.sh [tests|bench|firmware]...    (default: all three)
#
# CORAL_MAP picks the memory map: highmem (default; 1 MB ITCM and DTCM,
# runtime/link.ld, into build/coral/) or m3 (the M3 as built: 8 KB ITCM and
# 32 KB DTCM, runtime/link_m3.ld, into build/coral_m3/). A program that does
# not fit is reported and skipped, and the script exits non-zero at the end.
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
        -e CORAL_MAP="${CORAL_MAP:-highmem}" \
        ubuntu:24.04 bash coral/tools/build.sh "$@"
fi

case "${CORAL_MAP:-highmem}" in
highmem) out=build/coral; ldscript=coral/runtime/link.ld ;;
m3) out=build/coral_m3; ldscript=coral/runtime/link_m3.ld ;;
*)
    echo "unknown CORAL_MAP: $CORAL_MAP" >&2
    exit 1
    ;;
esac
# The ISA reef_perf simulates and the M3 implements; Coral's own builds add
# extensions the model does not run.
arch=(-march=rv32imf_zicsr_zifencei_zbb_zve32f -mabi=ilp32 -mcmodel=medany)
# -ffp-contract=off: every float operation rounds on its own, as in the
# host reference. No exceptions, RTTI or C library: these run bare-metal.
cflags=("${arch[@]}" -std=c++17 -O3 -ffp-contract=off -fno-math-errno
        -fno-exceptions -fno-rtti -fno-asynchronous-unwind-tables
        -ffunction-sections -fdata-sections -Wall -Wextra -Werror
        -Icoral/include -Icoral/tests -Icoral/firmware -Ibuild/coral)
ldflags=("${arch[@]}" -nostdlib -nostartfiles -T"$ldscript"
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

failed=0

# link SOURCE ELF: compile one program and link it with the library.
link() {
    mkdir -p "$(dirname "$2")"
    "$cxx" "${cflags[@]}" -c "$1" -o "${2%.elf}.o"
    if ! "$cxx" "${ldflags[@]}" "${objects[@]}" "${2%.elf}.o" -lgcc -o "$2"; then
        echo "failed to link $2" >&2
        failed=1
        return 0
    fi
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

exit "$failed"

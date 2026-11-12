#!/usr/bin/env bash
# Cross-compiles coral/ for the Coral M3 core into build/coral/:
#   tests/<name>.elf     golden tests (needs build/coral/golden; see test.sh)
#   bench/<name>.elf     reef-perf benchmarks
#   firmware/<name>.elf  the firmware
#   dist/tp_rank.elf     one rank of tensor-parallel GPT-Neo (dist-coral/)
#   lib/                 for firmware built outside this repo: crt0.o,
#                        libcoral_kernels.a (kernels and libc), and flags.sh,
#                        the compile and link flags as bash arrays
#                        (coral_arch, coral_cflags, coral_ldflags) without
#                        include paths or the linker script
#
#   coral/tools/build.sh [tests|bench|firmware|dist|lib]...
#                        (default: all but lib)
#
# Uses Coral's GCC (toolchain_coralnpu_v2) from $CORAL_TOOLCHAIN/bin, or
# from PATH. Without it, reruns itself in a linux/amd64 container that
# mounts the coral-toolchain Docker volume (see setup_toolchain.sh).
set -euo pipefail

engine="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$engine"

cxx="${CORAL_TOOLCHAIN:+$CORAL_TOOLCHAIN/bin/}riscv64-unknown-elf-g++"
ar="${CORAL_TOOLCHAIN:+$CORAL_TOOLCHAIN/bin/}riscv64-unknown-elf-ar"
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
codegen=("${arch[@]}" -std=c++17 -O3 -ffp-contract=off -fno-math-errno
         -fno-exceptions -fno-rtti -fno-asynchronous-unwind-tables
         -ffunction-sections -fdata-sections -Wall -Wextra -Werror)
cflags=("${codegen[@]}" -Icoral/include -Icoral/tests -Icoral/firmware
        -Idist-coral/include -I"$out")
linking=("${arch[@]}" -nostdlib -nostartfiles -Wl,--gc-sections)
ldflags=("${linking[@]}" -Tcoral/runtime/link.ld)

mkdir -p "$out/obj"
"$cxx" "${arch[@]}" -c coral/runtime/crt0.S -o "$out/obj/crt0.o"
"$cxx" "${cflags[@]}" -fno-tree-loop-distribute-patterns \
    -c coral/runtime/libc.cc -o "$out/obj/libc.o"
objects=("$out/obj/crt0.o" "$out/obj/libc.o")
library=("$out/obj/libc.o")
for source in coral/kernels/*.cc coral/firmware/gpt_neo.cc; do
    [[ -f "$source" ]] || continue
    object="$out/obj/$(basename "${source%.cc}").o"
    "$cxx" "${cflags[@]}" -c "$source" -o "$object"
    objects+=("$object")
    [[ "$source" == coral/kernels/* ]] && library+=("$object")
done

# array NAME VALUE...: one bash array assignment, safely quoted.
array() {
    printf '%s=(' "$1"
    shift
    printf ' %q' "$@"
    printf ' )\n'
}

# link SOURCE ELF: compile one program and link it with the library.
link() {
    mkdir -p "$(dirname "$2")"
    "$cxx" "${cflags[@]}" -c "$1" -o "${2%.elf}.o"
    "$cxx" "${ldflags[@]}" "${objects[@]}" "${2%.elf}.o" -lgcc -o "$2"
    echo "built $2"
}

groups=("$@")
[[ ${#groups[@]} -gt 0 ]] || groups=(tests bench firmware dist)
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
    dist)
        "$cxx" "${cflags[@]}" -c dist-coral/firmware/tp_rank.cc \
            -o "$out/obj/tp_rank.o"
        objects+=("$out/obj/tp_rank.o")
        link dist-coral/firmware/tp_main.cc "$out/dist/tp_rank.elf"
        ;;
    lib)
        mkdir -p "$out/lib"
        cp "$out/obj/crt0.o" "$out/lib/crt0.o"
        rm -f "$out/lib/libcoral_kernels.a"
        "$ar" rcs "$out/lib/libcoral_kernels.a" "${library[@]}"
        {
            echo "# coral/tools/build.sh lib: the flags coral's kernels were"
            echo "# built with. Add -I.../coral/include and"
            echo "# -T.../coral/runtime/link.ld; link crt0.o first."
            array coral_arch "${arch[@]}"
            array coral_cflags "${codegen[@]}"
            array coral_ldflags "${linking[@]}"
        } >"$out/lib/flags.sh"
        echo "built $out/lib"
        ;;
    *)
        echo "unknown group: $group" >&2
        exit 2
        ;;
    esac
done

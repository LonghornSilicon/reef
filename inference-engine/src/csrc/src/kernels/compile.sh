#!/usr/bin/env bash
# Compiles one kernel for reef_perf: ./compile.sh linear  ->  linear.elf
# Uses Coral M3's riscv32-unknown-elf-g++ from PATH, or else the copy in the
# reef-coralnpu Docker volume.
set -e
cd "$(dirname "$0")"
root="$(git rev-parse --show-toplevel)"
here="${PWD#"$root"/}"
flags=(-march=rv32imf_zve32f_zicsr -mabi=ilp32 -O2 -nostartfiles
       -T reef-perf/src/reef_perf/experiments/run/workloads/link.ld
       "$here/$1.cc" -o "$here/$1.elf" "${@:2}")
if command -v riscv32-unknown-elf-g++ >/dev/null; then
    cd "$root" && riscv32-unknown-elf-g++ "${flags[@]}"
else
    docker run --rm --platform linux/amd64 -v reef-coralnpu:/work -v "$root":/src -w /src \
        coralnpu-reef /work/m3tc/toolchain_kelvin_v2/bin/riscv32-unknown-elf-g++ "${flags[@]}"
fi

#!/usr/bin/env bash
# Builds every kernel ELF the matrix-engine experiments run
# (docs/matrix-experiments.md), from inference-engine/coral at PR #57.
#
#   build_kernels.sh SRC [OUT]
#
# SRC is a directory holding coral/ from inference-engine at commit 83f7152
# (branch e2e-npu-firmware), for example:
#   git -C <reef clone> archive 83f7152 inference-engine/coral \
#       | tar -x -C /tmp/pr57
#   build_kernels.sh /tmp/pr57/inference-engine
# OUT defaults to build/e2e under reef-perf/. Coral's GCC must be in
# $CORAL_TOOLCHAIN/bin (coral/tools/setup_toolchain.sh DIR installs it).
#
# Writes one subdirectory per build (the names the sweep configs use):
#   t8     defaults: 8 tokens at position 0
#   t8p8   -DE2E_START_POS=8          (second prefill block)
#   t4     -DE2E_TOKENS=4
#   t2     -DE2E_TOKENS=2
#   t1     -DE2E_TOKENS=1 -DE2E_START_POS=16   (decode token 17)
#   t1p79  -DE2E_TOKENS=1 -DE2E_START_POS=79   (decode token 80)
#   epi_t8, epi_t1  as t8 and t1 with e6_epilogue.patch (mcycle reads
#                   around each epilogue in linear.cc)
#   b16    a 16-token block (-DE2E_BLOCK_TOKENS=16 -DE2E_TOKENS=16): the
#          whole model and standalone fc1 and qkv; link logs only, since a
#          build that does not fit the DTCM fails
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
module="$(cd "$here/../../../../.." && pwd)"
src="$(cd "${1:?usage: build_kernels.sh SRC [OUT]}" && pwd)"
out="$(mkdir -p "${2:-$module/build/e2e}" && cd "${2:-$module/build/e2e}" && pwd)"
: "${CORAL_TOOLCHAIN:?set CORAL_TOOLCHAIN to the Coral GCC directory}"
[[ -x "$src/coral/tools/build.sh" ]] || {
    echo "error: $src has no coral/tools/build.sh" >&2
    exit 2
}

build() { # name, then build.sh flags
    local name=$1
    shift
    (cd "$src" && E2E_OUT="$out/$name" bash coral/tools/build.sh "$@") \
        >"$out/$name.log" 2>&1
    echo "built $name"
}

build t8
build t8p8 -DE2E_START_POS=8
build t4 -DE2E_TOKENS=4
build t2 -DE2E_TOKENS=2
build t1 -DE2E_TOKENS=1 -DE2E_START_POS=16
build t1p79 -DE2E_TOKENS=1 -DE2E_START_POS=79

# The epilogue builds compile a patched copy, so SRC stays untouched.
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cp -r "$src/coral" "$work/coral"
patch -p1 -d "$work" --quiet <"$here/e6_epilogue.patch"
src_epi="$work"
(cd "$src_epi" && E2E_OUT="$out/epi_t8" bash coral/tools/build.sh) \
    >"$out/epi_t8.log" 2>&1
echo "built epi_t8"
(cd "$src_epi" && E2E_OUT="$out/epi_t1" bash coral/tools/build.sh \
    -DE2E_TOKENS=1 -DE2E_START_POS=16) >"$out/epi_t1.log" 2>&1
echo "built epi_t1"

# 16-token block (E2.7). build.sh stops at the whole model, so the
# standalone kernels are linked here with build.sh's flags.
b16="$out/b16"
mkdir -p "$b16/obj"
flags=(-DE2E_BLOCK_TOKENS=16 -DE2E_TOKENS=16)
if (cd "$src" && E2E_OUT="$b16" bash coral/tools/build.sh "${flags[@]}") \
    >"$b16/e2e_build.log" 2>&1; then
    echo "b16: whole model linked"
else
    echo "b16: whole model did not link (b16/e2e_build.log)"
fi
cxx="$CORAL_TOOLCHAIN/bin/riscv64-unknown-elf-g++"
arch=(-march=rv32im_zicsr_zifencei_zbb_zve32x -mabi=ilp32 -mcmodel=medany)
cflags=("${arch[@]}" -std=c++17 -Os -fno-exceptions -fno-rtti
    -fno-asynchronous-unwind-tables -ffunction-sections -fdata-sections
    -fno-tree-loop-distribute-patterns -Wall -Wextra -Werror "${flags[@]}")
ldflags=("${arch[@]}" -nostdlib -nostartfiles -T"$src/coral/runtime/link.ld"
    -Wl,--gc-sections)
"$cxx" "${arch[@]}" -c "$src/coral/runtime/crt0.S" -o "$b16/obj/crt0.o"
"$cxx" "${cflags[@]}" -c "$src/coral/runtime/libc.cc" -o "$b16/obj/libc.o"
lib=("$b16/obj/crt0.o" "$b16/obj/libc.o")
for kernel in embed layernorm linear attention lm_head; do
    "$cxx" "${cflags[@]}" -c "$src/coral/kernels/$kernel.cc" \
        -o "$b16/obj/$kernel.o"
    lib+=("$b16/obj/$kernel.o")
done
for kernel in fc1 qkv; do
    define="KERNEL_$(echo "$kernel" | tr '[:lower:]' '[:upper:]')"
    "$cxx" "${cflags[@]}" "-D$define" -c "$src/coral/kernels/standalone.cc" \
        -o "$b16/obj/standalone_$kernel.o"
    if "$cxx" "${ldflags[@]}" "${lib[@]}" "$b16/obj/standalone_$kernel.o" \
        -o "$b16/$kernel.elf" >"$b16/${kernel}_link.log" 2>&1; then
        echo "b16: standalone $kernel linked"
    else
        echo "b16: standalone $kernel did not link (b16/${kernel}_link.log)"
    fi
done

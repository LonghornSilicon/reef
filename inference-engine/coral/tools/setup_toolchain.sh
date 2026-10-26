#!/usr/bin/env bash
# Installs Coral's GCC (toolchain_coralnpu_v2), the archive CoralNPU's own
# Bazel build pins in rules/repo_defs.bzl. The binaries are x86-64 Linux.
#
#   coral/tools/setup_toolchain.sh        Docker volume coral-toolchain, which
#                                         build.sh mounts (macOS, or any host
#                                         without the compiler)
#   coral/tools/setup_toolchain.sh DIR    GCC alone, stripped, into DIR (x86-64
#                                         Linux, as CI does); then build with
#                                         CORAL_TOOLCHAIN=DIR
#
# Checks the archive's sha256 either way and does nothing if already set up.
set -euo pipefail

url=https://storage.googleapis.com/shodan-public-artifacts/toolchain_coralnpu_v2-2026-06-29.tar.xz
sha256=de06690c2da5cd783d76b2998208bd4db4dcdc22dec146c7b0a5ee1af40d3db7

if [[ $# -eq 1 ]]; then
    mkdir -p "$1"
    dir="$(cd "$1" && pwd)"
    if [[ -x "$dir/bin/riscv64-unknown-elf-g++" ]]; then
        echo "$dir already holds Coral's GCC"
        exit 0
    fi
    work="$(mktemp -d)"
    trap 'rm -rf "$work"' EXIT
    curl -fsSL -o "$work/toolchain.tar.xz" "$url"
    echo "$sha256  $work/toolchain.tar.xz" | sha256sum -c -
    # The archive also holds clang, gdb and LTO; GCC needs only these. Keep
    # every bin/riscv64-unknown-elf-* tool: the archive stores g++ and
    # riscv64-unknown-elf/bin/* as hard links to them.
    top=toolchain_coralnpu_v2
    tar -xJf "$work/toolchain.tar.xz" -C "$dir" --strip-components=1 \
        --wildcards \
        --exclude="$top/libexec/gcc/*/lto1" \
        --exclude="$top/bin/*-gdb*" --exclude="$top/bin/*-lto-dump" \
        --exclude="$top/bin/*-run" \
        "$top/bin/riscv64-unknown-elf-*" "$top/libexec/gcc" \
        "$top/lib/gcc" "$top/riscv64-unknown-elf"
    # The host binaries ship with debug info, 700 MB each for cc1/cc1plus.
    find "$dir/bin" "$dir/libexec" "$dir/riscv64-unknown-elf/bin" \
        -type f -perm -u+x -exec strip --strip-unneeded {} + 2>/dev/null || true
    echo "installed Coral's GCC in $dir ($(du -sh "$dir" | cut -f1))"
    exit 0
fi

# Docker volume. Reuse the archive from a CoralNPU Bazel cache volume
# (coralnpu-cache) when there is one.
cached=/cache/bazel/_bazel_builder/cache/repos/v1/content_addressable/sha256/$sha256/file
mounts=(-v coral-toolchain:/toolchain)
if docker volume inspect coralnpu-cache >/dev/null 2>&1; then
    mounts+=(-v coralnpu-cache:/cache:ro)
fi
docker volume create coral-toolchain >/dev/null
docker run --rm "${mounts[@]}" alpine sh -euc "
    if [ -x /toolchain/toolchain_coralnpu_v2/bin/riscv64-unknown-elf-g++ ]; then
        echo 'coral-toolchain is already set up'
        exit 0
    fi
    archive=$cached
    if [ ! -f \"\$archive\" ]; then
        archive=/tmp/toolchain.tar.xz
        wget -q -O \"\$archive\" $url
    fi
    echo \"$sha256  \$archive\" | sha256sum -c
    tar -xJf \"\$archive\" -C /toolchain
    echo 'coral-toolchain is ready'
"

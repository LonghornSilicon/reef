#!/usr/bin/env bash
# Checks the C++ code against the LLVM coding style without modifying files:
# scripts/check_style.sh (names, header guards, file banners), clang-format,
# the build (warnings are errors, see CMakeLists.txt), and clang-tidy (naming
# and llvm-* checks), one file per build job.
#
#   scripts/check_cpp.sh [BUILD_DIR]     (default: build, shared with pytest)
#
# Run inside the Docker image (../tools/docker.sh shell).
set -euo pipefail

cd "$(dirname "$0")/.."
build_dir="${1:-build}"

sources=(model/*.cpp model/*/*.cpp tools/*/*.cpp tests/cpp/*/*.cpp)
headers=(model/*.h model/*/*.h)

bash scripts/check_style.sh

uv run clang-format --dry-run --Werror "${sources[@]}" "${headers[@]}"

cmake -S . -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$build_dir"

# Each clang-tidy process parses Sparta and Spike (~1 GB), so it shares the
# build's job limit. The "N warnings generated" lines count warnings in
# third-party headers, which HeaderFilterRegex already hides.
clang_tidy="$(uv run which clang-tidy)"
printf '%s\n' "${sources[@]}" |
    xargs -P "${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}" -n 1 \
        "$clang_tidy" --quiet -p "$build_dir" 2>&1 |
    { grep -v '^[0-9]* warnings\? generated\.$' || true; }

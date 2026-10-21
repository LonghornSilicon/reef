#!/usr/bin/env bash
# Checks the parts of the LLVM coding style that clang-format and clang-tidy
# cannot check here:
#   - file names: UpperCamelCase .h/.cpp named after their class (tool mains
#     like main.cpp and tools/*/*.cpp excepted); tests end in Test.cpp;
#   - header guards: REEF_PERF_<DIR>_<FILE>_H, from the path under model/;
#   - file headers: an 80-column "//===- File - Description ---===//" banner,
#     with "-*- C++ -*-" in headers;
#   - doc comments use ///, not /** */; no #pragma once.
# Prints every problem and exits non-zero if there is any.
set -uo pipefail

cd "$(dirname "$0")/.."

status=0
fail() {
    echo "$1: $2"
    status=1
}

shopt -s nullglob
headers=(model/*.h model/*/*.h)
sources=(model/*.cpp model/*/*.cpp)
tests=(tests/cpp/*/*.cpp)
tools=(tools/*/*.cpp)

for f in model/*.hpp model/*/*.hpp; do
    fail "$f" "use .h for headers"
done

for f in "${headers[@]}" "${sources[@]}"; do
    base=$(basename "$f")
    [[ "$base" == main.cpp ]] && continue
    [[ "$base" =~ ^[A-Z][A-Za-z0-9]*\.(h|cpp)$ ]] ||
        fail "$f" "file name should be UpperCamelCase, e.g. ResourcePool.cpp"
done

for f in "${tests[@]}"; do
    [[ "$(basename "$f")" =~ ^[A-Z][A-Za-z0-9]*Test\.cpp$ ]] ||
        fail "$f" "test file name should be <Name>Test.cpp"
done

for f in "${headers[@]}"; do
    rel=${f#model/}
    guard="REEF_PERF_$(echo "${rel%.h}" | tr '[:lower:]/' '[:upper:]_')_H"
    grep -qx "#ifndef $guard" "$f" || fail "$f" "missing '#ifndef $guard'"
    grep -qx "#define $guard" "$f" || fail "$f" "missing '#define $guard'"
    [[ "$(grep -v '^$' "$f" | tail -n 1)" == "#endif // $guard" ]] ||
        fail "$f" "should end with '#endif // $guard'"
done

for f in "${headers[@]}" "${sources[@]}" "${tests[@]}" "${tools[@]}"; do
    first=$(head -n 1 "$f")
    base=$(basename "$f")
    if [[ ${#first} -ne 80 || "$first" != "//===- $base - "*"===//" ]]; then
        fail "$f" "first line should be an 80-column '//===- $base - ... ===//' banner"
    elif [[ "$f" == *.h && "$first" != *"-*- C++ -*-===//" ]]; then
        fail "$f" "header banner should end in '-*- C++ -*-===//'"
    fi
    if grep -n '/\*\*' "$f" >/dev/null; then
        fail "$f" "use /// doc comments, not /** */"
    fi
    if grep -n '^#pragma once' "$f" >/dev/null; then
        fail "$f" "use an #ifndef header guard, not #pragma once"
    fi
done

exit $status

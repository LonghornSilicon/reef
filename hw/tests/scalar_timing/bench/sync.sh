#!/usr/bin/env bash
# Copy the canonical bench sources into the build tree (new files only live in
# tests/cocotb/scalar_timing/; nothing else in the tree is touched).
set -euo pipefail
cd "$(dirname "$0")"
DST="${CORALNPU:?set CORALNPU to a CoralNPU checkout at tag M3-2026-04-27}/tests/cocotb/scalar_timing"
mkdir -p "$DST"
shopt -s nullglob
for f in BUILD *.cc *.h *.S *.py; do
  cp -v "$f" "$DST/$f"
done

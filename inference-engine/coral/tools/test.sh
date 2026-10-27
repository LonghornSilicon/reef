#!/usr/bin/env bash
# Checks every Coral kernel, and the firmware's whole forward pass, against
# the host reference bit for bit:
#   1. builds gen_golden on the host and writes build/coral/golden/,
#   2. cross-compiles coral/tests/ (build.sh),
#   3. runs each test on reef_perf (run.sh).
# Exits non-zero if any output differs from the reference in a single bit.
set -euo pipefail

engine="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$engine"

cmake -S . -B build/coral-host -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build/coral-host --target gen_golden >/dev/null
build/coral-host/bin/gen_golden build/coral

coral/tools/build.sh tests
coral/tools/run.sh build/coral/tests/*.elf

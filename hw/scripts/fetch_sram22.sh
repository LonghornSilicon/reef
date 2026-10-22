#!/usr/bin/env bash
# Fetch the SRAM22 sky130 macro views that are not in the repo (Liberty, LEF, optionally GDS) at the
# pinned commit (rtl/sram/sram22/UPSTREAM.md) and check their sha256. The Verilog model is vendored in
# rtl/sram/sram22/ and checked here too.
#
#   scripts/fetch_sram22.sh [--gds] [OUT_DIR]     default OUT_DIR: build/sram22
#
# Output: OUT_DIR/sram22_512x128m4w8{_ss_100C_1v60,_tt_025C_1v80,_ff_n40C_1v95}.lib, .lef (+ .gds.gz).
# Already-downloaded files with the right hash are kept.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMIT=de0a9a7cdabf941c807a5ab35864f80d97fccfea
MACRO=sram22_512x128m4w8
URL="https://raw.githubusercontent.com/ucb-substrate/sram22_sky130_macros/$COMMIT/$MACRO"

gds=0
[ "${1:-}" = --gds ] && { gds=1; shift; }
OUT="${1:-$REPO_ROOT/build/sram22}"
mkdir -p "$OUT"

files=(
  "68652bd0f6176239cff8b5550db3715d043a2eaff466d0c781830ce421401f54  ${MACRO}_ss_100C_1v60.lib"
  "f694737128f096a11fb8651ddd79e401716b6cce14aaec7e4fbc138e92c44beb  ${MACRO}_tt_025C_1v80.lib"
  "2fc7fd451c7a46d220cc883ca5bdeda46d655a237a8f12d3ce51b9cb6f949f03  ${MACRO}_ff_n40C_1v95.lib"
  "2d59a7eb3c3b16fbccd1c31fe7841c412023224621e63a3f75fa542c605bc291  ${MACRO}.lef"
)
[ $gds = 1 ] && files+=("fd5c6b82cc305360b0ce20d856fad4e5024c14e2fbf90c76c6f0584a26cc30ba  ${MACRO}.gds.gz")

for entry in "${files[@]}"; do
  sum="${entry%%  *}"; name="${entry##*  }"
  if ! echo "$sum  $OUT/$name" | sha256sum -c --quiet >/dev/null 2>&1; then
    curl -fsSL --retry 3 -o "$OUT/$name.part" "$URL/$name"
    mv "$OUT/$name.part" "$OUT/$name"
    echo "$sum  $OUT/$name" | sha256sum -c --quiet || { echo "sha256 mismatch: $name" >&2; exit 1; }
  fi
done
echo "f6ad924d2ce42c71617b02ac2931cd3dabbb8db09a7fe27acf2807cac28002db  $REPO_ROOT/rtl/sram/sram22/$MACRO.v" \
  | sha256sum -c --quiet || { echo "vendored $MACRO.v does not match the pinned commit" >&2; exit 1; }
echo "SRAM22 $MACRO @ ${COMMIT:0:7}: ${#files[@]} file(s) in ${OUT#$REPO_ROOT/}"

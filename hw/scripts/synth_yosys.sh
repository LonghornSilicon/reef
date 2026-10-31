#!/usr/bin/env bash
# Quick synthesis of the vendored CoralNPU RTL for area estimates (not a PD flow).
#
#   scripts/synth_yosys.sh [--lib nangate45|sky130hd|/path/to.lib] [--out DIR] [-DFOO ...] [-UFOO ...]
#
# -UFOO drops FOO from the default define set (e.g. -UZVE32F_ON: no vector FP32).
# --exclude-list FILE drops the listed vendored files from synth.f
#   (no vector FP: -UZVE32F_ON --exclude-list configs/synth_exclude_nozve32f.txt).
# --period-ps N makes abc map for an N ps delay target (timing-driven; default: area only).
# --top reef_top synthesizes the test-chip top (core + rtl/reef/reef.f) instead of the core.
# --node sky130 reads configs/filelists/reef_top_sky130.f (PD Drop 0): TCMs from the SRAM22 macro
#   wrapper and the clock gate from sky130_fd_sc_hd__sdlclkp_4, linked as liberty cells (the macro
#   .lib: SKY130_SRAM_LIB, default the tt .lib from scripts/fetch_sram22.sh, fetched on first use).
#   TCM=openram uses the OpenRAM fallback list instead (macro .lib default
#   $LIB_DIR/sky130_sram_2kbyte_1rw1r_32x512_8_TT_1p8V_25C.lib, from the open PDK).
#   The macros are listed in the report but their area is not in the std-cell totals.
#
# Yosys + yosys-slang (Yosys's own parser can't read the RVV backend / fpnew SV).
# Uses the synth define set from docs/baseline-m3.md (TB_SUPPORT included on purpose).
#
# Notes:
# - synth.f must be read as ONE compilation unit (--single-unit): fpnew uses macros from
#   registers.svh without `include-ing it.
# - Sram is blackboxed (generic node): the TCMs become macros in PD, so their area is NOT in the totals.
#   ITCM = Sram 512x128 (8 KB), DTCM = Sram 2048x128 (32 KB).
# - Hierarchy is kept, and yosys-slang uniquifies every module by instance path, so
#   scripts/area_report.py can roll area up by block.
# - Needs OSS CAD Suite (yosys with the slang plugin). Set OSS_CAD_SUITE or put yosys on PATH.
#   The full design takes ~1 GB to elaborate; run it on an ECE server, not a laptop.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CORALNPU_RTL="$REPO_ROOT/rtl/vendor/coralnpu" REEF_RTL="$REPO_ROOT/rtl/reef" REEF_ROOT="$REPO_ROOT"
LIB_DIR="${LIB_DIR:-$REPO_ROOT/build/liberty}"
# Open liberty files from OpenROAD-flow-scripts (pinned commit), fetched on first use.
ORFS=https://raw.githubusercontent.com/The-OpenROAD-Project/OpenROAD-flow-scripts/5a76f84edc02baf49b75ccbd1a3a9e6a458a8d90/flow/platforms
[ -n "${OSS_CAD_SUITE:-}" ] && export PATH="$OSS_CAD_SUITE/bin:$PATH"

lib=nangate45
out=""
extra_defines=()
undefs=()
exclude_list=""
period_ps=""
top=RvvCoreMiniAxi
node=generic
while [ $# -gt 0 ]; do
  case "$1" in
    --lib) lib="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --exclude-list) exclude_list="$2"; shift 2 ;;
    --period-ps) period_ps="$2"; shift 2 ;;
    --top) top="$2"; shift 2 ;;
    --node) node="$2"; shift 2 ;;
    -D*)   extra_defines+=("$1"); shift ;;
    -U*)   undefs+=("${1#-U}"); shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

case "$lib" in
  nangate45) LIB="$LIB_DIR/NangateOpenCellLibrary_typical.lib"; NAND2=NAND2_X1
             URL="$ORFS/nangate45/lib/NangateOpenCellLibrary_typical.lib" ;;
  sky130hd)  LIB="$LIB_DIR/sky130_fd_sc_hd__tt_025C_1v80.lib"; NAND2=sky130_fd_sc_hd__nand2_1
             URL="$ORFS/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib" ;;
  *)         LIB="$lib"; NAND2="${NAND2:?set NAND2=<cell> for a custom liberty}"; lib="$(basename "$lib" .lib)" ;;
esac
if [ ! -f "$LIB" ] && [ -n "${URL:-}" ]; then
  mkdir -p "$LIB_DIR" && curl -fsSL -o "$LIB" "$URL"
fi
[ -f "$LIB" ] || { echo "liberty not found: $LIB (set LIB_DIR)" >&2; exit 1; }
command -v yosys >/dev/null || { echo "yosys not on PATH (set OSS_CAD_SUITE)" >&2; exit 1; }

out="${out:-$REPO_ROOT/build/synth/$lib}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"

case "$node" in
  generic) filelist="$CORALNPU_RTL/synth.f"; core_prefix='${CORALNPU_RTL}/'
           read_libs=""; blackbox="--blackboxed-module Sram" ;;
  sky130)  core_prefix='${REEF_ROOT}/rtl/vendor/coralnpu/'
           case "${TCM:-sram22}" in
             sram22)  filelist="$REPO_ROOT/configs/filelists/reef_top_sky130.f"
                      if [ -z "${SKY130_SRAM_LIB:-}" ]; then
                        "$REPO_ROOT/scripts/fetch_sram22.sh" >/dev/null
                        SKY130_SRAM_LIB="$REPO_ROOT/build/sram22/sram22_512x128m4w8_tt_025C_1v80.lib"
                      fi ;;
             openram) filelist="$REPO_ROOT/configs/filelists/reef_top_sky130_openram.f"
                      SKY130_SRAM_LIB="${SKY130_SRAM_LIB:-$LIB_DIR/sky130_sram_2kbyte_1rw1r_32x512_8_TT_1p8V_25C.lib}" ;;
             *) echo "TCM must be sram22 or openram" >&2; exit 2 ;;
           esac
           SRAM_LIB="$SKY130_SRAM_LIB"
           [ -f "$SRAM_LIB" ] || { echo "SRAM macro liberty not found: $SRAM_LIB (set SKY130_SRAM_LIB)" >&2; exit 1; }
           read_libs="read_liberty -lib $LIB
read_liberty -lib $SRAM_LIB"; blackbox="" ;;
  *) echo "--node must be generic or sky130" >&2; exit 2 ;;
esac
if [ -n "$exclude_list" ]; then
  src_filelist="$filelist"; filelist="$out/synth.f"
  grep -v '^#' "$exclude_list" | sed '/^$/d' | sed "s|^|$core_prefix|" > "$out/exclude.txt"
  grep -vxF -f "$out/exclude.txt" "$src_filelist" > "$filelist"
  echo "synth: excluded $(( $(wc -l < "$src_filelist") - $(wc -l < "$filelist") )) files"
fi

DEFINES=""
for d in USE_GENERIC TB_SUPPORT VLEN_128 ZVE32F_ON SYNTHESIS; do
  [[ " ${undefs[*]:-} " == *" $d "* ]] || DEFINES+="-D$d "
done
DEFINES+="${extra_defines[*]:-}"

reef_f=""; [ "$top" = reef_top ] && [ "$node" = generic ] && reef_f="-f $REEF_RTL/reef.f"

cat > "$out/synth.ys" <<EOF
$read_libs
read_slang --single-unit -f $filelist $reef_f $DEFINES --top $top \\
  --ignore-assertions --keep-hierarchy $blackbox
hierarchy -check -top $top
synth -top $top
dfflibmap -liberty $LIB
abc ${period_ps:+-D $period_ps} -liberty $LIB
opt_clean -purge
tee -o $out/stat.txt stat -liberty $LIB
tee -o $out/stat.json stat -json -liberty $LIB
EOF

echo "synth: lib=$lib node=$node${TCM:+ TCM=$TCM} out=$out defines: $DEFINES${period_ps:+ abc -D $period_ps}"
start=$(date +%s)
/usr/bin/time -v yosys -m slang -q -l "$out/yosys.log" -s "$out/synth.ys" 2> "$out/time.txt"
echo "synth: done in $(( $(date +%s) - start ))s, peak RSS $(awk '/Maximum resident/{printf "%.1f GB", $NF/1048576}' "$out/time.txt")"

python3 "$REPO_ROOT/scripts/area_report.py" "$out/stat.json" --liberty "$LIB" --nand2 "$NAND2" --top "$top" \
  | tee "$out/area_by_block.md"

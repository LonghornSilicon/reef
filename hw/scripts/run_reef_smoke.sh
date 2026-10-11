#!/usr/bin/env bash
# Build and run the reef_top smoke test (Verilator, pins-only host over SPI).
#
#   scripts/run_reef_smoke.sh
#
# Runs: the pass program at f_clk/f_sck = 16 (the spec minimum) and 32, and the bus-error
# program (off-chip load -> DECERR -> trap -> fault_o) at 16.
#
# Needs: Verilator >= 5.032 and riscv-none-elf-gcc (GCC >= 14 for the RVV intrinsics) on PATH,
# or VERILATOR / RISCV_PREFIX set. The core model build takes ~10 min and several GB; run it on
# an ECE server. Output: build/reef_smoke/.
#
# NODE=sky130 builds the PD Drop 0 configuration from configs/filelists/reef_top_sky130_sim.f: TCMs
# from SRAM22 macros (rtl/sram/Sram_sram22.sv + SRAM22's macro model) and the clock gate from
# sky130_fd_sc_hd__sdlclkp_4 (rtl/clock, our model of the cell), instead of upstream's generic
# Sram.v and ClockGate.sv. Output: build/reef_smoke_sky130/.
# NODE=sky130 TCM=openram uses the OpenRAM fallback TCM (reef_top_sky130_openram_sim.f) instead.
# Output: build/reef_smoke_sky130_openram/. SRAM=sky130 is an old name for NODE=sky130.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CORALNPU_RTL="$REPO_ROOT/rtl/vendor/coralnpu" REEF_RTL="$REPO_ROOT/rtl/reef"
export SRAM_RTL="$REPO_ROOT/rtl/sram" CLOCK_RTL="$REPO_ROOT/rtl/clock" REEF_ROOT="$REPO_ROOT"
SRC="$REPO_ROOT/tests/reef_smoke"
NODE="${NODE:-${SRAM:-generic}}"
case "$NODE" in
  generic) OUT="${OUT:-$REPO_ROOT/build/reef_smoke}"
           rtl_f=(-f "$CORALNPU_RTL/sim.f" -f "$REEF_RTL/reef.f") ;;
  sky130)  case "${TCM:-sram22}" in
             sram22)  OUT="${OUT:-$REPO_ROOT/build/reef_smoke_sky130}"
                      rtl_f=(-f "$REPO_ROOT/configs/filelists/reef_top_sky130_sim.f") ;;
             openram) OUT="${OUT:-$REPO_ROOT/build/reef_smoke_sky130_openram}"
                      rtl_f=(-f "$REPO_ROOT/configs/filelists/reef_top_sky130_openram_sim.f") ;;
             *) echo "TCM must be sram22 or openram" >&2; exit 2 ;;
           esac ;;
  *) echo "NODE must be generic or sky130" >&2; exit 2 ;;
esac
VERILATOR="${VERILATOR:-verilator}"
RISCV_PREFIX="${RISCV_PREFIX:-riscv-none-elf-}"
JOBS="${JOBS:-16}"
mkdir -p "$OUT"

# Program: M3's toolchain ISA string (toolchain/cc_toolchain_config.bzl).
MARCH="-march=rv32imf_zve32f_zicsr_zifencei_zbb -mabi=ilp32"
for prog in smoke smoke_buserr; do
  extra=""; [ "$prog" = smoke_buserr ] && extra="-DTEST_BUS_ERROR"
  # shellcheck disable=SC2086
  "${RISCV_PREFIX}gcc" $MARCH $extra -O2 -nostdlib -nostartfiles -ffreestanding -T "$SRC/link.ld" \
    -o "$OUT/$prog.elf" "$SRC/start.S" "$SRC/smoke.c"
  "${RISCV_PREFIX}objcopy" -O binary -j .text "$OUT/$prog.elf" "$OUT/$prog.bin"
  "${RISCV_PREFIX}objdump" -d "$OUT/$prog.elf" > "$OUT/$prog.dis"
done
result_addr=$("${RISCV_PREFIX}nm" "$OUT/smoke.elf" | awk '$3=="smoke_result"{print $1}')
echo "program: $(stat -c %s "$OUT/smoke.bin") bytes, smoke_result @0x$result_addr"

# Model: sim filelist + upstream's Verilator defines/waivers (M3 BUILD vopts).
if [ ! -x "$OUT/obj/Vreef_top" ] || [ -n "$(find "$REEF_RTL" "$SRAM_RTL" "$CLOCK_RTL" "$REPO_ROOT/configs/filelists" "$SRC/tb_reef_top.cpp" "$SRC/upstream" -newer "$OUT/obj/Vreef_top")" ]; then
  # shellcheck disable=SC2086
  "$VERILATOR" --cc --exe --build -j "$JOBS" -O3 --x-assign fast --x-initial fast \
    --top-module reef_top -Mdir "$OUT/obj" \
    "${rtl_f[@]}" \
    -DUSE_GENERIC -DTB_SUPPORT -DVLEN_128 -DZVE32F_ON \
    -Wno-WIDTH -Wno-CASEINCOMPLETE -Wno-LATCH -Wno-SIDEEFFECT -Wno-MULTIDRIVEN \
    -Wno-UNOPTFLAT -Wno-BLKANDNBLK -Wno-CASEX -Wno-ASCRANGE -Wno-WIDTHEXPAND \
    -Wno-WIDTHTRUNC -Wno-UNSIGNED -Wno-PINCONNECTEMPTY -Wno-PINMISSING -Wno-TIMESCALEMOD \
    -CFLAGS "-I$SRC/upstream" \
    "$SRC/tb_reef_top.cpp" "$SRC/upstream/sram_backdoor.cc" > "$OUT/verilate.log" 2>&1 \
    || { tail -30 "$OUT/verilate.log"; exit 1; }
fi

status=0
run() {  # prog ratio mode
  echo "=== $1, f_clk/f_sck=$2, $3, NODE=$NODE${TCM:+ TCM=$TCM}"
  "$OUT/obj/Vreef_top" "$OUT/$1.bin" "$result_addr" "$2" "$3" | tee "$OUT/$1_r$2.log" || status=1
}
run smoke 16 pass
run smoke 32 pass
run smoke_buserr 16 buserr
[ $status = 0 ] && echo "ALL SMOKE RUNS PASS" || echo "SOME SMOKE RUNS FAILED"
exit $status

# reef_top: Reef v1 test-chip top (v0)

**Status:** v0 draft for review, 2026-10-01. Unowned item from the week-1 list ("reef_top
integration shell for DV and PD"). Core: CoralNPU M3 `72d700e` `RvvCoreMiniAxi`, unmodified.

RTL: `rtl/reef/` (`reef_top.sv`, `reef_spi2axi.sv`, `reef_axi_err_slave.sv`, `reef_rst_sync.sv`),
filelist `rtl/reef/reef.f`. Lint: `scripts/lint_reef.sh`. Test: `scripts/run_reef_smoke.sh`.

## Why a shell is needed

`RvvCoreMiniAxi` exposes two full AXI4 ports with 128-bit data (about 600 signal pins) plus 300
sim-trace outputs. A test chip on a ~44-GPIO frame (e.g. ChipFoundry openframe) cannot bond that
out. `reef_top` reduces the core to 10 pins and gives the bring-up host a way in.

## Pins

| Pin | Dir | Function |
|---|---|---|
| `clk_i` | in | core clock |
| `rst_ni` | in | active-low async reset (the core's `RstSync` and `reef_rst_sync` sync the deassert) |
| `spi_sck_i`, `spi_csn_i`, `spi_mosi_i` | in | host SPI (mode 0), oversampled in `clk_i` |
| `spi_miso_o` | out | host SPI |
| `test_en_i` | in | DFT test enable: core `io_te`, clock-gate override, reset-sync bypass. Tie low in functional mode |
| `halted_o`, `fault_o`, `wfi_o` | out | core status |

## Blocks

**`reef_spi2axi`**: SPI slave to AXI4 master on the core's AXI slave port (ITCM, DTCM, CSR).
- Wire protocol = upstream `Spi2TLULV2` (M3 `hdl/chisel/src/bus/Spi2TLULV2.scala`), so upstream's
  host tools (`sw/utils/nexus_loader`, `coralnpu_test_utils/ftdi_spi_master.py`) work as they are.
  Frame: `op` (1 read, 2 write), 4 address bytes, 2 length bytes (beats - 1), then data. Reads
  return `0xFE` + 16 bytes per beat.
- Differs from upstream in one way: the SPI pins are **oversampled in `clk_i`** (2-flop
  synchronizers) instead of clocking logic with SCK. No CDC FIFO, no SCK clock tree, and STA has
  one clock. Cost: **f_clk >= 16 x f_sck** (e.g. 100 MHz core -> 6.25 MHz SPI; 40 KB loads in ~50 ms).
  CS must stay high for at least 4 `clk_i` cycles between frames so the parser sees the frame end
  (the testbench uses 4 SCK periods).
- Each beat is a single AXI transfer of 16 bytes at `addr + 16*i` with strobe
  `0xFFFF << addr[3:0]`. The address is passed through unaligned on purpose: `CoreAxiCSR` decodes
  the exact address (`PC_START` is a write to +0x4 with data in bits 63:32).

**`reef_axi_err_slave`**: terminates the core's AXI master. There is no off-chip memory in v1, so
any access outside ITCM/DTCM/CSR gets DECERR (bursts honored) and the core takes an access fault.

**`reef_rst_sync`**: async assert, sync deassert for the logic `reef_top` adds; bypassed when
`test_en_i = 1`.

## Core tie-offs

| Core port | Tied to | Note |
|---|---|---|
| `io_boot_addr` | `0` (ITCM base) | the host writes `PC_START` anyway |
| `io_irq`, `io_timer_irq`, `io_software_irq` | `0` | no interrupt sources on the v1 chip |
| `io_dm_req_valid` / `io_dm_rsp_ready` | `0` / `1` | debug module is reachable over AXI at CSR +0x800 |
| `io_debug_*` (300 outputs) | open | sim trace only |

## Boot sequence (host, over SPI)

Same as upstream `execute_from()` (`coralnpu_test_utils/core_mini_axi_interface.py`):

1. Write the program into ITCM at `0x0000_0000` (and data into DTCM at `0x0001_0000` if needed).
2. Write `PC_START` (`0x0003_0004`).
3. Write `RESET` (`0x0003_0000`) = 1: clock gate released, still in reset.
4. Write `RESET` = 0: run.
5. Poll `halted_o` (or `STATUS` at `0x0003_0008`: bit 0 halted, bit 1 fault). A program ends with
   `mpause` (`0x08000073`, clean halt) or `ebreak` (halt + fault), as in M3's crt.
6. Read results back from DTCM.

## Verification so far

`scripts/run_reef_smoke.sh` (Verilator, M3 sim filelist and upstream defines). The testbench drives
only `reef_top`'s pins, the way the bring-up host will. 3/3 runs PASS on 2026-10-01: the program
(RVV int32 add, RVV FP32 reduction, scalar int and FP) at SCK ratios 16 and 32, and a bus-error
program that gets DECERR, traps with mcause 5 and raises `fault_o`. Log: `results/reef_smoke.md`.
`scripts/lint_reef.sh`: our blocks are clean under Verilator `-Wall`; reef_top + core passes.

## Area

Std cells, sky130hd, quick Yosys flow, no timing target: `reef_spi2axi` 0.035 mm^2,
`reef_axi_err_slave` 0.001 mm^2, `reef_rst_sync` < 0.001 mm^2. Together about 0.5% of the core
(7.2 mm^2). Most of the bridge is its two 2-entry beat queues.

## Open decisions (owner)

- SPI vs UART for the host link, and the SCK ratio (Peter + Arch; v0 assumes SPI, 16x).
- Interrupt pins: none in v1 (Arch).
- Does Arch want an off-chip memory port? If yes, the AXI master needs a narrow external bus
  instead of the DECERR sink (Arch + PD, pin budget).
- Scan pin sharing in test mode, see `docs/dft-v0.md` (PD).
- Pad ring, IO cells and their drive strengths (PD).

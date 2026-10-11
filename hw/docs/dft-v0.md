# Reef v1 DFT, MBIST and scan interface (v0)

**Status:** draft for review, 2026-10-01. Unowned item from the week-1 list ("MBIST and logic-scan
interface document"). Baseline: CoralNPU M3 `72d700e`, top `reef_top`.

Guidance we are following (Sirish Oruganti, #rtl-design, 9/30 and 10/1):

- Test chips usually get a full boundary scan, plus some internal bits; if not pin-limited, bring
  out a memory bus.
- BIST "can introduce more errors than it tests for if done incorrectly". Keep the first chip simple
  until any BIST scheme is verified in RTL and netlist simulation.
- Start from both ends: silicon limits (area, IO pins, std cells, memory) and what the chip must do.

## Decision proposed for v1

| Item | v1 proposal | Why |
|---|---|---|
| Memory test (ITCM 8 KB, DTCM 32 KB) | **No MBIST.** Test the SRAMs functionally from the host over SPI: write patterns, read back (march-style patterns in host software). | Every TCM byte is already reachable through `reef_spi2axi` -> core AXI slave. Zero added RTL and nothing new to verify. The smoke test already does write/readback of ITCM and DTCM. |
| Logic test | **Full-scan insertion by the PD tool**, if the flow supports it; otherwise functional test only. | No hand-written scan logic in RTL. The RTL only has to be scan-friendly (below). |
| Boundary scan (JTAG 1149.1) | **Not in v1.** | The pad ring has ~10 functional pins; a TAP adds 4-5 pins and a TAP controller for little coverage on a pad-limited test chip. Revisit if the package has spare pins. |
| Debug | AXI-only debug (CSR window +0x800 through SPI), `io_dm_*` tied off. | No JTAG DTM needed (see `docs/rtl-map-v0.md` block 9). |

## What the RTL already provides

- `test_en_i` pin -> core `io_te`, which forces the core's clock gates (`ClockGate.sv`) open and is
  passed through `RstSync`. In `reef_rst_sync`, `test_en_i` also bypasses the reset synchronizer so
  scan can control reset directly from `rst_ni`.
- One clock domain (`clk_i`). SPI SCK is oversampled, so it is not a clock and needs no scan clock.
- No latches except inside the two clock gates, which the sky130 build maps to the ICG cell
  `sky130_fd_sc_hd__sdlclkp_4` with its scan enable `SCE` on `io_te` (`rtl/clock/`).
- Asynchronous reset only at the reset synchronizers; internal logic uses synchronous reset
  (`docs/rtl-map-v0.md` block 11).

## Scan pins if PD inserts scan (proposal)

Pins are scarce, so share them with SPI in test mode (`test_en_i = 1`):

| Pin | Functional | Scan mode |
|---|---|---|
| `spi_csn_i` | SPI chip select | `scan_en` |
| `spi_mosi_i` | SPI MOSI | `scan_in` (1 chain) |
| `spi_miso_o` | SPI MISO | `scan_out` |
| `spi_sck_i` | SPI clock (oversampled) | unused (scan clock = `clk_i`) |

This needs a 2:1 mux on `spi_miso_o` in `reef_top`, added once PD confirms the scan flow and chain
count. The core has about 66k flops and latches (sky130hd netlist from `scripts/synth_yosys.sh`,
2026-10-01). One chain of 66k bits at 50 MHz shifts in ~1.3 ms per pattern: slow, but fine for a
test chip. More chains need more shared pins.

## Open questions for PD (Nathan)

1. Does the PD flow (OpenROAD/sky130 or the commercial flow) do scan insertion? If not, v1 is
   functional test only and the scan mux above is dropped.
2. Package pin count: is there room for a JTAG TAP or a second scan chain?

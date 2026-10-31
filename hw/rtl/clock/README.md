# Clock gate

M3's core has two `ClockGate` instances (`rtl/vendor/coralnpu/ClockGate.sv`): one inside
`RstSync` (holds the clock off for 2 cycles after reset) and `cg` in `RvvCoreMiniAxi` (CSR clock
gate, WFI, interrupts, debug halt request). Both have `te` = the core's `io_te` (`test_en_i` on
`reef_top`). Under `USE_GENERIC` the vendored file is a behavioral latch + AND; upstream has cell
branches for TSMC12FFC and GF22 only.

## sky130: `ClockGate_sky130.sv`

One `sky130_fd_sc_hd__sdlclkp_4`: `GCLK = CLK & latch(GATE | SCE)`, latch open while `CLK` is low
(`.lib`: `clock_gating_integrated_cell : "latch_posedge_precontrol"`). That is the same function as
the generic branch, with `te` on the scan enable `SCE`. Drive strength 4 is the largest `sdlclkp`;
CTS buffers the gated clock after it.

The vendored `ClockGate.sv` is not edited. sky130 builds drop it from the filelist and add ours,
the same way as the TCMs (`rtl/sram/README.md`):

```bash
grep -v -e '/Sram.v$' -e '/ClockGate.sv$' $CORALNPU_RTL/synth.f > synth_sky130.f
# synthesis:  -f synth_sky130.f -f $SRAM_RTL/sky130.f -f $CLOCK_RTL/sky130.f  + sky130_fd_sc_hd .lib/.db
# Verilator:  -f sim_sky130.f   -f $SRAM_RTL/sky130_sim.f -f $CLOCK_RTL/sky130_sim.f
```

`sim/sky130_fd_sc_hd__sdlclkp_4.sv` is our model of the cell for Verilator: the PDK model is built
from a UDP latch, which Verilator does not take.

### Test

`scripts/run_clock_gate_sky130_tb.sh` (VCS): `tests/clock_gate_sky130/tb_clock_gate.sv` checks
`clk_o` every 0.1 ns against the contract (enable/te pass while `clk_i` is low, hold while it is
high, so no glitch or short pulse), with enable and te changing at random times in both clock
phases. It runs on the vendored generic `ClockGate.sv`, on the wrapper with the PDK cell model
(functional and timing variants) and with our model; a mutant without the latch must fail.
Results: `results/clock_gate_sky130.md`.

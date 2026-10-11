# sky130 TCM from OpenRAM macros: function and timing

**Wrapper:** `rtl/sram/Sram_sky130.sv` at commit `fd6feb9` (design notes in `rtl/sram/README.md`).
Pinned baseline M3 `72d700e`. Macro: `sky130_sram_2kbyte_1rw1r_32x512_8` from `sky130_sram_macros`
in the open_pdks build in `fossi-foundation/ciel-releases` (same release as
`results/sta_m3_corners_sky130hd.md`). 20 macros: ITCM 4, DTCM 16 (4 banks x 4).

## 1. Function

All on koopa, from a fresh clone of `fd6feb9` unless noted.

| Check | Tool | Result |
|---|---|---|
| Unit test, ITCM + DTCM shapes, OpenRAM's own macro model (with its delays) | VCS T-2022.06, `scripts/run_sram_sky130_tb.sh` | **PASS**: ITCM 7,553 checks, DTCM 6,017 checks, 0 errors (2,879 / 2,869 random reads) |
| Same test on our Verilator cycle model of the macro | same | **PASS**, same counts |
| Test sensitivity: wrapper without the hold register (`rdata = dout`) | same (working copy, same RTL) | FAIL on both models (ITCM 4,669 / DTCM 3,143 errors: X or poison) |
| Test sensitivity: DTCM bank select stuck at 0 | same (working copy, same RTL) | FAIL on both models (DTCM 4,492 errors) |
| reef_top smoke test, TCMs = sky130 wrapper + cycle model | Verilator 5.052, `SRAM=sky130 scripts/run_reef_smoke.sh` | **3/3 PASS** (result words as in `results/reef_smoke.md`); `main()` 1,655 core cycles, total 267,845 / 534,430 / 284,300 |
| Same smoke test, upstream generic `Sram.v` (script change is a no-op for it) | same, `scripts/run_reef_smoke.sh` | 3/3 PASS, identical cycle counts (1,655; 267,845 / 534,430 / 284,300) |
| Lint (WSL, Verilator 5.032 from M3's Bazel cache) | `scripts/lint_reef.sh` | PASS: `Sram_sky130` `-Wall` at 512 and 2048 entries; reef_top + core + sky130 TCMs (synth set) |

The unit test checks every cycle, 1 ns before the rising edge: read data one cycle after a read,
`rdata` held over idle cycles and writes (including a write to the address just read), byte masks
(random, all-zero, all-one), and `rvalid` = `enable` delayed one cycle. The mutation rows show the
checks are not vacuous; they also caught a testbench bug (static initializers in the random loop,
so the first version did no random reads) that is fixed in `fd6feb9`.

The OpenRAM model needs one mechanical fix to compile in VCS (it uses `mem` before declaring it);
the script compiles a copy with only that declaration moved up and saves the diff.

## 2. Synthesis with the macros (Design Compiler)

**Runs:** 2026-10-04 on koopa, `scripts/synth_dc.sh --sram-db sky130_sram_2kbyte_1rw1r_32x512_8_TT_1p8V_25C.db`
(DC T-2022.03-SP2, `compile_ultra -no_autoungroup`), top `RvvCoreMiniAxi`, same defines and
constraints as `results/synth_m3_dc_sky130hd_10ns.md` (5% uncertainty, IO 20%). RTL = `fd6feb9`
except the text of one elaboration `$error` message. The macro `.lib` (TT only) compiled with
Library Compiler T-2022.03-SP5 (one warning: no `internal_power_calculation` attribute).
Elaborate/link: 0 errors, 20 macros linked as library cells. Re-timed with PrimeTime
(`scripts/sta_pt.sh --extra-db <macro .db>`, `io_te` = 0) on each DC netlist.

| Run (std-cell corner / clock) | Std cells (mm^2) | Macros (mm^2) | Total (mm^2) | Setup, all reg paths (PT) | Macro -> core (half cycle) | Core -> macro | DC wall |
|---|---:|---:|---:|---|---:|---:|---:|
| tt / 20 ns | 4.819 | 5.691 | 10.510 | met (0.00) | +0.01 ns | +5.20 ns | 4,357 s |
| tt / 10 ns | 5.066 | 5.691 | 10.756 | reg-to-reg met (0.00); 84 recovery violations, -1.05 ns (note 1) | +0.00 ns | +0.37 ns | 6,122 s |
| **ss / 20 ns** (macro TT) | **5.086** | 5.691 | **10.776** | **met (0.00)** | +0.00 ns | +0.49 ns | 5,689 s |
| tt 20 ns netlist re-timed at ss | | | | **fails, -21.8 ns** (42,048 endpoints) | -9.46 ns | -7.54 ns | |

Full Drop 0 top (`reef_top` with these TCMs and the sky130 clock gates, ss/20 ns, DC + PT;
`results/synth_reef_top_sky130_node.md`): 5.07 + 5.69 = **10.76 mm^2**, met with 0 violating
endpoints; the binding path is again ITCM `dout0` -> fetch `instructionBuffer` (+0.007 ns), and
core -> macro has +0.08 ns.

Macro area is exact: 20 x 284,538 um^2 (`.lib` area). Same corner and clock with `Sram` as a black
box: 5.052 mm^2 (tt 10 ns) and 5.07 mm^2 (ss 20 ns), so the wrapper plus timing the TCM paths
adds about **0.015 mm^2** of standard cells.

### What the timing says

- **The half-cycle read path is the binding one, and it is in instruction fetch.** The worst
  macro -> core path is ITCM -> `core/score/fetch/instructionBuffer` in both the tt/10 ns and
  ss/20 ns runs. At ss/20 ns:
  launch on the falling edge at 10.00 ns, macro `clk0 -> dout0` 0.39 ns (OpenRAM's number), then
  8.4 ns of fetch logic, arrival 18.83 ns = required time. At tt/10 ns the same path has 4.0 ns of
  logic in a 5 ns half cycle. DC sized it to exactly zero slack in both runs.
- So **there is no margin for a slower macro.** Every ns that the real `clk0 -> dout0` is slower
  than OpenRAM's characterized 0.39 ns comes straight off this path. The TT-only, model-generated
  `.lib` is the weakest number in this run.
- Core -> macro inputs have a full cycle; worst slack +0.49 ns at ss/20 ns (LSU address registers ->
  DTCM macro address pins), +0.37 ns at tt/10 ns.
- A netlist synthesized at tt does not hold at ss (-21.8 ns at 20 ns, consistent with
  `results/sta_m3_corners_sky130hd.md`). Synthesis for PD has to target the slow corner.
- Note 1: the 84 tt/10 ns violations are async-reset recovery checks (debug module `dmcontrol`
  -> RVV FMA reservation-station flop resets), which PT checks and DC does not by default. They
  have nothing to do with the SRAMs.
- Before P&R: ideal clock, wire-load model, no macro placement. Max-transition / max-capacitance
  violations remain (high-fanout nets, e.g. inside `u_fma_rs`), as in the black-box runs; P&R
  buffering fixes these. Hold is not meaningful before CTS.

### Options if the read path does not close after P&R or with real macro timing

1. Lower the clock: the logic budget after the macro output is about half the period minus
   uncertainty and setup (8.4 ns at 20 ns), so each extra ns of period adds ~0.5 ns to it.
2. Ask PD for a macro `.lib` at ss (re-characterize with OpenRAM, or use the older
   `sram_1rw1r_32_256_8` libs that do have SS/FF corners as a sanity check of the derating).
3. Change the core's fetch to accept ITCM data one cycle later. This is a Chisel change in the
   vendored core, so it is the last resort.

## Bottom line

The sky130 TCMs work functionally with the M3 core (unit test on OpenRAM's model, smoke 3/3), and
the whole M3 core plus TCMs synthesizes to **~10.8 mm^2 at the ss corner at 50 MHz** (5.1 std cells
+ 5.7 macros, before utilization and halo), with timing met before P&R. The open risk is the
half-cycle ITCM -> fetch path, which closes with zero margin against a TT-only, model-generated
macro timing.


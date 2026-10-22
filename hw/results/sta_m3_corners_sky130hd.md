# Does M3 RvvCoreMiniAxi run at 100 MHz on sky130? Corner check + real SRAM macros

**Run:** 2026-10-03 on koopa, repo commit `e773f17` (DC runs), `0ccd881` (PT runs, `io_te` = 0),
pinned baseline M3 `72d700e`.
Libraries: sky130_fd_sc_hd `ss_100C_1v60` and `ff_n40C_1v95` from the open_pdks build in
`fossi-foundation/ciel-releases` tag `sky130-1689ac3f…` (the tt `.lib` used so far is ORFS's copy;
it differs from the ciel tt file only in formatting and pg-pin attributes), compiled to `.db` with
Library Compiler T-2022.03-SP5, 0 errors. SRAM: `sky130_sram_macros` (OpenRAM) from the same release.

**Short answer: no, not at the slow corner.** The 10 ns result in `synth_m3_dc_sky130hd_10ns.md`
holds only at tt. At ss the same netlist needs about 21 ns (~48 MHz); re-synthesizing at ss gets
to ~17.5 ns (~57 MHz) at best, and 20 ns (50 MHz) closes at 5.07 mm^2. All of this is pre-P&R
(ideal clock, wire-load model), so the post-route number will be lower.

## 1. The tt-synthesized netlist re-timed at ss (PrimeTime)

`scripts/sta_pt.sh --netlist build/dc/core_10ns/netlist.v --db <corner>.db --period 10`.
Same constraints as DC (10 ns, 0.5 ns uncertainty, IO 20%), library default wire load, ideal clock,
`io_te` = 0 (functional mode). Sram black boxes: TCM paths unconstrained.

| Corner | reg->reg WNS | worst arrival | all-path WNS | violating endpoints | min period |
|---|---:|---:|---:|---:|---:|
| tt 25C 1.80 V | +0.00 | 9.41 ns | +0.00 | 0 | 10 ns (by construction) |
| **ss 100C 1.60 V** | **-10.85** | 20.03 ns | -10.85 | 73,206 | **~20.9 ns (~48 MHz)** |

At ss the worst path moves to the RVV MUL/MAC (`u_mulmac/.../u_mul8_delay` -> `gen_res_ff[*]`,
45 cells deep); in->reg -0.83, reg->out -7.71. ss is about 2.1x slower than tt on this netlist.

Note on the first PT run: without `io_te` = 0, PT reported -1.33 ns at tt on `io_te` -> async reset
pins (recovery checks, which DC does not time by default). That is test mode only; `sta_pt.sh` now
holds the test enable at 0.

## 2. DC re-synthesis at ss

`scripts/synth_dc.sh --db sky130_fd_sc_hd__ss_100C_1v60.db --period {10,20} --out
build/dc/core_ss_{10,20}ns`, same flow and constraints as the tt run (`compile_ultra
-no_autoungroup`, Sram black boxes). Timing re-checked with `sta_pt.sh` at the same corner; the
numbers agree with DC.

| DC target (ss) | Std-cell area | Setup at ss | Worst arrival | max_transition viol. | Wall |
|---|---:|---|---:|---:|---:|
| 10 ns | **5.482 mm^2** | **fails, WNS -7.45 ns**, 45,891 endpoints | 16.66 ns | 2,579 | 6087 s |
| 20 ns | **5.069 mm^2** | **met**, slack 0.00 | 18.65 ns | 19,486 | 5457 s |
| (tt run, for reference: 10 ns at tt) | 5.052 mm^2 | met at tt, -10.85 at ss | 20.03 ns at ss | 203 | 5530 s |

- Pushing DC hard (10 ns target) brings the ss critical path down to 16.66 ns arrival, i.e. a
  **minimum period of ~17.5 ns (~57 MHz)** at ss with an ideal clock and wire-load model. It costs
  +8% area (5.48 vs 5.07 mm^2) and still fails 10 ns by 7.45 ns: sizing cannot close a 2x gap.
- **20 ns (50 MHz) closes at ss** with essentially the tt area (5.07 mm^2), but leaves ~19k
  max-transition violations that P&R buffering must fix.
- In the over-constrained run the worst path is no longer in RVV: it is in the scalar core
  (`score/retirement_buffer` -> `csr/minstret`), and reg->out paths fail by the same amount. Extra
  pipelining in one RVV unit would not be enough to reach 100 MHz on sky130.
- Hold is not meaningful before CTS (the 20 ns run reports 20k hold paths, worst -0.31 ns).

## 3. TCM with real sky130 SRAM macros (instead of the ~8 mm^2 estimate)

The only sky130 SRAM macros in the open PDK are OpenRAM `1rw1r` (one read/write + one read port):

| Macro | Size (um) | Area (mm^2) |
|---|---|---:|
| `sky130_sram_2kbyte_1rw1r_32x512_8` | 683.1 x 416.5 | 0.285 |
| `sky130_sram_1kbyte_1rw1r_32x256_8` | 479.8 x 397.5 | 0.191 |

M3's TCMs are ITCM 512x128 (8 KB) and DTCM 2048x128 (32 KB). With the 2 KB macro (512x32):
ITCM = 4 macros, DTCM = 16 macros (4 wide x 4 deep, plus a 4:1 read mux), **20 macros,
5.69 mm^2 before halo** (~6.1 mm^2 with a 10 um halo per side). That is below the ~8 mm^2 estimate,
but the macro is not a drop-in:

- **Timing data are not trustworthy for sign-off.** The `.lib` is TT only (no ss, no ff), and its
  numbers (min period 1.96 ns, clk->dout 0.38-0.53 ns) are OpenRAM model output, not silicon.
- **Read data launch on the falling clock edge** (`timing_type : falling_edge`), so the core has
  only half a cycle (5 ns at 100 MHz) from dout to its capture flop.
- It needs a new `Sram.v` branch for sky130 (the wrapper's TODO in the README) and a DTCM bank mux.

## What this changes

- "100 MHz on sky130" is **not** supported. Plan on **40-50 MHz** for sky130 (20 ns closes at ss
  before P&R; wires and clock skew will take some of that) until real SRAM timing and post-route
  STA say otherwise. Getting to 100 MHz would need pipelining changes in both RVV and the scalar
  core (RTL changes to M3, not in scope for an SV-first vendor flow) or a faster node.
- Area: closing timing at ss does not grow the logic much (5.07-5.48 mm^2 vs 5.05 at tt), and real
  macros make the TCM smaller than the estimate. The area conclusion of `docs/area-feasibility.md`
  (fits at ~19-24 mm^2 if our logic numbers are right) does not change.

## Still not verified

P&R (placement, CTS, routing, real parasitics), hold after CTS, SRAM timing at ss, gate-level
simulation or LEC of the netlist, and the reconciliation with Nathan's 21 mm^2.

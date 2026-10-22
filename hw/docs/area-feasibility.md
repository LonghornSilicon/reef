# Area feasibility v0: does Reef fit the die?

**Status:** v0, 2026-10-01, timing/SRAM notes added 2026-10-03. Inputs for the node/scope decision; **not a sign-off number.**
Baseline: CoralNPU M3 `72d700e` `RvvCoreMiniAxi` (scalar + scalar FPU + RVV VLEN 128 + vector
FP32, ITCM 8 KB, DTCM 32 KB). No VME/matrix engine: M3 does not have one.

## The question

On 9/30 Nathan (PD) synthesized "core, rvvcore and vme" on sky130 at a 10 ns clock and got about
**21 mm^2** of standard cells, about **30 mm^2** at 70% utilization, and concluded CoralNPU does not
fit a **32 mm^2** die (ChipFoundry double-wide openframe, per Alan). He also budgets ~2 mm^2 for
the IO ring and ~2 mm^2 for halo/blockages.

## Our numbers (std cells only, TCM SRAMs excluded)

| Variant | sky130hd (mm^2) | NanGate45 (kGE) | Source |
|---|---:|---:|---|
| M3 RvvCoreMiniAxi, default, area-only mapping | **7.20** | 1,798 | `results/synth_m3_sky130hd.md`, `synth_m3_nangate45.md` |
| same, **Design Compiler `compile_ultra`, 10 ns, timing met at tt** | **5.05** | - | `results/synth_m3_dc_sky130hd_10ns.md` |
| same, DC at the **ss** corner, 20 ns (met) / 10 ns (fails by 7.45 ns) | **5.07** / 5.48 | - | `results/sta_m3_corners_sky130hd.md` |
| without vector FP32 (`ZVE32F_ON` off) | **5.18** | 1,291 | `results/synth_m3_nozve32f.md` |
| scalar only (default minus RVV unit, from the block table) | ~1.2 | ~300 | derived, not synthesized |
| whole `reef_top` (core + shell), area-only | **7.24** | - | `results/synth_reef_top_sky130hd.md` |

Block split (sky130hd, default): RVV 83%, scalar core 13%, scalar FPU 2%, AXI/fabric 1%, the rest
< 1% each. Inside RVV the largest units are the vector FMA (13% of the core), divider (9%),
MUL/MAC (9%), ALU (8%) and VRF (6%).

Why the Yosys number is a lower bound: generic `synth` + `abc`, no SDC, no buffering or sizing for
real wire loads, no placement. Timing-driven synthesis and P&R usually add area.

A Yosys run with `abc -D 10000` (10 ns target) produced exactly the same netlist as the area-only
run. That does **not** mean 10 ns is met: with hierarchy kept, abc maps each module on its own,
and no module-local path is near 10 ns, so the target never binds. Cross-module paths are only seen
by a full-design tool, hence the Design Compiler run (`scripts/synth_dc.sh`, sky130 tt .lib
compiled to .db with Library Compiler on the ECE servers).

## Budget at 32 mm^2 (sky130)

| Item | Low | High | Note |
|---|---:|---:|---|
| Logic, our numbers / 0.7 utilization | 7.2 | 10.3 | DC 10 ns 5.05 / 0.7; Yosys 7.20 / 0.7 |
| Logic, Nathan's number / 0.7 | - | 30 | 21 / 0.7 |
| TCM SRAM, 40 KB | ~8 | ~9.4 | Nathan's ~8 mm^2 estimate; Aarush's commercial macro at 0.118 mm^2 per 4 kbit |
| IO ring | 2 | 2 | Nathan |
| Halo, blockages | 2 | 2 | Nathan |
| **Total with our logic numbers** | **~19** | **~24** | die = 32 |
| Total with Nathan's logic number | | ~43 | |

So **it fits if our logic numbers are close to right, and it does not fit if Nathan's is**. Two
independent tools put the pinned M3 core at 5-7 mm^2, and the timing-driven DC run (5.05 mm^2 at
10 ns, timing met at tt) is *smaller* than the area-only Yosys run, so timing-driven sizing does not
explain a 3-4x gap. The likeliest explanation is that his run was a different design (his note says
"vme", which is not in M3; it may also be upstream main or a different config) or a different flow
setting (flattened, no SRAM black boxes, slow corner, high utilization assumptions).

Timing caveat (checked 10/3, `results/sta_m3_corners_sky130hd.md`): 10 ns holds **only at tt**.
At the ss corner (100C, 1.60 V) the tt netlist needs ~21 ns; re-synthesized at ss, 20 ns (50 MHz)
closes and ~17.5 ns is the best DC reaches. All pre-P&R (ideal clock, wire-load model). On sky130,
plan for **40-50 MHz**, not 100 MHz.

SRAM check: the open sky130 PDK's OpenRAM macro `sky130_sram_2kbyte_1rw1r_32x512_8` is 0.285 mm^2,
so 40 KB is 20 macros = 5.7 mm^2 before halo, below the ~8 mm^2 estimate used in the budget. Its
`.lib` is TT only and model-generated, and read data launch on the falling clock edge.
**With the macros in synthesis (10/4, `results/sram_sky130.md`):** M3 core + 20 macros at ss, 20 ns:
5.09 mm^2 std cells + 5.69 mm^2 macros = **10.78 mm^2**, timing met before P&R. The ITCM -> fetch
read path (half a cycle, macro data launch on the falling edge) closes with zero margin.

## What to do

1. **Reconcile the two runs first** (Nathan + Samuel + Aaron): same top (`RvvCoreMiniAxi` at M3),
   same defines, same liberty, SRAMs blackboxed. Compare per-block (`scripts/area_report.py`).
   Until this is done, neither "fits" nor "doesn't fit" is established.
2. If area is tight after that, the cheapest cut is **vector FP32: -28% of logic** (-2.0 mm^2
   area-only), and only a define change. Arch decides whether the workloads need vector float
   (TinyStories int8 inference may not). The no-FP config is untested upstream, so it needs the
   RVV regression before we commit to it.
3. Node: v1 is sky130 (decided). A later move to 65 nm would shrink the logic by a large factor,
   but it is not an option for this chip.

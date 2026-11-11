# Matrix engine: correlation log

How the matrix module got to its current state: steps A-D, each with the
RTL measurements, waveform readings and held-out predictions behind it.
Kept as the evidence for [matrix.md](matrix.md), which says what the module
does now and how much to trust it. "Case N" refers to the open cases there.

## Steps A-D, with their results

The work toward three goals, in order. "Case N" refers to the open cases
below.

Every step has to pass the same checks:

- `bash scripts/lint.sh`, ctest and `uv run pytest` pass;
- `tests/golden/m3_results.json` is unchanged. No step here is meant to change
  M3 timing. A step that changes it has a bug, or needs the owner of the module
  it touches to decide;
- the matrix microbenchmark ROIs before and after are recorded in the commit
  message whenever they move.

### A. Keep the rest of the model unchanged

| # | Step | Touches | Owner sign-off | Verified by |
| --- | --- | --- | --- | --- |
| A1 | **Done.** Zvt opt-in flag (case 16): `top.frontend.params.zvt`, default off; `configs/machines/m3_vme.yaml` turns it on. With it off, `FuncSim` and `decodeInst` never call `decodeZvt`, so an M3 run takes exactly the old code path, and a Zvt word traps in Spike as it would on the M3 RTL. | `frontend/Fetch.*`, `frontend/FuncSim.*`, `frontend/InstDecode.*`, `configs/machines/m3_vme.yaml` | frontend | Golden unchanged; `ZvtTest.InstDecodeIgnoresZvtWhenOff`; `test_zvt_traps_without_vme_config` |
| A2 | **Done for `mac_chain`, `mac_indep`, `tile_load`; not possible for `tile_store`, `gemm_tile`.** End each matrix ROI on the last result (case 2). Before `ROI_END`, each microbenchmark reads back a result that depends on the last matrix op (for example `vtmv.v.t` and then `vmv.x.s`), so neither the model nor the RTL closes the ROI early. This changes no core rule. | `microbenchmarks/matrix/*.S` | none | ROIs include the tail; `test_matrix_microbenchmarks.py` updated |
| A3 | **Done.** `interfaces.md`, "Matrix-engine edits to other modules (for review)": every edit outside `matrix/`, by module, with when it fires and what verifies it. Each owner still has to review their part. **List every edit to another module's file** in `interfaces.md` with the condition it fires under (a Zvt instruction, or a matrix writer), for review. | `docs/interfaces.md` | frontend, core, vector, common | Review |

### B. Mirror CoralNPU's engine (from RTL source; no measurement needed)

All inside the matrix code, so the golden cannot move. Each gets a unit test
built from the cited RTL behavior.

| # | Step | Case | Files |
| --- | --- | --- | --- |
| B1 | **Done.** `mtx_mapping`, default `coral`; `mac_indep_tm8tn8` (2.0 → 4.0) and the new `mac_indep_tn4` (predicted 4.0 before its RTL run) match. **Coral tile mapping.** A `mtx_mapping` parameter: `coral` (busy = min(4, ceil(tm/2)), no `tn` term, `zvt_pe_array.sv:216`) or `generic` (today's formula, for the sweep presets). `coral` is the default. | 3 | `matrix/MatrixFabric.*`, `configs/matrix/*.yaml` |
| B2 | **Done.** `FuncSim` checks `vs1`/`vs2` alignment by element width (int8: register bits 2:1 = 0; bf16: bits 2 and 0; fp32: bits 1:0) and retires a misaligned `vtmm` as a no-op. `vtmm_misaligned` (held out): no traps on the RTL, op counts as predicted, 6.25 against 6.0 cycles per iteration (+4%, the ROB-bound loop, not the no-ops). **Misaligned `vs1`/`vs2` become no-ops** (`rvv_backend_decode_unit_ari.sv:3219-3233`). | 8 | `frontend/FuncSim.cpp` (behind the A1 flag) |
| B3 | **Done.** `mtx_lsu_fifo_entries`; tile loads and stores run beat by beat through the FIFOs. No change on `tile_load` or `tile_store` (the LSU, at 5 cycles per beat, is slower than the controller, so the FIFOs never fill); `tile_load_behind_mac` exercises them. **2-deep `lsu2vme`/`vme2lsu` FIFOs** between the LSU and the controller (`zvt.sv:163-206`). | 6 | `matrix/Matrix.*` |
| B4 | **Done.** `mtx_acc_raw_cycles`; `mac_chain` now matches the RTL (4.0 cycles per MAC) with the reference core's 16-entry ROB, which also brings tile load and store to +45% and +23%. **Per-tile MAC hazard.** Replace the whole-group wait with the RTL's comparison against in-flight adder stages (`zvt_pe_array.sv:195-204`). The exact back-to-back timing still needs a trace (C3). | 9 | `matrix/Matrix.*` |
| B5 | **Done (matrix side).** `vtdiscard` no longer holds the controller; it completes after every older matrix result. `vtdiscard_after_mac` (held out) shows the core interlock it still lacks: 24.0 cycles per iteration on the RTL, 5.5 in the model (case 7). **`vtdiscard` inside the engine.** Only the matrix-side rules. Dispatching alone and waiting for an empty retirement buffer are core rules and stay open. | 7 | `matrix/Matrix.*` |

### C. Correlate against the VME RTL

| # | Step | Notes |
| --- | --- | --- |
| C1 | **Done (2026-11-06);** the CoralNPU changes are in `docs/coralnpu_vme_verification.patch`. Also needed a testbench fix: with verification on, the VME core has extra retirement-lane debug outputs (`mtype`, `tileWrites`) that `core_mini_axi_tb.{h,cc}` did not bind, so SystemC refused to start. Build the VME verification simulator. In `~/coralnpu-pinned/tests/verilator_sim/BUILD`, add `vme_core_mini_verification_axi_tb` and `_sim` next to the existing local `vme_core_mini_axi_*` entries, depending on `//hdl/chisel/src/coralnpu:vme_core_mini_verification_axi_cc_library`. Then `bazel build -c opt //tests/verilator_sim:vme_core_mini_verification_axi_sim`. | About 20 minutes (`baseline/ASYNC.md`). Runs in the background. |
| C2 | **Done.** `ITERS`, set with `build_microbenchmark(..., {"ITERS": n})`; the `matrix_diff` experiment builds N and 2N. Give each microbenchmark an iteration count set at build time (today it is fixed at 200), and build N and 2N copies of each with the CoralNPU toolchain. | Needed for the differencing in "Correlating it", step 3 |
| C3 | **Done for the four looping programs** (`microbenchmarks/matrix/rtl_vme.csv`; results below). Waveforms not kept yet. Run them on the RTL and record the results in `microbenchmarks/matrix/rtl_vme.csv`. Keep one waveform of `mac_chain` and one of `tile_store` for cases 5 and 9. | Needs C1 and C2 |
| C4 | **Tile path: LSU cost done** (`docs/lsu-tile-beats.md`): the VME RTL charges a tile load or store 2 cycles per instruction + 1 per beat. On that core a vector load costs the same and a vector store one cycle more; the model's vector cost is M3's LSU (it matches the M3 RTL), so the gap was mostly the LSU version. The core's LSU now gives tile beats their own cost (`lsu_tile_cost`; off by default and in `m3_vme.yaml`, which keep M3's LSU; on for correlation in `REFERENCE_CORE_PARAMS`), pending the core owner's review. The unrolled `*_burst` programs match within 2%. The loop-per-row `tile_load`/`tile_store` are frontend-bound in the model, so they are not used for this. **MAC pipeline and command path done** from VME waveforms (`mac_chain`, `tile_load_behind_mac`): `mtx_cmd_latency` 6 → 4, `mtx_pe_pipeline_depth` 6 → 9, the controller starts the cycle after the last MAC's tile write, and a `lsu2vme` beat is popped no earlier than the cycle after its push; `tile_load_behind_mac` now matches (32.0). **Remaining PLACEHOLDERs done (2026-11-10)** with 14 held-out programs (results below): `mtx_move_latency` 1 → 0, a controller op starts on its first uop instead of after all its uops issue, and the other values are confirmed and retagged. Left: `mtx_queue_entries`, Dispatch's 4 matrix credits, which cap one-uop ops at about 0.8 per cycle where the RTL takes one per cycle (core; the engine's own 8-uop reservation station is modeled since D4), and `gemm_tile` (vector loads, core). | Matrix parameters only |
| C5 | **Done (2026-11-10).** `mtx_complete_latency` stays 1, now measured at `8a5e1b31`: `mac_rob`, `zero_rob` and `move_tv_rob` match, and 0 or 2 would each be off by one cycle per op. In the waveforms, `vmeRtVld` rises 2 cycles after an op's last controller cycle (a register, then the `mtInfoRs` FIFO, `zvt.sv:247-276`, `:535-570`) and the core retires the op in that cycle. | From C3 |

#### First per-op results (C3)

`python -m reef_perf.experiments.matrix_diff.main compare`, model defaults
against CoralNPU `8a5e1b31`, `(ROI(2N) - ROI(N)) / ops`:

| Program | Op | Model | RTL | Error |
| --- | --- | --- | --- | --- |
| `mac_chain` | dependent `vtmmu` | 10.0 | 4.0 | +150% |
| `mac_indep` | independent `vtmmu` | 4.125 | 4.0 | +3.1% |
| `tile_load` | `vtle8` | 6.19 | 3.44 | +80% |
| `tile_store` | `vtse8` | 7.25 | 3.56 | +104% |

Instruction counts match the RTL exactly. MAC throughput is right. Dependent
MACs on the RTL issue every 4 cycles, as fast as independent ones, so the
model's wait for the previous MAC's tile write (busy + 6-stage pipeline) is
wrong: this settles case 9's direction, and B4 should let a MAC accumulate
into a tile still in the pipeline. The tile load/store path is about twice too
slow, which is calibration (C4).

After B4, with the reference core's ROB (`REFERENCE_CORE_PARAMS` in
`matrix_diff/main.py`: 16 entries, CoralNPU `Parameters.scala:141`, against
M3's 8):

| Program | Op | Model | RTL | Error |
| --- | --- | --- | --- | --- |
| `mac_chain` | dependent `vtmmu` | 4.0 | 4.0 | 0% |
| `mac_indep` | independent `vtmmu` | 4.0 | 4.0 | 0% |
| `tile_load` | `vtle8` | 5.0 | 3.44 | +45% |
| `tile_store` | `vtse8` | 4.38 | 3.56 | +23% |
| `tile_load_behind_mac` | `vtmmu` + 8 `vtle8` + `lw`, per iteration (after B3) | 44.0 | 32.0 | +38% |

After the LSU tile cost (C4, `docs/lsu-tile-beats.md`), same ROB:

| Program | Op | Model | RTL | Error |
| --- | --- | --- | --- | --- |
| `tile_load_burst` | `vtle8`, unrolled | 3.0 | 3.0 | 0% |
| `tile_store_burst` | `vtse8`, unrolled | 3.06 | 3.0 | +2% |
| `tile_load32_burst` | `vtle32` (4 beats) | 6.0 | 6.0 | 0% |
| `tile_store32_burst` | `vtse32` (4 beats) | 6.0 | 6.0 | 0% |
| `tile_load` | `vtle8`, one branch per row | 4.38 | 3.44 | +27% (frontend) |
| `tile_store` | `vtse8`, one branch per row | 4.38 | 3.56 | +23% (frontend) |
| `tile_load_behind_mac` | per iteration | 28.0 | 32.0 | -12.5% |

After the waveform fixes (C4: command path, MAC pipeline, controller drain,
FIFO pop latency), same ROB and LSU:

| Program | Op | Model | RTL | Error |
| --- | --- | --- | --- | --- |
| `mac_chain` | dependent `vtmmu` | 4.0 | 4.0 | 0% |
| `mac_indep` | independent `vtmmu` | 4.0 | 4.0 | 0% |
| `tile_load_behind_mac` | per iteration | 32.0 | 32.0 | 0% |
| `tile_{load,store}_burst` | one beat | 3.0 / 3.0 | 3.0 / 3.0 | 0% |
| `tile_{load,store}32_burst` | four beats | 6.0 / 6.0 | 6.0 / 6.0 | 0% |

Raw ROIs agree within 8 cycles for these programs, so one-time costs line
up too, but only differences are compared.

#### Held-out programs

Written after calibration, with the model's numbers recorded before each
RTL run (`microbenchmarks/matrix/heldout_predictions.csv`). Cycles per op:

| Program | Tests | Predicted | RTL |
| --- | --- | --- | --- |
| `mac_chain_tm4`, `mac_indep_tm4` | tm 4 (2 busy cycles), dependent / independent | 4.0 / 2.0 | 4.0 / 2.0 |
| `mac_chain_tn4` | tn 4, dependent | 4.0 | 4.0 |
| `mac_indep_tm8tn8` | 8 x 8 tile | 2.0 (4.0 after B1) | 4.0 |
| `mac_indep_tn4` | tn 4, independent (added after B1) | 4.0 | 4.0 |
| `mac_{chain,indep}_fp32` | fp32 `vtfmm.tvv` | 4.0 / 4.0 | 4.0 / 4.0 |
| `mac_{chain,indep}_bf16` | bf16 `vtfmm.alt` | 4.0 / 4.0 | 4.0 / 4.0 |
| `gemm_multi` | 32 x 32 x 64, four accumulators | 2142 | 1378 (+55%) |
| `vtmm_misaligned` | two misaligned `vtmmu` (no-ops) + one with odd, aligned operands, per iteration (added after B2) | 6.25 | 6.0 (+4%) |
| `vtdiscard_after_mac` | `vtmmu` + `vtdiscard`, per iteration (added after B5; the core interlock is not modeled) | 5.48 | 24.0 (-77%) |
| `gemm_longk` | 16 x 16 x 256 | 2790 | 1672 (+67%) |

The dependent tm-4 chain confirms the 4-cycle hitRaw window when the array
is busy only 2 cycles (case 9). The GEMM errors are the vector load path,
as for `gemm_tile` below: with the newer LSU's vector costs approximated,
`gemm_multi` comes to about +3% and `gemm_longk` to about +11% (0% with a
32-entry ROB).

#### C4 and C5: the remaining PLACEHOLDERs

Fourteen programs, each putting one PLACEHOLDER or `mtx_complete_latency`
on the critical path, with the model's predictions recorded before their
RTL run (`heldout_predictions.csv`, 2026-11-10; one-parameter nudges showed
which program each value moves). Cycles per op, same settings as above:

| Program | Tests | Predicted | After | RTL |
| --- | --- | --- | --- | --- |
| `zero_burst` | `vtzero` (4 tiles) | 4.0 | 4.0 | 4.0 |
| `move_tv_burst` | `vtmv.t.v`, 32-bit (4 uops) | 5.0 | 4.0 | 4.0 |
| `move_vt_burst` | `vtmv.v.t`, 32-bit (4 uops) | 4.0 | 4.0 | 4.0 |
| `move_tv8_burst`, `move_vt8_burst` | 8-bit moves (1 uop) | 2.0 / 1.53 | 1.5 / 1.5 | 1.19 / 1.19 |
| `move_vt_use` | `vtmv.v.t` → `vmv.x.s` → next TSS | 16.0 | 14.0 | 13.0 |
| `load_then_mac` | `vtle32` → `vtmmu`, same tiles | 18.0 | 18.0 | 18.0 |
| `move_then_mac` | `vtmv.t.v` → `vtmmu`, same tiles | 19.0 | 18.0 | 18.0 |
| `mac_then_store` | `vtmmu` → `vtse32`, same tiles | 18.0 | 18.0 | 18.0 |
| `load_then_store` | `vtle32` → `vtse32`, same row | 13.0 | 13.0 | 14.0 |
| `mac_rob`, `zero_rob`, `move_tv_rob` | one op + 15 `addi` (ROB-bound) | 22 / 12 / 14 | 22 / 12 / 12 | 22 / 12 / 12 |
| `mac_queue` | 4 `vtmmu` + 8 `addi` | 19.0 | 19.0 | 19.0 |

What changed, from waveforms (`move_tv8_burst`, `move_tv_rob`,
`move_vt_use`, `load_then_store`):

- **A controller op starts on its first uop.** Dispatch to first move uop is
  5 cycles for a 4-uop move, as for a 1-uop one; the controller then takes
  one uop per cycle, slower than the 3 per cycle that issue. A MAC still
  waits for all 4 uops (`isPe` over the uop lanes, `zvt_ctrl.sv:108-119`),
  which is the 6 measured before. `Matrix.cpp` now starts controller ops at
  `CmdStart + 1`.
- **`mtx_move_latency` 1 → 0.** A move's tile write is a register
  (`zvt_mt_reg.sv`), readable the cycle after its last uop, which the model
  already counts.

The remaining gaps are outside the matrix module:

- **Queue depth (core).** One-uop moves run one per controller cycle on the
  RTL, 16 back to back. The model's 4 matrix credits in Dispatch, back
  about 5 cycles after dispatch, cap them at 0.8 per cycle. (The engine's
  8-uop reservation station, `ZVT_RS_DEPTH`, is modeled since D4; it holds
  8 one-uop ops but only 2 MACs.) Letting one-uop ops through needs
  Dispatch to count credits in uops, or more of them
  (`Dispatch.cpp:219-255`).
  With 8 entries the model gives 1.275; the last 0.09 is the loop branch
  (3 cycles on the RTL, about 4.4 in the model: the frontend gap above).
- **`vmv.x.s` of a move result (vector).** On the RTL it retires in the
  cycle the move does; the model is 1 cycle later (`move_vt_use`).
- **A tile store holds the LSU while it waits for data (core, case 5).** In
  `load_then_store` the LSU takes the store as soon as the load's last beat
  is pushed and waits 2 cycles for the controller to finish the load's pops;
  the model books the LSU only once data is ready, 1 cycle per pair fast.

#### D4: a 128-PE CoralNPU array

CoralNPU sizes its array as `NUM_PE = TE * TE * COMPRATIO` and walks a tile
in `PROCESS_DELAY = 1 / COMPRATIO` steps (`rvv_backend_define.svh:210-220`).
At VLEN 128 the default is 64 PEs and 4 steps. Setting `COMPRATIO` to
`8/TE` and `PROCESS_DELAY` to `TE/8` builds a 128-PE, 2-step array; every
use of `COMPRATIO` is a product, so the textual macro arithmetic holds. 256
PEs (1 step) does not build: the step counter is `$clog2(PROCESS_DELAY)` = 0
bits wide (`zvt_pe_array.sv:183`). The build is a separate CoralNPU worktree
(`~/coralnpu-pe128`, `8a5e1b31` + `docs/coralnpu_vme_verification.patch` +
the two defines); its results are in `microbenchmarks/matrix/rtl_vme_pe128.csv`.

Two model changes, both in the matrix module:

- **The Coral mapping scales with the PE count**: `PROCESS_DELAY = 256 /
  (mtx_pe_rows x mtx_pe_cols)` steps of `8 / PROCESS_DELAY` rows per half
  (`MatrixFabric.cpp`), where it was fixed at `min(4, ceil(tm / 2))`.
  Unchanged at 64 PEs. `mtx_zero_cycles` is also `PROCESS_DELAY` and has to
  be set with it.
- **The uop reservation station** (`mtx_uop_buffer_entries`, 8 =
  `ZVT_RS_DEPTH`). Dispatch pushes up to 3 uops a cycle, each push lane
  checking the room left at the start of the cycle, before that cycle's pop
  (`almost_full`, `rvv_backend.sv:776-807`). A MAC pops its 4 uops with its
  last step (`peCmdRdy = canStart && cntCr`); the controller pops one per
  cycle. The model used to give each instruction its own `ceil(uops / 3)`
  issue cycles, which capped MACs at one per 2 cycles; the 128-PE waveform
  of `mac_indep_tm4` shows 2 MACs every 3 cycles, the FIFO count cycling 3,
  6, 4. At 64 PEs it changes only `vtmm_misaligned` (6.25 to 5.83; RTL 6.0).

Predictions were recorded before the 128-PE run
(`heldout_predictions_pe128.csv`). Cycles per op:

| Program | 64-PE RTL | 128-PE RTL | 128-PE model |
| --- | --- | --- | --- |
| `mac_indep` and its tn4, 8 x 8, fp32, bf16 variants | 4.0 | 2.0 | 2.0 |
| `mac_chain` and all its variants | 4.0 | 4.0 | 4.0 (hitRaw does not depend on the array size) |
| `mac_indep_tm4` | 2.0 | 1.755 | 1.835 (2.0 before the reservation station) |
| `zero_burst` | 4.0 | 2.0 | 2.0 |
| `load_then_mac`, `move_then_mac`, `mac_then_store` | 18 | 16 | 16 |
| `mac_rob`, `zero_rob` | 22 / 12 | 20 / 10 | 20 / 10 |
| `mac_rob_tm4`, `mac_rob_tn4`, `mac_rob_tm8tn8` | 20 / 22 / 22 | 19 / 20 / 20 | 19 / 20 / 20 |
| `tile_load_behind_mac` | 32 | 30 | 30 |
| `mac_queue` | 19 | 15 | 15.36 |

The known gaps (vector loads, frontend, queue depth, case 5, `vtdiscard`)
are the same at both sizes. `mac_indep_tm4` and `mac_queue` are ROB-bound
short loops at 128 PEs and stay 2-5% slow. The engine is not the cause:
`mac_rob_tm4` (the same MAC's dispatch-to-retire time) matches, and a
shorter MAC pipeline that would close it is ruled out by the three
`mac_rob_*` programs. The frontend's loop cost is the likeliest cause; not
checked.

This validates the Coral mapping at other sizes. It says nothing about the
systolic and tree presets, which no RTL has.

#### D4: systolic and tree arrays

No RTL has these arrays, so their formulas (`getMacCost`, generic mapping)
were checked two ways.

**A cycle-level reference** (`python/reef_perf/experiments/array_ref/ref.py`)
schedules each pass of a MAC: when each operand reaches each PE (from where it
enters and whether it is forwarded or broadcast, worked out separately for the
two operands), each multiply, each partial-sum hop, each tree input and each
stationary load. It books every PE port, tree input and column load path per
cycle, starts each pass at the earliest cycle nothing collides, fails if the
two operands of a multiply do not meet, and checks that every product of the
tile is computed exactly once. `array_ref.main` runs it over 1,536 cases
(output-stationary 8 and 16 square, 1 or 4 lanes, every delivery pair, hop 1
or 2, depth 1 or 3; weight-stationary chain 2/4/16 x 8/16, hop 1 or 2, load 0
or 6, single or double buffered; trees 2/8 x 4/8, 1 or 4 lanes, level 1 or 2;
eight tile shapes up to 16 x 16 x 4) and writes
`tests/cpp/matrix/FabricReference.inc`. `MatrixFabricTest.ReferenceArrays`
checks every row: latency exactly, back-to-back occupancy as the ceiling of
the reference's interval (alternating buffers can make it a fraction).

It found two errors, both fixed:

- **Weight-stationary latency counted the row skew twice**, once as fill (a
  forwarded column operand) and once as the partial-sum chain. The weights do
  not move during compute, so the skew is the chain alone. `ws_systolic_16x16`
  on a 16 x 16 x 4 MAC: 42 to 39 cycles.
- **Stationary loads and back-to-back MACs.** The model charged every MAC
  one load and nothing else. Double buffered, the next MAC's load overlaps
  this one's compute, limited by the PEs (`Temporal`), the column load path
  (`Load`) and buffer reuse two passes later (`Temporal + Load + chain
  skew` per pair of passes): `ws_systolic_16x16` 20 to 16 cycles per MAC,
  `tree_8x8` 33 to 32. Single buffered, a column reloads only after its last
  row's use, which trails the first by the chain: 20 to 23. A lone MAC's
  latency still includes its first load. `MatrixFabric.cpp` now separates
  the two (`Busy` for a lone MAC, the interval for occupancy).

Output-stationary arrays, single-buffered trees and every latency outside
the weight-stationary chain matched before the fix.

**SCALE-Sim 3.0.0** (an open-source systolic-array simulator) independently
gives the same cycles as the reference on 11 single-fold GEMMs, output- and
weight-stationary, 4 x 8 to 16 x 16
(`python -m reef_perf.experiments.array_ref.scalesim_check`, in a throwaway
environment; it is not a dependency). That includes the weight-stationary
skew counted once. SCALE-Sim cannot check the rest: it skews and loads over
the whole array even for a partial tile, and runs folds one after another
with full fill and drain, so it does not model overlapping passes or MACs.

**What is assumed, not checked** (no RTL):

- A pass's results go straight to tile storage (each PE writes its own C
  element; for chains and trees, the bottom of each column). Arrays that
  shift results out through the array would add a drain per pass and could
  not overlap passes as freely.
- Skew and loads span only the PEs a tile uses.
- Stationary loads move one PE row per cycle down each column, in step with
  that column's wavefront.
- Everything outside the array is Coral's: one tile per `vtmm`, the uop
  path, the controller. Under Zvt, weight reuse across instructions does not
  exist (case 12).
- Input-stationary arrays are not in the reference.

#### Held-out check: `gemm_tile`

One 16x16x64 int8 GEMM (16 k-steps of 8 `vle8` + `vtmmu`, then 16 rows of
`vtmv.v.t` + `vse32`), never used for fitting. Per GEMM, by differencing
10 and 20 GEMMs:

| Model setting | Model | RTL |
| --- | --- | --- |
| `REFERENCE_CORE_PARAMS` (vector accesses on M3's LSU) | 870 | 520 |
| + vector loads at the newer LSU's 3 cycles, `vse32` (4 registers) bracketed at 5 and 9 (RTL 7) | 539 to 595, about 567 | 520 |
| + a 32-entry ROB | 486 to 550, about 518 | 520 |

The matrix side matches the waveform: a MAC starts 3 cycles after its last
operand's writeback, retires 14 cycles after its first step, and the
read-back rows are paced by the store. The rest is the vector load path:
M3's LSU in throughput (5 cycles per `vle8` against 3), and in latency, which
keeps each MAC in the 16-entry ROB 3 cycles longer than on the RTL. A
k-step is 19 instructions, so that stalls dispatch for 3 cycles per k-step
(48 per GEMM); a 32-entry ROB hides it. Both are the core owner's
(`docs/lsu-tile-beats.md`). The approximate vector costs above are not in
`REFERENCE_CORE_PARAMS`: the LSU's vector parameters cannot express a
per-instruction overhead, so they cannot match one- and four-register
accesses at once.

Running it found one matrix-interface bug: Dispatch held a vector reader of
a `vtmv.v.t` result until the result was ready, so each read-back row took
13 cycles against the RTL's 7. That rule now lives in the vector module
(interfaces.md, "Matrix → vector").

The ROB matters because a Zvt instruction holds its retirement-buffer entry
until its tile write lands (`RetirementBuffer.scala:540-549`). With M3's 8
entries, only about 2.7 `mac_chain` iterations fit, and the chain runs at
7.0 cycles per MAC, limited by the ROB, not the engine. Which ROB Reef's core
has is the core owner's call; the matrix numbers above assume 16.

### D. Configurable for the sweep

What to sweep, across modules, is collected in `sweep-candidates.md`.

| # | Step | Verified by |
| --- | --- | --- |
| D1 | **Done (2026-11-11).** Every `configs/matrix/*.yaml` runs `mac_indep` and `gemm_tile` on top of `m3_vme.yaml`, within its peak MAC rate (`test_presets_run_and_respect_compute_bound`). New preset `coral_128pe.yaml`, the RTL-checked 128-PE array. | New pytest |
| D2 | **Done (2026-11-11).** A zero row, column, lane, array, divider, queue, issue-width or FIFO count, a uop buffer smaller than a MAC, an unknown dataflow, delivery, reduction or mapping name, or a Coral PE count other than 32-256 stops the run with the parameter's name in the error (`checkFabricConfig`; `MatrixFabricTest`, `test_bad_matrix_parameters_fail_by_name`). | ctest |
| D3 | **A sweep script** that runs a grid of `-p` overrides and writes ROI cycles, `mac_utilization` and the cost estimates to CSV. | Runs on the presets |
| D4 | **Coral-shaped arrays done (2026-11-11):** the Coral mapping scales with the PE count and matches a 128-PE CoralNPU build (results below; `test_matches_rtl_at_128_pes`). **Systolic and tree arrays: formulas checked (2026-11-11)** against a cycle-level reference and SCALE-Sim; two formula errors fixed (results below). No RTL has them. 256 PEs is out of scope for Reef. | Per GEMM shape |

### Left for other owners or the spec

These need an interface change or a decision outside the matrix module, so
they stay as open cases until the owner takes them up:

- the shared RVV dispatch and the 8-uop reservation station (case 4: vector);
- the LSU waiting for tile-store data (case 5: core);
- a dedicated memory port (case 13: core and memory);
- same-cycle WAR on vector registers (case 15: vector);
- the CSR empty-Rob rule (case 2: core);
- tile data and `mstatus.MS` (case 10);
- Spike's handling of altfmt (case 11);
- weight reuse across instructions (case 12: spec);
- cost parameters (case 14: PD).

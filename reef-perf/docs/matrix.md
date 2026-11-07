# The matrix module

The matrix module (`top.matrix`, `model/matrix/`) models a matrix engine
behind CoralNPU's Zvt instructions (spec v0.4, CoralNPU `8a5e1b31`). It
answers two questions:

1. **Coral correlation:** with its defaults, how close is it to CoralNPU's own
   engine? The VME RTL is the only matrix engine we can measure.
2. **Design-space exploration:** keeping Zvt as the programmer's interface,
   how fast is a different compute fabric (systolic, adder tree, a larger
   array) behind it?

M3 has no matrix engine. Zvt programs run only on CoralNPU builds with VME,
so every number here belongs to `8a5e1b31`, not to M3.

## What runs where

```
 ELF ─► FuncSim ─────────────► Dispatch ──► matrix module ─────────────────────────┐
        Zvt runs here, not in     │          1. command path: cmd_latency,        │
        Spike (frontend/Zvt.h):   │             uops_per_cycle, in order,         │
        mtype, vtype/vl, rd,      │             after vector sources are ready    │
        tile group, Mem           │          2. execution, in order:              │
                                  │             vtmm   → MAC arrays (MatrixFabric)│
        mset* → VSET → vector     │             vtzero, vtmv → controller         │
                                  │             vtle/vtse → core LSU, 16 B beats  │
                                  │          3. tile scoreboard (mt0-mt15)        │
                                  │          4. complete_latency → core Rob ◄─────┘
                                  └─ tile loads/stores also take an LSU credit
```

- **Functional side.** Spike does not know Zvt. FuncSim skips Spike for Zvt
  words and applies their effects itself: `mtype` (tm, tk, mtwiden), and
  `vtype`/`vl` through Spike's vector unit, so ordinary vector instructions
  around them run correctly. Tile contents and `vtmv.v.t` results are **not**
  computed.
- **mset\*** are vector configuration (`InstClass::VSET`), timed by the vector
  module like `vsetvli`. That is what CoralNPU does: the RVV front end
  executes them and sends no command to the backend (`RvvFrontEnd.sv:637-643`).
- **Every other Zvt instruction** is `InstClass::MATRIX`, with the details in
  `Inst::Mtx` (op, tile group, `tm`/`tn`/`tk`, input type).
- **Traps** match the RTL: every tile instruction traps on `vill` or an
  unconfigured `mtype` (`RvvFrontEnd.sv:624-633`, `Decode.scala:573-579,
  :1052-1073`). **Rejected shapes** retire as no-ops (`MatrixOp::Nop`), as the
  RVV decoder does (`rvv_backend_decode_unit.sv:92-96`): vl = 0, a misaligned
  tile group, vtmm with `tm`, `tn` or `tk` of 0 or a type configuration other
  than int8 {SEW8, x4}, bf16 {SEW16, x2, altfmt} or fp32 {SEW32, x1}.

Zvt is off unless `top.frontend.params.zvt` is true. With it off (M3, and
every config without `m3_vme.yaml`), `FuncSim` and `decodeInst` never look
for Zvt: a Zvt word goes to Spike and traps, as on the M3 RTL, and non-Zvt
programs take exactly the code path they took before the matrix module
existed.

Run it:

```
reef_perf --elf prog.elf -c configs/machines/m3.yaml \
          -c configs/machines/m3_vme.yaml                                  # Coral engine
reef_perf --elf prog.elf -c configs/machines/m3.yaml \
          -c configs/machines/m3_vme.yaml \
          -c configs/matrix/ws_systolic_16x16.yaml                         # another array
reef_perf ... -p top.matrix.params.mtx_pe_rows 32                          # one value
```

The JSON summary has a `matrix` object: ops by kind, useful and capacity
MACs, `mac_utilization`, array busy cycles, memory beats, 16-byte tile reads
and writes, and `area_um2`/`energy_pj` once the cost parameters are set.

## Parameter map

Each parameter is tagged with where its default comes from:

- **RTL**: read from CoralNPU `8a5e1b31` at the cited place.
- **TRACE**: measured in a CoralNPU simulation, but at checkout `08eadd1c`
  (`ZVT_ISA_REFERENCE.md`), not re-measured at `8a5e1b31`.
- **INFER**: our reading of the RTL; not checked in a waveform.
- **PLACEHOLDER**: no source; calibrate it.
- **DSE**: a design choice for the sweep. CoralNPU's value is the default.

### Core interface and command path

| Parameter | Default | Tag | Meaning |
| --- | --- | --- | --- |
| `mtx_queue_entries` | 4 | PLACEHOLDER | Matrix command-queue entries, counted in instructions. CoralNPU's Zvt reservation station holds 8 uops (`ZVT_RS_DEPTH`, `rvv_backend_define.svh:201`): 2 vtmm or 8 vtzero (open case 4). |
| `mtx_cmd_latency` | 4 | TRACE | Cycles from the unit receiving an instruction to issuing its uops. On CoralNPU this is the RVV front end, decode and dispatch. VME waveform of `mac_chain`: a MAC into an idle engine starts its first step 6 cycles after the core dispatches it = 4 + 2 issue cycles (4 uops at 3 per cycle). Was a PLACEHOLDER 6. |
| `mtx_uops_per_cycle` | 3 | RTL | Uops issued per cycle (`DISPATCH3` → `NUM_DP_UOP = 3`, `RvvCore.scala:639`). |
| `mtx_complete_latency` | 1 | TRACE | Result written to retirement (vtmmu: `vmeRt` at 20, retired at 21). |

### Compute fabric (`MatrixFabric`)

| Parameter | Default | Tag | Meaning |
| --- | --- | --- | --- |
| `mtx_num_arrays` | 1 | DSE | Independent arrays; MACs to different tiles can overlap. |
| `mtx_pe_rows`, `mtx_pe_cols` | 8, 8 | RTL + INFER | 64 PEs (`NUM_PE = TE*TE*COMPRATIO`, `rvv_backend_define.svh:212-222`). The 8 x 8 shape is ours; CoralNPU has 4 blocks (`NUM_BLK`) and a different row mapping (open case 3). |
| `mtx_macs_per_pe_int8` / `_bf16` / `_fp32` | 4 / 2 / 1 | RTL + INFER | K lanes per PE: 4 int8, 2 bf16 or 1 fp32 in one 32-bit word, matching KMAX (`RvvFrontEnd.sv:374-379`). |
| `mtx_dataflow` | `output` | DSE | Stationary operand: `output` (C stays; K streams), `weight` (B stays; M streams), `input` (A stays; N streams). |
| `mtx_row_operand`, `mtx_col_operand` | `broadcast` | DSE | `broadcast` or `forwarded` (one PE per hop: systolic skew). |
| `mtx_hop_latency` | 1 | DSE | Cycles per PE-to-PE hop. |
| `mtx_mapping` | `coral` | RTL + TRACE | How a tile maps onto the array's steps. `coral`: CoralNPU's PE array walks the tile's rows two at a time and stops at 4 steps, whatever `tn` is: busy = min(4, ceil(tm/2)) (`ROWS_PER_CNT`, `cntMax00`, `zvt_pe_array.sv:216-217`); output-stationary only. Confirmed by `mac_indep_tm8tn8` and `mac_indep_tn4` (4.0 cycles each on the RTL; `generic` gives 2.0). `generic`: the rows x cols block formula below, for the sweep's other designs (the systolic and tree presets set it). |
| `mtx_reduction` | `in_place` | DSE | `in_place`, `chain` (partial sums move down columns), `tree` (adder tree). |
| `mtx_tree_level_latency` | 1 | DSE | Cycles per adder-tree level. |
| `mtx_pe_pipeline_depth` | 9 | RTL + TRACE | Stages after the last step: `MULBULKPIPENUM` 3 + the `handshake_ff` between multiply and add 1 (`zvt_pe_block.sv`) + `ADDERPIPENUM` 3 (`zvt_pe_array.sv:34-35`) + 2 for the skew registers that feed blocks 1-3 a cycle or two after block 0 (`pipeLa`/`pipeHa0`/`pipeHa1`, `zvt_pe_array.sv`). VME waveform of `mac_chain`: every MAC retires 14 cycles after its first step = 4 busy + 9 + `mtx_complete_latency` 1. Was 6 (multiply + adder only). Whether that is the whole MAC-to-tile path needs a trace. |
| `mtx_accumulate_latency` | 0 | PLACEHOLDER | Extra read-modify-write of the accumulator tile. |
| `mtx_acc_raw_cycles` | 4 | RTL | Cycles after a MAC starts until a MAC into the same tile may start. CoralNPU's hitRaw holds a MAC's first step while the previous MAC's first step is in multiply stages 1-3 (`MULBULKPIPENUM` + 1, `zvt_pe_array.sv:195-207`). Matches `mac_chain` (4.0 cycles per dependent MAC). 0 makes a MAC wait for the earlier MAC's tile write, for arrays that cannot forward the accumulator. Only after a MAC: after vtzero, a move or a tile load, a MAC waits for the write. |
| `mtx_stationary_load_cycles` | 0 | DSE | Load time per pass for weight/input stationary; 0 means one PE row per cycle. |
| `mtx_double_buffer_stationary` | false | DSE | Load the next pass's operand during this pass. |
| `mtx_pipeline_across_ops` | true | PLACEHOLDER | Fill, drain and pipeline overlap the next MAC. |
| `mtx_fabric_clock_divider` | 1 | DSE | Core cycles per fabric cycle. |

The timing of one MAC, for a tile of `TM x TN x TK` (`matrix/MatrixFabric.cpp`):

```
Lanes  = mtx_macs_per_pe_<type>;  KSteps = ceil(TK / Lanes)
output:  rows <- TM,      cols <- TN,  temporal = KSteps
weight:  rows <- KSteps,  cols <- TN,  temporal = TM
input:   rows <- KSteps,  cols <- TM,  temporal = TN
Passes  = generic: ceil(rows / pe_rows) * ceil(cols / pe_cols)
          coral:   min(4, ceil(TM / 2))   (output-stationary; tn does not count)
Busy    = Passes * temporal  (+ stationary loads: once if double-buffered, else per pass)
Fill    = (row operand forwarded ? (cols used - 1) * hop : 0) + (col operand forwarded ? (rows used - 1) * hop : 0)
Drain   = chain: (rows used - 1) * hop;  tree: ceil(log2(rows used * Lanes)) * tree_level_latency
Latency = Fill + Busy + Drain + pe_pipeline_depth + accumulate_latency
Occupancy (array unavailable) = pipeline_across_ops ? Busy : Latency
All three are multiplied by the clock divider.
```

Dimensions of 0 are charged as 1 (the frontend turns those instructions into
no-ops before they get here). With the defaults, a full int8 tile
(16 x 16 x 4) is 4 busy cycles at 256 MACs/cycle. That matches the 4 PE-busy cycles measured for vtmmu (TRACE) and
the `PROCESS_DELAY = 4` counter in the RTL.

### Tile storage and data movement

| Parameter | Default | Tag | Meaning |
| --- | --- | --- | --- |
| `mtx_zero_cycles` | 4 | RTL | vtzero holds the controller for `PROCESS_DELAY` cycles (`zvt_ctrl.sv:435-445`, `:613`). |
| `mtx_move_cycles_per_uop` | 1 | PLACEHOLDER | Controller cycles per vtmv uop (1/2/4 uops at SEW 8/16/32). |
| `mtx_move_latency` | 1 | PLACEHOLDER | Last move uop to result written. |
| `mtx_tile_write_latency` | 1 | PLACEHOLDER | Last load beat to tile written. |
| `mtx_tile_read_latency` | 1 | PLACEHOLDER | Tile read before a store's first beat. |
| `mtx_lsu_fifo_entries` | 2 | RTL | Entries in each FIFO between the LSU and the controller: `lsu2vme` for tile loads, `vme2lsu` for tile stores (`multi_fifo` `DEPTH 2`, `zvt.sv:163-206`). A load's LSU half runs at most this many beats ahead of the controller; a store's controller stalls when the LSU is this many beats behind. A push may share a cycle with the pop that frees its entry. The RTL holds a finished load beat in the LSU until the FIFO has room; the model starts the beat late instead (INFER: same stall, the LSU's hold not modeled). |
| `mtx_ctrl_waits_for_array_drain` | true | RTL + TRACE | vtzero, vtmv.\*, vtle and vtse wait for `!peBusy` (`zvt_ctrl.sv:105-106, :130-134`), and start the cycle after the last MAC's tile write (VME waveform of `tile_load_behind_mac`: `peBusy` falls 13 cycles after the MAC's first step, the first pop is at 14). |

Tile loads and stores use the core's LSU, one 16-byte beat per uop
(`rvv_backend_decode_unit_lsu.sv:3048-3070`), with the LSU's own
`lsu_vector_*` costs. `vl` is clamped to 16 (`Lsu.scala:328`).

The controller serves one head uop at a time, so an instruction starts only
after the previous one releases it: a MAC when its compute steps end
(`canStart && cntCr`, `zvt_pe_array.sv:209`), vtzero after its counter,
loads, stores and moves after one controller cycle per uop.

### Fixed by the ISA

These are not parameters. Changing them changes Zvt and VLEN:

- tile edge 16 elements (`TE = VLEN/8`);
- 16 tiles (`NUM_MT`);
- int8 → int32, bf16 → fp32 and fp32 → fp32 only;
- one tile (at most 16 x 16 x KMAX) per `vtmm`.

An array larger than 16 x 16 under this ISA is under-used, and
`mac_utilization` shows by how much.

### Cost (for weighing designs)

| Parameter | Default | Meaning |
| --- | --- | --- |
| `mtx_area_per_pe_um2` | 0 | Area of one PE. Total = arrays x rows x cols x this + tile storage. |
| `mtx_tile_storage_area_um2` | 0 | Area of the tile storage. |
| `mtx_energy_per_mac_pj` | 0 | Energy per useful MAC. |
| `mtx_energy_per_subtile_access_pj` | 0 | Energy per 16-byte tile read or write. |
| `mtx_energy_per_mem_beat_pj` | 0 | Energy per 16-byte memory beat. |

All zero until the PD team has sky130 numbers. The JSON reports `null` for
area or energy while their parameters are all zero.

## Presets (`configs/matrix/`)

| Overlay | Fabric | Status |
| --- | --- | --- |
| `coral_vme.yaml` | CoralNPU: 64 PEs, 4 int8 lanes, output-stationary, broadcast | The defaults, written out with sources |
| `ws_systolic_16x16.yaml` | 16 x 16 scalar PEs, weight-stationary, forwarded operands, partial-sum chain, double-buffered weights | Illustrative; not validated |
| `os_systolic_16x16.yaml` | 16 x 16 scalar PEs, output-stationary, forwarded operands | Illustrative; not validated |
| `tree_8x8.yaml` | 8 x 8 PEs of 4-wide dot products, weight-stationary, adder tree | Illustrative; not validated |

## Correlating it

1. **Build a VME verification simulator.** In the CoralNPU checkout, add
   `vme_core_mini_verification_axi_tb` and `..._sim` to
   `tests/verilator_sim/BUILD`, pointing at the existing
   `//hdl/chisel/src/coralnpu:vme_core_mini_verification_axi_cc_library`.
   Without `--enableVerification` the trace has no `mcycle` values, so no
   ROI can be measured (`baseline/ASYNC.md`).
2. **Run `microbenchmarks/matrix/*.S` on it**, measured as roadmap ticket B01
   does, and record the results in `microbenchmarks/matrix/rtl_vme.csv`.
3. **Take differences.** The model's core is M3's, and the VME build's core
   is `8a5e1b31`'s, which is 15-60% different (`baseline/ASYNC.md`). So
   compare per-instruction costs, (2N - N iterations), not whole ROIs.
   Calibrate the PLACEHOLDER parameters from: `mac_chain` (MAC latency),
   `mac_indep` (MAC occupancy), `tile_load` and `tile_store` (beat path), and
   `gemm_tile` (vector to matrix to vector).
4. **When a difference does not explain itself, read a waveform.** Build
   the program with a small `ITERS` (`build_microbenchmark(..., {"ITERS":
   4})`), run the VME simulator with `--trace_file w.fst`, and convert it
   with `fst2vcd`. Under
   `VVmeCoreMiniVerificationAxi.core.VmeCoreMiniVerificationAxi.core`, the
   signals used so far: `clock`; `score.dispatch.io_rvv_<k>_{valid,ready,bits_pc}`
   (dispatch to the RVV backend); `io_debug_rb_inst_<k>_{valid,bits_pc}`
   (retirement); `rvvCore.rvvCoreWrapper.core.backend.vme.zvtPeArray.{canStart,cnt,hitRaw}`
   (MAC steps); `...vme.peBusy`; `...vme.{uop_lsu2vme_vld,uop_lsu2vme_rdy}`
   (LSU push) and `...vme.{vmelsuresVld,vmelsuresRdy}` (controller pop);
   `score.lsu.slot.io_uop_{valid,ready}` (the LSU slot taking an
   instruction). Sample on the rising edge of `clock`. This is how the MAC
   pipeline (14 cycles from first step to retire) and the command path (6
   cycles from dispatch to first step) were measured.
5. **Cross-check the non-Coral presets** against a cycle-level array
   simulator or an RTL prototype, GEMM shape by GEMM shape, before using them
   in the sweep.

## Next steps

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
| C1 | **Done (2026-11-06).** Also needed a testbench fix: with verification on, the VME core has extra retirement-lane debug outputs (`mtype`, `tileWrites`) that `core_mini_axi_tb.{h,cc}` did not bind, so SystemC refused to start. Build the VME verification simulator. In `~/coralnpu-pinned/tests/verilator_sim/BUILD`, add `vme_core_mini_verification_axi_tb` and `_sim` next to the existing local `vme_core_mini_axi_*` entries, depending on `//hdl/chisel/src/coralnpu:vme_core_mini_verification_axi_cc_library`. Then `bazel build -c opt //tests/verilator_sim:vme_core_mini_verification_axi_sim`. | About 20 minutes (`baseline/ASYNC.md`). Runs in the background. |
| C2 | **Done.** `ITERS`, set with `build_microbenchmark(..., {"ITERS": n})`; the `matrix_diff` experiment builds N and 2N. Give each microbenchmark an iteration count set at build time (today it is fixed at 200), and build N and 2N copies of each with the CoralNPU toolchain. | Needed for the differencing in "Correlating it", step 3 |
| C3 | **Done for the four looping programs** (`microbenchmarks/matrix/rtl_vme.csv`; results below). Waveforms not kept yet. Run them on the RTL and record the results in `microbenchmarks/matrix/rtl_vme.csv`. Keep one waveform of `mac_chain` and one of `tile_store` for cases 5 and 9. | Needs C1 and C2 |
| C4 | **Tile path: LSU cost done** (`docs/lsu-tile-beats.md`): the VME RTL charges a tile load or store 2 cycles per instruction + 1 per beat. On that core a vector load costs the same and a vector store one cycle more; the model's vector cost is M3's LSU (it matches the M3 RTL), so the gap was mostly the LSU version. The core's LSU now gives tile beats their own cost (`lsu_tile_cost`; off by default and in `m3_vme.yaml`, which keep M3's LSU; on for correlation in `REFERENCE_CORE_PARAMS`), pending the core owner's review. The unrolled `*_burst` programs match within 2%. The loop-per-row `tile_load`/`tile_store` are frontend-bound in the model, so they are not used for this. **MAC pipeline and command path done** from VME waveforms (`mac_chain`, `tile_load_behind_mac`): `mtx_cmd_latency` 6 → 4, `mtx_pe_pipeline_depth` 6 → 9, the controller starts the cycle after the last MAC's tile write, and a `lsu2vme` beat is popped no earlier than the cycle after its push; `tile_load_behind_mac` now matches (32.0). Remaining: the other matrix-side PLACEHOLDERs and `gemm_tile`. **Calibrate the PLACEHOLDERs** from the per-iteration differences, in the order `mac_indep`, `mac_chain`, `tile_load`, `tile_store`. `gemm_tile` is the held-out check. Retag each value TRACE with its benchmark. | Matrix parameters only |
| C5 | **Re-measure the TRACE value** (`mtx_complete_latency`, from checkout `08eadd1c`) at `8a5e1b31`. | From C3 |

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
| D1 | **Test every preset:** each `configs/matrix/*.yaml` loads on top of `vme.yaml` and runs `gemm_tile`. | New pytest |
| D2 | **Validate parameters at construction:** zero rows, columns, lanes, arrays or divider, and unknown dataflow, delivery, reduction or mapping names, each fail with the parameter's name in the error. | ctest |
| D3 | **A sweep script** that runs a grid of `-p` overrides and writes ROI cycles, `mac_utilization` and the cost estimates to CSV. | Runs on the presets |
| D4 | **Check the non-Coral formulas** (case 17) against a reference array simulator or RTL prototype before using the sweep's numbers. | Per GEMM shape |

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

## Open cases

Things this draft does not settle, roughly by how much they matter for
correlation. Each was checked against CoralNPU `8a5e1b31` where the source
can answer it.

1. **No RTL numbers yet.** Every PLACEHOLDER is a guess, and the TRACE value
   (`mtx_complete_latency`) is from an older checkout. It needs the VME
   verification simulator (step 1 above).
2. **The ROI ends too early in the model (settled: the RTL differs).** On
   CoralNPU a CSR read waits for an empty retirement buffer
   (`Decode.scala:707-708`), and a tile instruction stays there until its
   tile write lands (`RetirementBuffer.scala:431, :540-562`). So the RTL's
   closing `csrr mcycle` waits for older matrix work. The model's core does
   not have that rule (a core-module change), so its matrix ROIs miss the
   tail. Until the core models it, compare total work per iteration (step 3)
   or add a dependence on the last result before `ROI_END`.
3. **Partial tiles (settled, B1).** CoralNPU's PE-busy count is
   min(4, ceil(tm/2)), with no `tn` term (`zvt_pe_array.sv:216`). The generic
   formula gives ceil(tm/8) x ceil(tn/8). `mtx_mapping: coral` (the default)
   uses CoralNPU's count; the held-out `mac_indep_tm8tn8` and `mac_indep_tn4`
   confirm it on the RTL. The 8 x 8 x 4-lane shape stands in for CoralNPU's
   4 blocks of 16 PEs: same peak rate; `mac_utilization` shows the wasted
   steps of a partial tile.
4. **Zvt shares the RVV backend on CoralNPU (confirmed).** Its uops are
   dispatched by the RVV dispatch into a Zvt reservation station of 8 uops
   (`rvv_backend_dispatch.sv:498-512`, `rvv_backend.sv:782`). The model gives
   the matrix module its own instruction-counted queue and issue path, so
   vector and matrix work never contend. Fixing this is an interface change
   between the vector and matrix modules.
5. **Tile stores hold the LSU.** CoralNPU's LSU takes the store's address and
   waits idle for the data (`Lsu.scala:1077-1101`). The model books the LSU
   only once the data is ready, so it underestimates LSU contention. How
   long the hold lasts needs a trace.
6. **Tile loads start at once (settled, B3).** The LSU half starts the
   cycle the instruction reaches the matrix module, limited by the 2-deep
   `lsu2vme` FIFO (`mtx_lsu_fifo_entries`). The RTL holds a finished beat in
   the LSU while the FIFO is full; the model delays the beat's start instead,
   so the LSU is free for other work a little earlier than on the RTL. A
   beat is popped no earlier than the cycle after its push (the FIFO's
   count is a register, `multi_fifo.sv:107`; seen in the waveform).
   `tile_load_behind_mac`, which fills the FIFO behind a MAC, matches the
   RTL exactly, so the hold approximation costs nothing there.
7. **`vtdiscard` (matrix side settled, B5; core side open).** The engine
   never sees it: the RVV decoder rejects its encoding and retires it
   (`rvv_backend_decode_unit_ari.sv:3600-3607`), so it does not hold the
   controller; the model keeps only the engine's share of the core's
   interlock (complete after every older matrix result). Its real cost is
   the core's: it dispatches alone from slot 0 (`Decode.scala:197`) and only
   with an empty retirement buffer, the RVV backend idle and the LSU idle
   (`:601-607`), and it sets `mstatus.MS` to Initial (`Csr.scala:632`).
   Not modeled; `vtdiscard_after_mac` measures it: 24.0 cycles per
   `vtmmu` + `vtdiscard` on the RTL against the model's 5.5. Kernels issue
   it rarely (between kernels or on a context switch).
8. **vtmm register alignment (settled, B2).** A `vtmm` whose `vs1` or `vs2`
   is misaligned for its element width retires as a no-op
   (`rvv_backend_decode_unit_ari.sv:3219-3233`, `rvv_backend_decode.sv:54-62`);
   `FuncSim` applies the same check, and `vtmm_misaligned` confirms it on the
   RTL. For int8 the check is register bits 2:1, so odd registers such as v9
   are legal. Whether the scalar LSU still takes a queue entry for a vl = 0
   tile load or store is untraced (the model takes none).
9. **The hitRaw rule (settled, B4).** CoralNPU holds a MAC only while the
   previous MAC into the same tile group has its first step in the multiply
   stages (`zvt_pe_array.sv:195-207`), so dependent MACs start 4 cycles apart
   (`mtx_acc_raw_cycles`). For a full tile that equals the array's busy time,
   so `mac_chain` cannot tell whether the window is 4 or less; partial tiles
   (busy 1 or 2 cycles) would, and are not measured yet.
10. **Tile data is not computed.** Programs whose control flow depends on
    matrix results take the wrong path. Spike's `minstret` does not count
    Zvt instructions. `mstatus.MS` is not modeled, so a program that turns
    MS off traps on the RTL but not in the model.
11. **A standard `vsetvli` with altfmt (bit 28)** is legal at SEW8/16 on
    CoralNPU (`RvvFrontEnd.sv:208`), but Spike without Zvfbfa sets vill, so
    later vector instructions trap in the model.
12. **No reuse across instructions.** Each vtmm reloads its stationary
    operand. That penalizes weight- and input-stationary arrays, because
    Zvt has one tile per instruction. A real engine could keep weights
    across vtmm with the same B; deciding that is part of the spec.
13. **No dedicated memory port.** CoralNPU's tile loads use the LSU. A
    matrix-owned port (`RequesterKind::Matrix`) needs Dispatch to skip the
    LSU credit and the matrix module to call `MemoryInterface`; that is an
    interface change.
14. **Cost parameters are zero**, and fmax per topology is only the clock
    divider. These need PD estimates.
15. **Same-cycle WAR on vector registers.** If a younger vector instruction,
    dispatched in the same cycle, overwrites a register an older matrix
    instruction reads, the matrix instruction waits for the younger write
    (interfaces.md, "Vector → matrix"). The other direction, a vector
    reader of a `vtmv.v.t` result, now waits inside the vector module
    (interfaces.md, "Matrix → vector").
16. **Zvt is off by default (settled, step A1).** It runs only with
    `m3_vme.yaml` (`top.frontend.params.zvt`). On a VME build the RTL also routes
    some standard OPMVV funct6 111100 encodings (the vwmaccu family) to the
    matrix engine (`RvvFrontEnd.sv:633`, `rvv_backend_decode_unit_ari_de2.sv
    :506-509`), a CoralNPU decode collision the model does not reproduce.
17. **Fill, drain and tree formulas** are analytic and unchecked for any
    setting but CoralNPU's.

    
    
0. Commit what's pending. The N/2N work is still uncommitted: ITERS, the readbacks, matrix_diff, rtl_vme.csv, the doc updates and the preset comments. I'd make it one commit: "reef-perf: N/2N matrix measurement against the VME RTL". The 13 inference-engine/coral/bench files stay out, as before.

1. B4: per-tile MAC hazard. This is the biggest error (mac_chain +150%). Replace the wait for the previous MAC's full latency with the RTL's rule: compare against the adder stages still in flight (zvt_pe_array.sv:195-204). The change is in Matrix.cpp. Re-run matrix_diff. mac_chain should come down to about 4 cycles per MAC, and mac_indep must stay at about 4.

2. Tile load/store path: B3, then C4 and C5.
- B3 is a structural fix: add the 2-deep lsu2vme/vme2lsu FIFOs from zvt.sv.
- C4: once the structure matches, calibrate the PLACEHOLDER values against tile_load (+80%) and tile_store (+104%). Retag each calibrated value TRACE. gemm_tile is the held-out check, so don't tune against it.
- C5: re-measure mtx_complete_latency on the current CoralNPU commit.

3. Remaining CoralNPU behaviors, all inside the matrix code:
- B1: a mtx_mapping: coral option for how a tile maps onto the array;
- B2: misaligned vs1/vs2 become no-ops;
- B5: matrix-side vtdiscard.

4. A3: review list. In docs/interfaces.md, list every edit to another module's file and the condition it fires under. The frontend, core, vector and common owners need this to sign off.

5. Sweep readiness (D). The v2 spec is due Dec 6.
- D1: test that every preset loads and runs gemm_tile.
- D2: invalid parameters give errors that name the parameter.
- D3: a sweep script that runs a grid of -p overrides and writes a CSV.
- D4: check the non-Coral formulas against a reference before trusting their numbers.

6. Ship.
- Save the CoralNPU testbench patch as docs/coralnpu_vme_verification.patch, so others can rebuild the VME sim.
- git push -u origin akul/matrix_engine.
- Open a draft PR with base yuvesh-kernel-integration.
- Get sign-off from the module owners.

# The matrix module

The matrix module (`top.matrix`, `model/matrix/`) models a matrix engine
behind CoralNPU's Zvt instructions (spec v0.4, CoralNPU `8a5e1b31`). It does
two jobs:

1. **CoralNPU's engine, checked against its RTL.** With its defaults it is
   CoralNPU's VME engine, the only matrix engine we can measure.
2. **Other engines behind the same instructions**: larger Coral arrays,
   systolic arrays, adder trees, more arrays, deeper pipelines, for the
   architecture sweep.

M3 has no matrix engine. Zvt programs run only with `m3_vme.yaml`, which
turns Zvt on; without it a Zvt instruction traps, as on the M3 RTL, and M3
timing is untouched.

The history behind the numbers here (each step, its RTL runs, waveforms and
held-out predictions) is in [matrix-correlation-log.md](matrix-correlation-log.md).

## What to trust

| Claim | Evidence |
| --- | --- |
| The Coral engine's timing matches CoralNPU's RTL | 44 programs differenced at N and 2N iterations on the VME RTL. 30 are in the RTL gate (`test_matches_rtl`), within 2% (one within 5%); 23 of them were written after calibration, with the model's prediction recorded before the RTL run (`heldout_predictions.csv`). The other 14 differ for reasons outside the matrix module (below). |
| Coral arrays of other sizes are right | A 128-PE CoralNPU build matches on 28 programs (`test_matches_rtl_at_128_pes`); every predicted change landed. 32 PEs uses the same rule but has no RTL run. 256 PEs is out of scope. |
| The systolic and tree formulas are right under stated assumptions | The model matches a cycle-level reference on 1,536 array and shape cases (`MatrixFabricTest.ReferenceArrays`), and the reference matches SCALE-Sim on 11 single-fold cases. No RTL has these arrays; the assumptions are under "Checking it". |
| Every preset runs real instruction mixes; impossible configurations fail by name | `test_presets_run_and_respect_compute_bound`, `test_bad_matrix_parameters_fail_by_name` |
| M3 timing is unaffected | `tests/golden/m3_results.json`, unchanged |

What is not right yet. None of it is in the matrix module; each owner has
the details:

| Gap | Effect | Owner |
| --- | --- | --- |
| Vector loads use M3's LSU; the VME RTL has a newer, faster one | Whole GEMMs 55-67% slower than the VME RTL; about 3-11% with the newer LSU's costs approximated (`docs/lsu-tile-beats.md`) | core |
| ROB size: M3 has 8 entries, the VME RTL 16 | A MAC holds its entry until its tile write lands: dependent MACs 7.0 cycles each with 8, 4.0 with 16 | core |
| A tile store holds the LSU while it waits for data (case 5) | about 7% fast on load-then-store chains | core |
| `vtdiscard`'s dispatch interlock (case 7) | 77% fast on a loop of `vtmmu` + `vtdiscard` | core |
| The M3 frontend is slower on short loops than the VME core's | about 25% slow on a loop with a branch every 5 instructions | frontend |
| A vector reader of a `vtmv.v.t` result | 1 cycle late | vector |
| Matrix and vector work share dispatch on CoralNPU (case 4) | Mixed vector and matrix kernels come out fast | vector |
| Area and energy | Not estimated: the cost parameters are 0 until physical design has sky130 numbers | PD |

**Using the numbers:**

- Run each result at both values of the open core choices, ROB 8 and 16 and
  `lsu_tile_cost` off and on (next section), and report the range.
- Compare designs with each other, under the same core settings, rather than
  quoting absolute cycles.
- Say which designs are RTL-checked (Coral) and which are formula-checked
  (systolic, tree), and that formula-checked designs carry the assumptions
  listed under "Checking it".

## Using it

### Run

```
# CoralNPU's engine (the defaults)
reef_perf --elf prog.elf -c configs/machines/m3.yaml -c configs/machines/m3_vme.yaml

# another array: add a preset
reef_perf --elf prog.elf -c configs/machines/m3.yaml -c configs/machines/m3_vme.yaml \
          -c configs/matrix/ws_systolic_16x16.yaml

# one value
reef_perf ... -p top.matrix.params.mtx_num_arrays 2
```

From Python, `reef_perf.sim.run_sim(elf, overlay_configs=(...), params={...})`
does the same. A program must turn Zvt's `mstatus.MS` on and set `mtype`
before its first tile instruction; `microbenchmarks/matrix/zvt.inc` has the
instructions as assembler macros, and `microbenchmarks/matrix/*.S` are
working examples.

### Presets (`configs/matrix/`)

| Preset | Engine | Checked by |
| --- | --- | --- |
| `coral_vme.yaml` | CoralNPU: 64 PEs (8 x 8), 4 int8 lanes, output-stationary, operands broadcast, results in place. The defaults, written out with their sources. | RTL |
| `coral_128pe.yaml` | The same engine with 128 PEs (8 x 16): 2 steps per tile instead of 4 | RTL (a 128-PE build) |
| `os_systolic_16x16.yaml` | 16 x 16 scalar PEs, output-stationary, operands forwarded one PE per cycle | Formulas (reference and SCALE-Sim) |
| `ws_systolic_16x16.yaml` | 16 x 16 scalar PEs, weight-stationary, operands forwarded, partial sums down each column, double-buffered weights | Formulas |
| `tree_8x8.yaml` | 8 x 8 PEs of 4-wide dot products, weight-stationary, broadcast, an adder tree per column | Formulas |

Any parameter below can be changed on top of a preset. A Coral array must
have 32, 64, 128 or 256 PEs; `mtx_zero_cycles` follows the PE count unless
set.

### Settings outside the module that change its numbers

| Setting | Values | Why it matters |
| --- | --- | --- |
| `top.core.rob.params.rob_entries` | 8 (M3, `m3.yaml`); 16 (the VME RTL) | A tile instruction holds its entry until its tile write lands, so the ROB limits how many MACs are in flight. Undecided for Reef: run both. |
| `top.core.lsu.params.lsu_tile_cost` with `lsu_tile_inst_overhead 1`, `lsu_tile_beat_overhead 0`, `lsu_tile_switch_cycles 1` | off (M3's LSU, `m3_vme.yaml`); on (the VME RTL's LSU) | A one-beat tile load costs 5 cycles on M3's LSU and 3 on the newer one. Undecided for Reef: run both. |
| `top.frontend.params.zvt` | set by `m3_vme.yaml` | Turns the Zvt instructions on |

`matrix_diff.REFERENCE_CORE_PARAMS` holds the VME RTL's values (ROB 16,
`lsu_tile_cost` on), which every RTL comparison uses.

### Output

The JSON summary's `matrix` object has the ops executed by kind, useful and
capacity MACs, `mac_utilization`, array busy cycles, 16-byte memory beats,
16-byte tile reads and writes, and `area_um2` and `energy_pj` (`null` while
their cost parameters are 0).

### Errors

A configuration the engine cannot be stops the run with the parameter's name:
a zero size, lane, array, divider, queue, issue-width or FIFO count, a uop
buffer smaller than a MAC's 4 uops, an unknown dataflow, delivery, reduction
or mapping name, a Coral array that is not 32-256 PEs, or the Coral mapping
with a dataflow other than output-stationary.

## How it works

```
 ELF ─► FuncSim ─────────────► Dispatch ──► matrix module ───────────────────────────┐
        Zvt runs here, not in     │          1. command path: cmd_latency, then the   │
        Spike (frontend/Zvt.h):   │             uop buffer (8 uops, 3 pushed a cycle) │
        mtype, vtype/vl, rd,      │          2. execution, in order:                  │
        tile group, Mem           │             vtmm   → MAC arrays (MatrixFabric)    │
                                  │             vtzero, vtmv → controller             │
        mset* → VSET → vector     │             vtle/vtse → core LSU, 16 B beats,     │
                                  │                         through 2-entry FIFOs     │
                                  │          3. tile scoreboard (mt0-mt15)            │
                                  │          4. complete_latency → core ROB ◄─────────┘
                                  └─ 8 matrix credits; tile loads/stores also take an LSU credit
```

- **Functional side.** Spike does not know Zvt. FuncSim skips Spike for Zvt
  words and applies their effects itself: `mtype` (tm, tk, mtwiden), and
  `vtype`/`vl` through Spike's vector unit, so ordinary vector instructions
  around them run correctly. Tile contents and `vtmv.v.t` results are **not**
  computed.
- **mset\*** are vector configuration (`InstClass::VSET`), timed by the vector
  module like `vsetvli`, as on CoralNPU (`RvvFrontEnd.sv:637-643`).
- **Every other Zvt instruction** is `InstClass::MATRIX`, with the details in
  `Inst::Mtx` (op, tile group, `tm`/`tn`/`tk`, input type).
- **Traps and no-ops** match the RTL: every tile instruction traps on `vill`
  or an unconfigured `mtype` (`RvvFrontEnd.sv:624-633`, `Decode.scala:573-579,
  :1052-1073`). Shapes the RVV decoder rejects retire as no-ops
  (`rvv_backend_decode_unit.sv:92-96`): vl = 0, a misaligned tile group or
  `vs1`/`vs2`, `vtmm` with `tm`, `tn` or `tk` of 0, or a type configuration
  other than int8 {SEW8, x4}, bf16 {SEW16, x2, altfmt} or fp32 {SEW32, x1}.
- **The command path.** An instruction's uops enter an 8-uop buffer
  (`ZVT_RS_DEPTH`) `mtx_cmd_latency` cycles after dispatch, up to 3 a cycle;
  each push checks the room left at the start of the cycle. A MAC needs all
  its 4 uops and leaves the buffer with its last step; vtzero, moves, tile
  loads and stores start on their first uop and leave one per cycle.
- **The controller** serves one instruction at a time and, except for MACs,
  waits for the array to drain. Tile loads run their LSU half as soon as the
  instruction arrives, at most 2 beats ahead of the controller.

## Parameters

Each default is tagged with where it comes from:

- **RTL**: read from CoralNPU `8a5e1b31` at the cited place.
- **TRACE**: measured in a VME RTL simulation at `8a5e1b31` (a waveform or
  N/2N differencing); the program is named.
- **REF**: checked against the cycle-level array reference (D4).
- **PLACEHOLDER**: no source yet.
- **DSE**: a design choice for the sweep; CoralNPU's value is the default.

### Command path

| Parameter | Default | Tag | Meaning |
| --- | --- | --- | --- |
| `mtx_queue_entries` | 8 | DSE | Matrix credits in Dispatch, in instructions: how far the core runs ahead of the engine. CoralNPU has no such instruction limit; its limit is the 8-uop buffer, so 8 credits never bind before it does. With 4, one-uop moves ran at 0.8 per cycle against the RTL's 1. |
| `mtx_cmd_latency` | 4 | TRACE | Dispatch to a uop entering the buffer. A MAC's first step comes 6 cycles after dispatch (its 4 uops take 2 cycles), a move's first uop 5 (`mac_chain`, `move_tv_rob`). |
| `mtx_uops_per_cycle` | 3 | RTL | Uops pushed into the buffer per cycle (`NUM_DP_UOP`). |
| `mtx_uop_buffer_entries` | 8 | RTL + TRACE | The Zvt reservation station (`ZVT_RS_DEPTH`): 2 MACs or 8 one-uop ops. Pushes see the room before that cycle's pops (`rvv_backend.sv:776-807`; 128-PE `mac_indep_tm4`). |
| `mtx_complete_latency` | 1 | TRACE | Result to the core seeing completion: `vmeRtVld` 2 cycles after an op's last controller cycle, retired that cycle (`mac_rob`, `zero_rob`, `move_tv_rob`). |

### Array (`MatrixFabric`)

| Parameter | Default | Tag | Meaning |
| --- | --- | --- | --- |
| `mtx_num_arrays` | 1 | DSE | Independent arrays; MACs into different tiles overlap. No RTL has more than one. |
| `mtx_pe_rows`, `mtx_pe_cols` | 8, 8 | RTL + TRACE | 64 PEs (`NUM_PE = TE*TE*COMPRATIO`). 8 x 16 matches a 128-PE build. |
| `mtx_macs_per_pe_int8` / `_bf16` / `_fp32` | 4 / 2 / 1 | RTL | K elements per PE per cycle: one 32-bit word (KMAX, `RvvFrontEnd.sv:374-379`). |
| `mtx_mapping` | `coral` | RTL + TRACE | `coral`: CoralNPU's PE array takes 256 / PEs steps for a full tile (`PROCESS_DELAY`) of 8 / steps tile rows per half, rows 8-15 in parallel, whatever `tn` is (`cntMax00`, `zvt_pe_array.sv:216-217`); output-stationary only. `generic`: the tile is cut into array-sized blocks (REF). |
| `mtx_dataflow` | `output` | DSE | The stationary operand: `output` (C stays; K streams), `weight` (B stays; M streams), `input` (A stays; N streams). Output and weight are checked (REF); **input is not**. |
| `mtx_row_operand`, `mtx_col_operand` | `broadcast` | DSE, REF | `broadcast` (every PE in the row or column at once) or `forwarded` (one PE per hop: systolic skew). |
| `mtx_hop_latency` | 1 | DSE, REF | Cycles per PE-to-PE hop. |
| `mtx_reduction` | `in_place` | DSE, REF | Where partial sums are added: `in_place`, `chain` (down each column), `tree` (an adder tree per column). |
| `mtx_tree_level_latency` | 1 | DSE, REF | Cycles per adder-tree level. |
| `mtx_pe_pipeline_depth` | 9 | RTL + TRACE | Stages after a MAC's last step: multiply 3 + handshake 1 + adder 3 + block skew 2. A full tile retires 14 cycles after its first step (`mac_chain`). |
| `mtx_accumulate_latency` | 0 | TRACE | Extra accumulator read-modify-write; folded into the pipeline depth. |
| `mtx_acc_raw_cycles` | 4 | RTL + TRACE | Cycles after a MAC starts until a MAC into the same tile may start (hitRaw, `zvt_pe_array.sv:195-207`; `mac_chain`). 0: wait for the earlier MAC's tile write. |
| `mtx_stationary_load_cycles` | 0 | DSE, REF | Loading the stationary operand per pass; 0 is one PE row per cycle. |
| `mtx_double_buffer_stationary` | false | DSE, REF | Load the next pass's operand while this one computes. |
| `mtx_pipeline_across_ops` | true | TRACE | MACs overlap in the pipeline (`mac_indep`: 4.0 cycles per MAC; without overlap 13). |
| `mtx_fabric_clock_divider` | 1 | DSE | Core cycles per array cycle. The clock an array can reach needs PD. |

One MAC of `TM x TN x TK` (`matrix/MatrixFabric.cpp`):

```
Lanes = mtx_macs_per_pe_<type>;  KSteps = ceil(TK / Lanes)
output:  rows <- TM,      cols <- TN,  temporal = KSteps
weight:  rows <- KSteps,  cols <- TN,  temporal = TM
input:   rows <- KSteps,  cols <- TM,  temporal = TN
Passes = coral:   ceil(min(TM, 8) / (8 / Steps)),  Steps = 256 / PEs
         generic: ceil(rows / pe_rows) * ceil(cols / pe_cols)

output stationary:  Busy = Interval = Passes * temporal
weight and input:   Load = stationary_load_cycles, or the rows used
                    Wait = chain ? (rows used - 1) * hop : 0
  single buffered:  Interval = Passes * (temporal + Load + Wait);  Busy = Interval - Wait
  double buffered:  Pair     = max(2 * max(temporal, Load), temporal + Load + Wait)
                    Busy     = Load + temporal + (Passes > 1 ? max(temporal, Load) + ceil((Passes - 2) * Pair / 2) : 0)
                    Interval = ceil(Passes * Pair / 2)

Fill    = (row operand forwarded ? (cols used - 1) * hop : 0)
        + (output stationary and col operand forwarded ? (rows used - 1) * hop : 0)
Drain   = chain: (rows used - 1) * hop;  tree: ceil(log2(rows used * Lanes)) * tree_level
Latency = Fill + Busy + Drain + pe_pipeline_depth + accumulate_latency
Occupancy (the array is unavailable) = pipeline_across_ops ? Interval : Latency
All are multiplied by the clock divider.
```

`Busy` is a lone MAC's time on the array; `Interval` is the spacing of
back-to-back MACs, shorter when the next MAC's stationary load overlaps this
one's compute. A full int8 tile on the defaults is 4 steps at 256 MACs per
cycle.

### Controller and tile storage

| Parameter | Default | Tag | Meaning |
| --- | --- | --- | --- |
| `mtx_ctrl_waits_for_array_drain` | true | RTL + TRACE | vtzero, moves, tile loads and stores wait until no MAC is in flight, and start the cycle after the last MAC's tile write (`zvt_ctrl.sv:105-106, :130-134`; `tile_load_behind_mac`). |
| `mtx_zero_cycles` | 0 (derived) | RTL + TRACE | Cycles vtzero holds the controller. 0 derives it: `PROCESS_DELAY` (256 / PEs) for the Coral mapping, 4 otherwise (`zvt_ctrl.sv:435-445`; `zero_burst` at 64 and 128 PEs). |
| `mtx_move_cycles_per_uop` | 1 | RTL + TRACE | Controller cycles per `vtmv` uop (1, 2 or 4 uops at SEW 8, 16, 32) (`zvt_ctrl.sv:605-610`; `move_tv_burst`, `move_vt_burst`). |
| `mtx_move_latency` | 0 | RTL + TRACE | Extra cycles before a move's result can be read; the tile write is a register (`zvt_mt_reg.sv`; `move_then_mac`). |
| `mtx_tile_write_latency` | 1 | RTL + TRACE | A tile load's last beat to the tile written (`load_then_mac`, `load_then_store`). |
| `mtx_tile_read_latency` | 1 | RTL | Reading a tile for a store: the read is combinational, the 1 is the `vme2lsu` FIFO's register (`mac_then_store`). |
| `mtx_lsu_fifo_entries` | 2 | RTL | The `lsu2vme` and `vme2lsu` FIFOs (`zvt.sv:163-206`): how far tile loads and stores run ahead of the controller. |

Tile loads and stores use the core's LSU, one 16-byte beat per uop, at the
LSU's cost (`lsu_tile_cost`, above).

### Fixed by the ISA

Not parameters; changing them changes Zvt or VLEN: tile edge 16 elements, 16
tiles, the type pairs int8 → int32, bf16 → fp32 and fp32 → fp32, and one
tile (at most 16 x 16 x KMAX) per `vtmm`. An array larger than 16 x 16 is
under-used under this ISA; `mac_utilization` shows how much.

### Cost

| Parameter | Default | Meaning |
| --- | --- | --- |
| `mtx_area_per_pe_um2` | 0 | Area of one PE; total = arrays x PEs x this + tile storage. |
| `mtx_tile_storage_area_um2` | 0 | Area of the tile storage. |
| `mtx_energy_per_mac_pj` | 0 | Energy per useful MAC. |
| `mtx_energy_per_subtile_access_pj` | 0 | Energy per 16-byte tile read or write. |
| `mtx_energy_per_mem_beat_pj` | 0 | Energy per 16-byte memory beat. |

All 0 until PD has sky130 numbers.

## Checking it

### The tests

| Test | Checks |
| --- | --- |
| `test_matrix_correlation.py::test_matches_rtl` | 30 programs against the 64-PE VME RTL (`microbenchmarks/matrix/rtl_vme.csv`), each within its tolerance |
| `test_matrix_correlation.py::test_matches_rtl_at_128_pes` | 28 programs against a 128-PE build (`rtl_vme_pe128.csv`) |
| `test_matrix_correlation.py::test_every_program_is_classified` | Each program is either correlated or a known gap with its reason (`KNOWN_GAP`) |
| `MatrixFabricTest.ReferenceArrays` | The fabric formulas against 1,536 cases of the cycle-level reference (`tests/cpp/matrix/FabricReference.inc`) |
| `test_matrix_microbenchmarks.py` | Every program runs and executes the expected ops; every preset runs `mac_indep` and `gemm_tile` within its peak MAC rate; bad parameters fail by name |

A change to the matrix module has to keep all of them and the M3 golden
passing. A change that moves a correlated program's cost has to say why in
its commit.

### Measuring against the RTL

1. **Build a VME verification simulator.** In a CoralNPU checkout at
   `8a5e1b31`, `git apply` [`coralnpu_vme_verification.patch`](coralnpu_vme_verification.patch)
   and `bazel build -c opt //tests/verilator_sim:vme_core_mini_verification_axi_sim`
   (about 20 minutes). For the 128-PE engine, also apply
   [`coralnpu_pe128.patch`](coralnpu_pe128.patch), in a separate worktree so
   the 64-PE build stays.
2. **Record the model's prediction first** for a new program
   (`heldout_predictions.csv`), then run it:
   ```
   uv run python -m reef_perf.experiments.matrix_diff.main build
   uv run python -m reef_perf.experiments.matrix_diff.main model
   PYTHONPATH=python python3 -m reef_perf.experiments.matrix_diff.main rtl --sim <sim> [--record]
   uv run python -m reef_perf.experiments.matrix_diff.main compare
   ```
   The model steps run in the Docker image, the RTL step on the host.
   Each program is built with N and 2N iterations (`ITERS`), and
   `(ROI(2N) - ROI(N)) / ops` is compared, which cancels the one-time costs
   and most of the M3-against-`8a5e1b31` core difference.
3. **When a difference does not explain itself, read a waveform.** Build the
   program with a small `ITERS` (1-6; `fst2vcd` refuses large traces), run the
   simulator with `--trace_file w.fst`, convert with `fst2vcd`, and sample on
   the rising edge of `clock` under
   `VVmeCoreMiniVerificationAxi.core.VmeCoreMiniVerificationAxi.core`:

   | Signal | What it shows |
   | --- | --- |
   | `score.dispatch.io_rvv_<k>_{valid,ready,bits_pc}` | Dispatch to the RVV backend |
   | `io_debug_rb_inst_<k>_{valid,bits_pc}` | Retirement |
   | `rvvCore.rvvCoreWrapper.core.backend.{rs_valid_dp2zvt,zvt_rs_almost_full,u_zvt_rs.entry_count,pop_zvt2rs}` | The uop buffer: pushes, room, fill, pops |
   | `...backend.vme.zvtPeArray.{canStart,cnt,hitRaw,peCmdVld}` | MAC steps and the uops present |
   | `...backend.vme.zvtCtrl.{isZero,isMv2Vme,isMv2Rvv,isLsu2Vme,isVme2Lsu}` | What the controller is doing |
   | `...backend.vme.{peBusy,vmeRtVld}` | Array busy; completion to the core |
   | `...backend.vme.{uop_lsu2vme_vld,uop_lsu2vme_rdy}` | Tile-load beats from the LSU |
   | `score.lsu.slot.io_uop_{valid,ready,bits_pc}` | The LSU taking an instruction |

### The non-Coral arrays

No RTL has them. `python -m reef_perf.experiments.array_ref.main` regenerates
the reference table; `array_ref.scalesim_check` reruns the SCALE-Sim
comparison (in a throwaway environment; it is not a dependency). The
reference, and so the model, assumes:

- a pass's results go straight to tile storage (each PE writes its own C
  element; for chains and trees, the bottom of each column), with no limit on
  tile write ports;
- skew and loads span only the PEs a tile uses;
- stationary operands load one PE row per cycle down each column, in step
  with that column's wavefront;
- everything outside the array is Coral's: one tile per `vtmm`, the uop
  buffer, one controller.

Input-stationary arrays are not in the reference.

## Open cases

Questions about CoralNPU's behavior, settled or open, roughly by how much
they matter. Settled ones say how. Each was checked against CoralNPU `8a5e1b31` where the source
can answer it.

1. **RTL numbers (settled, C3-C5, D4).** Every Coral value is now from the
   RTL source or a VME waveform. `mtx_queue_entries` (Dispatch's matrix
   credits) has no RTL counterpart; it defaults to 8 so that the 8-uop
   buffer, which does, is the limit.
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
   the matrix module its own issue path into its own model of that 8-uop
   station (`mtx_uop_buffer_entries`), so vector and matrix work never
   contend for dispatch. Fixing this is an interface change between the
   vector and matrix modules.
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
   so `mac_chain` alone cannot tell whether the window is 4 or less; partial
   tiles can, and `mac_chain_tm4` (2 busy cycles) and the 128-PE build
   (1 or 2) both show 4.0 cycles per dependent MAC.
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
17. **Fill, drain and tree formulas (settled for the formulas, D4).**
    Checked against a cycle-level reference on 1,536 array and shape cases
    and against SCALE-Sim on single-fold shapes. No RTL; the assumptions are
    listed under "Checking it".

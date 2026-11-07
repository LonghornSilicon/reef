# LSU cost of Zvt tile beats

For the core module's owner. The matrix module needs the core's LSU to time
Zvt tile loads and stores (`vtle*`, `vtse*`). This note shows that CoralNPU's
LSU charges a tile beat much less than a vector register. It describes the
stopgap now in the model (option 1, off unless a config turns it on) and the
investigation that should replace it (method 2). Nothing here changes M3.

## What the RTL does

Measured on the CoralNPU `8a5e1b31` VME verification simulator
(`vme_core_mini_verification_axi_sim`) by differencing N and 2N iterations
(`python -m reef_perf.experiments.matrix_diff.main`; raw ROIs in
`microbenchmarks/matrix/rtl_vme.csv`). Each program runs 16 tile
instructions per loop iteration with no branch between them, so the
per-instruction cost is the LSU's, not the frontend's:

| Program | Instruction | 16-byte beats | RTL cycles per instruction |
| --- | --- | --- | --- |
| `tile_load_burst` | `vtle8` | 1 | 3.0 |
| `tile_store_burst` | `vtse8` | 1 | 3.0 |
| `tile_load32_burst` | `vtle32` | 4 | 6.0 |
| `tile_store32_burst` | `vtse32` | 4 | 6.0 |

So a tile load or store costs **2 cycles per instruction + 1 per beat**,
loads and stores alike. Three beats more cost three cycles more: after the
first, a beat costs only its DTCM line.

The model's LSU charged each tile beat what it charges a vector register
(`model/core/Lsu.cpp`, `accessVectorRegister`), with the `lsu_vector_*`
values fitted to RTL `vle8`/`vse8` in B14:

| Per beat | Load | Store |
| --- | --- | --- |
| `lsu_vector_{load,store}_reg_overhead` | 3 | 2 |
| DTCM line (memory module) | 1 | 1 |
| `lsu_vector_switch_cycles`, after the instruction's last beat | 1 | 1 |

That is 5 (load) and 4 (store) cycles for a one-beat tile instruction and 17
and 13 for a four-beat one, against 3 and 6 on the RTL. Those vector values
are right for vector registers; tile beats take a cheaper path through the
same LSU.

## Option 1: what is in the model now

The matrix module marks its requests as tile beats, and the LSU gives them
their own cost when asked to.

| File | Change |
| --- | --- |
| `model/common/Interfaces.h` | `VectorLsuRequest::IsTile`, default false. Only the matrix module sets it (`Matrix::runTileLoad`, `runTileStore`). |
| `model/core/Lsu.h` | `lsu_tile_cost` (bool, default **false**), `lsu_tile_inst_overhead`, `lsu_tile_beat_overhead`, `lsu_tile_switch_cycles` (default 0). |
| `model/core/Lsu.cpp` | When `IsTile && lsu_tile_cost`: overhead = `lsu_tile_beat_overhead` on every beat + `lsu_tile_inst_overhead` on the first, and `lsu_tile_switch_cycles` after the last, in place of the vector values. Otherwise unchanged. |
| `configs/machines/m3_vme.yaml` | `lsu_tile_cost: true`, inst 1, beat 0, switch 1. |
| `docs/interfaces.md` | The new field, in "Vector → core (LSU)". |

With these values a tile instruction of n beats holds the LSU 1 + n + 1
cycles, which is 3 for one beat and 6 for four, matching all four programs
above within 2%.

**Why M3 does not change.** Three things each keep the M3 path as it was:

1. `lsu_tile_cost` defaults to false in the code, and `m3.yaml` does not set
   it. With it false, `accessVectorRegister` computes exactly what it did
   before.
2. Only the matrix module sets `IsTile`, and the matrix module only receives
   instructions when Zvt is on (`top.frontend.params.zvt`, set by
   `m3_vme.yaml`). Vector and scalar requests never take the new branch.
3. `tests/golden/m3_results.json`, which pins cycles and stall counts of
   every M3 microbenchmark, is unchanged.

**How to turn it off.** Pass `-p top.core.lsu.params.lsu_tile_cost false`
(or set it in a config after `m3_vme.yaml`). Tile beats then cost a vector
register again. `test_tile_beats_take_their_own_lsu_cost` checks both
settings (6 and 17 cycles per `vtle32`).

**What option 1 does not know.**

- How the 2 cycles per instruction split between before the first beat
  (`lsu_tile_inst_overhead`) and after the last (`lsu_tile_switch_cycles`).
  Throughput cannot tell them apart; latency to the tile write can. The 1 +
  1 split is our guess, copying the vector path's 1-cycle switch.
- Latency: only throughput was fitted. When a tile beat's data reaches the
  tile, or a store completes, still follows the vector rule (data usable at
  memory latency − 2 after the access starts).
- `tile_load_behind_mac` (a MAC, then 8 `vtle8` and a `lw`) now runs 12.5%
  fast (28.0 against 32.0 cycles per iteration). The FIFO hold (case 6 in
  `matrix.md`) or the split above may explain it; not investigated.

## What the core owner is asked to do

1. **Review the interface change**: the `IsTile` field in
   `VectorLsuRequest` and its row in `docs/interfaces.md`.
2. **Review `Lsu.h` and `Lsu.cpp`**: the four parameters, their default-off
   behavior, and their names. The LSU is your module; rename or restructure
   as you like, as long as the matrix module can still mark a tile beat.
3. **Decide where the values live.** They are in `m3_vme.yaml` because they
   only matter with Zvt. If you would rather keep every LSU value in the
   core's own config, they can move.
4. **Take method 2** (below), or tell us the RTL reason if you already know
   it, so the fitted values can be replaced by the mechanism.
5. **Two related findings for the core and frontend:**
   - The VME RTL's core is CoralNPU `8a5e1b31`, with a 16-entry retirement
     buffer (`Parameters.scala:141`); M3 has 8. A Zvt instruction holds its
     entry until its tile write lands, so with 8 entries the ROB, not the
     engine, limits a chain of MACs (7.0 cycles per MAC against 4.0). The
     matrix correlation runs with 16 (`REFERENCE_CORE_PARAMS` in
     `matrix_diff/main.py`); Reef's ROB size is your call
     (`docs/sweep-candidates.md`).
   - `tile_load` and `tile_store` take one branch per tile instruction (5
     instructions per loop). With the LSU's cost removed, the model runs that
     loop at 4.375 cycles per iteration, frontend-bound; the RTL runs it at
     about 3.5. The `8a5e1b31` core's frontend handles this loop faster than
     the M3 model's. That is the frontend owner's; the matrix correlation
     uses the unrolled `*_burst` programs to stay clear of it.

## Method 2: find the mechanism in the RTL

**The question.** CoralNPU's LSU runs tile beats through the same slot as
RVV registers. Why does a tile instruction cost 2 + 1 per beat when a vector
register costs about 4 to 5 each?

**What the RTL source shows** (`hdl/chisel/src/coralnpu/scalar/Lsu.scala`):

- The slot carries an `isVme` flag (set from `LsuOp.isTile`, `:1676`).
- Store data comes from the `vme2lsu` FIFO for a tile and from the RVV
  register file for a vector store, and the vector path also waits for the
  register and mask data to be valid (`:1077-1101`).
- Load data returns on `vmeWriteback` into `lsu2vme` for a tile and through
  the RVV writeback for a vector load (`:1916-1943`).
- There is a tile-specific no-op path (`isTileNoop`, `:2047`).

**Candidate causes** (not checked): the per-register handshakes with the RVV
backend (`rvv2lsu` in, `lsu2rvv` out) that a tile beat skips; the wait for
vector register and mask data; how the slot reloads between instructions
(`LsuSlot.fromLsuUOp`) and between beats of one instruction.

**How to measure it.**

1. Dump waveforms of `tile_load32_burst` and `vec_add_m4` (or a `vle8`
   loop) with the VME simulator's `--trace_file <path>.fst` flag (FST;
   open in GTKWave or Surfer). Use small `ITERS` (`--defsym ITERS=2`) to keep
   the file small; `matrix_diff build` shows how to build with it.
2. For each beat or register, record the cycle the slot accepts it, the DTCM
   request, the data return, and the writeback handshake (`lsu2vme` valid
   and ready for tiles, the RVV writeback for vectors), plus when the slot
   takes the next instruction.
3. Attribute the vector path's extra cycles to those steps.

**How to fold it back.** Replace the fitted `lsu_tile_*` values (or the
`lsu_vector_*` ones too, if the vector fit improves) with parameters named
after the steps, each tagged RTL or TRACE with its waveform. The matrix side
does not change: it keeps setting `IsTile`, and `matrix_diff` re-checks the
four `*_burst` programs plus `gemm_tile`.

**Done when** the four `*_burst` programs still match, the
inst/switch split is set from a waveform rather than guessed, tile-write
latency is checked against the trace, and `tile_load_behind_mac` is within a
few percent or its gap is explained.

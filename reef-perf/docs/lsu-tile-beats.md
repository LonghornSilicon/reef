# LSU cost of Zvt tile beats

For the core module's owner. The matrix module needs the core's LSU to time
Zvt tile loads and stores (`vtle*`, `vtse*`). This note shows what the
CoralNPU VME RTL charges for them, why most of the model's error was the LSU
version rather than anything tile-specific, what is in the model now
(option 1, off unless a config turns it on), and the investigation that
should replace it (method 2). Nothing here changes M3.

## What the RTL does

### Tile and vector beats on the same core

Measured on the CoralNPU `8a5e1b31` VME verification simulator
(`vme_core_mini_verification_axi_sim`) by differencing N and 2N iterations
(`python -m reef_perf.experiments.matrix_diff.main`; raw ROIs in
`microbenchmarks/matrix/rtl_vme.csv`). Every program runs 16 memory
instructions per loop iteration with no branch between them, so the cost is
the LSU's, not the frontend's. The `vec_*_burst` programs have the same
shape as the `tile_*_burst` ones, with RVV instead of Zvt.

| Access | Tile (Zvt) | RTL | Vector (RVV) | RTL |
| --- | --- | --- | --- | --- |
| load, 1 x 16 bytes | `tile_load_burst` (`vtle8`) | 3.0 | `vec_load_burst` (`vle8.v`, m1) | 3.0 |
| load, 4 x 16 bytes | `tile_load32_burst` (`vtle32`) | 6.0 | `vec_load4_burst` (`vle8.v`, m4) | 6.0 |
| store, 1 x 16 bytes | `tile_store_burst` (`vtse8`) | 3.0 | `vec_store_burst` (`vse8.v`, m1) | 4.0 |
| store, 4 x 16 bytes | `tile_store32_burst` (`vtse32`) | 6.0 | `vec_store4_burst` (`vse8.v`, m4) | 7.0 |

Cycles per instruction. On this core:

- **Loads cost 2 cycles per instruction + 1 per 16-byte beat, tile or
  vector alike.** There is no tile-specific load cost.
- **Stores cost the same + 1 for a vector store**: 2 + 1 per beat for a
  tile, 3 + 1 per register for a vector. A tile store's data comes from the
  matrix controller's `vme2lsu` FIFO, a vector store's from the RVV backend
  (see method 2), which plausibly explains the extra cycle; not checked.

### The newer LSU is faster than M3's, for every access

The model's LSU parameters are fitted to the **M3** RTL (`lsu_vector_*` in
B14, `lsu_scalar_*` before). The VME RTL is a newer core. The existing
vector and scalar microbenchmarks, unchanged, on both RTLs (M3 from
`microbenchmarks/rtl_m3.csv`; VME measured 2026-11-07 with the same ELFs,
built by `build_microbenchmark`, retired-instruction counts equal):

| Program | M3 RTL | VME RTL (`8a5e1b31`) | |
| --- | --- | --- | --- |
| `vec_memcpy` | 18467 | 6721 | 2.7x faster |
| `vec_add_m4` | 61544 | 28206 | 2.2x faster |
| `vec_add_m1` | 123025 | 81966 | 1.5x faster |
| `load_stream` (scalar loads) | 8204 | 6158 | 1.3x faster |
| `vec_pingpong` | 8505 | 8509 | same |

The `vec_*_burst` programs on both RTLs and on the model's vector path
(the model runs `m3.yaml` + `m3_vme.yaml`; the M3 RTL is
`rvv_core_mini_verification_axi_sim` from `~/coralnpu-m3` at `72d700ef`,
see `baseline/ASYNC.md`):

| Program | M3 RTL | Model (M3-fitted LSU) | VME RTL (`8a5e1b31`) |
| --- | --- | --- | --- |
| `vec_load_burst` (1 register) | 5.06 | 5.0 | 3.0 |
| `vec_store_burst` (1 register) | 4.06 | 4.0 | 4.0 |
| `vec_load4_burst` (4 registers) | 17.0 | 17.0 | 6.0 |
| `vec_store4_burst` (4 registers) | 13.0 | 13.0 | 7.0 |

The model's LSU matches M3 within 1% on programs it was not fitted to. So
when the model's tile beats were 5 cycles against the VME RTL's 3, the
comparison was M3's LSU against a newer LSU, not a vector path against a
tile path.

The M3-fitted cost, per 16-byte register:

| Per register | Load | Store |
| --- | --- | --- |
| `lsu_vector_{load,store}_reg_overhead` | 3 | 2 |
| DTCM line (memory module) | 1 | 1 |
| `lsu_vector_switch_cycles`, after the instruction's last register | 1 | 1 |

The newer LSU instead pays its overhead once per instruction: a 4-register
access costs 3 more than a 1-register one, not 4 times as much.

## Option 1: what is in the model now

The matrix module marks its requests as tile beats, and the LSU gives them
their own cost when asked to.

| File | Change |
| --- | --- |
| `model/common/Interfaces.h` | `VectorLsuRequest::IsTile`, default false. Only the matrix module sets it (`Matrix::runTileLoad`, `runTileStore`). |
| `model/core/Lsu.h` | `lsu_tile_cost` (bool, default **false**), `lsu_tile_inst_overhead`, `lsu_tile_beat_overhead`, `lsu_tile_switch_cycles` (default 0). |
| `model/core/Lsu.cpp` | When `IsTile && lsu_tile_cost`: overhead = `lsu_tile_beat_overhead` on every beat + `lsu_tile_inst_overhead` on the first, and `lsu_tile_switch_cycles` after the last, in place of the vector values. Otherwise unchanged. |
| `configs/machines/m3_vme.yaml` | `lsu_tile_cost: false` (M3 + Zvt keeps M3's LSU), with the newer LSU's values (inst 1, beat 0, switch 1) listed for `-p top.core.lsu.params.lsu_tile_cost true`. |
| `python/reef_perf/experiments/matrix_diff/main.py` | `REFERENCE_CORE_PARAMS` turns `lsu_tile_cost` on with those values, next to the 16-entry ROB, so the correlation runs the reference core's LSU. |
| `docs/interfaces.md` | The new field, in "Vector → core (LSU)". |

With these values a tile instruction of n beats holds the LSU 1 + n + 1
cycles, which is 3 for one beat and 6 for four, matching the four
`tile_*_burst` programs within 2%.

**What the values mean, given the comparison above.** They are the
`8a5e1b31` LSU's cost for a tile access, not "how much cheaper a tile beat
is than a vector register on M3's LSU". Of the 2-cycle gap on a one-beat
load (5 against 3), all of it is the LSU version; on a store, one cycle is
the LSU version and one is the tile path. They are right for correlating
against the VME RTL. Whether they are right for Reef depends on which LSU
Reef's core has. No RTL has Zvt on M3's LSU, but on the newer core a tile
load costs exactly a vector load and a tile store a vector store less one
cycle. If that holds on M3's LSU, an M3 tile load would cost what an M3
vector load does (5 cycles for one beat, 17 for four: `lsu_tile_cost`
false), and these values would be optimistic by about 40% to 65%. Like the
16-entry ROB, they describe the reference core, so they are on only for
correlation (`REFERENCE_CORE_PARAMS`), and `m3_vme.yaml` keeps M3's LSU
until the core owner decides (below).

**Why M3 does not change.** Three things each keep the M3 path as it was:

1. `lsu_tile_cost` defaults to false in the code, and `m3.yaml` does not set
   it. With it false, `accessVectorRegister` computes exactly what it did
   before; with it false the tile programs reproduce the model's earlier
   cycles exactly.
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
- `tile_load_behind_mac` (a MAC, then 8 `vtle8` and a `lw`) ran 12.5% fast
  until its waveform showed the cause was on the matrix side (MAC pipeline
  depth, command path, controller drain, FIFO pop latency; `matrix.md`, C4).
  It now matches, so the 1 + 1 split does not show there; it is still a
  guess.

## What the core owner is asked to do

1. **Decide which LSU Reef models.** The LSU parameters fit M3's RTL; Zvt
   exists only on the newer core, whose LSU is 1.3x to 2.7x faster on the
   programs above and charges its overhead per instruction instead of per
   register. Reef's v2 numbers depend on which one Reef's core has. If it
   is the newer one, the vector and scalar fits need redoing against it, and
   the tile cost can follow from the same model.
2. **Confirm where the tile values live.** They are now on **for
   correlation only**: `REFERENCE_CORE_PARAMS` (`matrix_diff/main.py`) turns
   them on next to the 16-entry ROB, and `m3_vme.yaml` keeps M3's LSU
   (`lsu_tile_cost` false), so M3 + Zvt prices a tile beat like a vector
   register. That is right if Reef keeps M3's LSU. If Reef takes the newer
   LSU, turn them on in `m3_vme.yaml` (or the core's own config) and refit
   the vector path to match.
3. **Review the interface change**: the `IsTile` field in
   `VectorLsuRequest` and its row in `docs/interfaces.md`. The store
   difference (1 cycle) is a real tile-specific cost on the newer core, so
   the LSU needs to know a beat is a tile beat in either choice above.
4. **Review `Lsu.h` and `Lsu.cpp`**: the four parameters, their default-off
   behavior, and their names. Rename or restructure as you like, as long as
   the matrix module can still mark a tile beat.
5. **Take method 2** (below), or tell us the RTL reason if you already know
   it, so the fitted values can be replaced by the mechanism.
6. **Two related findings for the core and frontend:**
   - The VME RTL's core has a 16-entry retirement buffer
     (`Parameters.scala:141`); M3 has 8. A Zvt instruction holds its entry
     until its tile write lands, so with 8 entries the ROB, not the engine,
     limits a chain of MACs (7.0 cycles per MAC against 4.0). The matrix
     correlation runs with 16 (`REFERENCE_CORE_PARAMS`); Reef's ROB size is
     your call (`docs/sweep-candidates.md`).
   - `tile_load` and `tile_store` take one branch per tile instruction (5
     instructions per loop). With the LSU's cost removed, the model runs that
     loop at 4.375 cycles per iteration, frontend-bound; the RTL runs it at
     about 3.5. The newer core's frontend handles this loop faster than the
     M3 model's. That is the frontend owner's; the matrix correlation uses
     the unrolled `*_burst` programs to stay clear of it.

## Method 2: find the mechanism in the RTL

**The questions.**

1. Why does the newer LSU charge 2 per instruction + 1 per register, where
   M3's charges about 4 per register? This is a difference between two LSU
   versions, and it covers vector and scalar accesses too.
2. Why is a tile store 1 cycle per instruction cheaper than a vector store
   on the newer LSU, when tile and vector loads cost the same?

**What the RTL source shows** (`hdl/chisel/src/coralnpu/scalar/Lsu.scala`
at `8a5e1b31`):

- Every memory instruction becomes one `LsuUOp` (`:2046`). For a tile, its
  length comes from the tile element width alone (`:296-302`: 8-bit is one
  16-byte register, 32-bit is four), so a `vtle32` enters the slot once and
  the slot walks its four beats.
- The slot takes the next instruction only when the current one is done
  (`io.uop.ready := stateFromAction.isDone`, near `:1962`): a cost paid once
  per instruction.
- The slot carries an `isVme` flag (set from `LsuOp.isTile`, `:1676`).
  Store data comes from the `vme2lsu` FIFO for a tile and from the RVV
  register file for a vector store; the vector path also waits for the
  register and mask data to be valid (`:1077-1101`). A tile's mask is all
  ones (`Mux(isVmeInst, true.B, ...)`), so there is no mask to wait for.
- Load data returns on `vmeWriteback` into `lsu2vme` for a tile and through
  the RVV writeback (`lsu2rvv`) for a vector load (`:1902-1918`).
- There is a tile-specific no-op path (`isTileNoop`, `:2047`).

**Where to start: the CoralNPU commits between M3 and `8a5e1b31`.** In
`~/coralnpu-pinned`, `git log 72d700ef..8a5e1b31 --
hdl/chisel/src/coralnpu/scalar/Lsu.scala` lists the LSU's history since M3.
Three changes match the measured costs. The matching is from the commit
messages and the cost shapes, not from waveforms:

| Commit | What it says | What it would explain |
| --- | --- | --- |
| `15425970` "Initial implementation of LsuV3"; `0b2924d8` "Switch to LsuV3"; `d57a9342` "Decommission LsuV2" | LsuV3 "redesigns the LSU to improve pipelining within the same instruction". M3 runs LsuV2. | Question 1: the newer LSU takes a whole instruction as one operation and streams its 16-byte registers one per cycle, so its overhead is paid once per instruction (about 2 cycles + 1 per register) instead of per register (about 4 each on M3). A 4-register load goes from 17 cycles to 6. |
| `3ca02cc6` "LSU skip handshake" (fixed in `ae57d1bd`) | "Skip vector data handshake for US (including WR and Mask) and CS loads. This allows LSU to start and complete bus tx before RVV starts asking." | Question 1 for loads: a one-register load goes from 5 to 3. Question 2: the skip covers loads only, so a vector store still waits for its data from the RVV backend (one cycle more than a load), while a tile store's data comes from the matrix controller's `vme2lsu` FIFO and costs the same as a load. |
| `76418e1a` "Precompute bus transaction size and offset during instruction initialization"; `912ba1b5` "Speculatively evaluate next rowAddr and leadIndex in LsuSuperSlot" | Timing work inside the LsuV3 slot. | Probably what keeps one register per cycle inside an instruction; not tied to a cycle. |

The VME tile path itself came in with `e37e48fd` ("Wire VME LSU load and
store channels"), `e2e1d8ab` and `27d00090` ("Implement VME tile
load/store"), after LsuV3, which is why no RTL has Zvt on M3's LsuV2. The
LsuV3 message also lists "add a second super slot to improve pipelining
between instructions" as later work; `57375f85` ("prepare for dual execution
slots") suggests it is not in `8a5e1b31`, so the model's one-slot LSU still
holds. Check both in the waveform.

**Candidate causes** (not checked): for question 1, LsuV2 stepping through
a vector instruction's registers with per-register handshakes with the RVV
backend (`rvv2lsu` in, `lsu2rvv` out), against LsuV3's per-instruction walk
and the load handshake skip above; for question 2, the store's wait for RVV
register and mask data, which a tile store's FIFO data skips.

**How to measure it.**

1. Dump waveforms of `vec_store_burst` and `tile_store_burst`, and of
   `vec_load4_burst`, with the VME simulator's `--trace_file <path>.fst` flag
   (FST; open in GTKWave or Surfer). Use small `ITERS` (`--defsym ITERS=2`)
   to keep the files small; `matrix_diff build` shows how to build with it.
   For question 1, run the same `vec_*_burst` ELFs on the M3 RTL simulator
   (`~/coralnpu-m3/bazel-bin/tests/verilator_sim/rvv_core_mini_verification_axi_sim`,
   which also takes `--trace_file`) for the old LSU's side; their cycle
   counts are in the table above.
2. For each register or beat, record the cycle the slot accepts the
   instruction, each DTCM request, the data arrival (RVV or `vme2lsu`), the
   data return or writeback handshake, and when the slot takes the next
   instruction.
3. Attribute each extra cycle to one of those steps.

**How to fold it back.** Replace the fitted `lsu_tile_*` values, and the
`lsu_vector_*` ones if Reef takes the newer LSU, with parameters named after
the steps, each tagged RTL or TRACE with its waveform. The matrix side does
not change: it keeps setting `IsTile`, and `matrix_diff` re-checks the
`tile_*_burst` and `vec_*_burst` programs plus `gemm_tile`.

**Done when** the eight `*_burst` programs match on the LSU version Reef
models, the inst/switch split is set from a waveform rather than guessed,
and tile-write latency is checked against the trace.

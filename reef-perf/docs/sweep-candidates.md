# Sweep candidates

Design choices worth varying in the Nov–Dec architecture sweep (v2 spec due
Dec 6), collected as we come across them. Add a row whenever a correlation
step, an RTL read or a result shows that a choice moves performance, area or
energy. This file only proposes; the module's owner decides what is swept and
what Reef ships.

Each row gives:

- **Knob**: the parameter path, or a short name if no parameter exists yet.
- **Reference**: the value in M3 (`configs/machines/m3.yaml`) and in the
  CoralNPU VME RTL (`8a5e1b31`) when they differ.
- **Why it matters**: what we saw, with the benchmark or RTL source.
- **Status**: **ready** (sweep it today with `-p`), **model work** (the
  model needs a change first), or **decision** (needs the spec or another
  owner before it can be a knob).

Run a point with `-p` on top of the three configs, for example:

```
reef_perf --elf x.elf -c configs/machines/m3.yaml \
    -c configs/machines/m3_vme.yaml -c configs/matrix/coral_vme.yaml \
    -p top.core.rob.params.rob_entries 16
```

## Core (owner: core)

| Knob | Reference | Why it matters | Status |
| --- | --- | --- | --- |
| `top.core.rob.params.rob_entries` | M3 8 (`Parameters.scala:95` at the M3 checkout); VME RTL 16 (`Parameters.scala:141`) | A Zvt instruction holds its entry until its tile write lands (`RetirementBuffer.scala:540-549`), so the ROB bounds the MACs in flight. `mac_chain`: 7.0 cycles/MAC with 8 entries, 4.0 (array-bound, matches RTL) with 16. Found in B4, 2026-11-07. | ready |
| `top.core.lsu.params.lsu_queue_entries` | 4 (`Lsu.scala:835`) | Tile loads and stores use the core's LSU, one 16-byte beat at a time; scalar and vector memory traffic shares it. Not measured for matrix work yet. | ready |
| `top.core.lsu.params.lsu_tile_cost` and `lsu_tile_*` | Off in the code, `m3.yaml` and `m3_vme.yaml` (M3's LSU); on, with 1 + 0 per beat + 1, only in `matrix_diff`'s `REFERENCE_CORE_PARAMS`. VME RTL: 2 cycles per tile instruction + 1 per beat | How fast the LSU moves tile data, and so how much of a GEMM's time goes to operand loads. Off, a one-beat tile load costs 5 cycles (M3's LSU, as for a vector register); on, 3 (the newer LSU). The difference is the LSU version, not the tile path (`docs/lsu-tile-beats.md`). A dedicated matrix port (below) is the structural alternative. | ready |
| LSU version: M3's vs CoralNPU `8a5e1b31`'s | Model fits M3 (matches the M3 RTL within 1% on `vec_*_burst`) | The newer LSU charges its overhead once per instruction instead of per register: 6 cycles for a 4-register `vle8` against M3's 17, and 1.3x to 2.7x faster on the existing vector and scalar memory programs. Zvt only exists on the newer core. Affects every memory-bound kernel. Owner: core. | decision (core owner) |
| Frontend (fetch, branch redirect) | M3 model vs CoralNPU `8a5e1b31` | A loop with a branch every 5 instructions runs at 4.375 cycles per iteration in the model and about 3.5 on the VME RTL (`tile_load`), so the newer frontend is faster. Matters for kernels with short inner loops. Owner: frontend. | decision (frontend owner) |
| Tile store holds the LSU | CoralNPU's LSU takes the address and waits for the data (`Lsu.scala:1077-1101`) | The model books the LSU only once data is ready, so it underestimates contention (matrix.md case 5). Matters if matrix stores overlap other memory traffic. | model work |

## Matrix (owner: matrix)

| Knob | Reference | Why it matters | Status |
| --- | --- | --- | --- |
| Array preset (`configs/matrix/*.yaml`): `mtx_pe_rows`, `mtx_pe_cols`, `mtx_dataflow`, `mtx_row_operand`, `mtx_col_operand`, `mtx_reduction` | Coral: 8 x 8 PEs x 4 int8 lanes, output-stationary, broadcast, in place | The main fabric choice. Coral has RTL behind it (64 and 128 PEs); the systolic and tree presets have formulas checked against a cycle-level reference and SCALE-Sim, but no RTL, and their connection to Zvt (one tile per instruction, the uop path) is assumed (D4). Under Zvt one `vtmm` is at most 16 x 16 x KMAX, so arrays bigger than that are under-used (`mac_utilization` shows it). | ready (Coral: RTL; others: formula-checked) |
| Coral array size: `mtx_pe_rows` x `mtx_pe_cols` with `mtx_mapping: coral`, and `mtx_zero_cycles` = 256 / PEs | 64 PEs | Fewer steps per tile. 128 PEs matches a 128-PE CoralNPU build (matrix.md, D4); 32 is the formula's other buildable size; 256 is out of scope. | ready (64, 128) |
| `top.matrix.params.mtx_uop_buffer_entries` | 8 (`ZVT_RS_DEPTH`) | How many uops wait for the engine: 2 MACs. Limits small-tile MAC rates once the array is fast. | ready |
| `top.matrix.params.mtx_num_arrays` | 1 | MACs to different tiles overlap with more arrays. `mac_indep` is array-bound at 4.0 cycles/MAC on one array. | ready |
| `top.matrix.params.mtx_acc_raw_cycles` | 4 (hitRaw, `zvt_pe_array.sv:195-207`) | Whether a MAC can accumulate into a tile still in the pipeline. 0 (no forwarding) makes `mac_chain` 10 cycles/MAC instead of 4. Matters for any GEMM that accumulates over K into one tile. | ready |
| `top.matrix.params.mtx_queue_entries` | 4 instructions; CoralNPU's Zvt reservation station holds 8 uops = 2 `vtmm` (`ZVT_RS_DEPTH`) | How far the core can run ahead of the engine. | ready (counted in instructions, not uops) |
| `mtx_pe_pipeline_depth`, `mtx_fabric_clock_divider` | 9 (multiply 3 + handshake 1 + adder 3 + block skew 2), 1 | Deeper pipelines or a slower fabric clock trade latency for fmax; with `mtx_acc_raw_cycles` and the ROB they set how many MACs must be in flight. fmax per design needs PD (case 14). | ready; fmax is a decision |
| `mtx_double_buffer_stationary`, `mtx_stationary_load_cycles` | Not used by Coral (output-stationary) | Load cost of weight/input-stationary arrays. | ready |
| Weight reuse across `vtmm` | None: each `vtmm` reloads its stationary operand | Penalizes weight- and input-stationary arrays under Zvt's one-tile-per-instruction (case 12). | decision (spec) |
| `top.matrix.params.mtx_lsu_fifo_entries` | 2: `lsu2vme`/`vme2lsu` (`zvt.sv:163-206`) | How far tile loads and stores run ahead of the controller (B3). | ready |
| Area and energy (`mtx_area_per_pe_um2`, `mtx_energy_*`) | 0 (left out of the summary) | Needed to weigh designs, not just rank them by cycles. | decision (PD sky130 numbers) |

## Across modules

| Knob | Reference | Why it matters | Status |
| --- | --- | --- | --- |
| Shared vs separate RVV dispatch for Zvt | CoralNPU sends Zvt uops through the RVV backend's dispatch (`rvv_backend_dispatch.sv:498-512`); the model gives matrix its own path | With sharing, vector and matrix work contend (case 4). Matters for kernels that mix vector and matrix ops, such as attention and layernorm around GEMMs. | model work (vector + matrix interface) |
| Dedicated matrix memory port | None: tile loads use the core's LSU | A matrix-owned port would take tile traffic off the LSU (case 13). | model work (core, memory, matrix interface) |

## Fixed for now

Changing these changes the ISA or VLEN, so they are outside this sweep unless
the spec reopens them: tile edge 16 (`TE = VLEN/8`), 16 tiles, the supported
type pairs (int8 → int32, bf16 → fp32, fp32 → fp32), one tile per `vtmm`.

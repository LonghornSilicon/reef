# Matrix (Zvt) microbenchmarks

Programs that exercise the CoralNPU Zvt matrix extension (spec v0.4, CoralNPU
8a5e1b31). They only run on a CoralNPU build with VME, not on M3. Like the
frontend set, they are not part of the M3 correlation report:
`list_microbenchmarks()` only globs `microbenchmarks/*.S`.

`zvt.inc` defines the Zvt instructions as `.insn` macros.

| Microbenchmark | Measures |
| --- | --- |
| `mac_chain` | Dependent-MAC rate: `vtmmu` into one accumulator |
| `mac_indep` | MAC throughput: `vtmmu` rotating over four accumulators |
| `tile_load`, `tile_store` | `vtle8` / `vtse8` throughput through the LSU |
| `tile_load_burst`, `tile_store_burst` | `vtle8` / `vtse8` unrolled, 16 per loop branch: the LSU's cost of a one-beat tile instruction, without the frontend |
| `tile_load32_burst`, `tile_store32_burst` | `vtle32` / `vtse32` unrolled: 4 beats each, which separates the per-instruction from the per-beat cost (`docs/lsu-tile-beats.md`) |
| `vec_load_burst`, `vec_store_burst`, `vec_load4_burst`, `vec_store4_burst` | The same shapes with RVV (`vle8.v`/`vse8.v` at LMUL 1 and 4): the vector reference for the four tile `*_burst` programs, so the two LSU paths compare on one core. No matrix instructions |
| `mac_{chain,indep}_tm4`, `mac_chain_tn4`, `mac_indep_tn4`, `mac_indep_tm8tn8` | Partial tiles (held out): the PE-busy count for small tm and tn, and the hitRaw window when the array is busy fewer than 4 cycles |
| `mac_{chain,indep}_fp32`, `mac_{chain,indep}_bf16` | fp32 `vtfmm.tvv` and bf16 `vtfmm.alt` MACs (held out) |
| `vtmm_misaligned` | `vtmmu` with misaligned operands, which retire as no-ops, next to one with odd but aligned operands (held out, B2) |
| `vtdiscard_after_mac` | `vtmmu` + `vtdiscard`: the core's vtdiscard interlock, which the model does not have (held out, B5) |
| `gemm_multi`, `gemm_longk` | 32 x 32 x 64 and 16 x 16 x 256 int8 GEMMs (held out) |
| `tile_load_behind_mac` | Tile loads whose controller half waits for a MAC to drain while the LSU runs ahead through the 2-entry `lsu2vme` FIFO |
| `gemm_tile` | A 16x16x64 int8 tile GEMM with operand loads and read-back |
| `zero_burst` | `vtzero` throughput (C4) |
| `move_tv_burst`, `move_tv8_burst`, `move_vt_burst`, `move_vt8_burst` | `vtmv.t.v` / `vtmv.v.t` throughput at 4 uops (32-bit) and 1 uop (8-bit) per move (C4) |
| `move_vt_use` | Latency from `vtmv.v.t` to a `vmv.x.s` of its result, carried around the loop (C4) |
| `load_then_mac`, `move_then_mac`, `mac_then_store`, `load_then_store` | Tile write and read latencies: each op depends on the one before through the same tiles (C4) |
| `mac_rob`, `zero_rob`, `move_tv_rob` | One matrix op + 15 `addi`: ROB-bound, so the op's dispatch-to-retire time, including `mtx_complete_latency`, sets the loop (C5) |
| `mac_rob_tm4`, `mac_rob_tn4`, `mac_rob_tm8tn8` | `mac_rob` with partial tiles: a partial-tile MAC's dispatch-to-retire time (D4) |
| `mac_queue` | 4 independent `vtmmu` + 8 `addi`: how far dispatch runs ahead of the engine (C4) |
| `illegal_shapes` | Tile instructions with vl = 0, which retire as no-ops |

Run them with Zvt on: `-c configs/machines/m3.yaml -c
configs/machines/m3_vme.yaml`. With `m3.yaml` alone they trap, as on M3.

The looping programs (all but `gemm_tile` and `illegal_shapes`) take their loop count
from the `ITERS` assembler symbol (`--defsym ITERS=<n>`), so the
`matrix_diff` experiment can build N and 2N and take the per-op difference.
`microbenchmarks/matrix/rtl_vme.csv` holds those programs' ROIs on the
CoralNPU `8a5e1b31` VME verification simulator, at N and 2N;
`rtl_vme_pe128.csv` the same on a 128-PE build (docs/matrix.md, D4). All but
`tile_store` end by reading their last result back into a scalar, so the model's `ROI_END`
waits for it; `tile_store` and `gemm_tile` end on a store, which the model
cannot wait for (docs/matrix.md, open case 2).

The operands of `vtmmu` in `mac_chain` and `mac_indep` are never loaded: the
model does not time tile or register contents. On the RTL that is also fine;
the results are just meaningless.

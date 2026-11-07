# Matrix (Zvt) microbenchmarks

Programs that exercise the CoralNPU Zvt matrix extension (spec v0.4, CoralNPU
8a5e1b31). They only run on a CoralNPU build with VME, not on M3. Like the
frontend set, they are not part of the M3 correlation report:
`list_microbenchmarks()` only globs `microbenchmarks/*.S`.

`zvt.inc` defines the Zvt instructions as `.insn` macros.

| Microbenchmark | Measures |
| --- | --- |
| `mac_chain` | MAC latency: dependent `vtmmu` into one accumulator |
| `mac_indep` | MAC throughput: `vtmmu` rotating over four accumulators |
| `tile_load`, `tile_store` | `vtle8` / `vtse8` throughput through the LSU |
| `gemm_tile` | A 16x16x64 int8 tile GEMM with operand loads and read-back |
| `illegal_shapes` | Tile instructions with vl = 0, which retire as no-ops |

Run them with Zvt on: `-c configs/machines/m3.yaml -c
configs/machines/m3_vme.yaml`. With `m3.yaml` alone they trap, as on M3.

`mac_chain`, `mac_indep`, `tile_load` and `tile_store` take their loop count
from the `ITERS` assembler symbol (`--defsym ITERS=<n>`), so the
`matrix_diff` experiment can build N and 2N and take the per-op difference.
`microbenchmarks/matrix/rtl_vme.csv` holds those programs' ROIs on the
CoralNPU `8a5e1b31` VME verification simulator, at N and 2N. The first three
end by reading their last result back into a scalar, so the model's `ROI_END`
waits for it; `tile_store` and `gemm_tile` end on a store, which the model
cannot wait for (docs/matrix.md, open case 2).

The operands of `vtmmu` in `mac_chain` and `mac_indep` are never loaded: the
model does not time tile or register contents. On the RTL that is also fine;
the results are just meaningless.

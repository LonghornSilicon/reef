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

There are no RTL numbers yet (no `rtl_vme.csv`). To get them, build a VME
*verification* simulator (`vme_core_mini_verification_axi_sim`; docs/matrix.md)
and measure each ELF as roadmap ticket B01 does. Until then
`tests/python/integration/test_matrix_microbenchmarks.py` only checks that
they run and that the model's results are self-consistent.

The operands of `vtmmu` in `mac_chain` and `mac_indep` are never loaded: the
model does not time tile or register contents. On the RTL that is also fine;
the results are just meaningless.

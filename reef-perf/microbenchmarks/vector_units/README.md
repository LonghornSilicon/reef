# Vector functional unit microbenchmarks

One dependent chain and one independent stream per M3 vector functional unit
type, plus LMUL=4 streams for how instructions split into uops. They
calibrate the vector module's functional unit parameters (roadmap B15) and
are not part of the correlation report: `list_microbenchmarks()` only globs
`microbenchmarks/*.S`, so the sweep, the correlation experiment and the
golden file skip this directory.

Each loop runs 250 times, 8 vector instructions per iteration (6 at LMUL=4),
starts on a 16-byte boundary and uses `vl` = VLMAX at e32 (e16 for the
widening reductions). `rtl_m3.csv` holds the RTL ROI cycles, measured as in
roadmap ticket B01 at CoralNPU `M3-2026-04-27`;
`tests/python/integration/test_vector_unit_microbenchmarks.py` checks the
model against them, within 10%. The `.S` files were generated from one
template; edit them by hand from now on.

A chain costs the unit's latency + 2 cycles per instruction: dispatch, the
unit, a cycle in the result FIFO, then retire, and the dependent instruction
dispatches in that retire cycle. An independent stream costs the cycles a uop
keeps its unit busy, when that is more than the frontend's limit of about
1.4 cycles per instruction.

| Unit (count, reservation station) | Microbenchmark | RTL cycles/instruction | Parameter |
| --- | --- | --- | --- |
| ALU (2, 8) | `vadd_chain`, `vand_chain` | 4, 3 | `vec_addsub_latency: 2`, `vec_alu_latency: 1` |
| | `vadd_m4_indep` | 2.0 (4 uops on 2 ALUs) | |
| | `vmseq_m4_indep` | 4.0: compares run only on the first ALU | |
| MUL (2, 8) | `vmul_chain`, `vmacc_chain` | 4, 4 | `vec_mul_latency: 2` |
| | `vmul_m4_indep` | 2.0 | |
| DIV (1, 4) | `vdiv_chain`, `vdiv_indep` | 15, 13 (31-bit dividends) | `vec_div_cycles_per_uop: 13` |
| | `vdiv_small_chain`, `vdiv_small_indep` | 5, 3 (1-bit dividends): not modeled | |
| | `vfdiv_chain`, `vfdiv_indep` | 11, 8 | `vec_fdiv_latency: 9`, `vec_fdiv_cycles_per_uop: 8` |
| FMA (2, 8) | `vfadd_chain`, `vfmacc_chain` | 6, 6 | `vec_fp_latency: 4` |
| Permute/reduction (1, 4) | `vredsum_chain`, `vredsum_m4_indep` | 5, 4.0 (1 uop per cycle) | `vec_reduce_latency: 3` |
| | `vwredsum_indep`, `vwredsum_m4_indep` | 2.0, 8.0 (2 cycles per uop) | `vec_widening_reduce_cycles_per_uop: 2` |
| | `vslidedown_chain`, `vslidedown_indep`, `vrgather_chain` | 5, 5, 5: not pipelined | `vec_permute_latency: 3`, `vec_permute_cycles_per_uop: 5` |
| | `vfredusum_chain`, `vfredusum_indep` | 15, 12 | `vec_fp_reduce_latency: 13`, `vec_fp_reduce_cycles_per_uop: 12` |

`vadd_indep`, `vmul_indep`, `vredsum_indep`, `vfadd_indep` and `vcpop_indep`
are limited by fetch and the scalar retirement buffer, not by their unit; the
model is 8-9% fast on them.

The integer divider retires 3 quotient bits per cycle after skipping the
dividend's leading zeros, and reuses its previous result when the next divide
has the same operands (`rvv_backend_div_unit_divider.sv`). The divide
microbenchmarks therefore divide by -1, so every divide sees new operands.
The model has no operand values and uses the 31-bit time; the two
`vdiv_small_*` rows are marked `not modeled` in `rtl_m3.csv` and the test
skips them.

Build one by hand (the sweep doesn't), inside the Docker image:

```sh
mkdir -p build/microbenchmarks/vector_units
uv run python -c "from pathlib import Path; from reef_perf.microbenchmarks import build_microbenchmark; print(build_microbenchmark('vector_units/vadd_chain', Path('build/microbenchmarks')))"
```

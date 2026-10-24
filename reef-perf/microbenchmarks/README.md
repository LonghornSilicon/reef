# Microbenchmarks

Small assembly programs, each stressing one feature, plus `gemv_int8`, a
kernel shaped like LLM decode. They link to the Reef M3 default memory map
(code at `0x0`, data at `0x10000`) and end with `mpause`, so the same ELF runs
on the simulator and on the M3 RTL simulator.
Each loop is bracketed by `csrr mcycle` into `s10`/`s11`, so its RTL cycle
count is `s11 - s10`; the model reports the same difference as `roi_cycles`.
`rtl_m3.csv` holds each microbenchmark's ROI cycles and instruction count on
the M3 RTL (roadmap ticket B01), which the correlation experiment compares
against.

| Microbenchmark | Stresses |
| --- | --- |
| `alu_indep`, `alu_chain` | Fetch/dispatch width; ALU latency |
| `branch_tight`, `branch_group` | Taken-branch cost; a branch ends the dispatch group |
| `mul_chain`, `div_chain` | Multiplier latency; data-dependent divider |
| `fp_chain`, `fp_mix` | FPU latency; FP dispatches alone |
| `load_stream`, `load_chain`, `addr_forward` | LSU throughput; load-to-use latency; address forwarding |
| `vec_add_m1`, `vec_add_m4`, `vec_memcpy` | Vector ALU and vector memory at different LMUL |
| `vec_pingpong` | Scalar to vector to scalar round trip (`vmv.s.x`, `vmv.x.s`) |
| `gemv_int8` | int8 matrix-vector product (widening multiply + reduction) |

To add one, drop a `<name>.S` file here that uses `common.inc`, measure it on
the RTL and add its row to `rtl_m3.csv`. The tests, experiments and golden file pick it up
automatically (regenerate the golden file, see
`tests/python/integration/test_golden.py`).

`reef_perf.microbenchmarks` assembles them with `riscv64-linux-gnu-as`/`ld`.

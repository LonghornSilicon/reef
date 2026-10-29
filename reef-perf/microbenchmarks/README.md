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
| `gemv_int8` | int8 matrix-vector product (widening multiply + reduction) |

To add one, drop a `<name>.S` file here that uses `common.inc`, measure it on
the RTL and add its row to `rtl_m3.csv`. The tests, experiments and golden file pick it up
automatically (regenerate the golden file, see
`tests/python/integration/test_golden.py`).

`reef_perf.microbenchmarks` assembles them with `riscv64-linux-gnu-as`/`ld`.

## Parameterized RTL training and testing sweeps

Every `.S` file (including those in `frontend/`) now has overridable
assembler constants. The default values reproduce the original workload, so
ordinary `build_microbenchmark()` calls and the original `rtl_m3.csv` remain
valid. For example, `alu_chain.S` takes `ITERATIONS` and `ADDS`:

```sh
riscv64-linux-gnu-as -march=rv32imf_zicsr_zifencei_zve32f -mabi=ilp32f \
    -I microbenchmarks --defsym=ITERATIONS=750 --defsym=ADDS=12 \
    microbenchmarks/alu_chain.S -o /tmp/alu_chain.o
```

`microbenchmarks/sweeps.json` defines two *nonoverlapping* sets of parameters
for **all 20** benchmarks (15 main and 5 frontend): 40 training configurations
and 40 held-out testing configurations. Use smaller or larger valid values as
needed, but keep the two sets disjoint. Values are checked for positivity,
array/buffer sizes, and GEMV vector-length limits.

### Generate RTL CSVs

Build the M3 verification Verilator simulator in the **CoralNPU** source tree
at tag `M3-2026-04-27`, as described in
[`docs/roadmap.md`](../docs/roadmap.md) (ticket B01):

```sh
cd /path/to/CoralNPU
bazel build -c opt //tests/verilator_sim:rvv_core_mini_verification_axi_sim
```

In an environment where `riscv64-linux-gnu-as`, `riscv64-linux-gnu-ld`,
`uv`, and the resulting simulator executable are accessible (such as the
Reef Docker environment with the simulator also mounted), run from
`reef-perf/`:

```sh
# Inspect all planned configurations; no toolchain or RTL required.
uv run python -m reef_perf.experiments.correlation.rtl_sweep --list

# Optional assembly/link verification of all 80 parameter configurations.
uv run python -m reef_perf.experiments.correlation.rtl_sweep --compile-only

# Build + run each ELF on Verilator; extract cycle counts from --instr_trace.
uv run python -m reef_perf.experiments.correlation.rtl_sweep \
    --simulator /absolute/path/to/rvv_core_mini_verification_axi_sim
```

The script writes **`microbenchmarks/rtl_m3_train.csv`** and
**`microbenchmarks/rtl_m3_test.csv`**, ready to commit with the PR. Each row
identifies a microbenchmark and variant, its parameters (numeric columns),
`rtl_roi_cycles`, and `rtl_retired_insts`. Missing parameters are blank.
The original single-workload `rtl_m3.csv` is kept unchanged for the existing
correlation experiment. Train/test files intentionally contain multiple rows
per benchmark and therefore should **not** be passed to the legacy
`load_rtl_results()` function, which expects one row per benchmark.

Build artifacts and raw simulator traces are saved in ignored
`results/rtl_sweep/elfs/` and `results/rtl_sweep/logs/`. All runs must complete
and reach `mpause` before either generated CSV is published. If any assembly,
Verilator run, or trace validation fails, the script stops with an error rather
than producing fabricated or partial measurements. Options include
`--split train`, `--split test`, `--timeout SECONDS`, `--sweeps FILE`,
`--out DIR`, and `--csv-dir DIR`. Keep the train/test datasets separate when
using them to calibrate or evaluate `reef_perf`.

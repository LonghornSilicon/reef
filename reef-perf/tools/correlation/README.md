# Correlation tools

Helpers for comparing the model with the M3 RTL beyond the microbenchmarks:
microbenchmark variants with their RTL measurements, a script that compares
model builds against them, and a way to read RTL waveforms as per-cycle
tables.

| Path | What it is |
| --- | --- |
| `variants/vadd_loop/` | `vsetvli` + `vadd.vv` loop variants (29) |
| `variants/vec_add/` | `vec_add_m1` (`va1_*`) and `vec_add_m4` (`va4_*`) with parts of the loop removed (17) |
| `rtl_variants.csv` | ROI cycles of each variant on the M3 RTL, and its loop iterations |
| `compare_rtl.py` | runs the microbenchmarks and variants on model builds and prints the error against the RTL |
| `fstr.c` | a minimal reader for the simulator's FST waveforms |
| `wave_table.py` | prints chosen waveform signals as one row per clock cycle |

The variants use the microbenchmarks' `common.inc` and `link.ld`.

## The variants

**`vadd_loop`**: isolated why `vsetvli` plus `vadd.vv` costs 3 cycles per
iteration on the RTL while each alone costs nothing (deviation U8). Starting
from `base` (`vsetvli t0, a3, ...; vadd.vv; ` the scalar bookkeeping):

| Variants | Change |
| --- | --- |
| `vset_only` | no `vadd` |
| `ivli`, `ivli_noadd` | `vsetivli zero, 4`: no scalar result |
| `rd_unused`, `rd_unused_noadd` | `vsetvli t5, ...`: a scalar result nothing reads |
| `pos1` ... `pos5`, `pos6_top` | `vadd` moved 1 to 6 instructions later |
| `two_vadds` | two independent `vadd`s |
| `spK_add`, `spK_noadd` | K independent `addi`s after the `vadd`, K = 0 to 32 |
| `wave_base`, `wave_pp` | short versions of `base` and `vec_pingpong` for waveforms (no RTL measurement) |

**`vec_add`**: `full` is the microbenchmark's loop; `noadd`, `nostore`,
`loads`, `store`, `addonly` and `vset` keep only some of its vector
instructions, and `nv_*` hoist the `vsetvli` out of the loop.

`rtl_variants.csv` comes from `rvv_core_mini_verification_axi_sim --instr_trace`
at CoralNPU `M3-2026-04-27`, built with `-c opt`. Per-iteration cycles are
`rtl_roi_cycles / iterations`.

## Measuring on the RTL

Build the simulator as in roadmap B01, assemble the variant, and run:

```sh
bazel-bin/tests/verilator_sim/rvv_core_mini_verification_axi_sim \
    --binary base.elf --backdoor_load --instr_trace > base.log
```

`reef_perf.experiments.correlation.rtl_trace.measure_log("base.log")` reads
the ROI cycles from the log. Ignore a `trap=yes` row after `mpause`
(deviation M8).

## Comparing with the model

From `reef-perf/`, inside Docker:

```sh
uv run python tools/correlation/compare_rtl.py
uv run python tools/correlation/compare_rtl.py --only sp \
    --model before=/tmp/reef_perf_base:/tmp/m3_base.yaml \
    --model after=build/bin/reef_perf
```

Each `--model` is `LABEL=BINARY[:CONFIG]`. To compare before and after a
change, copy the old binary and its `m3.yaml` aside before rebuilding. Some
results depend on unmerged work elsewhere (for example the fetch fixes
B04-B06); check out a branch that has it when that matters.

## Reading waveforms

`--trace` makes the simulator write `/tmp/<model>.core.fst` (for this build
`/tmp/VRvvCoreMiniVerificationAxi.core.fst`), overwritten by each run. Trace a
short program: a few loop iterations are enough, and the file stays small.

Build `fstr` once in the CoralNPU container (the command is at the top of
`fstr.c`), then:

```sh
export FSTR=/path/to/fstr
python3 wave_table.py core.fst --list rd_valid_rob2rt
python3 wave_table.py core.fst --first 70 --last 82 \
    -s pc=score.dispatch.io_inst_0_bits_addr \
    -s valid=score.dispatch.io_inst_0_valid \
    -s ready=score.dispatch.io_inst_0_ready \
    -s rob2rt=backend.rd_valid_rob2rt_o \
    -s rob_n=retirement_buffer.instBuffer.io_nEnqueued
```

A name can be a unique suffix of the full hierarchical name. Signals that
were useful for the vector unit (`backend` is
`core.rvvCore.rvvCoreWrapper.core.backend`):

| Signal | Meaning |
| --- | --- |
| `score.dispatch.io_inst_N_{valid,ready,bits_addr}` | scalar dispatch slot N; dispatched when valid and ready |
| `retirement_buffer.instBuffer.io_nEnqueued`, `retirement_buffer.io_nRetired` | scalar retirement buffer occupancy and retires |
| `frontend.valid_inst_q` | RvvFrontEnd's register stage |
| `backend.pop_de2cq`, `backend.push_de2uq` | DE1 pops the command queue; DE2 pushes the uop queue |
| `backend.rs_valid_dp2alu`, `backend.pop_alu2rs` | dispatch into, and pop from, the ALU reservation station |
| `backend.res_valid_alu`, `backend.res_valid_arb2rob` | ALU result; result written into the ROB |
| `backend.rd_valid_rob2rt_o`, `backend.rt_xrf_valid_rvv2rvs` | vector ROB retires; write to a scalar register |
| `score.dispatch.io_undefFault_0`, `score.fault_manager.io_out_valid` | undefined instruction in slot 0; fault raised |

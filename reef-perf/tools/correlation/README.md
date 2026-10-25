# Correlation tools

Helpers for comparing the model with the M3 RTL beyond the microbenchmarks:
microbenchmark variants with their RTL measurements, a script that compares
model builds against them, and a way to read RTL waveforms as per-cycle
tables.

| Path | What it is |
| --- | --- |
| `variants/vadd_loop/` | `vsetvli` + `vadd.vv` loop variants (29) |
| `variants/vec_add/` | `vec_add_m1` (`va1_*`) and `vec_add_m4` (`va4_*`) with parts of the loop removed (17) |
| `variants/waveforms/` | short versions of microbenchmarks, a few iterations each, for RTL waveforms (no RTL measurement) |
| `rtl_variants.csv` | ROI cycles of each variant on the M3 RTL, and its loop iterations |
| `compare_rtl.py` | runs the microbenchmarks and variants on model builds and prints the error against the RTL |
| `measure_on_rtl.sh` | assembles programs and measures their ROI cycles on the RTL simulator |
| `sync_edits_to_correlation_testbench.sh` | copies uncommitted edits onto a test bench checkout that has unmerged work, rebuilds and tests it |
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

**`waveforms`**: each is a microbenchmark cut to a few iterations, so its
waveform stays small: `wave_gemv` and `wave_gemv_loads` (`gemv_int8`,
`gemv_loads`), `wave_va1` and `wave_va4` (`vec_add_m1`, `vec_add_m4`),
`wave_vadd_m4` (`vadd_m4_indep`), `wave_vdiv_m8` (`vdiv_m8_indep`), and
`wave_vse_m8`, a 3-instruction loop around an LMUL=8 store that shows RTL bug
RB1 (`rtl_bugs.md` at the top of the Reef repository).

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

`measure_on_rtl.sh` does all of this from the host when the model and the
RTL live in two Docker containers (the reef image and a CoralNPU build). It
assembles in the reef container, runs up to 6 programs at a time in the
CoralNPU one, and prints each program's ROI cycles:

```sh
bash reef-perf/tools/correlation/measure_on_rtl.sh \
    microbenchmarks/vector_units/vdiv_m8_indep.S \
    tools/correlation/variants/vec_add/va1_full.S
```

Paths are relative to `reef-perf/`. The container names and the simulator
path come from `REEF_CONTAINER`, `CORALNPU_CONTAINER` and `RTL_SIM` (the
script's header lists the defaults). `RTL_SIM_FLAGS=--trace` with one program
also writes its waveform.

In a vector loop shorter than the 8-entry retirement buffer, the ROI can end
while vector work is still in flight, because of RTL bug RB1
(`rtl_bugs.md`). The ROI cycles then understate the work.

## Comparing with the model

From `reef-perf/`, inside Docker:

```sh
uv run python tools/correlation/compare_rtl.py
```

Each `--model` is `LABEL=BINARY[:CONFIG]`. To compare before and after a
change, copy the old binary and its `m3.yaml` aside before rebuilding:

```sh
cp build/bin/reef_perf /tmp/reef_perf_before_my_change
cp configs/machines/m3.yaml /tmp/m3_before_my_change.yaml
# ... make the change and rebuild ...
uv run python tools/correlation/compare_rtl.py --only sp \
    --model before=/tmp/reef_perf_before_my_change:/tmp/m3_before_my_change.yaml \
    --model after=build/bin/reef_perf
```

Some results depend on unmerged work elsewhere (for example the fetch fixes
B04-B06); check out a branch that has it when that matters.

`sync_edits_to_correlation_testbench.sh` keeps such a branch handy as a
**test bench**: a git worktree in the reef container (`TESTBENCH`, default
`/tmp/rtl-correlation-testbench`) whose commits are that unmerged work. Run it
on the host, from the checkout you're editing, after each edit: it resets the
test bench to its own commits, applies that checkout's uncommitted diff,
rebuilds and runs the C++ tests.
Only uncommitted edits travel, so cherry-pick into the test bench any commit
your later edits build on. New files need `git add -N` first, and staged
renames or deletions must be unstaged.

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

`--dispatch` adds a column with the PCs the scalar core dispatched each cycle,
and `--active` drops the cycles where nothing dispatched and every signal is 0:

```sh
python3 wave_table.py core.fst --dispatch --active \
    -s take_v=lsu.io_rvv2lsu_0_valid -s take_r=lsu.io_rvv2lsu_0_ready \
    -s store_done=lsu.io_storeComplete_valid \
    -s retired=retirement_buffer.io_nRetired
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
| `score.dispatch.io_retirement_buffer_nSpace` | free scalar retirement buffer entries, as dispatch sees them |
| `retirement_buffer.io_writeDataVector_0_{valid,bits_uop_pc,bits_last_uop_valid}` | a vector result reaching the retirement buffer, the PC it is matched by, and whether it's the instruction's last uop |
| `lsu.io_rvv2lsu_0_{valid,ready}` | the LSU takes a vector register when both are 1 |
| `lsu.io_lsu2rvv_0_valid`, `lsu.io_storeComplete_{valid,bits}` | the LSU hands a loaded register back; a store (each vector store register) completes, with its PC |
| `backend.rs_valid_dp2lsu`, `backend.res_valid_lsu` | dispatch into the vector LSU reservation station; LSU result into the vector ROB |
| `backend.pop_div2rs`, `backend.res_valid_div` | a uop enters the divider; divider result |

# Beginner Tickets: Make the Coarse Model Match the M3 RTL

`reef_perf` runs real programs, but its timing is **deliberately generic**: a
plain 4-wide in-order core with guessed latencies. Reef's baseline is the
CoralNPU M3 release, so these tickets turn the model into a first-order model
of the **M3 RTL**, one behaviour at a time.

Each ticket:

- explains the hardware behaviour in plain terms;
- points to the RTL code that proves it (paths are relative to the CoralNPU
  checkout at tag `M3-2026-04-27`);
- names the model file to change and a workload that shows the effect;
- says how to check your work against the RTL.

Paths to model code are relative to `reef-perf/`. Headers are in
`src/csrc/include/reef_perf/` and implementations in `src/csrc/src/`. Run
everything inside the Docker image (`tools/docker.sh shell`).

"First order" means we care about effects big enough to move a workload's
cycle count by roughly 5% or more. Don't chase single cycles.

---

## Before you start

1. Read the [README](../README.md), including the "Reading the summary"
   section.
2. Skim [implementation-plan.md](implementation-plan.md) §2, which describes
   how the M3 baseline works.
3. Do **B00**.

### Ground rules

- **One behaviour per PR.**
- **Every new timing number comes from somewhere.** Either the RTL (cite
  `file:line`) or a measurement (cite the B01 result). Put that source in a
  comment next to the value in `configs/m3.yaml`.
- **New behaviour goes behind a parameter.** Its default (in the `.hpp`) must
  keep the old super-coarse behaviour, and `configs/m3.yaml` turns it on. That
  way old results stay reproducible, and we can switch a rule off to see what it
  costs.
- **Every PR includes a before/after table** from the `run` experiment:
  `uv run python -m reef_perf.experiments.run.run --config
  src/reef_perf/experiments/run/configs/all.yaml`. After B03 exists, use the
  `correlate` experiment's table instead.
- **Every PR passes `bash tools/lint.sh` and `uv run pytest`.** New C++
  behaviour gets a Google Test in `tests/cpp/`, and every new function or member
  gets a Doxygen comment (private ones included).

### Order

```
B00 ─┬─ B01 ─ B02 ─ B03     (measurement: do these first, can be parallel with B19/B20)
     ├─ B19, B20             (tooling: any time)
     └─ then any of B04..B18 (model fixes); within a group, go in order
```

| Group | Tickets |
| --- | --- |
| 0. Measure | B00 run the model · B01 RTL cycle counts · B02 region of interest · B03 `correlate` experiment |
| 1. Fetch | B04 fetch every other cycle · B05 static branch prediction · B06 aligned fetch blocks |
| 2. Dispatch | B07 branch ends the group · B08 dispatch-alone instructions · B09 no address forwarding · B10 CSRs drain the Rob |
| 3. Execute | B11 multiplier and FPU latency · B12 data-dependent divider · B13 LSU timing · B14 vector memory through the LSU |
| 4. Vector | B15 real vector functional units · B16 vector decode/dispatch widths · B17 scalar↔vector crossing · B18 vector ROB |
| 5. Tooling | B19 per-instruction timing trace · B20 complete the decoder tests |

---

## Group 0: Measure

### B00: Build and run the model

**Goal.** Get the model running and understand its output.

**Steps**
1. Follow the README quick start: `tools/docker.sh build`, then
   `tools/docker.sh shell`, then build with CMake.
2. Run the `run` experiment's sweep:
   `uv run python -m reef_perf.experiments.run.run --config
   src/reef_perf/experiments/run/configs/all.yaml`.
3. Run `gemv_int8` with the full summary:
   `build/bin/reef_perf --elf build/workloads/gemv_int8.elf -c configs/m3.yaml`.
4. Run `build/bin/reef_trace build/workloads/gemv_int8.elf 60` and match the
   printed instructions to
   `src/reef_perf/experiments/run/workloads/gemv_int8.S`.
5. Change one parameter with `-p` (e.g. `top.backend.execute.params.vec_units 1`)
   and explain the change in cycles to a teammate.
6. Run `uv run pytest` and make sure it passes.

**Done when** you can explain what each line of the summary means.

---

### B01: Measure the workloads on the M3 RTL

**Why.** Correlation needs ground truth. Every workload brackets its main loop
with `csrr s10, mcycle` … `csrr s11, mcycle`. On the RTL, the loop's cycle count
is `s11 − s10`.

**Steps**
1. In the M3 checkout (`git checkout M3-2026-04-27`), build the verification
   simulator. It prints every retired instruction together with the value it
   wrote:
   ```
   bazel build //tests/verilator_sim:rvv_core_mini_verification_axi_sim
   ```
   The build is heavy. Use `--jobs=4` on an 8 GB machine.
2. Run each workload (the ELFs are in `reef-perf/build/workloads/` after the
   `run` experiment has run once):
   ```
   bazel-bin/tests/verilator_sim/rvv_core_mini_verification_axi_sim \
     --binary <path>/<name>.elf --backdoor_load --instr_trace > <name>.log
   ```
3. Write `src/reef_perf/experiments/correlate/rtl_trace.py`, a library module
   (no `main`). It finds the two `csrr` writes to `s10` (x26) and `s11` (x27)
   in a log and returns `s11 − s10`. Add a unit test for it in `tests/python/`
   using a small sample log.
4. Record the results in `src/reef_perf/experiments/correlate/data/rtl_m3.csv`
   with the columns `workload,rtl_roi_cycles,rtl_retired_insts,notes`. Put the
   retired-instruction count between the two markers in `rtl_retired_insts`.
5. For each workload, check that `rtl_retired_insts` equals the number of
   instructions Spike executes between the markers (`reef_trace`). If they
   differ, Spike and the RTL disagree functionally. Report it; don't fix it
   here.

**Notes**
- If a workload doesn't reach `mpause` on the RTL, look at what it does
  differently: memory map, `mstatus` setup, or an instruction the RTL doesn't
  support. Record it in `notes`.
- Deviation M1 in deviations.md says the verification build should have the
  same timing as the normal build at M3. As a spot check, also run 2–3
  workloads on `rvv_core_mini_axi_sim` (no `--instr_trace`) and compare total
  cycles.

**Done when** `rtl_m3.csv` has a row for every workload, and the extraction
module and its test are committed.

---

### B02: Measure only the region of interest in the model

**Why.** The RTL number from B01 covers only the loop between the two `mcycle`
reads. The model currently reports the whole program, including setup. To
compare like with like, the model needs to report the same region.

**Where**
- `src/csrc/src/rob.cpp`: retirement is where "time" is counted.
- `src/csrc/src/reef_sim.cpp`: the summary and JSON output.
- `src/reef_perf/sim.py`: `SimResult`, which parses the JSON.

**Steps**
1. In `Rob::retire_insts()`, detect a retiring instruction that is `csrr` of `mcycle`
   (`inst->mnemonic` starts with `csrr` and the CSR number, bits 31:20 of
   `inst->encoding`, is `0xB00`).
   - The first one's retire cycle is `roi_start`.
   - The second one's retire cycle is `roi_end`.
   - Also count the instructions retired in between.
2. Report `roi_cycles = roi_end − roi_start` and `roi_instructions` in the
   summary and in the JSON output, and add them to `SimResult`.
3. Show `roi_cycles` in the `run` experiment's table and `summary.csv`.
4. Extend `tests/python/test_sim.py` to check the ROI numbers for `alu_chain`.

**Check.** For `alu_chain` (16,000 dependent adds, plus loop overhead),
`roi_cycles` should be a little over 16,000.

**Done when** every workload reports ROI numbers.

---

### B03: The `correlate` experiment

**Why.** This is the scoreboard for every other ticket.

**Steps.** Create a new experiment, `src/reef_perf/experiments/correlate/`,
following the layout in `src/reef_perf/experiments/README.md` (use `run/` as
the example):
1. `main.py`: for one workload, read its `run` artifact and its RTL row from
   `data/rtl_m3.csv`, and write a correlation artifact with RTL cycles, model
   ROI cycles, `error% = (model − rtl) / rtl × 100`, and the model's top stall
   reason.
2. `run.py`: given a config, use the artifacts of a `run` sweep (it may call
   `reef_perf.experiments.run.run.run_sweep` first), correlate every workload,
   and print:
   ```
   workload       rtl_cycles  model_cycles  error%   model top stall
   ```
   with the mean absolute error at the bottom.
3. `plot.py`: a bar chart of error% per workload.
4. `configs/m3.yaml`: the workloads and simulation config to correlate.
5. `README.md`, and `tests/python/experiments/test_correlate_experiment.py`.
6. Commit the first run's table as `correlate/baseline.md`. That's the
   super-coarse model's starting error.

**Done when** `uv run python -m reef_perf.experiments.correlate.run --config
src/reef_perf/experiments/correlate/configs/m3.yaml` prints the table, its test
passes, and the baseline is committed. From now on, every model PR pastes its
before/after table.

---

## Group 1: Fetch

### B04: M3 fetches only every other cycle (~1 day)

**Hardware.**
- The M3 fetch unit reads 16 bytes (4 instructions) per request.
- After a response arrives, it waits a cycle before sending the next request,
  and only one request is ever in flight.
- So instructions arrive at most 4 every 2 cycles, and straight-line code tops
  out at about 2 IPC even though dispatch is 4 wide.

**RTL evidence.**
- `hdl/chisel/src/coralnpu/scalar/UncachedFetch.scala:194-195`: new fetches are
  blocked while `io.fetchData.valid` ("Wait one cycle for next fetch").
- Deviation D7 in deviations.md.

**Model.** `Fetch` already has `fetch_interval`.

**Steps**
1. Set `fetch_interval: 2` in `configs/m3.yaml`, with the RTL citation.
2. Run `alu_indep` before and after.

**Check.** `alu_indep` IPC should drop from about 3.2 to about 2.0, and the
main stall reason should become `ibuf_empty`. Compare against the B01 RTL
number.

**Done when** `alu_indep` is within 10% of the RTL, or you've written down why
not.

---

### B05: Static branch prediction instead of a flat redirect penalty (~3–5 days)

**Hardware.** The fetch unit guesses branch directions from the instruction bits
alone:

- `jal`: always predicted taken.
- Conditional branch with a **negative** offset (backward, typically a loop):
  predicted taken.
- Conditional branch with a positive offset: predicted not taken.
- `jalr`: never predicted.

A correctly predicted taken branch still costs a small bubble while fetch
restarts at the target. A wrong guess (or any `jalr`) costs more: the branch has
to execute first, then fetch restarts.

**RTL evidence.**
- `scalar/UncachedFetch.scala:87-93` (`PredictJump`).
- `scalar/Bru.scala`.
- Deviations D6 and U1.

**Model.** `Fetch` currently charges `redirect_penalty` for every taken
branch/jump, and nothing for not-taken branches.

**Steps**
1. Add a parameter `predictor` (string: `none` | `static_btfn`). Default `none`
   keeps today's behaviour.
2. Add parameters `taken_bubble` (cost of a correctly predicted taken branch)
   and `mispredict_penalty`.
3. In `Fetch::fetch_group()` (`src/csrc/src/fetch.cpp`), for each control-flow
   instruction:
   - work out the prediction from `inst->encoding` (the sign bit, bit 31, gives
     the offset direction);
   - compare it with the actual outcome (`inst->redirected()`);
   - apply the right cost.
   Note that a mispredicted **not-taken** branch (predicted taken but fell
   through) also costs cycles.
4. Count `num_predicted_taken`, `num_mispredicts` and `num_jalr` as counters.
5. Measure `taken_bubble` with `branch_tight`: the loop body is 2 instructions,
   so RTL cycles per iteration minus 1 is roughly the bubble. Measure
   `mispredict_penalty` with a new workload that has a forward-taken branch in a
   loop (add `src/reef_perf/experiments/run/workloads/branch_fwd_taken.S`).

**Done when** `branch_tight`, `branch_group` and the new workload are within 10%
of the RTL.

---

### B06: Fetch blocks are 16-byte aligned (~2–3 days)

**Hardware.**
- A fetch reads an aligned 16-byte block. If the program jumps to an address in
  the middle of a block, that fetch delivers only the instructions from the
  target to the end of the block (e.g. a branch target at offset 8 delivers 2
  instructions, not 4).
- A group also can't continue into the next block within the same fetch.

**RTL evidence.** `UncachedFetch.scala`, the predecode logic around line 120:
`startElem` and `validsIn`.

**Model.** In `Fetch::fetch_group()`, stop the group at the end of the 16-byte block
that contains the first instruction's PC. Put this behind a parameter
`aligned_blocks` (default `false`).

**Check.** Loops whose start isn't 16-byte aligned should get slower. Add a
workload that pads the loop start with 1–3 `nop`s and compare aligned vs
unaligned on the RTL and on the model.

**Done when** the model shows the same aligned/unaligned difference as the RTL,
within 1 cycle per iteration.

---

## Group 2: Dispatch rules

All four tickets change `Dispatch::can_dispatch()` / `Dispatch::dispatch_group()`
in `src/csrc/src/dispatch.cpp`. The RTL for every rule is `DispatchV2` in
`hdl/chisel/src/coralnpu/scalar/Decode.scala` (lines ~301-530 at M3); read the
`canDispatch` expression there first. Add one stall reason per rule to
`StallReason` so the summary shows the rule's cost.

### B07: A conditional branch ends the dispatch group (~1–2 days)

**Hardware.** When a conditional branch dispatches, nothing after it dispatches
in the same cycle.

**RTL evidence.**
- `Decode.scala:322-327` (`branchInterlock`).
- Deviation D2 (the docs say otherwise).

**Steps**
1. Add a parameter `branch_ends_group` (default `false`).
2. When it's on, stop the dispatch loop right after dispatching a `BRANCH`
   instruction.

**Check.** `branch_group` (`[alu, beq, alu, alu]` repeated) should go from about
4 instructions per dispatch cycle to about 2. Compare with the RTL.

---

### B08: Some instructions dispatch alone, from slot 0 (~2 days)

**Hardware.** These instructions can only dispatch as the *first* instruction of
a cycle, and nothing else dispatches with them:

- all scalar FP instructions, including `flw`/`fsw`;
- vector instructions with an FP scalar operand (`.vf` forms, `vfmv.f.s`,
  `vfmv.s.f`);
- CSR instructions;
- `fence.i`, `ebreak`, `wfi`, `mpause`.

**RTL evidence.**
- `Decode.scala:163-169` (`forceSlot0Only`) and the `slot0Interlock` block.
- Deviation D3.

**Steps**
1. Add a `dispatch_alone` flag to `Inst` (`inst.hpp`), set in
   `src/csrc/src/inst_decode.cpp` for the instructions above, and add decoder
   tests for it in `tests/cpp/test_inst_decode.cpp`. Hint: FP-scalar vector ops
   are `OPFVF` (funct3 = 5) plus `vfmv.f.s`.
2. Add a parameter `enforce_dispatch_alone` (default `false`).
3. When it's on: an instruction with `dispatch_alone` can only dispatch if it's
   the first of the cycle, and the cycle ends right after it.

**Check.** `fp_mix` (`[fadd.s, addi, addi, addi]`) should need about 2 dispatch
cycles per group instead of 1. Compare with the RTL.

---

### B09: Load/store addresses can't use a value produced the cycle before (~2 days)

**Hardware.** Most instructions can use a result the cycle after it's produced,
because of forwarding. But the base-address register of a load or store, and
the source register of a `jalr`, are read from the registered scoreboard: the
value must have been ready for one extra cycle. The data register of a scalar
store is treated the same way.

**RTL evidence.**
- `Decode.scala:347-356` (`usesRs1Regd`, `usesRs2Regd`, `readScoreboardRegd`).
- Deviation D4.

**Steps**
1. Add a parameter `addr_extra_cycle` (default `0`).
2. In `can_dispatch()`, for load/store base registers (the first `x` source of a
   memory instruction), `jalr`'s source, and a scalar store's data register,
   require `result_ready_cycle + addr_extra_cycle <= now`.
3. Record the stall as a new reason, `addr_not_registered`.

**Check.** `addr_forward` should slow down by about 1 cycle per
`addi → lw` pair. `load_chain` may also change. Compare with the RTL.

---

### B10: CSR instructions wait for an empty Rob; confirm the retire width (~2 days)

**Hardware.** A CSR instruction (including `csrr ..., mcycle`) doesn't dispatch
until every older instruction has retired.

**RTL evidence.**
- `Decode.scala:519`.
- Deviation M2.

This matters for B01/B02: every `mcycle` read drains the pipeline, so the model
has to do the same for ROI counts to match.

**Steps**
1. Dispatch needs to know when the Rob is empty. Add an `empty` flag that
   travels with the Rob credits: e.g. change the credit port to carry a small
   struct `{credits, empty}`.
2. Add a parameter `csr_waits_for_empty_rob` (default `false`) and a stall
   reason `csr_drain`.
3. Confirm `retire_width` from the RTL (`RetirementBuffer.scala`: how many
   entries can retire per cycle?). Update `configs/m3.yaml` with the citation.

**Check.** Short-loop workloads' ROI cycles should shift by a few cycles. The
drain happens at the ROI edges, not inside the loop.

---

## Group 3: Execute

### B11: Multiplier and FPU latencies (~1–2 days)

**Hardware.**
- **MLU:** an arbiter plus two queued stages (`scalar/Mlu.scala:67-101`), so the
  effective latency is 2 or 3 cycles.
- **FPU:** the fpnew unit with 3 pipeline registers (`float/FloatCore.scala:113`).
  Divide/sqrt is iterative.

**Steps**
1. From the B01 RTL numbers:
   - `mul_chain` cycles ÷ 4,000 ≈ multiply latency;
   - `fp_chain` cycles ÷ 4,000 ≈ FMA latency.
2. Set `mul_latency` and `fpu_latency` in `configs/m3.yaml`.
3. Add `src/reef_perf/experiments/run/workloads/fdiv_chain.S` (dependent
   `fdiv.s`) and set `fdiv_latency` and `fdiv_occupancy`.

**Done when** `mul_chain`, `fp_chain` and `fdiv_chain` are within 5% of the RTL.

---

### B12: The divider's latency depends on its operands (~3–5 days)

**Hardware.** The integer divider produces one bit per cycle, but first skips the
leading zeros of the dividend. Dividing a small number is much faster than
dividing a large one.

**RTL evidence.** `scalar/Dvu.scala:90-136` (`clz`, `count`).

**This ticket touches both halves of the model.**
1. **Driver** (`spike_driver.hpp` / `src/csrc/src/spike_driver.cpp`):
   - add `std::uint32_t rs1_val, rs2_val` to `InstRecord`;
   - in `SpikeDriver::Impl::step()`, fill them *before* `proc_->step(1)` from
     Spike's register file: `state->XPR[n]` for the encoding's rs1/rs2 fields.
2. **Model:**
   - copy the values into `Inst` (`src/csrc/src/func_sim.cpp`);
   - in `Execute::receive_inst()` for `DIV`, compute latency and occupancy as
     `div_base + (32 − clz(dividend))`. Use the unsigned or absolute value as the
     RTL does.
   - Keep the flat latency when a parameter `div_data_dependent` is `false`.
3. Fit `div_base` to the RTL using `div_chain`, which has large and small
   dividends.

**Done when** `div_chain` is within 5% of the RTL.

---

### B13: LSU timing: one memory instruction at a time (~3–5 days)

**Hardware.**
- The M3 LSU has a 4-entry queue and a single "slot". It handles one memory
  instruction at a time.
- Each bus transaction moves one 16-byte line, and read data comes back the
  cycle after the request.
- Several loads to the same line are still separate instructions, each taking
  its own turn.

**RTL evidence.**
- `scalar/Lsu.scala:824-1000` (`LsuV2`, `slot`, `readFired`).
- The M3 `doc/microarch/lsu.md`.
- Deviations D5 and U3.

**Steps**
1. From the RTL:
   - `load_chain` gives load-to-use latency (cycles per step, minus the loop
     overhead);
   - `load_stream` gives the throughput of back-to-back independent loads.
2. Set `lsu_latency` and `lsu_cycles_per_line`. If the independent-load
   throughput doesn't fit "one line per cycle", add a parameter
   `lsu_min_cycles_per_inst` (time the slot is busy per instruction, even for a
   1-line access) and use it in `Execute` (`src/csrc/src/execute.cpp`).
3. Check that stores behave the same way. Add a `store_stream` workload.

**Done when** `load_chain`, `load_stream` and `store_stream` are within 10% of
the RTL.

---

### B14: Vector loads/stores go through the LSU one register at a time (~3–5 days)

**Hardware.**
- A vector load/store is split per 16-byte vector register. For each register,
  the LSU slot:
  1. waits for the vector unit to send the mask/data (vector-update);
  2. issues one bus transaction per distinct 16-byte line;
  3. writes back.
- There is a fixed per-register overhead on top of the line transactions.
- Only one vector-to-LSU port is used at M3 (U7).

**RTL evidence.**
- `scalar/Lsu.scala:362-632` (`LsuSlot`, `vectorUpdate`, `vectorLoop`).
- `Lsu.scala:891`.
- Deviations U6 and U7.

**Steps**
1. Add a parameter `lsu_vector_reg_overhead` (cycles per register, default `0`).
2. In `Execute`, for `V_LOAD`/`V_STORE`, compute occupancy per register: group
   the element accesses by register (16 bytes of `vl × SEW` each), then add the
   overhead to each register's line count.
3. Fit it with `vec_memcpy` (LMUL=8, 8 registers per op) and `vec_add_m1`
   (1 register per op).
4. Optional: add a strided-load workload (stride 16 or more), where every element
   is its own line.

**Done when** `vec_memcpy`, `vec_add_m1` and `vec_add_m4` are within 10% of the
RTL on their memory-bound parts.

---

## Group 4: Vector unit

### B15: Model the real vector functional units (~1 week)

**Hardware.** The M3 vector backend has separate functional units, each with its
own reservation station:

| Unit | Count |
| --- | --- |
| ALU | 2 |
| Multiply/MAC | 2 |
| Integer divide | 1 |
| FP multiply-add | 2 |
| FP divide | 1 |
| Permute/reduction | 1 |

**RTL evidence.** `hdl/verilog/rvv/inc/rvv_backend_define.svh` (`NUM_ALU`,
`NUM_MUL`, `NUM_DIV`, `NUM_PMTRDT`, `NUM_FMA`, the RS depths under `DISPATCH3`).

**Steps**
1. In `Execute`, replace the single `vec_` `ResourcePool` with one pool per unit type,
   each with its own `count`, `latency` and `cycles_per_uop` parameters. Keep a
   parameter `vec_unified` (default `true`) that restores today's single pool.
2. Map the vector classes to pools: `V_ALU` → ALU, `V_MUL` → MUL, `V_DIV` → DIV,
   `V_FP` → FMA, `V_FDIV` → FDIV, `V_PERM` and `V_TO_SCALAR` → permute/reduce.
3. Latencies: write small workloads (one per class, dependent chain and
   independent stream) and measure them on the RTL. Copy the style of
   `mul_chain` / `alu_indep`.

**Done when** `vec_add_m1`, `vec_add_m4` and `gemv_int8` are within 15% of the
RTL, and the per-class workloads are within 10%.

---

### B16: Vector decode and dispatch widths (~1 week)

**Hardware.** Vector instructions go through their own pipeline:

1. an 8-entry command queue;
2. a decoder that takes **2 instructions per cycle** and splits each into uops
   (up to 6 uops per cycle);
3. a 16-entry uop queue;
4. dispatch of **3 uops per cycle** to the functional units.

A long LMUL=8 instruction can therefore hold up the ones behind it.

**RTL evidence.** `rvv_backend_define.svh` under `DISPATCH3` (`NUM_DE_INST`,
`NUM_DE_UOP`, `NUM_DP_UOP`, `UQ_DEPTH`) and `rvv_backend_config.svh:5`.

**Steps**
1. Create a new Sparta unit for these stages, between Dispatch and the vector
   pools: `src/csrc/include/reef_perf/vector.hpp` and `src/csrc/src/vector.cpp`
   (add it to `CMakeLists.txt` and bind its ports in `reef_sim.cpp`). Copy the
   structure of `Rob` (queue + tick event).
2. Move vector instructions to it: Dispatch sends vector ops to `Vector`
   instead of `Execute`. `Vector` issues uops into the B15 pools.
3. Parameters: `decode_width`, `max_uops_per_decode_cycle`,
   `uop_queue_entries`, `uop_dispatch_width`.

**Done when** vector workloads are no worse than after B15, and a new workload
with mixed LMUL=8 and LMUL=1 ops matches the RTL within 15%.

---

### B17: Crossing between the scalar core and the vector unit (~3–5 days)

**Hardware.**
- The scalar core hands vector instructions over through a queue.
- Results that come back to a scalar register (`vsetvli` writing `rd`,
  `vmv.x.s`, `vcpop`, reductions read with `vmv.x.s`) take extra cycles to cross
  back.
- `gemv_int8` does this once per row, so it matters for LLM-style kernels.

**RTL evidence.**
- `rvv/RvvCore.scala`, `hdl/verilog/rvv/design/RvvFrontEnd.sv`.
- Deviation U4.

**Steps**
1. Write a ping-pong workload:
   `vsetvli → vmv.s.x → vredsum → vmv.x.s → (scalar use) → loop`. Measure it on
   the RTL.
2. Set `vec_to_scalar_latency`, and add a `vset_latency` parameter for `VSET`
   (today it's timed as a 1-cycle ALU op).
3. Check `gemv_int8`.

**Done when** the ping-pong workload is within 10% and `gemv_int8` within 15%.

---

### B18: Vector reorder buffer (~2–3 days)

**Hardware.** Besides the 8-entry scalar retirement buffer, the vector backend has
its own **8-entry** reorder buffer. Long vector ops can fill it and stall new
vector work even when the scalar side has room.

**RTL evidence.** `rvv_backend_define.svh:35` (`ROB_DEPTH 8`); deviation U8.

**Steps**
1. Track in-flight vector uops (from issue until complete).
2. Stop issuing new vector uops when there are `vrob_entries` in flight.
3. Report a stall counter for it.

**Done when** a workload with long independent vector ops (e.g. `vdiv` at LMUL=8)
matches the RTL within 15%.

---

## Group 5: Tooling

### B19: Per-instruction timing trace (~2 days)

**Why.** When a workload doesn't correlate, you need to compare instruction by
instruction with the RTL's `--instr_trace` log.

**Steps**
1. Add an option `--inst-trace FILE` (`src/csrc/src/main.cpp`). When set,
   `Rob::retire_insts()` writes one line per retired instruction:
   `seq pc disasm fetch dispatch issue ready complete retire`.
2. Write `tools/diff_trace.py`. It lines up the model trace with the RTL
   `--instr_trace` log by PC sequence, and prints the first 20 places where the
   gap between consecutive retirements differs. Put the parsing logic in a
   `reef_perf` module so it can be unit-tested.

**Done when** you can point at the first instruction where the model and the RTL
diverge for any workload.

---

### B20: Complete the instruction decoder tests (~2–3 days)

**Why.** If `inst_decode.cpp` gets a register wrong, dependency stalls come out
wrong everywhere, and the error is silent.

`tests/cpp/test_inst_decode.cpp` already covers a first set of instructions
(integer ALU, mul/div, loads/stores, `csrr`, `fadd.s`, `vsetvli`, `vle8.v`,
`vadd.vv` at LMUL 1 and 4, masked ops, `vmacc.vv`, `vredsum.vs`, `vmv.x.s`).

**Steps**
1. Add Google Test cases to `tests/cpp/test_inst_decode.cpp` that build `Inst`
   objects from known encodings and check `cls`, `srcs` and `dsts`.
2. Get the encodings from `riscv64-linux-gnu-objdump -d` on the workloads
   (`build/workloads/*.elf`).
3. Cover at least the instructions not tested yet:
   - `add`, `addi`, `lw`, `sw`;
   - a branch, `jal`, `jalr`;
   - `mul`, `div`, `csrr`;
   - `flw`, `fsw`, `fadd.s`, `fmadd.s`, `feq.s`, `fcvt.w.s`, `fmv.w.x`;
   - `vsetvli`, `vle8.v`, `vse32.v`, `vadd.vv`, `vadd.vx`, `vmacc.vv`,
     `vwmul.vv`, `vwredsum.vs`, `vmv.x.s`, `vmv.s.x`, `vfmacc.vf`;
   - a masked op (`vm = 0`).
4. Fix any bugs you find in `inst_decode.cpp`. Each fix is its own commit with
   the test that caught it.

**Done when** the tests pass under `ctest --test-dir build` and `uv run pytest`.

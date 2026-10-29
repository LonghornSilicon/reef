# reef_perf Roadmap

The tickets that take reef_perf from a super-coarse model to one that
correlates within ±10% against the M3 RTL (Reef's CoralNPU baseline).

- **[Part 1: Beginner tickets](#part-1-beginner-tickets) (B00–B20)** are the
  near-term work for new team members: each fixes one first-order behavior.
- **[Part 2: Long-term tickets](#part-2-long-term-tickets) (T01–T25)** are the
  full plan, epic by epic. The beginner tickets are its first steps.

Background: [overview.md](overview.md) (what the model is and how M3 works),
[interfaces.md](interfaces.md) (module boundaries). IDs such as `D7` or `U3`
refer to entries in [deviations.md](deviations.md).

## Part 1: Beginner tickets

`reef_perf` runs real programs, but its timing is **deliberately generic**: a
plain 4-wide in-order core with guessed latencies. Reef's baseline is the
CoralNPU M3 release, so these tickets turn the model into a first-order model
of the **M3 RTL**, one behavior at a time.

Each ticket:

- explains the hardware behavior in plain terms;
- points to the RTL code that proves it (paths are relative to the CoralNPU
  checkout at tag `M3-2026-04-27`);
- names the model file to change and a microbenchmark that shows the effect;
- says how to check your work against the RTL.

Paths to model code are relative to `reef-perf/`. C++ headers and sources
sit together in `model/<module>/`. Run everything inside the Docker image
(`../tools/docker.sh shell`).

"First order" means we care about effects big enough to move a microbenchmark's
cycle count by roughly 5% or more. Don't chase single cycles.

---

### Before you start

1. Read the [README](../README.md), including the "Reading the summary"
   section.
2. Skim [overview.md](overview.md) §2, which describes
   how the M3 baseline works.
3. Do **B00**.

#### Ground rules

- **One behavior per PR.**
- **Every new timing number comes from somewhere.** Either the RTL (cite
  `file:line`) or a measurement (cite the B01 result). Put that source in a
  comment next to the value in `configs/machines/m3.yaml`.
- **New behavior goes behind a parameter.** Its default (in the `.h`) must
  keep the old super-coarse behavior, and `configs/machines/m3.yaml` turns it on. That
  way old results stay reproducible, and we can switch a rule off to see what it
  costs.
- **Every PR shows its before/after table.** The correlation bot posts it on
  the PR; to see it locally, run
  `uv run python -m reef_perf.experiments.correlation.main`.
- **Every PR passes `bash scripts/lint.sh` and `uv run pytest`.** New C++
  behavior gets a Google Test in `tests/cpp/`, and every new public
  declaration gets a `///` comment.

#### Order

```
B00 ─┬─ B01 ─ B02 ─ B03     (measurement: do these first, can be parallel with B19/B20)
     ├─ B19, B20             (tooling: any time)
     └─ then any of B04..B18 (model fixes); within a group, go in order
```

| Group | Tickets |
| --- | --- |
| 0. Measure | B00 run the model · B01 RTL cycle counts · B02 region of interest · B03 `correlation` experiment |
| 1. Fetch | B04 fetch every other cycle · B05 static branch prediction · B06 aligned fetch blocks |
| 2. Dispatch | B07 branch ends the group · B08 dispatch-alone instructions · B09 no address forwarding · B10 CSRs drain the Rob |
| 3. Execute (core, memory) | B11 multiplier and FPU latency · B12 data-dependent divider · B13 LSU timing · B14 vector memory through the LSU |
| 4. Vector | B15 real vector functional units · B16 vector decode/dispatch widths · B17 scalar↔vector crossing · B18 vector ROB |
| 5. Tooling | B19 per-instruction timing trace · B20 complete the decoder tests |

---

### Group 0: Measure

#### B00: Build and run the model

**Goal.** Get the model running and understand its output.

**Steps**
1. Follow the README quick start: `tools/docker.sh build`, then
   `tools/docker.sh shell`, then build with CMake.
2. Run the `sweep` experiment:
   `uv run python -m reef_perf.experiments.sweep.sweep --config
   python/reef_perf/experiments/sweep/sweeps/all.yaml`.
3. Run `gemv_int8` with the full summary:
   `build/bin/reef_perf --elf build/microbenchmarks/gemv_int8.elf -c configs/machines/m3.yaml`.
4. Run `build/bin/reef_trace build/microbenchmarks/gemv_int8.elf 60` and match the
   printed instructions to
   `microbenchmarks/gemv_int8.S`.
5. Change one parameter with `-p` (e.g. `top.vector.params.vec_alu_count 1`)
   and explain the change in cycles to a teammate.
6. Run `uv run pytest` and make sure it passes.

**Done when** you can explain what each line of the summary means.

---

#### B01: Measure the microbenchmarks on the M3 RTL — done

Orhan Unuvar ran every microbenchmark on `rvv_core_mini_verification_axi_sim
--instr_trace` (CoralNPU tag `M3-2026-04-27`, `--backdoor_load`). Every run
reached `mpause`, and the ROI instruction counts and PC sequences match Spike.
The results are in `microbenchmarks/rtl_m3.csv`; `rtl_trace.py` in the
correlation experiment extracts them from a trace log:

```
bazel build -c opt //tests/verilator_sim:rvv_core_mini_verification_axi_sim
bazel-bin/tests/verilator_sim/rvv_core_mini_verification_axi_sim \
  --binary <path>/<name>.elf --backdoor_load --instr_trace > <name>.log
```

Build with `-c opt`. Without it Bazel uses its default `fastbuild` mode, which
compiles the Verilated model without optimization; CoralNPU's README omits the
flag. The optimized simulator is about 19x faster (`vec_memcpy`: 6.0 s instead
of 114.5 s) and gives identical traces apart from the junk instruction word in
the final `trap=yes` row after `mpause`.

The ELFs are in `reef-perf/build/microbenchmarks/` after an experiment has run
once. The build is heavy (about 18 minutes with `--jobs=6`); use `--jobs=4` on
an 8 GB machine. Deviation M1 records the check that the verification build has
production timing.

When a microbenchmark is added or changed, measure it the same way and update
its row.

---

#### B02: Measure only the region of interest in the model — done

The RTL number from B01 covers only the loop between the two `mcycle` reads.
The model now reports the same region: `Rob` records the issue cycle of every
retired `csrr …, mcycle` (`Rob::getMcycleReads()`), and the summary and JSON
report `roi_cycles` (last read − first read), which `SimResult` parses. For
`alu_chain` (16,000 dependent adds plus loop overhead) it is a little over
16,000.

Still open: an `roi_instructions` count, if B01 finds Spike and the RTL
disagree on what runs between the markers.

---

#### B03: The `correlation` experiment — done

`python/reef_perf/experiments/correlation/` runs every microbenchmark and
prints, per microbenchmark, the model's ROI cycles, the RTL's (from
`microbenchmarks/rtl_m3.csv`), the error, both sides' ROI instruction counts,
total cycles and the change against a baseline, with the mean absolute error at the bottom. A GitHub
workflow runs it on every push to a PR that changes `reef-perf/` and posts the
table as a PR comment, so every model PR shows its before/after automatically.

Still open: a top-stall column and a per-microbenchmark plot are welcome
additions.

---

### Group 1: Fetch

#### B04: M3 fetches only every other cycle (~1 day)

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
1. Set `fetch_interval: 2` in `configs/machines/m3.yaml`, with the RTL citation.
2. Run `alu_indep` before and after.

**Check.** `alu_indep` IPC should drop from about 3.2 to about 2.0, and the
main stall reason should become `ibuf_empty`. Compare against the B01 RTL
number.

**Done when** `alu_indep` is within 10% of the RTL, or you've written down why
not.

---

#### B05: Static branch prediction instead of a flat redirect penalty (~3–5 days)

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
   keeps today's behavior.
2. Add parameters `taken_bubble` (cost of a correctly predicted taken branch)
   and `mispredict_penalty`.
3. In `Fetch::fetchGroup()` (`model/frontend/Fetch.cpp`), for each control-flow
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
   `mispredict_penalty` with a new microbenchmark that has a forward-taken branch in a
   loop (add `microbenchmarks/branch_fwd_taken.S`).

**Done when** `branch_tight`, `branch_group` and the new microbenchmark are within 10%
of the RTL.

---

#### B06: Fetch blocks are 16-byte aligned (~2–3 days)

**Hardware.**
- A fetch reads an aligned 16-byte block. If the program jumps to an address in
  the middle of a block, that fetch delivers only the instructions from the
  target to the end of the block (e.g. a branch target at offset 8 delivers 2
  instructions, not 4).
- A group also can't continue into the next block within the same fetch.

**RTL evidence.** `UncachedFetch.scala`, the predecode logic around line 120:
`startElem` and `validsIn`.

**Model.** In `Fetch::fetchGroup()`, stop the group at the end of the 16-byte block
that contains the first instruction's PC. Put this behind a parameter
`aligned_blocks` (default `false`).

**Check.** Loops whose start isn't 16-byte aligned should get slower. Add a
microbenchmark that pads the loop start with 1–3 `nop`s and compare aligned vs
unaligned on the RTL and on the model.

**Done when** the model shows the same aligned/unaligned difference as the RTL,
within 1 cycle per iteration.

---

### Group 2: Dispatch rules

All four tickets change `Dispatch::canDispatch()` / `Dispatch::dispatchGroup()`
in `model/core/Dispatch.cpp`. The RTL for every rule is `DispatchV2` in
`hdl/chisel/src/coralnpu/scalar/Decode.scala` (lines ~301-530 at M3); read the
`canDispatch` expression there first. Add one stall reason per rule to
`StallReason` so the summary shows the rule's cost.

#### B07: A conditional branch ends the dispatch group (~1–2 days)

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

#### B08: Some instructions dispatch alone, from slot 0 (~2 days)

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
1. Add a `dispatch_alone` flag to `Inst` (`Inst.h`), set in
   `model/frontend/InstDecode.cpp` for the instructions above, and add decoder
   tests for it in `tests/cpp/test_inst_decode.cpp`. Hint: FP-scalar vector ops
   are `OPFVF` (funct3 = 5) plus `vfmv.f.s`.
2. Add a parameter `enforce_dispatch_alone` (default `false`).
3. When it's on: an instruction with `dispatch_alone` can only dispatch if it's
   the first of the cycle, and the cycle ends right after it.

**Check.** `fp_mix` (`[fadd.s, addi, addi, addi]`) should need about 2 dispatch
cycles per group instead of 1. Compare with the RTL.

---

#### B09: Load/store addresses can't use a value produced the cycle before (~2 days)

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
   require `ResultReadyCycle + addr_extra_cycle <= Now`.
3. Record the stall as a new reason, `addr_not_registered`.

**Check.** `addr_forward` should slow down by about 1 cycle per
`addi → lw` pair. `load_chain` may also change. Compare with the RTL.

---

#### B10: CSR instructions wait for an empty Rob; confirm the retire width (~2 days)

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
   entries can retire per cycle?). Update `configs/machines/m3.yaml` with the citation.
   **Done:** 8, the whole buffer, in one cycle.

**Check.** Short-loop microbenchmarks' ROI cycles should shift by a few cycles. The
drain happens at the ROI edges, not inside the loop.

---

### Group 3: Execute (core and memory modules)

#### B11: Multiplier and FPU latencies (~1–2 days)

**Hardware.**
- **MLU:** an arbiter plus two queued stages (`scalar/Mlu.scala:67-101`), so the
  effective latency is 2 or 3 cycles.
- **FPU:** the fpnew unit with 3 pipeline registers (`float/FloatCore.scala:113`).
  Divide/sqrt is iterative.

**Steps**
1. From the B01 RTL numbers:
   - `mul_chain` cycles ÷ 4,000 ≈ multiply latency;
   - `fp_chain` cycles ÷ 4,000 ≈ FMA latency.
2. Set `mul_latency` and `fpu_latency` in `configs/machines/m3.yaml`.
3. Add `microbenchmarks/fdiv_chain.S` (dependent
   `fdiv.s`) and set `fdiv_latency` and `fdiv_occupancy`.

**Done when** `mul_chain`, `fp_chain` and `fdiv_chain` are within 5% of the RTL.

---

#### B12: The divider's latency depends on its operands (~3–5 days)

**Hardware.** The integer divider produces one bit per cycle, but first skips the
leading zeros of the dividend. Dividing a small number is much faster than
dividing a large one.

**RTL evidence.** `scalar/Dvu.scala:90-136` (`clz`, `count`).

**This ticket touches both halves of the model.**
1. **Driver** (`model/frontend/FuncSim.h` / `.cpp`):
   - add `uint32_t Rs1Val, Rs2Val` to `Inst`;
   - in `FuncSim::execute()`, fill them *before* `Proc->step(1)` from
     Spike's register file: `state->XPR[n]` for the encoding's rs1/rs2 fields.
2. **Model:**
   - copy the values into `Inst` (`model/frontend/FuncSim.cpp`);
   - in `ScalarExec::receiveInst()` (`model/core/ScalarExec.cpp`) for `DIV`, compute latency and occupancy as
     `div_base + (32 − clz(dividend))`. Use the unsigned or absolute value as the
     RTL does.
   - Keep the flat latency when a parameter `div_data_dependent` is `false`.
3. Fit `div_base` to the RTL using `div_chain`, which has large and small
   dividends.

**Done when** `div_chain` is within 5% of the RTL.

---

#### B13: LSU timing: one memory instruction at a time (~3–5 days)

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
2. Set `top.memory.params.dtcm_latency` and `dtcm_cycles_per_line`. If the
   independent-load throughput doesn't fit "one line per cycle", add a
   parameter `dtcm_min_cycles_per_access` (time the port is busy per access,
   even for a 1-line access) and use it in `TcmModel`
   (`model/memory/MemTiming.cpp`).
3. Check that stores behave the same way. Add a `store_stream` microbenchmark.

**Done when** `load_chain`, `load_stream` and `store_stream` are within 10% of
the RTL.

---

#### B14: Vector loads/stores go through the LSU one register at a time (~3–5 days)

**Status.** Done. Vector loads and stores pass through the vector pipeline
(B16/B18), one uop per register, wait in a 4-entry LSU reservation station,
and book the core's LSU per register through `VectorLsuInterface`
(docs/interfaces.md, "Vector → core (LSU)"). From RTL waveforms of
`vec_add_m1` and `vec_add_m4`: the LSU takes a load register every 4 cycles
and a store register every 3, plus 1 cycle to move on to the next
instruction (`lsu_vector_switch_cycles`); uop *i* of a dependent instruction
waits only for register *i* of its source; and a store frees its scalar
retirement-buffer entry once its first register's store completes. On the
correlation test bench (with B04-B06), `vec_add_m1` went from -20.9% to
-0.1%, `vec_add_m4` from +10.4% to -0.1%, `vec_memcpy` from -1.1% to +0.4%
and `gemv_loads` from -4.7% to +1.5%. Only unit-stride e32 accesses are
measured (deviation U6).

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
1. Add a parameter `lsu_vector_reg_overhead` (cycles per register, default `0`)
   to the LSU (`model/core/Lsu.h`).
2. In `Lsu::receiveInst()`, for `V_LOAD`/`V_STORE`, compute occupancy per register: group
   the element accesses by register (16 bytes of `vl × SEW` each), then add the
   overhead to each register's line count.
3. Fit it with `vec_memcpy` (LMUL=8, 8 registers per op) and `vec_add_m1`
   (1 register per op).
4. Optional: add a strided-load microbenchmark (stride 16 or more), where every element
   is its own line.

**Done when** `vec_memcpy`, `vec_add_m1` and `vec_add_m4` are within 10% of the
RTL on their memory-bound parts.

---

### Group 4: Vector unit

#### B15: Model the real vector functional units (~1 week)

**Status.** Done for arithmetic, in `model/vector/Vector.cpp` and
`VectorPipeline.cpp`: one pool and reservation station per functional unit
type, M3's uop counts (by EMUL, not `vl`), one vector ROB entry for compares
and reductions, and latencies from `microbenchmarks/vector_units/` (all
within 0.3% of the RTL except the small-dividend divides, deviation U5).
With the fetch fixes B04-B06 applied, `gemv_int8` went from -29.5% to
-11.2%, and to -7.5% with B14's vector loads. The rest is in the scalar loop
tail (a branch alone in its dispatch group, the taken-branch redirect).

Instead of a `vec_unified` switch, the organization is configuration: the
reservation stations, the pools they feed, and the pool each instruction
group runs on (`vec_stations`, `vec_pools`, `vec_<group>_pool`).
`configs/machines/m3_unified_vector.yaml`, applied after `m3.yaml`, is the
old design: two general-purpose lanes behind one shared station.

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
1. In `Vector` (`model/vector/Vector.cpp`), replace the single `lanes_` `ResourcePool` with one pool per unit type,
   each with its own `count`, `latency` and `cycles_per_uop` parameters. Keep a
   parameter `vec_unified` (default `true`) that restores today's single pool.
2. Map the vector classes to pools: `V_ALU` → ALU, `V_MUL` → MUL, `V_DIV` → DIV,
   `V_FP` → FMA, `V_FDIV` → FDIV, `V_PERM` and `V_TO_SCALAR` → permute/reduce.
3. Latencies: write small microbenchmarks (one per class, dependent chain and
   independent stream) and measure them on the RTL. Copy the style of
   `mul_chain` / `alu_indep`.

**Done when** `vec_add_m1`, `vec_add_m4` and `gemv_int8` are within 15% of the
RTL, and the per-class microbenchmarks are within 10%.

---

#### B16: Vector decode and dispatch widths (~1 week) — done

**Status.** Done, together with B18, in
`model/vector/VectorPipeline.cpp`: one in-order pipeline that computes each
instruction's path when it arrives, instead of a separate decode unit (a
port between stages would only add latency). Vector loads and stores pass
through it too (B14). Check: `microbenchmarks/vector_units/vmixed_lmul`
(LMUL=8 and LMUL=1 adds, a `vsetvli` between them) is within 0.1% of the RTL.

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
1. Add a Sparta unit for these stages in front of the vector lanes:
   `model/vector/VectorDecode.h` and `.cpp`. The vector module then has two
   units, so give it a `Module` subclass like `Core` (see
   docs/interfaces.md, "Adding a unit to a module") and point its public
   input port at the new unit. Copy the structure of `Rob` (queue + tick
   event).
2. It issues uops into the B15 pools. Nothing outside the vector module
   changes.
3. Parameters: `decode_width`, `max_uops_per_decode_cycle`,
   `uop_queue_entries`, `uop_dispatch_width`.

**Done when** vector microbenchmarks are no worse than after B15, and a new microbenchmark
with mixed LMUL=8 and LMUL=1 ops matches the RTL within 15%.

---

#### B17: Crossing between the scalar core and the vector unit (~3–5 days)

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
1. Write a ping-pong microbenchmark:
   `vsetvli → vmv.s.x → vredsum → vmv.x.s → (scalar use) → loop`. Measure it on
   the RTL.
2. Set `vec_to_scalar_latency`, and add a `vset_latency` parameter for `VSET`
   (today it's timed as a 1-cycle ALU op).
3. Check `gemv_int8`.

**Done when** the ping-pong microbenchmark is within 10% and `gemv_int8` within 15%.

---

#### B18: Vector reorder buffer (~2–3 days) — done

**Status.** The vector ROB is modeled with B16 (8 uops, 4 retired per
cycle), and a vector instruction completes for the core's Rob when its last
uop retires from it (deviation U8), except a store, which completes with its
first register (B14). Check: `vdiv_m8_indep` and `vdiv_m8_then_vadd_m8` in
`microbenchmarks/vector_units/` are within 3.2% of the RTL. They showed that
the scalar retirement buffer matches vector writes by PC, so a loop's later
iteration retires when an earlier one with the same PC does. That is an RTL
bug (RB1 in `rtl_bugs.md` at the top of the Reef repository, fixed upstream
after M3); the model reproduces it, and `vec_completion_matches_pc: false`
turns it off. The same waveforms showed that a vector load or store takes a
scalar LSU queue entry at dispatch, until its first register starts, which
took `gemv_loads` from +1.5% to -0.6%.

**Hardware.** Besides the 8-entry scalar retirement buffer, the vector backend has
its own **8-entry** reorder buffer. Long vector ops can fill it and stall new
vector work even when the scalar side has room.

**RTL evidence.** `rvv_backend_define.svh:35` (`ROB_DEPTH 8`); deviation U8.

**Steps**
1. Track in-flight vector uops (from issue until complete).
2. Stop issuing new vector uops when there are `vrob_entries` in flight.
3. Report a stall counter for it.

**Done when** a microbenchmark with long independent vector ops (e.g. `vdiv` at LMUL=8)
matches the RTL within 15%.

---

### Group 5: Tooling

#### B19: Per-instruction timing trace (~2 days)

**Why.** When a microbenchmark doesn't correlate, you need to compare instruction by
instruction with the RTL's `--instr_trace` log.

**Steps**
1. Add an option `--inst-trace FILE` (`model/main.cpp`). When set,
   `Rob::retireInsts()` writes one line per retired instruction:
   `seq pc disasm fetch dispatch issue ready complete retire`.
2. Write `scripts/diff_trace.py`. It lines up the model trace with the RTL
   `--instr_trace` log by PC sequence, and prints the first 20 places where the
   gap between consecutive retirements differs. Put the parsing logic in a
   `reef_perf` module so it can be unit-tested.

**Done when** you can point at the first instruction where the model and the RTL
diverge for any microbenchmark.

---

#### B20: Complete the instruction decoder tests (~2–3 days)

**Why.** If `inst_decode.cpp` gets a register wrong, dependency stalls come out
wrong everywhere, and the error is silent.

`tests/cpp/test_inst_decode.cpp` already covers a first set of instructions
(integer ALU, mul/div, loads/stores, `csrr`, `fadd.s`, `vsetvli`, `vle8.v`,
`vadd.vv` at LMUL 1 and 4, masked ops, `vmacc.vv`, `vredsum.vs`, `vmv.x.s`).

**Steps**
1. Add Google Test cases to `tests/cpp/test_inst_decode.cpp` that build `Inst`
   objects from known encodings and check `cls`, `srcs` and `dsts`.
2. Get the encodings from `riscv64-linux-gnu-objdump -d` on the microbenchmarks
   (`build/microbenchmarks/*.elf`).
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

---

## Part 2: Long-term tickets

**Status after the initial infrastructure PR:**

| Status | Tickets |
| --- | --- |
| Done | T01 (Docker image + pinned submodules), T04 (`FuncSim`), T07 (CMake/Sparta project, tests, lint, Doxygen), T08 (execute-at-fetch end to end) |
| Obsolete | T06: Spike already implements Zve32f |
| Partly done | T03 (microbenchmarks in `microbenchmarks/`), T10–T15 (super-coarse versions of every unit), T19 (the `correlation` experiment and its PR bot exist; RTL numbers come from B01) |
| Open | Everything else |

**Sizes** are rough, for one engineer:

| Size | Effort |
| --- | --- |
| S | Up to 3 days |
| M | 1–2 weeks |
| L | 2–4 weeks |

**Epics:**

| Epic | Tickets | Covers |
| --- | --- | --- |
| A | T01–T03 | Setup and reference |
| B | T04–T06 | Functional simulator (Spike) |
| C | T07–T15 | Model implementation |
| D | T16–T19 | RTL measurement and the correlation loop |
| E | T20–T25 | Calibration and sign-off |

**Dependency overview:**

```
T01 ─┬─ T02 ─┬─ T16 ─┬─ T17 ──────────────┐
     │       │       └─ T18               │
     │       └─ T03 ────────────┐         │
     ├─ T04 ─┬─ T08 ─┬─ T09     │         │
     │       │       ├─ T10..T14 ─ T19 ───┴─ T20..T24 ─ T25
     │       │       └─ T15                              ▲
     │       └─ T05                                      │
     ├─ T06 ─────────────────────────────────────────────┘
     └─ T07 ─ T08
```

---

### Epic A: Setup and reference

#### T01: Pin all sources and create a reproducible build environment (S) — done

**Why.** Correlation only means something if everyone compares against the same
RTL and the same simulator (deviations.md, M7).

**What was done.** Spike (`fd72ee2d`, the commit M3 pins), Sparta
(`map_v3.0.2`) and yaml-cpp (`0.8.0`) are submodules in `reef-perf/ext/`. The
Docker image (`Dockerfile`, `tools/docker.sh` at the repository root) builds them, with no
sudo and no host packages. The `coralnpu` M3 checkout (tag `M3-2026-04-27`) is
only needed for correlation (T02) and is not part of the build.

**Depends on:** none.

---

#### T02: Build and run the M3 RTL simulators (M)

**Why.** The RTL simulation is our ground truth. We need it running before we can
compare anything.

**Tasks**
- Build the M3 Verilator simulators for `CoreMini` and `RvvCoreMini`, in both the
  normal and the `Verification` flavor, using the **highmem (1 MB/1 MB)**
  variants. The default 8 KB instruction memory is too small for real kernels.
- Build one trivial test program with the M3 toolchain, run it on each simulator,
  and read back its `mcycle` value.
- Write down the exact commands and any problems hit, e.g. Bazel output paths
  and backdoor ELF loading.

**Done when** `scripts/run_rtl.sh <elf> <config>` prints a cycle count for all
four simulator builds.

**Depends on:** T01.

---

#### T03: Microbenchmark corpus v0 (M)

**Why.** We need a fixed set of programs that runs on both the RTL and Spike,
from tiny loops up to LLM-shaped kernels. At M3 the M3 repo has few ML kernels
(M5).

**Already there:** 15 microbenchmarks plus `gemv_int8` in
`microbenchmarks/`, assembled by `microbenchmarks.py`.

**Tasks**
- Add C microbenchmarks built with the M3 toolchain
  (`-march=rv32imf_zve32f_zicsr_zifencei_zbb`).
- **Microbenchmark set.** Small loops, each isolating one feature. Examples:
  - an independent ALU stream;
  - a dependent ALU chain;
  - the branch patterns `[alu, beq, alu, alu]`, taken and not taken;
  - an FP stream;
  - load chains;
  - stride sweeps;
  - `vadd` at each LMUL;
  - `vle`/`vse` at each addressing mode.
- **Kernel set.**
  - M3's `rvv_matmul` (int8).
  - Integer GEMV and matmul at transformer shapes taken from
    `reef/microbenchmarks` / `parameter.py` (e.g. d_model × d_ff slices sized to fit
    DTCM).
  - Post-M3 Gemma kernels ported to M3 where they don't need BF16:
    `rms_norm`, `residual_add`, int8 matmul.
- Tag each microbenchmark `scalar`, `rvv-int` or `rvv-fp`.

**Done when** every microbenchmark builds, runs to completion on the M3 RTL simulator
with a self-check passing, and is listed in the microbenchmark table in
`python/reef_perf/experiments/run/README.md`.

**Depends on:** T02.

---

### Epic B: Functional simulator (Spike)

#### T04: Spike step API for the perf model (M) — done

**Why.** Execute-at-fetch needs to tell the functional simulator "execute the
next instruction and tell me everything about it".

**What was done.** `FuncSim` (`model/frontend/FuncSim.h`) embeds Spike as a
library. `next()` returns a decoded `Inst`: PC, encoding, disassembly, next PC,
per-element memory accesses and registers written (from Spike's commit log),
`vl`/SEW/LMUL. It stops
at `mpause` and throws on traps. `tools/reef_trace/reef_trace.cpp` prints the stream.

**Depends on:** T01.

---

#### T05: Check Spike against the M3 RTL architecturally (M)

**Why.** If Spike computes something differently from the RTL (a different branch
outcome, a different `vl`), the model times the wrong instruction stream, and no
calibration can fix that (F3, F5).

**Tasks**
- Use the M3 verification simulator's retire trace (`--instr_trace`, which gives
  PC, instruction and written value) and the T04 record stream.
- Compare them instruction by instruction on the whole T03 corpus.
- Report the first mismatch per microbenchmark.
- File a bug for each real Spike or RTL difference. Known candidates:
  - `flushat`/`flushall` (M3 decodes them; Spike doesn't);
  - CoralNPU's custom CSRs (F2);
  - CSR read values and `misa`.

**Done when** every corpus microbenchmark retires the same PC sequence and register
writes in both, or each difference has a filed issue and a documented
workaround.

**Depends on:** T02, T03, T04.

---

#### T06: Vector FP32 support in the functional simulator — obsolete

This ticket existed because MPACT lacked Zve32f. Spike implements it, and the
driver's default ISA string (`DefaultISA`) enables it, so FP kernels such as
`rvv_float_matmul` run today.

---

### Epic C: Model implementation

#### T07: Sparta project skeleton (S) — done, except CI

**What was done.** `reef-perf/CMakeLists.txt` builds against Sparta from
`ext/map`; layout as in overview.md §4.5; `reef_perf` takes
`--elf`, `--json`, `-c` and Sparta's standard options. Tests, lint
(clang-tidy, clang-format, Ruff) and Doxygen run under `uv run pytest`.

**Remaining:** a GitHub Actions job that builds the Docker image (cache the
`deps` stage) and runs `uv run pytest`.

**Depends on:** T01.

---

#### T08: Execute-at-fetch driver and end-to-end run (M) — done

**What was done.** Fetch calls `FuncSim::next()` once per instruction; Dispatch,
Execute and Rob carry it to retirement; the run stops at `mpause`.
`tests/python/integration/test_sim.py` checks that the retired count equals Spike's.

**Depends on:** T04, T07.

---

#### T09: Instruction classification table (M)

**Why.** The timing units need to know what kind of instruction each one is: which
unit it uses, whether a special dispatch rule applies, and how many vector uops it
becomes. Keeping this in data rather than code makes adding ISA extensions cheap
(overview.md §4.3).

**Tasks**
- Write `isa/instructions.yaml` covering every mnemonic in RV32IMF, Zbb, Zicsr,
  Zifencei, Zve32x and Zve32f. Fields:
  - `class`;
  - `alone` (must dispatch alone from slot 0);
  - `ends_group`;
  - `mem` mode;
  - `uops` rule;
  - optional `latency_model`.
- Write a loader that validates the table at startup.
- Make any mnemonic that isn't in the table a hard error at run time.
- Seed the flags from the M3 RTL:
  - every F instruction is `alone` (D3);
  - every RVV instruction with an FP scalar operand is `alone`;
  - CSR ops are `alone` with `csr_drain`;
  - conditional branches `ends_group` (D2).

**Done when** the model runs the whole corpus with no unknown-mnemonic errors, and
a unit test checks the flags for a sample of instructions.

**Depends on:** T08.

---

#### T10: Fetch unit model (M)

**Why.** At M3, fetch is likely the single largest limit on scalar speed. It can
start a read only every other cycle, which caps IPC near 2 (D7). Branch handling
adds more lost cycles (D6, U1).

**Tasks**
- Model 16-byte fetch blocks. Instructions that start partway into a block only
  deliver the rest of that block.
- Parameters:
  - `fetch_issue_interval`;
  - `max_outstanding_fetches`;
  - an instruction buffer of `ibuf_entries`.
- Static prediction (`static_btfn`): `jal` taken, backward conditional branches
  taken, forward ones not taken. `jalr` is never predicted.
- On a mispredict (known immediately from the Spike record), stall fetch until
  the branch resolves in the BRU, plus `mispredict_penalty`.
- On a correctly predicted taken branch, add `taken_bubble`.
- Add counters: fetch blocks, taken branches, mispredicts, cycles with an empty
  buffer.

**Done when** unit tests with hand-computed expected cycles pass (e.g. a
straight-line ALU stream reaches exactly 2 IPC with the M3 seed values), and the
parameters appear in the config.

**Depends on:** T08, T09.

---

#### T11: Dispatch unit model (M)

**Why.** Dispatch applies all of the M3 baseline's issue rules. Several of them aren't in
the docs (D2, D3, D4) and have a large effect on speed.

**Tasks**
- Each cycle, consider the oldest `width` instructions in order. Stop at the
  first that can't dispatch.
- Implement these rules, each switchable by a config flag:
  - integer and FP register scoreboards (RAW and WAW hazards);
  - `branch_ends_group`;
  - `alone` classes only from slot 0, with nothing else that cycle;
  - `addr_needs_registered_value`: a load/store base register or `jalr` source
    produced last cycle blocks dispatch;
  - per-unit per-cycle limits (e.g. 1 MLU);
  - LSU queue space;
  - vector command-queue space;
  - retirement-buffer space;
  - CSR waits for an empty retirement buffer.
- Record one stall reason per cycle from the list in overview.md
  §4.2.

**Done when** unit tests covering each rule pass, and the stall-reason counters
add up to (total cycles − cycles with at least one instruction dispatched).

**Depends on:** T08, T09.

---

#### T12: Scalar execution units and retirement buffer (S)

**Why.** Once dispatched, an instruction must take the right time to produce its
result, and the retirement buffer limits how many can be in flight.

**Tasks**
- Model ALU, BRU, MLU, DVU and FPU pools, each with `count`, `latency` and
  `issue_interval`.
- Model the DVU `latency_model: dvu_clz`, which computes latency from the operand
  values, which `FuncSim` reads from Spike's registers (the divider skips
  leading zeros; U10).
- Model FPU div/sqrt as non-pipelined.
- Add a retirement buffer with `rob_entries` and `retire_width`. Instructions
  retire in order once complete. Stores complete when the LSU reports it.
- Report results back to the scoreboard at `latency`.

**Done when** unit tests pass for dependent chains of each unit type (e.g. a chain
of N dependent `mul`s takes N × latency), and the stall counter for a full
retirement buffer works.

**Depends on:** T11.

---

#### T13: LSU and memory model (M)

**Why.** Every load and store, scalar or vector, goes through one LSU that handles
one instruction at a time (D5). For LLM workloads, memory behavior usually
decides performance.

**Tasks**
- Add a queue of `queue_entries` and a single slot processing one memory
  instruction at a time.
- Cost of an instruction:
  - a scalar access is one line transaction, or two if it crosses a 16-byte
    boundary;
  - a vector access is, for each register's worth of elements, the number of
    distinct 16-byte lines touched plus `vector_per_register_overhead`.
- Use the per-element addresses in `Inst::Mem` to count lines exactly.
- Memory regions:
  - ITCM and DTCM: fixed latency;
  - external: `ext_latency` plus bytes ÷ `ext_bytes_per_cycle`.
- Loads write the scoreboard at `load_to_use`.
- Add counters: lines per instruction, bytes moved per region, cycles the LSU is
  busy.

**Done when** unit tests pass for:
- a unit-stride aligned `vle8` at LMUL=1 and LMUL=4;
- a strided load with stride 16;
- a misaligned scalar load that crosses a line;
- a load from external memory.

The line counts must match hand calculations.

**Depends on:** T11.

---

#### T14: Vector unit model (L)

**Why.** Most ML work runs here. The unit is decoupled from the scalar core, so
its queues, uop splitting and functional-unit counts decide whether the scalar
core stalls.

**Tasks**
- Command queue with `cmd_queue_entries`, filled by Dispatch.
- Decode of `decode_width` instructions per cycle into uops using each class's
  `uops` rule. The rules:
  - `per_register`: `ceil(vl × SEW / 128)`, times 2 for widening or narrowing
    instructions;
  - `tree`: for reductions;
  - `fixed:N`.
- A uop queue, then dispatch of `dispatch_width` uops per cycle into functional
  unit pools (ALU, MUL, DIV, FMA, FDIV, permute/reduce). Each pool has `count`,
  `rs_entries`, `latency` and `issue_interval`.
- Uop dependencies through a vector register scoreboard.
- A vector ROB of `vrob_entries`.
- Vector loads and stores go to the LSU (T13), and complete when the LSU finishes
  them.
- Instructions that write a scalar or FP register (`vmv.x.s`, `vfmv.f.s`,
  `vsetvl*`, reductions to scalar) write the scalar scoreboard after
  `vector_to_scalar_latency`.
- Add counters: queue-full stalls, busy cycles per pool, uops per instruction.

**Done when** unit tests pass for:
- independent `vadd.vv` at LMUL 1 and 4 (throughput);
- a dependent `vmacc` chain (latency);
- a `vle8` → `vadd` → `vse8` sequence;
- a `vredsum` → `vmv.x.s` → scalar branch sequence.

**Depends on:** T11, T13.

---

#### T15: Statistics, pipeline trace and JSON report (S)

**Why.** Correlation and design exploration both depend on seeing *where* the
cycles went, not just the total.

**Tasks**
- End-of-run JSON summary:
  - cycles, instructions, IPC;
  - per-unit utilization;
  - the stall breakdown by reason;
  - memory bytes per region;
  - vector uop counts.
- Turn on Sparta's pipeline collection so a run can be viewed as a pipeline
  diagram. Document how to open it.
- Add `--roi-start`/`--roi-end` PC options, so only the region of interest is
  measured. This matches the `mcycle` brackets used on the RTL.

**Done when** a T03 kernel produces the JSON report and a viewable pipeline trace,
and ROI cycle counts exclude setup code.

**Depends on:** T08.

---

### Epic D: RTL measurement and the correlation loop

#### T16: RTL cycle-measurement harness (S)

**Why.** We need a trusted, repeatable way to get cycle counts from the RTL for any
microbenchmark.

**Tasks**
- Write a small C/assembly header with `ROI_BEGIN()`/`ROI_END()` macros that read
  `mcycle` and store the delta at a known memory address.
- Update the T03 microbenchmarks to use it.
- Extend `scripts/run_rtl.sh` to read the delta back and output JSON.
- Measure the fixed cost of the brackets themselves with an empty ROI, and
  subtract it. Reading `mcycle` drains the pipeline (M2), so microbenchmark loops
  must be long, e.g. ≥1000 iterations.

**Done when** repeated runs of the same microbenchmark give identical cycle counts, and
the harness output feeds T19.

**Depends on:** T02.

---

#### T17: Per-instruction cycle benchmarks on M3 (M)

**Why.** Calibration needs the latency and throughput of each instruction class
on the real RTL. M3 has no such benchmark (M4); `isa_cycle_bench` was added after
M3.

**Tasks**
- Backport `tests/cocotb/isa_cycle_bench.cc` from coralnpu main to build against
  M3.
- For every benchmarked instruction, add a **throughput** variant (independent
  instances) next to the existing **latency** variant (dependent chain).
- Cover every class in `instructions.yaml`, including vector ops at SEW 8/16/32
  and LMUL 1/2/4/8.
- Write the results into a table
  (`python/reef_perf/experiments/correlation/data/rtl_isa_table.json`) that later
  tickets use to fill in parameters.

**Done when** the table has latency and throughput for every instruction class,
with a note on anything that couldn't be measured.

**Depends on:** T16.

---

#### T18: Per-cycle event trace from the RTL (M)

**Why.** When the model and the RTL disagree, we need to see cycle by cycle what the
RTL did, to know which model unit is wrong.

**Tasks**
- Pick a small set of RTL signals: dispatch-fire per lane, instruction-buffer
  valid count, LSU bus request/response, vector command-queue and uop-queue
  occupancy, retirement-buffer occupancy.
- Get them per cycle, either from an FST waveform dump (`--trace`) with a Python
  extractor, or with a lightweight Verilator-side logger. Output a per-cycle CSV.
- Write a script that shows the model's per-cycle state next to the RTL's for
  the same microbenchmark.
- Use it to **confirm M1**: that the verification and normal RTL builds give the
  same cycle counts on the corpus.

**Done when** for one microbenchmark we can print "cycle N: RTL dispatched 2, model
dispatched 3", and M1 is marked measured in deviations.md.

**Depends on:** T16.

---

#### T19: Correlation runner and report (S)

**Why.** Automates the loop that all calibration tickets use: run everything,
compare, show the error.

**Tasks**
- Extend the `correlation` experiment (beginner ticket B03,
  `python/reef_perf/experiments/correlation/`). It runs each microbenchmark on
  the RTL (T16) and on the model (T15) with the matching preset, and produces a table: microbenchmark, RTL
  cycles, model cycles, error %, and the model's top 3 stall reasons.
- Add summary rows: mean absolute error per microbenchmark tag (`scalar`, `rvv-int`,
  `rvv-fp`).
- Run in CI on every reef-perf PR (done: `.github/workflows/reef-perf-correlation.yml`). Store results so trends are visible.
- Record the first report (seed parameters) as the baseline.

**Done when** CI posts the correlation table, and the baseline is committed in
the `correlation` experiment directory.

**Depends on:** T10–T16.

---

### Epic E: Calibration and sign-off

Each calibration ticket follows the same method:

1. Use T17 benchmarks and targeted T03 microbenchmarks to measure the behavior
   on the RTL.
2. Set the model parameters and rules to match.
3. Confirm the change with T19, and debug with T18 when it doesn't match.
4. Mark the related deviations.md entries `measured`, with a link to the result.

#### T20: Calibrate fetch and branches (M)

**Covers:** D6, D7, U1.

**Tasks**
- Measure:
  - straight-line code IPC, which confirms the fetch-every-other-cycle cap;
  - loop cost for bodies of 1–16 instructions;
  - forward-taken vs forward-not-taken branches;
  - backward-taken branches;
  - `jal`, and `jalr` call/return.
- Set `fetch_issue_interval`, `taken_bubble` and `mispredict_penalty`.

**Done when** the branch and loop microbenchmarks are within 1 cycle per iteration
of the RTL.

**Depends on:** T17, T19.

---

#### T21: Calibrate dispatch rules (M)

**Covers:** D2, D3, D4, M2.

**Tasks**
- Write microbenchmarks where each rule predicts a different cycle count
  depending on whether it's on or off. Examples:
  - `[alu, beq(not taken), alu, alu]` repeated;
  - `[fadd.s, alu, alu, alu]` repeated;
  - an address chain `addi a0 → lw 0(a0)` vs a value chain `addi a0 → add`;
  - 4 back-to-back independent loads;
  - a loop with a CSR read inside.
- Confirm each rule on the RTL and set the flags.

**Done when** all dispatch-rule microbenchmarks are within 5%, and the model's
stall breakdown for them names the expected rule.

**Depends on:** T17, T19.

---

#### T22: Calibrate scalar units and the retirement buffer (S)

**Covers:** D8, U2, U10, and the scalar part of U8.

**Tasks**
- From T17: set MLU, FPU (add/mul/fma/div/sqrt/convert) and DVU latency and issue
  interval.
- Fit the DVU `dvu_clz` formula against operand-size sweeps.
- Measure retirement-buffer stalls with a long-latency op (e.g. `div`) followed
  by N independent ALU ops, and set `rob_entries` behavior and `retire_width`.

**Done when** per-instruction latency and throughput match the T17 table exactly
for scalar instructions.

**Depends on:** T17, T19.

---

#### T23: Calibrate the LSU and memory (M)

**Covers:** D5, U3, U6, U7, M3.

**Tasks**
- Scalar: measure load-to-use latency, line-crossing cost, and store completion
  timing.
- Vector: measure per-register cost for unit-stride (aligned and misaligned),
  strided (stride sweep), indexed and segment loads and stores, at LMUL 1–8.
- Determine whether the backend's second LSU path matters. The scalar LSU only
  services one `rvv2lsu` port (U7).
- External memory: measure the testbench's latency and bandwidth, and set
  `ext_latency`/`ext_bytes_per_cycle`. Label them as testbench values.

**Done when** memory microbenchmarks are within 10%, and a DTCM-resident int8 GEMV
kernel is within 10%.

**Depends on:** T17, T19.

---

#### T24: Calibrate the vector unit (L)

**Covers:** U4, U5, U8 (vector part), U9.

**Tasks**
- Measure the scalar↔vector crossing costs: from dispatch to vector execution
  start, the cost of `vsetvli`, and vector→scalar writeback (`vmv.x.s`,
  reductions, `vfmv.f.s`).
- Fill in latency and throughput per functional-unit pool and per SEW/LMUL from
  T17.
- Check the uop-splitting rules against observed uop-queue activity from T18,
  especially for widening, narrowing, reductions, permutes and masks.
- Measure how an 8-entry vector ROB and 8-entry command queue limit overlap of
  long operations.
- Decide whether vector register-file read-port conflicts (U9) need modeling.
  Only add it if kernels are off by more than 5% because of it.

**Done when** vector microbenchmarks are within 10%, and `rvv_matmul` (int8) is
within 10%.

**Depends on:** T17, T18, T19.

---

#### T25: Kernel-level correlation sign-off (M)

**Why.** This closes the first model. It confirms the model is good enough to use
for design exploration, and it records how far it can be trusted.

**Tasks**
- Run the full corpus, including `rvv-fp` microbenchmarks once T06 is done.
- Reach **±10% total cycles** on every kernel (int8 matmul, int8 GEMV,
  `rms_norm`, `residual_add`, and float matmul).
- For each remaining miss, identify the cause with T18, and either fix it or
  record it as an accepted limitation.
- Set the error budget for each planned abstraction (deviations.md A1–A4).
- Write `docs/model-limitations.md`: what the model is trusted for, what it
  isn't, and which parameters were measured vs estimated.
- Tag the release `reef-perf-m3-v1`, with the correlation report attached.

**Done when** the kernel correlation table meets ±10%, or every exception is
documented and approved, and the tag exists.

**Depends on:** T06, T20–T24.

# reef_perf Roadmap Tickets (M3 baseline)

These 25 tickets take reef_perf from nothing to a coarse model that correlates
within ±10% against the M3 RTL (Reef's CoralNPU baseline) on our kernels.
The near-term work for new team members is in
[tickets-beginner.md](tickets-beginner.md); this file is the longer roadmap.

- Background for all of them: [implementation-plan.md](implementation-plan.md).
- IDs such as `D7` or `U3` refer to entries in [deviations.md](deviations.md).

**Status after the initial infrastructure PR:**

| Status | Tickets |
| --- | --- |
| Done | T01 (Docker image + pinned submodules), T04 (`SpikeDriver`), T07 (CMake/Sparta project, tests, lint, Doxygen), T08 (execute-at-fetch end to end) |
| Obsolete | T06: Spike already implements Zve32f |
| Partly done | T03 (workloads in `src/reef_perf/experiments/run/workloads/`), T10–T15 (super-coarse versions of every unit), T19 (the `run` experiment; correlation is beginner ticket B03) |
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

## Epic A: Setup and reference

### T01: Pin all sources and create a reproducible build environment (S) — done

**Why.** Correlation only means something if everyone compares against the same
RTL and the same simulator (deviations.md, M7).

**What was done.** Spike (`fd72ee2d`, the commit M3 pins), Sparta
(`map_v3.0.2`) and yaml-cpp (`0.8.0`) are submodules in `reef-perf/ext/`. The
Docker image (`reef-perf/Dockerfile`, `tools/docker.sh`) builds them, with no
sudo and no host packages. The `coralnpu` M3 checkout (tag `M3-2026-04-27`) is
only needed for correlation (T02) and is not part of the build.

**Depends on:** none.

---

### T02: Build and run the M3 RTL simulators (M)

**Why.** The RTL simulation is our ground truth. We need it running before we can
compare anything.

**Tasks**
- Build the M3 Verilator simulators for `CoreMini` and `RvvCoreMini`, in both the
  normal and the `Verification` flavour, using the **highmem (1 MB/1 MB)**
  variants. The default 8 KB instruction memory is too small for real kernels.
- Build one trivial test program with the M3 toolchain, run it on each simulator,
  and read back its `mcycle` value.
- Write down the exact commands and any problems hit, e.g. Bazel output paths
  and backdoor ELF loading.

**Done when** `tools/run_rtl.sh <elf> <config>` prints a cycle count for all
four simulator builds.

**Depends on:** T01.

---

### T03: Workload corpus v0 (M)

**Why.** We need a fixed set of programs that runs on both the RTL and Spike,
from tiny loops up to LLM-shaped kernels. At M3 the M3 repo has few ML kernels
(M5).

**Already there:** 15 microbenchmarks plus `gemv_int8` in
`src/reef_perf/experiments/run/workloads/`, assembled by `workloads.py`.

**Tasks**
- Add C workloads built with the M3 toolchain
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
    `reef/workloads` / `parameter.py` (e.g. d_model × d_ff slices sized to fit
    DTCM).
  - Post-M3 Gemma kernels ported to M3 where they don't need BF16:
    `rms_norm`, `residual_add`, int8 matmul.
- Tag each workload `scalar`, `rvv-int` or `rvv-fp`.

**Done when** every workload builds, runs to completion on the M3 RTL simulator
with a self-check passing, and is listed in the workload table in
`src/reef_perf/experiments/run/README.md`.

**Depends on:** T02.

---

## Epic B: Functional simulator (Spike)

### T04: Spike step API for the perf model (M) — done

**Why.** Execute-at-fetch needs to tell the functional simulator "execute the
next instruction and tell me everything about it".

**What was done.** `SpikeDriver` (`spike_driver.hpp`) embeds Spike as a
library. `step()` returns an `InstRecord`: PC, encoding, disassembly, next PC,
per-element memory accesses (from Spike's commit log), `vl`/SEW/LMUL. It stops
at `mpause` and throws on traps. `tools/reef_trace.cpp` prints the stream.

**Depends on:** T01.

---

### T05: Check Spike against the M3 RTL architecturally (M)

**Why.** If Spike computes something differently from the RTL (a different branch
outcome, a different `vl`), the model times the wrong instruction stream, and no
calibration can fix that (F3, F5).

**Tasks**
- Use the M3 verification simulator's retire trace (`--instr_trace`, which gives
  PC, instruction and written value) and the T04 record stream.
- Compare them instruction by instruction on the whole T03 corpus.
- Report the first mismatch per workload.
- File a bug for each real Spike or RTL difference. Known candidates:
  - `flushat`/`flushall` (M3 decodes them; Spike doesn't);
  - CoralNPU's custom CSRs (F2);
  - CSR read values and `misa`.

**Done when** every corpus workload retires the same PC sequence and register
writes in both, or each difference has a filed issue and a documented
workaround.

**Depends on:** T02, T03, T04.

---

### T06: Vector FP32 support in the functional simulator — obsolete

This ticket existed because MPACT lacked Zve32f. Spike implements it, and the
driver's default ISA string (`kDefaultIsa`) enables it, so FP kernels such as
`rvv_float_matmul` run today.

---

## Epic C: Model implementation

### T07: Sparta project skeleton (S) — done, except CI

**What was done.** `reef-perf/CMakeLists.txt` builds against Sparta from
`ext/map`; layout as in implementation-plan.md §4.5; `reef_perf` takes
`--elf`, `--json`, `-c` and Sparta's standard options. Tests, lint
(clang-tidy, clang-format, Ruff) and Doxygen run under `uv run pytest`.

**Remaining:** a GitHub Actions job that builds the Docker image (cache the
`deps` stage) and runs `uv run pytest`.

**Depends on:** T01.

---

### T08: Execute-at-fetch driver and end-to-end run (M) — done

**What was done.** Fetch calls `FuncSim::next()` once per instruction; Dispatch,
Execute and Rob carry it to retirement; the run stops at `mpause`.
`tests/python/test_sim.py` checks that the retired count equals Spike's.

**Depends on:** T04, T07.

---

### T09: Instruction classification table (M)

**Why.** The timing units need to know what kind of instruction each one is: which
unit it uses, whether a special dispatch rule applies, and how many vector uops it
becomes. Keeping this in data rather than code makes adding ISA extensions cheap
(implementation-plan.md §4.3).

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

### T10: Fetch unit model (M)

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

### T11: Dispatch unit model (M)

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
- Record one stall reason per cycle from the list in implementation-plan.md
  §4.2.

**Done when** unit tests covering each rule pass, and the stall-reason counters
add up to (total cycles − cycles with at least one instruction dispatched).

**Depends on:** T08, T09.

---

### T12: Scalar execution units and retirement buffer (S)

**Why.** Once dispatched, an instruction must take the right time to produce its
result, and the retirement buffer limits how many can be in flight.

**Tasks**
- Model ALU, BRU, MLU, DVU and FPU pools, each with `count`, `latency` and
  `issue_interval`.
- Model the DVU `latency_model: dvu_clz`, which computes latency from the operand
  values, which `SpikeDriver` reads from Spike's registers (the divider skips
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

### T13: LSU and memory model (M)

**Why.** Every load and store, scalar or vector, goes through one LSU that handles
one instruction at a time (D5). For LLM workloads, memory behaviour usually
decides performance.

**Tasks**
- Add a queue of `queue_entries` and a single slot processing one memory
  instruction at a time.
- Cost of an instruction:
  - a scalar access is one line transaction, or two if it crosses a 16-byte
    boundary;
  - a vector access is, for each register's worth of elements, the number of
    distinct 16-byte lines touched plus `vector_per_register_overhead`.
- Use the per-element addresses in `Inst::mem` to count lines exactly.
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

### T14: Vector unit model (L)

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

### T15: Statistics, pipeline trace and JSON report (S)

**Why.** Correlation and design exploration both depend on seeing *where* the
cycles went, not just the total.

**Tasks**
- End-of-run JSON summary:
  - cycles, instructions, IPC;
  - per-unit utilisation;
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

## Epic D: RTL measurement and the correlation loop

### T16: RTL cycle-measurement harness (S)

**Why.** We need a trusted, repeatable way to get cycle counts from the RTL for any
workload.

**Tasks**
- Write a small C/assembly header with `ROI_BEGIN()`/`ROI_END()` macros that read
  `mcycle` and store the delta at a known memory address.
- Update the T03 workloads to use it.
- Extend `tools/run_rtl.sh` to read the delta back and output JSON.
- Measure the fixed cost of the brackets themselves with an empty ROI, and
  subtract it. Reading `mcycle` drains the pipeline (M2), so microbenchmark loops
  must be long, e.g. ≥1000 iterations.

**Done when** repeated runs of the same workload give identical cycle counts, and
the harness output feeds T19.

**Depends on:** T02.

---

### T17: Per-instruction cycle benchmarks on M3 (M)

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
  (`src/reef_perf/experiments/correlate/data/rtl_isa_table.json`) that later
  tickets use to fill in parameters.

**Done when** the table has latency and throughput for every instruction class,
with a note on anything that couldn't be measured.

**Depends on:** T16.

---

### T18: Per-cycle event trace from the RTL (M)

**Why.** When the model and the RTL disagree, we need to see cycle by cycle what the
RTL did, to know which model unit is wrong.

**Tasks**
- Pick a small set of RTL signals: dispatch-fire per lane, instruction-buffer
  valid count, LSU bus request/response, vector command-queue and uop-queue
  occupancy, retirement-buffer occupancy.
- Get them per cycle, either from an FST waveform dump (`--trace`) with a Python
  extractor, or with a lightweight Verilator-side logger. Output a per-cycle CSV.
- Write a script that shows the model's per-cycle state next to the RTL's for
  the same workload.
- Use it to **confirm M1**: that the verification and normal RTL builds give the
  same cycle counts on the corpus.

**Done when** for one workload we can print "cycle N: RTL dispatched 2, model
dispatched 3", and M1 is marked measured in deviations.md.

**Depends on:** T16.

---

### T19: Correlation runner and report (S)

**Why.** Automates the loop that all calibration tickets use: run everything,
compare, show the error.

**Tasks**
- Extend the `correlate` experiment (beginner ticket B03,
  `src/reef_perf/experiments/correlate/run.py`). It runs each workload on the RTL (T16) and on the
  model (T15) with the matching preset, and produces a table: workload, RTL
  cycles, model cycles, error %, and the model's top 3 stall reasons.
- Add summary rows: mean absolute error per workload tag (`scalar`, `rvv-int`,
  `rvv-fp`).
- Run in CI on every reef-perf PR. Store results so trends are visible.
- Record the first report (seed parameters) as the baseline.

**Done when** CI posts the correlation table, and the baseline is committed in
the `correlate` experiment directory.

**Depends on:** T10–T16.

---

## Epic E: Calibration and sign-off

Each calibration ticket follows the same method:

1. Use T17 benchmarks and targeted T03 microbenchmarks to measure the behaviour
   on the RTL.
2. Set the model parameters and rules to match.
3. Confirm the change with T19, and debug with T18 when it doesn't match.
4. Mark the related deviations.md entries `measured`, with a link to the result.

### T20: Calibrate fetch and branches (M)

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

### T21: Calibrate dispatch rules (M)

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

### T22: Calibrate scalar units and the retirement buffer (S)

**Covers:** D8, U2, U10, and the scalar part of U8.

**Tasks**
- From T17: set MLU, FPU (add/mul/fma/div/sqrt/convert) and DVU latency and issue
  interval.
- Fit the DVU `dvu_clz` formula against operand-size sweeps.
- Measure retirement-buffer stalls with a long-latency op (e.g. `div`) followed
  by N independent ALU ops, and set `rob_entries` behaviour and `retire_width`.

**Done when** per-instruction latency and throughput match the T17 table exactly
for scalar instructions.

**Depends on:** T17, T19.

---

### T23: Calibrate the LSU and memory (M)

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

### T24: Calibrate the vector unit (L)

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
- Decide whether vector register-file read-port conflicts (U9) need modelling.
  Only add it if kernels are off by more than 5% because of it.

**Done when** vector microbenchmarks are within 10%, and `rvv_matmul` (int8) is
within 10%.

**Depends on:** T17, T18, T19.

---

### T25: Kernel-level correlation sign-off (M)

**Why.** This closes the first model. It confirms the model is good enough to use
for design exploration, and it records how far it can be trusted.

**Tasks**
- Run the full corpus, including `rvv-fp` workloads once T06 is done.
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

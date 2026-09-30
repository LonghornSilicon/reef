# Coarse Performance Model for CoralNPU: Implementation Plan

This document explains what we're building, how it works, and the order in which
we'll build it. It assumes you know basic computer architecture (pipelines,
caches, RISC-V). It doesn't assume you know CoralNPU, MPACT or Sparta.

- The tickets that carry out this plan are in [tickets.md](tickets.md).
- The known disagreements between the CoralNPU RTL and its docs and simulator are
  in [deviations.md](deviations.md).

---

## 1. What we are building, and why

A **performance model** is a program that predicts how many clock cycles a piece
of hardware takes to run a piece of software, and explains where the time goes.
It is much faster to change than RTL. To ask "what if the vector unit had 4
multipliers instead of 2?", you edit one number in a config file and rerun,
instead of redesigning hardware.

Ours will:

- **Take as input** a RISC-V ELF binary (the same program you'd run on the chip)
  and a YAML config file describing the hardware.
- **Produce:**
  - the total cycle count;
  - instructions per cycle (IPC);
  - how busy each hardware block was;
  - a **stall breakdown**: for every cycle the core couldn't issue work, *why*
    (e.g. "waiting on the load/store unit", "branch redirect", "vector queue
    full").

It models **CoralNPU at release `M3-2026-04-27`**. That's the RTL we can run in
simulation and compare against.

### "Coarse" and "configurable"

- **Coarse** means each hardware block is described by a handful of numbers
  (how long an operation takes, how many can run at once, how deep a queue is)
  instead of by its exact logic. We accept some error in exchange for a model
  that's small, fast, and easy to change.
- **Configurable** means every one of those numbers lives in YAML, not in C++.
  Changing the hardware you model should almost never need a code change.

### Correlation

**Correlation** means checking the model against the real design. We run the same
program on:

1. the model, and
2. a cycle-accurate simulation of the CoralNPU RTL (Verilator, which compiles the
   Verilog into a C++ simulator),

and compare cycle counts. The difference is the **correlation error**. The goal
of the ticket plan is to drive that error down, in this order:

| Level | Target |
| --- | --- |
| Microbenchmarks (tiny loops that stress one feature) | Within 1 cycle per loop iteration, or 5% |
| Kernels (matmul, GEMV, etc.) | Within **±10%** total cycles, with a stall breakdown that points the same way as the RTL |

These targets are what we mean by "decent correlation". We'll tighten them later
if design decisions need more precision.

---

## 2. The hardware we are modelling (CoralNPU M3), in plain terms

CoralNPU is a small RISC-V processor for ML at the edge. At M3 it has two parts.

### 2.1 The scalar core

A normal RISC-V processor that runs 32-bit integer and single-precision
floating-point code (RV32IMF, plus the Zbb bit-manipulation extension).

It's **in-order**: instructions start in program order. Instructions *after* a
stalled one wait, even if they're ready. It's **4-wide**: up to 4 instructions
can start in the same cycle.

The pipeline has these stages:

1. **Fetch.**
   - Reads 16 bytes (4 instructions) at a time from instruction memory into an
     8-entry instruction buffer.
   - At M3 it can only start a new read **every other cycle**. So it supplies
     at most ~2 instructions per cycle on average, which caps the core at about
     2 IPC even though the next stage is 4 wide.
   - It guesses branch directions (**static prediction**): jumps (`jal`) and
     backward branches (typically loops) are assumed taken, and everything else
     not taken.
   - A wrong guess costs a few cycles while the right instructions are fetched.
     Wrong-path instructions are thrown away, never executed.

2. **Dispatch**, also called decode/issue.
   - Each cycle it looks at the oldest 4 instructions in the buffer and sends as
     many as it can to the execution units, stopping at the first one that can't
     go (because the core is in-order).
   - An instruction can't go if:
     - it needs a register value that isn't ready yet. A **scoreboard** tracks
       which registers are waiting on a result.
     - its execution unit is busy or its queue is full.
     - a special rule applies. There are several; for example, a branch must be
       the last instruction sent that cycle, and floating-point instructions must
       be sent alone. The full list is in section 4.2.

3. **Execute.** The execution units are:

   | Unit | Count | What it does | Speed |
   | --- | --- | --- | --- |
   | ALU | 4 | add, shift, compare, bit-manip | 1 cycle |
   | BRU (branch unit) | 4 | branches and jumps | 1 cycle |
   | MLU (multiplier) | 1 | `mul*` | ~2–3 cycles, pipelined |
   | DVU (divider) | 1 | `div*`/`rem*` | up to ~32 cycles, one at a time |
   | FPU | 1 | FP32 math | ~3-cycle pipeline; divide/sqrt slower and iterative |
   | LSU (load/store unit) | 1 | all memory accesses | see below |

4. **Retirement buffer.**
   - An 8-entry list of in-flight instructions. Instructions leave it in program
     order as they finish.
   - If it's full, dispatch stops.
   - CSR instructions (reading or writing control registers, including the cycle
     counter) wait until it's completely empty. This matters for how we measure
     cycles; see section 6.

5. **LSU.**
   - Has a 4-entry queue but processes **one memory instruction at a time**.
   - Every memory access moves at most one 16-byte "line".
   - Tightly-coupled memories (TCMs), the on-core instruction and data SRAMs,
     answer in about a cycle. External memory (over the AXI bus) is slower.

### 2.2 The vector unit (RVV)

A separate engine that implements the standard RISC-V Vector extension (RVV 1.0,
integer and FP32). Each vector register is 128 bits (VLEN=128): for example
sixteen int8 values or four FP32 values.

Some RVV terms used below:

- **LMUL** lets one instruction operate on a group of 1, 2, 4 or 8 registers at
  once.
- **SEW** is the element size: 8, 16 or 32 bits.
- **`vl`** is how many elements the instruction processes.

Here's how a vector instruction flows through the unit:

1. The scalar core's dispatch stage puts vector instructions into an 8-entry
   **command queue**. The scalar core then carries on; it doesn't wait for vector
   work to finish.
2. The vector decoder takes up to 2 instructions per cycle and splits each one
   into **micro-ops (uops)**, roughly one per 128-bit register it touches. An
   LMUL=4 instruction becomes about 4 uops. Uops wait in a 16-entry uop queue.
3. Up to **3 uops per cycle** go to the vector functional units:

   | Functional unit | Count |
   | --- | --- |
   | Integer ALU | 2 |
   | Multiply / multiply-accumulate | 2 |
   | Divider | 1 |
   | FP multiply-add | 2 |
   | FP divider | 1 |
   | Permute / reduction | 1 |

   Each unit has a small reservation station (a waiting area) in front of it.

4. An 8-entry **vector reorder buffer** retires uops in order.
5. Vector loads and stores are handed to the **scalar LSU**, which processes them
   one 16-byte register at a time.
6. Results that go back to a scalar register (e.g. `vmv.x.s`, reductions) cross
   back into the scalar core, and that crossing costs cycles.

We still need to measure a lot of the exact timing (how many cycles each step
takes). Collecting those numbers is most of the correlation work.

---

## 3. How the model works

### 3.1 Two halves: functional and timing

The model splits the problem in two:

- **Functional:** *what* each instruction does. It computes results, memory
  addresses and branch directions. We use **MPACT**, Google's instruction-set
  simulator for CoralNPU (repo `coralnpu-mpact`). MPACT runs the program
  correctly but has no idea about time.
- **Timing:** *when* each instruction happens. We write this part, using
  **Sparta**.

### 3.2 Execute-at-fetch

When the timing model fetches an instruction, it immediately asks MPACT to
execute that instruction. From then on the timing model knows everything about
it:

- the opcode and registers;
- the memory address it will touch;
- whether a branch is taken;
- the current vector length.

The timing model then spends as many simulated cycles as the hardware would,
moving the instruction through fetch, dispatch, execute and retire.

This is simple and exact for CoralNPU because the core is **in-order and never
executes wrong-path instructions**. So we never have to undo anything MPACT did.
When the fetch unit guesses a branch wrong, the model just adds the lost cycles;
it doesn't have to simulate the wrong instructions.

Two more benefits:

- **Data-dependent timing is easy.** For example, the divider's latency depends
  on the operand values, and we have those values from MPACT.
- **This is the approach tt-rpm uses.** tt-rpm is Tenstorrent's open-source
  RISC-V performance model; it pairs Sparta with the Whisper instruction-set
  simulator. We follow its structure, with MPACT in place of Whisper.

### 3.3 Sparta, briefly

Sparta is a C++ framework for performance models, from the `sparcians/map`
project. It gives us:

- **Units:** C++ classes, one per hardware block (Fetch, Dispatch, LSU, ...).
- **Ports:** typed connections between units. Unit A sends an instruction to unit
  B with a delay of N cycles.
- **Events:** "do this at cycle X". This is how multi-cycle operations are
  modelled.
- **Parameters:** values for each unit, loaded from YAML at startup.
- **Counters and statistics:** reported automatically at the end of a run.
- **Pipeline collection:** a per-instruction, per-cycle log that can be viewed as
  a pipeline diagram (e.g. in Konata or Sparta's Argos viewer).

We don't write a simulation engine ourselves; we write units.

---

## 4. Model structure

### 4.1 Units and how they connect

```
             +---------+    MPACT (executes each instruction at fetch)
             |  Fetch  |<------------------------------------------+
             +----+----+                                           |
                  | instructions (with results already known)      |
             +----v-----+                                          |
             | Dispatch |---- stall reasons -> statistics          |
             +-+--+--+--+                                          |
   scalar ops  |  |  | vector ops                                  |
 +-------------v+ |  +-------------+                               |
 | Scalar Exec  | |  |  Vector     |                               |
 | ALU/BRU/MLU/ | |  |  (cmd queue,|                               |
 | DVU/FPU      | |  |  uops, FUs, |<-------+                      |
 +------+-------+ |  |  vec ROB)   |        | vector mem ops       |
        |       +-v--+-+-----------+        |                      |
        |       |  LSU  |<------------------+                      |
        |       +---+---+                                          |
        |           |                                              |
        |       +---v----+                                         |
        |       | Memory | (ITCM/DTCM latency, external latency/bw)|
        |       +--------+                                         |
        v                                                          |
   +-----------------+                                             |
   | Retire (ROB, 8) |-- redirect on mispredict --> Fetch ---------+
   +-----------------+
```

### 4.2 What each unit models, and its parameters

Every value below is a YAML parameter. The "M3 seed" column is our starting
estimate from reading the RTL. The calibration tickets replace those estimates
with measured numbers.

**Fetch**

| Parameter | Meaning | M3 seed |
| --- | --- | --- |
| `fetch_bytes` | Bytes per instruction-memory read | 16 |
| `fetch_issue_interval` | Minimum cycles between reads | 2 |
| `max_outstanding_fetches` | Reads in flight at once | 1 |
| `ibuf_entries` | Instruction buffer size | 8 |
| `predictor` | `static_btfn` (backward taken, forward not taken, `jal` taken) or `perfect` | `static_btfn` |
| `taken_bubble` | Cycles lost on a correctly predicted taken branch | TBD |
| `mispredict_penalty` | Cycles lost on a wrong guess or a `jalr` | TBD |

**Dispatch**

| Parameter | Meaning | M3 seed |
| --- | --- | --- |
| `width` | Instructions sent per cycle | 4 |
| `branch_ends_group` | A branch must be the last instruction sent that cycle | true |
| `alone_classes` | Instruction classes that must be sent alone from slot 0 | CSR, FP, vector-with-FP-scalar, fence |
| `addr_needs_registered_value` | Load/store addresses can't use a result produced the previous cycle | true |
| `csr_waits_for_empty_rob` | CSR instructions wait for the retirement buffer to empty | true |
| per-unit limits | e.g. at most 1 multiply per cycle | from the unit counts |

Each cycle, Dispatch records the reason it stopped. The possible reasons are:
`ibuf_empty`, `raw_hazard`, `unit_busy`, `lsu_queue_full`, `vec_queue_full`,
`rob_full`, `branch_rule`, `alone_rule`, `csr_drain`.

**Scalar Exec**

Each unit class has `count`, `latency` (cycles until the result is usable) and
`issue_interval` (cycles before the unit can take another operation). The divider
also has an optional latency formula that uses the operand values from MPACT.

**Retire**

| Parameter | Meaning | M3 seed |
| --- | --- | --- |
| `rob_entries` | Retirement buffer size | 8 |
| `retire_width` | Instructions retired per cycle | 4 (to confirm) |

**LSU**

| Parameter | Meaning | M3 seed |
| --- | --- | --- |
| `queue_entries` | Queue in front of the LSU | 4 |
| `slots` | Memory instructions processed at once | 1 |
| `line_bytes` | Bytes per memory access | 16 |
| `load_to_use` | Cycles from a load starting to its value being usable | TBD |
| `vector_per_register_overhead` | Fixed cycles per vector register handled | TBD |

Cost of one memory instruction = the number of distinct 16-byte lines it touches
× cycles per line + fixed overhead. For vector memory instructions this is
computed per register.

**Memory**

| Parameter | Meaning | M3 seed |
| --- | --- | --- |
| `itcm_latency`, `dtcm_latency` | TCM access latency | 1 |
| `ext_latency`, `ext_bytes_per_cycle` | External memory over AXI | TBD |
| `itcm_kb`, `dtcm_kb` | TCM sizes | 1024/1024 for highmem builds |

**Vector**

| Parameter | Meaning | M3 seed |
| --- | --- | --- |
| `cmd_queue_entries` | Command queue | 8 |
| `decode_width` | Instructions decoded per cycle | 2 |
| `uop_queue_entries` | Uop queue | 16 |
| `dispatch_width` | Uops sent per cycle | 3 |
| `vrob_entries` | Vector reorder buffer | 8 |
| per functional-unit pool | `count`, `rs_entries`, `latency`, `issue_interval` | from the RTL defines |
| `uop_rule` per instruction class | How many uops an instruction becomes, as a function of LMUL, SEW and `vl` | TBD |
| `scalar_to_vector_latency`, `vector_to_scalar_latency` | Cost of crossing between the scalar core and the vector unit | TBD |

### 4.3 The instruction table

One YAML file (`isa/instructions.yaml`) maps every instruction mnemonic to how the
model treats it:

```yaml
add:        { class: alu }
mul:        { class: mul }
div:        { class: div, latency_model: dvu_clz }
lw:         { class: load,  bytes: 4 }
fadd.s:     { class: fp_addmul, alone: true }
vadd.vv:    { class: v_alu, uops: per_register }
vmacc.vv:   { class: v_mul, uops: per_register }
vle8.v:     { class: v_load, mode: unit_stride }
vfmacc.vf:  { class: v_fma, uops: per_register, alone: true }
vredsum.vs: { class: v_reduce, uops: tree }
```

When MPACT executes an instruction that isn't in the table, the model stops with
an error. That way no instruction is silently timed as "1 cycle".

Adding a new instruction for ISA exploration takes two steps: implement its
behaviour in MPACT, then add one line here.

### 4.4 Configuration presets

`configs/coremini_m3.yaml` (scalar only) and `configs/rvvcoremini_m3.yaml`
(scalar + vector) hold the full parameter set for each M3 build.

For design exploration you copy a preset and change numbers, e.g.
`fetch_issue_interval: 1` or `vector.mul.count: 4`. Sparta also accepts
parameters on the command line, so a sweep script can vary them without writing
files.

### 4.5 Repository layout (`reef/perf-model/`)

```
perf-model/
  docs/                 plan, tickets, deviation register
  ext/                  sparta (sparcians/map), coralnpu-mpact @ a1d219e (git submodules)
  src/
    driver/             MPACT wrapper: steps one instruction, fills an InstRecord
    core/               Sparta units: Fetch, Dispatch, ScalarExec, Retire, Lsu, Memory
    vector/             Vector unit
    common/             InstRecord, instruction table loader, stall-reason enum
  isa/instructions.yaml
  configs/              coremini_m3.yaml, rvvcoremini_m3.yaml
  workloads/            microbenchmarks and kernels (source + build rules)
  correlation/          scripts to run RTL and model, compare, report
  tests/                unit tests for the model itself
```

The **InstRecord** is the object that flows through the model. It holds:

- PC and opcode;
- source and destination registers;
- memory address(es) and size;
- whether a branch was taken, and its target;
- `vl`, SEW and LMUL for vector instructions;
- the class from the instruction table;
- per-stage timestamps, used for the pipeline view.

---

## 5. Build order (milestones)

Each milestone ends with something that runs.

| Milestone | Result | Tickets |
| --- | --- | --- |
| **M-1: Foundations** | Everyone builds the same pinned sources. We can run the M3 RTL simulation and get a cycle count. The Sparta skeleton compiles. | T01–T03, T07 |
| **M-2: End-to-end at fixed IPC** | The model runs a real ELF to completion through MPACT, at a fake constant 1 IPC. Its instruction count matches MPACT's. The RTL cycle-measurement harness works. | T04, T08, T09, T15, T16 |
| **M-3: First coarse model** | All units exist, using the seed values. We get a first correlation report. Expect errors of 20–50% here; that's normal. | T10–T14, T19 |
| **M-4: RTL visibility** | Per-instruction cycle benchmarks and a per-cycle event trace from the RTL, so every model error can be explained. | T05, T17, T18 |
| **M-5: Calibration** | Each unit's parameters are replaced with measured values, and the corpus is widened to vector-FP kernels. | T06, T20–T24 |
| **M-6: Sign-off** | Kernels within ±10%. Remaining error sources are documented. | T25 |

Work can run in parallel across three tracks:

- **Model code:** T07–T15.
- **RTL measurement:** T02, T16–T18.
- **MPACT:** T04–T06.

The tracks meet at T19, the correlation report.

---

## 6. How we measure the RTL (and the traps)

- **Cycle counts** come from the RISC-V `mcycle` counter, read before and after
  the region of interest. Because reading `mcycle` is a CSR instruction, and CSR
  instructions wait for the pipeline to drain, every measurement includes a
  drain. Two consequences:
  - Measure loops long enough (thousands of iterations) that the drain doesn't
    matter.
  - Always subtract an empty-loop baseline.
- **Per-instruction detail** comes from the "verification" build of the RTL
  simulator (`--instr_trace`), which prints each retired instruction. At M3 this
  build should have the same timing as the normal build (see deviations.md, M1).
  Ticket T18 confirms it.
- **External memory** timing in the RTL simulation comes from the testbench's bus
  model, not a real DRAM. We correlate against that model, and we label
  external-memory results as such.
- **Programs** must run on both the RTL and MPACT. At the M3 pin, MPACT can't
  execute vector floating-point instructions. Until T06 lands, the corpus is
  scalar code plus integer vector kernels.

---

## 7. Out of scope for the first model

- Interrupts, traps, debug mode, multi-core. The workloads are bare-metal kernels
  that run to completion.
- The matrix engine (VME/Zvt) and BF16. They don't exist at M3, and will come
  with a future pin bump.
- Power and area.
- Detailed modelling of the vector register-file read ports, and of bus
  arbitration. These may be added if calibration shows they matter.

---

## 8. Glossary

| Term | Meaning |
| --- | --- |
| **CSR** | Control and status register, e.g. `mcycle` (the cycle counter) |
| **Dispatch** | The pipeline step that sends decoded instructions to execution units |
| **ELF** | The compiled program file |
| **In-order** | Instructions start in program order; a stalled instruction blocks the ones behind it |
| **IPC** | Instructions per cycle |
| **ITCM / DTCM** | On-core instruction and data memories (fast, fixed latency) |
| **LMUL / SEW / `vl`** | RVV register grouping / element width / number of active elements |
| **LSU** | Load/store unit |
| **MPACT** | Google's instruction-set simulator used as our functional model |
| **RAW hazard** | An instruction needs a register value that an earlier instruction hasn't produced yet |
| **ROB / retirement buffer** | Tracks in-flight instructions so they complete in order |
| **RTL** | The hardware design source (Chisel/SystemVerilog) |
| **Scoreboard** | A bitmask of registers still waiting for a result |
| **Slot 0** | The first of the 4 dispatch positions each cycle |
| **Sparta** | The C++ performance-modelling framework we build on |
| **Uop** | Micro-op: one internal piece of a vector instruction, usually one register's worth |
| **Verilator** | A tool that turns the Verilog RTL into a fast, cycle-accurate C++ simulator |

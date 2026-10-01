# coralnpu_perf: coarse performance model of CoralNPU

`coralnpu_perf` predicts how many clock cycles a RISC-V program takes on the
CoralNPU core, and explains where the time went.

**Status: super-coarse.** The model runs real programs end to end, but its
timing is a generic in-order core with guessed numbers. It does **not** yet
correlate with the CoralNPU M3 RTL. Closing that gap is the job of the
beginner tickets in [docs/tickets-beginner.md](docs/tickets-beginner.md).

Background reading:

- [docs/implementation-plan.md](docs/implementation-plan.md): what the model is
  and how CoralNPU M3 works.
- [docs/deviations.md](docs/deviations.md): where the RTL differs from its docs.

## How it works in one paragraph

The model has two halves.

- **MPACT**, Google's instruction-set simulator for CoralNPU, *runs* the
  program. It computes results, memory addresses and branch outcomes.
- **A Sparta model** decides *when* each instruction happens. Sparta is a C++
  framework for cycle-level models.

The key trick is **execute-at-fetch**. When the model's Fetch unit fetches an
instruction, it asks MPACT to execute it right away, so the timing model always
knows what the instruction does. CoralNPU is in-order and never executes
wrong-path instructions, so nothing ever has to be undone.

```
 ELF --> MPACT (driver/)                      execute-at-fetch
           |  one instruction record per step
           v
 +-------+     +----------+     +---------+     +-----+
 | Fetch | --> | Dispatch | --> | Execute | --> | Rob | --> retired
 +-------+     +----------+     +---------+     +-----+
     ^  credits    |  ^  credits (LSU, vector queues)  |
     +-------------+  +---------- credits -------------+
```

| Unit | File | What it does now |
| --- | --- | --- |
| Fetch | `src/Fetch.cpp` | Up to 4 instructions per cycle; a fixed penalty on every taken branch or jump |
| Dispatch | `src/Dispatch.cpp` | In order, up to 4 per cycle; scoreboard for register hazards; records why it stalled |
| Execute | `src/Execute.cpp` | Resource pools: ALU×4, MUL, DIV, FPU, FDIV, LSU, vector×2. Latency and occupancy per class |
| Rob | `src/Rob.cpp` | 8-entry retirement buffer, 4 retires per cycle |
| Decoder | `src/InstDecode.cpp` | Instruction class plus the registers it reads and writes, from the RISC-V encoding |
| MPACT driver | `driver/` | C library around MPACT; records each instruction's memory addresses |

## Quick start (Ubuntu 22.04 or WSL2)

```bash
git clone --recursive https://github.com/LonghornSilicon/reef.git
cd reef/perf-model
scripts/install_deps.sh   # once, needs sudo: compilers, Boost, clang, Java, bazelisk
scripts/build.sh          # builds everything; first run ~1 hour, then seconds
scripts/run_all.sh        # run every workload, one line each
```

Notes:

- **Already cloned without `--recursive`?** `scripts/build.sh` fetches the
  submodules for you (`git submodule update --init --recursive`).
- **Windows.** Run the scripts inside WSL (`wsl -d Ubuntu`). Cloning inside the
  Linux filesystem (`~/...`) is much faster than building from `/mnt/c/...`.
  When the checkout is on `/mnt/...`, build outputs go to
  `~/.cache/coralnpu_perf`; otherwise they go to `perf-model/build/`.
- **Memory.** Keep `JOBS=4` (the default) on an 8 GB machine; more parallel
  jobs can run WSL out of memory.

### What comes from where

Everything the model links against is pinned as a git submodule in `ext/`.
Nothing is downloaded by hand.

| Submodule | Pinned at | Why |
| --- | --- | --- |
| `ext/map` (Sparta) | `map_v3.0.2` | Simulation framework: units, ports, events, parameters, statistics |
| `ext/coralnpu-mpact` | `a1d219e` | MPACT functional simulator. This is the commit CoralNPU M3 itself pins (`coralnpu/rules/repos.bzl`) |
| `ext/yaml-cpp` | `0.8.0` | Sparta 3.x needs yaml-cpp ≥ 0.8; Ubuntu 22.04 ships 0.7 |

MPACT's own third-party dependencies (abseil, protobuf, ...) are fetched by
Bazel at the versions MPACT pins. System libraries (Boost, HDF5, SQLite,
RapidJSON) come from `apt` via `install_deps.sh`.

Run one workload with the full summary (the path printed by `build.sh`):

```bash
source scripts/env.sh
$MODEL --elf workloads/build/gemv_int8.elf -c configs/m3.yaml
```

Change a parameter without editing files:

```bash
$MODEL --elf x.elf -p top.core.fetch.params.fetch_interval 2
```

See every parameter and its current value:

```bash
$MODEL --elf x.elf --show-parameters --no-run
```

See the instruction stream MPACT produces (useful for checking a workload):

```bash
$DRIVER_DIR/cn_trace workloads/build/gemv_int8.elf 40
```

## Reading the summary

```
  instructions retired: 22009
  cycles              : 7011
  IPC                 : 3.139

  Dispatch: cycles with no dispatch, by cause
    (dispatched)               6007   85.7%
    ibuf_empty                 1003   14.3%
    raw_hazard                    0    0.0%
    ...
  Execute: utilisation (busy cycles / (units * cycles))
    alu      x4  ops      22009  util  78.5%
```

- **Dispatch table.** Every cycle is either a cycle where something dispatched,
  or a cycle blamed on exactly one reason. The rows add up to the total cycles.
  When the model disagrees with the RTL, this table tells you which part of the
  model to look at.
- **Execute table.** How busy each resource was.

`--json out.json` writes the same numbers for scripts.

## Workloads

`workloads/*.S` are small assembly programs, each stressing one feature (plus
`gemv_int8`, an LLM-decode-shaped kernel).

- They link to the **M3 default memory map** (code at `0x0`, data at `0x10000`)
  and end with `mpause`. The same ELF therefore runs on the model and on the M3
  RTL simulator.
- Each loop is bracketed by `csrr mcycle` into `s10`/`s11`, so the RTL cycle
  count of the loop is `s11 - s10`.

| Workload | Stresses |
| --- | --- |
| `alu_indep`, `alu_chain` | Fetch/dispatch width; ALU latency |
| `branch_tight`, `branch_group` | Taken-branch cost; branch ends the dispatch group |
| `mul_chain`, `div_chain` | Multiplier latency; data-dependent divider |
| `fp_chain`, `fp_mix` | FPU latency; FP dispatches alone |
| `load_stream`, `load_chain`, `addr_forward` | LSU throughput; load-to-use latency; address forwarding |
| `vec_add_m1`, `vec_add_m4`, `vec_memcpy` | Vector ALU and vector memory at different LMUL |
| `gemv_int8` | int8 matrix-vector product (widening multiply + reduction) |

## Adding things

- **A parameter:** add a `PARAMETER(...)` line to the unit's parameter set
  (e.g. `Fetch.hpp`), read it in the constructor, and add it to
  `configs/m3.yaml`.
- **A counter:** add a `sparta::Counter` member. It shows up with
  `--auto-summary on`.
- **A workload:** add `workloads/<name>.S` using `common.inc`; `build.sh` picks
  it up.
- **An ISA extension:** add the instruction to MPACT, then teach
  `src/InstDecode.cpp` its class and registers.

# reef_perf: coarse performance model of Reef

`reef_perf` predicts how many clock cycles a RISC-V program takes on the Reef
NPU core, and explains where the time went. Reef starts from the CoralNPU M3
release, so the first goal is to match the M3 RTL.

**Status: super-coarse.** The model runs real programs end to end, but its
timing is a generic in-order core with guessed numbers. It does **not** yet
correlate with the M3 RTL. Closing that gap is the job of the tickets in
[docs/roadmap.md](docs/roadmap.md).

Background reading:

- [docs/overview.md](docs/overview.md): what the model is and how the M3
  baseline works.
- [docs/interfaces.md](docs/interfaces.md): the five modules and how they talk.
- [docs/deviations.md](docs/deviations.md): where the RTL differs from its docs.

## How it works

The model has two halves:

- **Spike**, the RISC-V reference instruction-set simulator, *runs* the program:
  it computes results, memory addresses and branch outcomes.
- **A Sparta model** decides *when* each instruction happens. Sparta is a C++
  framework for cycle-level models.

The key idea is **execute-at-fetch**. When Fetch fetches an instruction, it
asks Spike to execute it right away, so the timing model always knows what the
instruction does. Reef is in-order and never executes wrong-path
instructions, so nothing ever has to be undone.

The model is split into **five modules**, each with its own directory, Sparta
subtree (`top.<module>`), config section and owner. Modules talk only through
the interfaces in [docs/interfaces.md](docs/interfaces.md).

```
 ELF --> Spike                                   execute-at-fetch
           |
 +---------v--------+  InstBuffer-   +----------------------------------+
 | frontend         |   Interface    | core                             |
 |  fetch, ibuf,    | <------------- |  dispatch --> scalar_exec        |
 |  Spike, decode   |   peek / pop   |     |    \--> lsu --------------+--> MemoryInterface
 +------------------+                |     |    \--> rob --> retired  |        |
                                     +-----|--------|------------------+        v
                          InstRef, credits |        | InstRef, credits  +---------------+
                                    +------v--+  +--v------+            | memory        |
                                    | vector  |  | matrix  |            |  ITCM, DTCM   |
                                    |  lanes  |  |  stub   |            |  AXI port     |
                                    +---------+  +---------+            +---------------+
```

| Module | Parameters | What it does now |
| --- | --- | --- |
| `frontend` | `top.frontend.params` | Runs each instruction in Spike as it is fetched (execute-at-fetch) and decodes it; up to 4 per cycle into an 8-entry instruction buffer; a fixed penalty on every taken branch or jump |
| `core` | `top.core.{dispatch,scalar_exec,lsu,rob}.params` | In-order dispatch with a register scoreboard and stall accounting; integer/FP pools; the LSU for scalar *and* vector memory ops; 8-entry retirement buffer |
| `vector` | `top.vector.params` | RVV command queue and 2 lanes; time per instruction from vl × SEW / VLEN |
| `matrix` | `top.matrix.params` | Stub with the vector module's interface; no matrix ISA yet, so it sees no traffic |
| `memory` | `top.memory.params` | ITCM and DTCM ports; an AXI port to off-core memory (black box) |

## Layout

```
reef-perf/
  model/<module>/        C++ model, one directory per module (headers beside sources)
  model/common/          shared types: Inst, ResourcePool, Module, interface types
  model/ReefSim.*        the top level, which wires the modules together; main.cpp
  tools/reef_trace/      prints Spike's instruction stream
  python/reef_perf/      runs the simulator, assembles microbenchmarks
    experiments/         sweep, correlation (the only standalone Python)
  microbenchmarks/       assembly programs, plus their M3 RTL results (rtl_m3.csv)
  configs/machines/      simulation configs: the machine being simulated (m3.yaml)
  tests/cpp/<module>/    Google Test
  tests/python/          pytest: unit/, integration/, experiments/
  tests/golden/          expected cycle counts (m3_results.json)
  scripts/               lint, C++ checks, dependency build
  docs/                  overview, interfaces, roadmap, deviations
  ext/                   pinned submodules: Spike, Sparta, yaml-cpp
```

The Docker image ([../Dockerfile](../Dockerfile)) and its launcher
([../tools/docker.sh](../tools/docker.sh)) are at the repository root, shared
with the rest of Reef.

## Requirements

**Docker. Nothing else.** Every build, test and experiment runs inside the
image defined by [../Dockerfile](../Dockerfile). You don't need sudo or any host
packages. Windows hosts are not supported directly; use Docker on Linux or
macOS, or from inside WSL.

Third-party code is pinned as git submodules in `ext/` and built into the
image:

| Submodule | Pinned at | Why |
| --- | --- | --- |
| `ext/spike` | `fd72ee2d` | Functional simulator. This is the Spike commit CoralNPU M3 itself pins (`coralnpu/rules/deps.bzl`) |
| `ext/map` (Sparta) | `map_v3.0.2` | Simulation framework: units, ports, events, parameters, statistics |
| `ext/yaml-cpp` | `0.8.0` | Sparta 3.x needs yaml-cpp ≥ 0.8; Ubuntu 22.04 ships 0.7 |

## Quick start

```sh
git clone --recursive https://github.com/LonghornSilicon/reef.git
cd reef
tools/docker.sh build     # once: builds Sparta and Spike into the image (~1 hour)
tools/docker.sh shell     # shell inside the image, with this checkout mounted
```

Inside the shell:

```sh
cd reef-perf
cmake -S . -B build -G Ninja && cmake --build build      # build reef_perf
uv run python -m reef_perf.experiments.sweep.sweep \
    --config python/reef_perf/experiments/sweep/sweeps/all.yaml  # every microbenchmark
```

Notes:

- **Already cloned without `--recursive`?** `tools/docker.sh build` fetches
  the submodules for you.
- **Memory.** `JOBS=4` (the default) suits an 8 GB Docker VM; Sparta needs
  about 1 GB per compile job. Use `JOBS=8 tools/docker.sh build` with more RAM.
- **TLS errors during the build** (`UnknownIssuer`, `certificate verify
  failed`). Your network or antivirus is intercepting HTTPS. Export its root
  certificate as a PEM file, then run
  `REEF_PERF_EXTRA_CA=/path/to/root.pem tools/docker.sh build`. The image then
  trusts it. Nothing is committed.

## Running the simulator

```sh
build/bin/reef_perf --elf build/microbenchmarks/gemv_int8.elf -c configs/machines/m3.yaml
build/bin/reef_perf --elf x.elf -p top.frontend.params.fetch_interval 2
build/bin/reef_perf --elf x.elf --show-parameters --no-run   # every parameter
build/bin/reef_trace build/microbenchmarks/gemv_int8.elf 40  # Spike's instruction stream
```

`configs/machines/m3.yaml` is the **simulation** config: the machine being
simulated, one section per module.
Every parameter is listed there, with the ticket that will correct it.

### Reading the summary

```
  instructions retired: 22009
  cycles              : 7011
  IPC                 : 3.139
  ROI cycles (mcycle) : 6004

  Dispatch: cycles with no dispatch, by cause
    (dispatched)               6007   85.7%
    ibuf_empty                 1003   14.3%
    raw_hazard                    0    0.0%
    ...
  Pools: utilization (busy cycles / (units * cycles))
    core.alu         x4  ops      22009  util  78.5%
```

- **Dispatch table.** Every cycle is either a cycle where something dispatched,
  or a cycle blamed on exactly one reason. When the model disagrees with the
  RTL, this table tells you which part of the model to look at.
- **ROI cycles.** Cycles between the program's first and last `csrr mcycle`:
  the microbenchmark's loop, directly comparable with the RTL.
- **Pools table.** How busy each resource was, by module.

`--json out.json` writes the same numbers for scripts.

## Experiments

Experiments are programs that run the simulator and produce artifacts. They
live in [`python/reef_perf/experiments/`](python/reef_perf/experiments/README.md):

- [`sweep`](python/reef_perf/experiments/sweep/README.md) runs the
  microbenchmarks under parameter variants and records cycles, IPC and stall
  breakdowns.
- [`correlation`](python/reef_perf/experiments/correlation/README.md) compares
  the model's ROI cycles with the M3 RTL. A GitHub workflow runs it on every
  push to a PR that changes `reef-perf/` and posts the table as a PR comment.

Both run the [microbenchmarks](microbenchmarks/README.md): small assembly
programs plus an LLM-decode-shaped int8 GEMV.

## Testing

Run everything from `reef-perf/` inside the image (`../tools/docker.sh shell`).

**Everything:**

```sh
uv run pytest
```

This builds the C++ targets (into `build/`), runs the Google Test suite
through CTest, runs the simulator end to end, checks that runs are
deterministic and can run in parallel, compares cycle counts with the golden
file, runs both experiments, and checks lint (Ruff, `scripts/check_cpp.sh`)
and Doxygen.

**Subsets**, by category mark:

```sh
uv run pytest -m unit               # fast, no C++ build
uv run pytest -m "not slow"         # skip the build-heavy tests
uv run pytest -m experiment         # experiment sweeps only
uv run pytest tests/python/integration/test_sim.py
```

**C++ tests alone:**

```sh
cmake -S . -B build -G Ninja && cmake --build build
ctest --test-dir build --output-on-failure
```

| Location | What it covers |
| --- | --- |
| `tests/cpp/<module>/` | Google Test: instruction decoding, routing, resource pools, memory and vector timing |
| `tests/python/unit/` | Packaging, microbenchmark assembly |
| `tests/python/integration/` | Simulator end to end (incl. parallel runs), golden cycle counts, CTest, lint, Doxygen |
| `tests/python/experiments/` | The `sweep` and `correlation` experiments |

Every pytest test must be marked `unit`, `integration` or `experiment`, and
long-running ones also `slow`. `tests/conftest.py` rejects unmarked tests.

**Golden timing.** `tests/python/integration/test_golden.py` checks every
microbenchmark's instructions, cycles, redirects and stall counters against
`tests/golden/m3_results.json`, exactly. A change that only moves code must
pass it unchanged. A change meant to alter timing regenerates the file in the
same commit and says why in the commit message:

```sh
REEF_PERF_UPDATE_GOLDEN=1 uv run pytest tests/python/integration/test_golden.py
```

## Lint and documentation

```sh
bash scripts/lint.sh        # apply Ruff + clang-format fixes, then run every check
bash scripts/check_cpp.sh   # C++ checks only, no edits
mkdir -p build && doxygen # C++ API docs into build/doxygen/html
```

C++ follows the [LLVM coding standards](https://llvm.org/docs/CodingStandards.html),
and the checks enforce it:

- **clang-format** ([.clang-format](.clang-format)): `BasedOnStyle: LLVM`,
  plus `RemoveBracesLLVM` (no braces on single-statement bodies).
- **clang-tidy** ([.clang-tidy](.clang-tidy)): llvm-project's own naming rules
  (types, variables, parameters and members `UpperCamelCase`, functions
  `lowerCamelCase`), the `llvm-*` checks, and two bug checks. The set is
  kept small so it stays fast; `scripts/check_cpp.sh` runs one file per core
  and only reports on our own headers, not Sparta's or Spike's.
- **Compiler warnings** (`-Wall -Wextra -Wpedantic`) are errors in every build
  of our targets (`CMakeLists.txt`).
- **[scripts/check_style.sh](scripts/check_style.sh)**: what the tools above can't
  check here: file names (`.h`/`.cpp`, `UpperCamelCase`, tests `*Test.cpp`),
  `REEF_PERF_<DIR>_<FILE>_H` header guards, the 80-column
  `//===- File - Description ---===//` banner, and `///` doc comments.
- **Doxygen** ([Doxyfile](Doxyfile)) fails on any undocumented public
  declaration. Private members need no Doxygen comment.

Deliberate deviations from LLVM: exceptions and RTTI are used (Sparta and
Spike need them); Sparta-defined names (`name`, `buildTree_`, `unit_port_set_`,
`PARAMETER` names, which are also config keys) keep Sparta's spelling, marked
with `NOLINT`; and the standard library is used instead of LLVM's ADT.

## Contributing

- Put C++ headers and sources together in `model/<module>/`, one class per
  `Name.h`/`Name.cpp` pair, and add the `.cpp` to `CMakeLists.txt`.
- Cover new C++ behavior with Google Test cases in `tests/cpp/<module>/`
  (`NameTest.cpp`), and new Python behavior with pytest cases in
  `tests/python/{unit,integration}/`.
- Don't declare copy or move constructors and assignments unless the
  implicit ones would be wrong, for example in a class that hands out
  pointers to its own members (`FuncSim`). Then delete just the copy pair,
  with a comment saying why; that also rules out moves.
- Document public declarations briefly with `///` (`\p Name` for parameters,
  `\returns` for results). Use plain `//` comments elsewhere, and only where
  the code is not obvious.
- **A new parameter:** add a `PARAMETER(...)` line (its description string is
  its documentation) to the unit's parameter set, read it in the constructor,
  and list it in `configs/machines/m3.yaml`.
- **A new microbenchmark:** see [microbenchmarks/README.md](microbenchmarks/README.md).
- **A new experiment:** add a directory under `python/reef_perf/experiments/`
  with its README and `main.py`, plus a test in `tests/python/experiments/`.
- **An ISA extension:** add the instruction to Spike (`ext/spike`, e.g. as an
  extension), then teach `model/frontend/InstDecode.cpp` its class and
  registers.

Before a PR is complete: `bash scripts/lint.sh` passes and `uv run pytest`
passes.

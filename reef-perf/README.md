# reef_perf: coarse performance model of Reef

`reef_perf` predicts how many clock cycles a RISC-V program takes on the Reef
NPU core, and explains where the time went. Reef starts from the CoralNPU M3
release, so the first goal is to match the M3 RTL.

**Status: super-coarse.** The model runs real programs end to end, but its
timing is a generic in-order core with guessed numbers. It does **not** yet
correlate with the M3 RTL. Closing that gap is the job of the beginner tickets
in [docs/tickets-beginner.md](docs/tickets-beginner.md).

Background reading:

- [docs/implementation-plan.md](docs/implementation-plan.md): what the model is
  and how the M3 baseline works.
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
 +---------v--------+  FetchPacket   +----------------------------------+
 | frontend         | -------------> | backend                          |
 |  fetch, Spike,   | <------------- |  dispatch --> scalar_exec        |
 |  decode          |    credits     |     |    \--> lsu --------------+--> MemoryInterface
 +------------------+                |     |    \--> rob --> retired  |        |
                                     +-----|--------|------------------+        v
                          InstPtr, credits |        | InstPtr, credits  +---------------+
                                    +------v--+  +--v------+            | mem           |
                                    | vector  |  | matrix  |            |  tcm (I/DTCM) |
                                    |  vxu    |  |  mxu    |            |  axi          |
                                    +---------+  +---------+            +---------------+
```

| Module | Units (`top.<module>.<unit>`) | What it does now |
| --- | --- | --- |
| `frontend` | `fetch` | Runs each instruction in Spike as it is fetched (execute-at-fetch) and decodes it; up to 4 per cycle; a fixed penalty on every taken branch or jump |
| `backend` | `dispatch`, `scalar_exec`, `lsu`, `rob` | In-order dispatch with a register scoreboard and stall accounting; integer/FP pools; the LSU for scalar *and* vector memory ops; 8-entry retirement buffer |
| `vector` | `vxu` | RVV command queue and 2 lanes; time per instruction from vl × SEW / VLEN |
| `matrix` | `mxu` | Stub with the vector module's interface; no matrix ISA yet, so it sees no traffic |
| `mem` | `tcm`, `axi` | ITCM and DTCM ports; an AXI port to off-core memory (black box) |

Code is laid out the same way: `src/csrc/include/reef_perf/<module>/`,
`src/csrc/src/<module>/` and `tests/cpp/<module>/`, plus `common/` for the
shared types (`Inst`, `ResourcePool`, the `Module` base class and the
interface types).

## Requirements

**Docker. Nothing else.** Every build, test and experiment runs inside the
image defined by [Dockerfile](Dockerfile). You don't need sudo or any host
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
cd reef/reef-perf
tools/docker.sh build     # once: builds Sparta and Spike into the image (~1 hour)
tools/docker.sh shell     # shell inside the image, with this checkout mounted
```

Inside the shell (`/app/reef/reef-perf`):

```sh
cmake -S . -B build -G Ninja && cmake --build build      # build reef_perf
uv run python -m reef_perf.experiments.run.run \
    --config src/reef_perf/experiments/run/configs/all.yaml  # run every workload
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
build/bin/reef_perf --elf build/workloads/gemv_int8.elf -c configs/m3.yaml
build/bin/reef_perf --elf x.elf -p top.frontend.fetch.params.fetch_interval 2
build/bin/reef_perf --elf x.elf --show-parameters --no-run   # every parameter
build/bin/reef_trace build/workloads/gemv_int8.elf 40        # Spike's instruction stream
```

`configs/m3.yaml` is the **simulation** config: the machine being simulated.
Every parameter is listed there, with the ticket that will correct it.

### Reading the summary

```
  instructions retired: 22009
  cycles              : 7011
  IPC                 : 3.139

  Dispatch: cycles with no dispatch, by cause
    (dispatched)               6007   85.7%
    ibuf_empty                 1003   14.3%
    raw_hazard                    0    0.0%
    ...
  Pools: utilisation (busy cycles / (units * cycles))
    backend.alu      x4  ops      22009  util  78.5%
```

- **Dispatch table.** Every cycle is either a cycle where something dispatched,
  or a cycle blamed on exactly one reason. When the model disagrees with the
  RTL, this table tells you which part of the model to look at.
- **Pools table.** How busy each resource was, by module.

`--json out.json` writes the same numbers for scripts.

## Experiments

Experiments are programs that run the simulator and produce artifacts. They
live in [`src/reef_perf/experiments/`](src/reef_perf/experiments/README.md).
The first one, [`run`](src/reef_perf/experiments/run/README.md), runs the
workloads and records cycles, IPC and stall breakdowns. Its workloads (small
assembly programs plus an LLM-decode-shaped int8 GEMV) are in
`src/reef_perf/experiments/run/workloads/`.

## Testing

Run everything from `reef-perf/` inside the image (`tools/docker.sh shell`),
or from the host with `tools/docker.sh test [pytest args]`.

**Everything:**

```sh
uv run pytest
```

This builds the C++ targets (into `build/pytest`), runs the Google Test
suite through CTest, runs the simulator end to end, checks that runs are
deterministic and can run in parallel, runs the full `run` experiment
sweep, and checks lint (Ruff, clang-format, clang-tidy) and Doxygen.

**Subsets**, by category mark:

```sh
uv run pytest -m unit               # fast, no C++ build
uv run pytest -m "not slow"         # skip the build-heavy tests
uv run pytest -m experiment         # experiment sweeps only
uv run pytest tests/python/test_sim.py
```

**C++ tests alone:**

```sh
cmake -S . -B build -G Ninja && cmake --build build
ctest --test-dir build --output-on-failure
```

| Location | What it covers |
| --- | --- |
| `tests/cpp/<module>/` | Google Test: instruction decoding, routing, resource pools, memory and vector timing |
| `tests/python/` | Packaging, workloads, simulator end to end (incl. parallel runs), golden cycle counts, CTest, lint, Doxygen |
| `tests/python/experiments/` | The `run` experiment's full sweep |

Every pytest test must be marked `unit`, `integration` or `experiment`, and
long-running ones also `slow`. `tests/conftest.py` rejects unmarked tests.

**Golden timing.** `tests/python/test_golden.py` checks every workload's
instructions, cycles, redirects and stall counters against
`tests/python/golden/m3.json`, exactly. A change that only moves code must
pass it unchanged. A change meant to alter timing regenerates the file in the
same commit and says why in the commit message:

```sh
REEF_PERF_UPDATE_GOLDEN=1 uv run pytest tests/python/test_golden.py
```

## Lint and documentation

```sh
bash tools/lint.sh        # apply Ruff + clang-format fixes, then run every check
bash tools/check_cpp.sh   # C++ checks only, no edits
mkdir -p build && doxygen # C++ API docs into build/doxygen/html
```

- **clang-tidy** rules are in [.clang-tidy](.clang-tidy), and warnings are
  errors.
- **Doxygen** ([Doxyfile](Doxyfile)) fails on any undocumented member,
  *including private ones*, and on any missing `@param` or `@return`.

## Contributing

- Put public C++ declarations in `src/csrc/include/reef_perf/` and their
  implementations in `src/csrc/src/`, and add the `.cpp` to `CMakeLists.txt`.
- Cover new C++ behaviour with Google Test cases in `tests/cpp/`, and new
  Python behaviour with pytest cases in `tests/python/`.
- Give every function, member and parameter a Doxygen comment
  (`/** ... */` or `///`, with `@param`/`@return`), private ones included.
- **A new parameter:** add a `PARAMETER(...)` line, with a `///` comment, to
  the unit's parameter set, read it in the constructor, and list it in
  `configs/m3.yaml`.
- **A new workload:** add `src/reef_perf/experiments/run/workloads/<name>.S`
  using `common.inc`.
- **A new experiment:** add a directory under `src/reef_perf/experiments/`
  with its README, `main.py`, `run.py`, `plot.py` and `configs/`, plus a test
  in `tests/python/experiments/`. Only experiments may run standalone from
  `src/`.
- **An ISA extension:** add the instruction to Spike (`ext/spike`, e.g. as an
  extension), then teach `src/csrc/src/inst_decode.cpp` its class and
  registers.

Before a PR is complete: `bash tools/lint.sh` passes and `uv run pytest`
passes. 

# Top-down analysis

Top-down analysis answers "what is this program waiting for?" with one
hierarchical breakdown that adds up to 100%. It is the method from Yasin's
*A Top-Down Method for Performance Analysis and Counters Architecture* (ISPASS
2014), adapted to Reef: an in-order core with no wrong-path execution.

Use it before changing the model or a kernel. It tells you which module to
look at, and the levels below tell you why.

## Slots

Every cycle, dispatch has `dispatch_width` **slots** (4 on M3). Each slot either
carries an instruction or is wasted, and every wasted slot is charged to
exactly one cause. Slots, not cycles, are the unit because a cycle that
dispatches one instruction out of four wastes three slots; counting it as a
busy cycle would hide that.

The dispatch unit (`model/core/Dispatch.cpp`) does the accounting. In each
cycle it dispatches in order until something stops it; the slots left over
are charged to whatever stopped it.

## The tree

```
slots                                   dispatch_width x cycles
├── retiring                            an instruction dispatched
├── frontend_bound                      the instruction buffer was empty
│   ├── branch_resteer                  refetching after a taken branch, jump or trap
│   └── fetch_bandwidth                 fetch hasn't caught up (width, interval, latency)
└── backend_bound                       the next instruction was blocked
    ├── core                            ... by scalar execution (ALU, MUL, DIV, FPU)
    ├── memory                          ... by the LSU or the memory behind it
    ├── vector                          ... by the vector engine
    └── matrix                          ... by the matrix engine
        └── raw_hazard | waw_hazard | rob_full | lsu/vector/matrix_queue_full
```

**Level 1: retiring, frontend bound, backend bound.**

- **Retiring.** Every dispatched instruction retires (execute-at-fetch never
  runs the wrong path), so retiring slots = instructions. A program that is
  100% retiring runs at the dispatch width.
- **Frontend bound.** Dispatch had a free slot but nothing to put in it.
- **Backend bound.** There was an instruction, but it couldn't go.

There is no *bad speculation* category: the model never executes wrong-path
instructions, so no slot is ever wasted on them. The cost of a taken branch
shows up as `frontend_bound.branch_resteer`, where Yasin's method also puts
resteer bubbles. When static branch prediction (roadmap B05) adds
mispredictions, their refetch bubbles land there too.

**Level 2: which part of the machine.**

- **Frontend bound** splits by what the frontend reports
  (`InstBufferInterface::getStall()`): a branch resteer if the latest fetch
  group ended in a redirect, and fetch bandwidth otherwise.
- **Backend bound** splits by the **unit responsible**, which is the
  module or owner to talk to:
  - a register hazard (RAW or WAW) is blamed on the unit producing the
    register, so a chain of dependent loads is *memory* bound even though
    the stalled instruction is an add;
  - a full Rob is blamed on the unit its oldest instruction waits for;
  - a full queue is blamed on that queue's unit.

  `memory` covers the LSU and the memory module together, as Yasin's memory
  bound does: on M3 every load and store, scalar or vector, goes through the
  one LSU.

**Level 3: the stall reason** (`raw_hazard`, `waw_hazard`, `rob_full`,
`lsu_queue_full`, `vector_queue_full`, `matrix_queue_full`), the same reasons
as the per-cycle dispatch table.

## Using it

The end-of-run summary prints levels 1 and 2:

```
  Top-down: dispatch slots (4 per cycle; docs/topdown.md)
    retiring                    6329   37.7%
    frontend_bound               185    1.1%
      branch_resteer               0    0.0%
      fetch_bandwidth            185    1.1%
    backend_bound              10270   61.2%
      core                      4269   25.4%
      memory                    6001   35.8%
      ...
```

`--json` writes every level under `"topdown"`, and `reef_perf.topdown` turns it
into a tree. The [`topdown` experiment](../python/reef_perf/experiments/topdown/README.md)
runs the microbenchmarks and writes each one's tree, a level-2 table and a
stacked bar chart:

```sh
uv run python -m reef_perf.experiments.topdown.main
uv run python -m reef_perf.experiments.topdown.main --microbenchmark load_chain
```

To drill down:

1. Find the biggest level-1 category, then its biggest child. That is the
   module to look at.
2. Level 3 says how that unit holds dispatch up: hazards (its latency),
   queue full (its throughput) or Rob full (both).
3. Change one parameter with `--param` and rerun. The category should shrink.
   If it doesn't, the bottleneck is elsewhere.
4. For individual instructions, use the per-instruction timing trace (roadmap
   B19).

## What it doesn't cover

- **The drain at the end.** Slots cover the cycles dispatch ran. The cycles
  after the last dispatch, while the pipeline drains, are not counted, so
  `slots / dispatch_width` is below `cycles` by up to the longest latency
  (33 cycles for `div_chain`, which ends on a divide).
- **The region of interest.** The tree covers the whole program, including
  setup. For the microbenchmarks the loop dominates, so the difference is
  small.
- **Overlap.** Each wasted slot is charged to the first thing that stopped
  dispatch. If a load is both waiting on a register and facing a full LSU
  queue, it counts as the hazard, so fixing the hazard can reveal the queue.

## Extending it

- **A new reason dispatch can stall** (for example M3's dispatch rules in
  B07/B08): add a `StallReason`, return it from `Dispatch::canDispatch()`
  with the unit to blame, and add a line here. The JSON, the summary and the
  experiment pick it up.
- **A new frontend cause** (for example fetch-block alignment in B06): add a
  `FrontendStall` and return it from `Fetch::getStall()`. That is an
  interface change, so update `interfaces.md` in the same PR.
- **A new backend unit**: it becomes a level-2 category once it has an
  `ExecTarget` and a name in `getBoundName()`.

The accounting invariant is tested: retiring slots equal retired
instructions, and every category adds up to its parent
(`tests/python/unit/test_topdown.py`,
`tests/python/experiments/test_topdown_experiment.py`).

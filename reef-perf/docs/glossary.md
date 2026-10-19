# Glossary

The terms reef-perf's code, docs, reviews and tickets use, each with one
meaning. If a word here doesn't fit what you're describing, use a different
word rather than stretching this one, and add the new term here in the same
PR.

## Structure of the model

From largest to smallest:

| Term | Means | In code |
| --- | --- | --- |
| **processor** | The whole Reef NPU being modeled. Not "the core": that's a module. | `ReefSim`, the Sparta tree `top` |
| **module** | One of the five owner-assignable parts: `frontend`, `core`, `vector`, `matrix`, `memory`. Each has an owner, a directory, a config section and a Sparta subtree. | `Module` subclass; `model/<module>/`; `top.<module>` |
| **unit** | A Sparta unit: a C++ class with its own parameters, ports, events and counters. A module has one or more. Always means this, never a hardware execution resource. | `sparta::Unit` subclass: `Fetch`, `Dispatch`, `ScalarExec`, `Lsu`, `Rob`, `Vector`, `Matrix`, `Memory`; `top.<module>[.<unit>]` |
| **pool** | A set of identical hardware resources that operations book for some cycles. | `ResourcePool` |
| **resource** | One member of a pool: one ALU, one vector lane, one memory port. | a `ResourcePool` entry |
| **functional unit (FU)** | A hardware execution resource for one kind of operation: ALU, multiplier, divider, FPU. Modeled as a pool, not as a unit. Always write the full term. | `ScalarExec`'s `ALU`, `MUL`, ... pools |
| **lane** | One execution resource of the vector or matrix engine. An instruction occupies one for its occupancy. | `vec_lanes`, `mtx_lanes` |
| **engine** | The hardware the vector or matrix module models ("the vector engine"). | the `vector` and `matrix` modules |

Two names coincide on purpose: the **core** is the module with dispatch,
scalar execution, the LSU and the Rob; the **LSU** (load/store unit) is both
a unit and the hardware it models.

Avoid **block** for parts of the model or hardware: say module, unit or
resource. "Block" means only an aligned range of memory (a fetch block, the
aligned lines `countDistinctBlocks()` counts). **Component**, **subsystem**
and **stage** aren't used; pick one of the terms above.

## Connections

| Term | Means |
| --- | --- |
| **port** | A Sparta port between units (`DataInPort`/`DataOutPort`), delivering a value a set number of cycles later. A hardware memory port is a **memory port** (ITCM port, DTCM port, AXI port) and is a resource. |
| **interface** | A synchronous C++ call from one module to another (`MemoryInterface`, `InstBufferInterface`). Modules connect only through public ports and interfaces (interfaces.md). |
| **booking** | How pools and interfaces time an operation: the call reserves the resources' future cycles and returns the start and latency at once, with no events. |
| **buffer** | Holds instructions in program order between steps of the pipeline: the **instruction buffer** (frontend), the **retirement buffer** (`Rob`). |
| **queue** | A waiting area in front of the unit that executes an instruction: the LSU queue, the vector and matrix **command queues**. Sized in **entries**. |
| **credit** | A free buffer or queue entry, sent back to the sender so it knows it may send again. |

## Instructions and time

| Term | Means |
| --- | --- |
| **instruction** | One dynamic instruction: an `Inst`, shared as an `InstRef`. |
| **class** | An instruction's coarse kind (`InstClass`: ALU, LOAD, V_ARITH, ...). |
| **exec target** | The unit that executes an instruction (`ExecTarget`), decided from its class. |
| **fetch, dispatch, issue, retire** | When the frontend fetches it, dispatch sends it to its exec target and the Rob, its execution starts, and it leaves the Rob. `Inst` records each as a `*Cycle`. |
| **result ready** | The first cycle a dependent instruction may dispatch. |
| **complete** | The first cycle the instruction may retire. |
| **cycle** | A point in simulated time (`Cycle`). A length of time is a **latency** (start to result) or an **occupancy** (how long it holds a resource). |
| **redirect** | A taken branch, jump or trap: fetch continues somewhere other than the next instruction. |

## Performance analysis

| Term | Means |
| --- | --- |
| **stall** | A cycle in which nothing dispatched, charged to one **stall reason** (`StallReason`). |
| **slot** | One of the `dispatch_width` chances per cycle to dispatch an instruction (topdown.md); **slot 0** is the first. Only dispatch slots are called slots, except when quoting the M3 LSU's RTL signal `slot`. |
| **retiring / frontend bound / backend bound** | The top-down categories a slot is charged to (topdown.md). **Core, memory, vector or matrix bound** names the part of the backend responsible. |
| **utilization** | A pool's busy cycles divided by its resources × cycles. |
| **IPC** | Instructions retired per cycle. |

## Programs, runs and results

| Term | Means |
| --- | --- |
| **microbenchmark** | A small assembly program in `microbenchmarks/` that stresses one behavior. Not "workload". |
| **kernel** | A real computation, such as an LLM layer's GEMV. `gemv_int8` is a kernel-shaped microbenchmark. |
| **program** | Any ELF the simulator runs. |
| **ROI** (region of interest) | The part of a microbenchmark between its two `csrr mcycle` reads; the part compared with the RTL. |
| **simulation config** | What machine to simulate: `configs/machines/*.yaml`. |
| **experiment** | A program in `python/reef_perf/experiments/` that runs the simulator and writes **artifacts** under `results/<experiment>/`. |
| **sweep** | Many runs of the same microbenchmarks under parameter **variants**; also the experiment that does it. |
| **correlation** | Comparing the model with the M3 RTL on the same programs. |
| **golden file** | `tests/golden/m3_results.json`: the cycle counts a refactor must preserve. |
| **owner** | The person responsible for a module, its section of the config and its side of each interface. |

In the Python package, **module** keeps its Python meaning (a `.py` file);
the context makes clear which is meant.

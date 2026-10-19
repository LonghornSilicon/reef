# Module interfaces

reef-perf is split into five modules so that each can have one owner who
changes it freely, as long as the interfaces in this document hold. This page
is the contract between them. An interface change is a PR that edits this
page and both sides of the interface together.

| Module | Sparta subtree | Code | Owner |
| --- | --- | --- | --- |
| Frontend | `top.frontend` | `frontend/` | TBD |
| Core (scalar pipeline) | `top.core` | `core/` | TBD |
| Vector engine | `top.vector` | `vector/` | TBD |
| Matrix engine | `top.matrix` | `matrix/` | TBD |
| Interconnect / memory | `top.memory` | `memory/` | TBD |

Shared types live in `common/` and belong to everyone: changing them needs
review from every module they touch.

`<module>/` means `model/<module>/` (headers and sources together) and
`tests/cpp/<module>/`.

The frontend, vector, matrix and memory modules are a single Sparta unit
each, so `top.<module>` *is* the unit (`UnitModule` in `common/Module.h`).
The core has several units and its own `Module` subclass.

## Rules

1. **Modules connect only through public ports, `InstBufferInterface` and
   `MemoryInterface`.** Each
   module lists its public ports as constants, e.g.
   `Core::OutVector` or `Vector::InInsts`. `ReefSim::bindTree_()` binds them; nothing else
   reaches into another module's units.
2. **A module wires its own units** in `bind()` and may add, remove or rename
   them freely, as long as its public port constants still point at the right
   ports.
3. **Unit names and pool names are unique across modules.** Sparta's resource
   set is shared, and pools appear unprefixed in the JSON output.
4. **Each module owns its section of `configs/machines/m3.yaml`**
   (`top.<module>.params.<name>`, or
   `top.core.<unit>.params.<name>`). Mark every value with where it came
   from (RTL `file:line`, a measurement, or `ASSUMPTION`).
5. **Each module reports its own statistics** through `Module::getPools()` and
   its Sparta counters. `ReefSim` only asks for the headline numbers
   (retired instructions, cycles, stall breakdown, redirects).
6. **Timing-preserving changes pass the golden test unchanged**
   (`tests/python/integration/test_golden.py`). Changes that move timing regenerate it in
   the same commit and say why.

## The instruction object

Every instruction is one `Inst` (`common/Inst.h`), shared between modules
as an `InstRef`. Any module may **read** any field. Each field has exactly
one **writer**:

| Fields | Written by |
| --- | --- |
| `Seq`, `PC`, `NextPC`, `Encoding`, `Disasm`, `Mnemonic`, `Mem`, `VL`, `SEWBytes`, `LMUL8`, `Class`, `Srcs`, `Dsts` | frontend, when the instruction is fetched |
| `FetchCycle` | frontend |
| `DispatchCycle`, `RetireCycle` | core |
| `IssueCycle`, `ResultReadyCycle`, `CompleteCycle` | the unit that executes the instruction (below) |

Within the frontend, `Mem` and `Dsts` come straight from Spike's commit log
(exact); `Class` and `Srcs` are decoded from the encoding, because Spike does
not log register reads.

`ResultReadyCycle` and `CompleteCycle` are `Inst::Never` until their
writer sets them. Readers must treat `Never` as "not yet": Dispatch stalls a
dependent instruction, and the Rob keeps waiting.

**Who executes what** is decided by `getExecTarget()` (`common/Interfaces.h`):

| `ExecTarget` | Unit | Instruction classes |
| --- | --- | --- |
| `Scalar` | `core.scalar_exec` | ALU, branch, jump, MUL, DIV, CSR, fence, system, FP, FP divide, unknown |
| `LSU` | `core.lsu` | every load and store, scalar, FP and vector (one LSU serves both, as on M3) |
| `Vector` | `vector` | `vset*` and vector arithmetic, permutes and vector-to-scalar moves |
| `Matrix` | `matrix` | `MATRIX` (nothing decodes to it until the matrix ISA exists) |

### Timing conventions

- A unit that starts an instruction at cycle `Start` with latency `L` sets
  `IssueCycle = Start`, `ResultReadyCycle = Start + L - 1` and
  `CompleteCycle = Start + L`. The `- 1` lets a dependent instruction
  dispatch in the cycle before the result exists, which stands in for
  operand forwarding.
- Instructions reach an execution unit one cycle after dispatch (input ports
  have a delay of 1).

## Credits

Every queue between modules uses the same protocol:

- The **receiver** sends its queue size as credits once, at startup
  (a Sparta `StartupEvent`).
- The **sender** spends one credit per instruction it sends and does not send
  without one.
- The receiver returns one credit when an instruction **leaves the queue**,
  i.e. when it starts executing (`out_credits.send(1, start - now)`).

## Interfaces

### Frontend → core

The instruction buffer belongs to the frontend (as on M3). The core takes
instructions through `InstBufferInterface` (`common/Interfaces.h`), a
synchronous interface like `MemoryInterface`:

| Call | Meaning |
| --- | --- |
| `peek(Now)` | the oldest instruction the core may take this cycle, or none |
| `pop(Now)` | the core took it |
| `isDrained(Now)` | the program is fully fetched and every instruction taken |

An instruction becomes visible the cycle after it is fetched, and an entry the
core frees can be refilled from the next cycle: the one-cycle hand-off in
each direction that a pair of ports would give. `Fetch` implements it;
`Core::setInstBuffer()` connects it.

Not modeled yet: a redirect/flush signal from the core. Today the frontend
charges a fixed `redirect_penalty` on every taken branch, because it already
knows the outcome from Spike.

### Backend → vector

| Port | Type | Direction |
| --- | --- | --- |
| `Core::OutVector` → `Vector::InInsts` | `InstRef` | instructions routed to `VECTOR` |
| `Vector::OutCredits` → `Backend::InVectorCredits` | `uint32_t` | free command-queue entries |

Vector loads and stores do **not** cross this interface; they go to the
core's LSU.

### Backend → matrix

Same shape as vector, so the two engines stay interchangeable from the
core's point of view:

| Port | Type | Direction |
| --- | --- | --- |
| `Backend::OutMatrix` → `Matrix::InInsts` | `InstRef` | instructions routed to `MATRIX` |
| `Matrix::OutCredits` → `Backend::InMatrixCredits` | `uint32_t` | free command-queue entries |

### Anything → memory

Requesters call `MemoryInterface::access(MemRequest)` and get a
`MemResponse` (`common/Interfaces.h`):

| `MemRequest` field | Meaning |
| --- | --- |
| `Requester` | `IFetch`, `LSU` or `Matrix` |
| `Accesses` | the bytes accessed (`Inst::Mem`, from Spike) |
| `Earliest` | first cycle the requester's own port is free |

| `MemResponse` field | Meaning |
| --- | --- |
| `Start` | cycle the transfer starts (≥ `Earliest`) |
| `Occupancy` | cycles the requester's port stays busy |
| `Latency` | cycles from `Start` until the data is usable or the write is done |

The call is **synchronous**: memory books its own resources and answers at
once ("booking" timing, like the execution pools). The memory owner may
replace it with request/response ports later; that is an interface change.

The memory module routes by the operation's first address, using the memory
map in `common/MemoryMap.h`, the same list Spike is given:

| Region | Default map | Timed by |
| --- | --- | --- |
| ITCM | `0x0`, 8 KB | `memory` (ITCM port) |
| DTCM | `0x10000`, 32 KB | `memory` (DTCM port) |
| anything else | — | `memory` (AXI): data channel, outstanding limit, fixed black-box latency |

Today only the LSU calls it. Instruction fetch is still free, and the matrix
engine does not touch memory yet.

## Known simplifications, by module

These are the obvious next steps for each owner. None of them changes an
interface unless noted.

- **Frontend:** no branch predictor (M3 predicts backward-taken,
  forward-not-taken); fetch does not go through memory (adds an `IFetch`
  requester); no redirect/flush signal to the core (interface change).
- **Backend:** Reef's special dispatch rules (a branch ends the group, FP and
  CSR dispatch alone, CSRs wait for an empty Rob); the LSU has one slot.
- **Vector:** one pool of identical lanes; no per-type units, chaining or
  vector ROB.
- **Matrix:** everything. The ISA encoding, an opaque-op extension in Spike,
  operands from vector registers (a vector → matrix interface) and a memory
  port (`RequesterKind::Matrix`).
- **Memory:** synchronous booking instead of request/response; one port per
  TCM, no banks; AXI read and write channels not separated; no arbitration
  between requesters; the AXI latency and outstanding limit are assumptions.
  No workload touches AXI memory yet, because Spike only backs the ITCM and
  DTCM regions.

## Adding a unit to a module

1. Write the unit (a `sparta::Unit` with a `ParameterSet` and a static
   `name`) in `<module>/`.
2. If the module is still a `UnitModule`, give it a `Module` subclass like
   `Core`. Register each unit's factory in `addFactories()`, list it in
   `getUnitNames()`, and bind the ports between units in `bind()`.
3. Add its source to `CMakeLists.txt` and its parameters to the module's
   section of `configs/machines/m3.yaml`.
4. If other modules need to reach it, add a public port constant and a line
   to this page.

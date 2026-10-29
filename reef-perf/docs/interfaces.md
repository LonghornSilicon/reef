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

The terms module, unit, pool, port and interface are defined in
[glossary.md](glossary.md).

Shared types live in `common/` and belong to everyone: changing them needs
review from every module they touch.

`<module>/` means `model/<module>/` (headers and sources together) and
`tests/cpp/<module>/`.

The frontend, vector, matrix and memory modules are a single Sparta unit
each, so `top.<module>` *is* the unit (`UnitModule` in `common/Module.h`).
The core has several units and its own `Module` subclass.

## Rules

1. **Modules connect only through public ports, `InstBufferInterface`,
   `VectorLsuInterface` and `MemoryInterface`.** Each
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
| `Seq`, `PC`, `NextPC`, `Encoding`, `Disasm`, `Mnemonic`, `Mem`, `VL`, `SEWBytes`, `LMUL8`, `Class`, `Srcs`, `Dsts`, `Rs1Val`, `Rs2Val` | frontend, when the instruction is fetched |
| `FetchCycle` | frontend |
| `DispatchCycle`, `RetireCycle` | core |
| `IssueCycle`, `ResultReadyCycle`, `CompleteCycle` | the unit that executes the instruction (below) |

Fields and interfaces use the purpose-defined types in `common/Types.h`:
`Cycle` for points in time (durations stay plain integers), `Address` for
byte addresses and PCs, and `RegId` for registers (`Srcs`, `Dsts`): a register
file and an index, with `getDense()` for arrays indexed by register.

Within the frontend, `Mem` and `Dsts` come straight from Spike's commit log
(exact); `Class` and `Srcs` are decoded from the encoding, because Spike does
not log register reads.

`ResultReadyCycle` and `CompleteCycle` are `Inst::Never` until their
writer sets them. Readers must treat `Never` as "not yet": Dispatch stalls a
dependent instruction, and the Rob keeps waiting.

Dispatch doesn't wait for vector registers, as on M3, whose scalar
scoreboards cover only x and f. Only vector instructions read and write them,
so the vector module tracks them itself, one register at a time.

**Who executes what** is decided by `getExecTarget()` (`common/Interfaces.h`):

| `ExecTarget` | Unit | Instruction classes |
| --- | --- | --- |
| `Scalar` | `core.scalar_exec` | ALU, branch, jump, MUL, DIV, CSR, fence, system, FP, FP divide, unknown |
| `LSU` | `core.lsu` | scalar and FP loads and stores |
| `Vector` | `vector` | every vector instruction: `vset*`, arithmetic, permutes, vector-to-scalar moves, loads and stores. Loads and stores reach `core.lsu` through `VectorLsuInterface`; one LSU serves both, as on M3 |
| `Matrix` | `matrix` | `MATRIX` (nothing decodes to it until the matrix ISA exists) |

### Timing conventions

- A unit that starts an instruction at cycle `Start` with latency `L` sets
  `IssueCycle = Start`, `ResultReadyCycle = Start + L - 1` and
  `CompleteCycle = Start + L`. The `- 1` lets a dependent instruction
  dispatch in the cycle before the result exists, which stands in for
  operand forwarding.
- The vector module is the exception: an instruction is complete when its
  last uop retires from the vector ROB (`CompleteCycle`), and its result is
  ready in that same cycle (`ResultReadyCycle`), as in the M3 RTL, where a
  dependent uop reads it from the ROB. So a vector instruction holds its Rob
  entry through the whole vector pipeline: 8 cycles for a vector add. A
  vector store is complete for the core's Rob a cycle after its first
  register's store completes, as on M3, where the retirement buffer matches
  the LSU's `storeComplete` by PC (`RetirementBuffer.scala`). The buffer
  matches vector writes by PC too, so an instruction is also complete when an
  older instance with the same PC, still in flight when it was dispatched,
  completes: in a short loop, a later iteration's vector instruction can
  retire long before its own uops finish. This is an M3 RTL bug (RB1 in
  `rtl_bugs.md` at the top of the Reef repository), fixed upstream after M3;
  `vec_completion_matches_pc: false` models the fix.
- Instructions reach their exec target one cycle after dispatch (input ports
  have a delay of 1).

## Credits

Every queue between modules uses the same protocol:

- The **receiver** sends its queue size as credits once, at startup
  (a Sparta `StartupEvent`).
- The **sender** spends one credit per instruction it sends and does not send
  without one.
- The receiver returns one credit when an instruction **leaves the queue**,
  i.e. when it starts executing (`out_credits.send(1, start - now)`). The
  vector module's queue is M3's command queue, which an instruction leaves
  when the first decode stage takes it.

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
| `getStall(Now)` | why `peek(Now)` found nothing (`FrontendStall`), for top-down accounting ([topdown.md](topdown.md)) |

An instruction becomes visible the cycle after it is fetched, and an entry the
core frees can be refilled from the next cycle: the one-cycle hand-off in
each direction that a pair of ports would give. `Fetch` implements it;
`Core::setInstBuffer()` connects it.

Not modeled yet: a redirect/flush signal from the core. Today the frontend
charges a fixed `redirect_penalty` on every taken branch, because it already
knows the outcome from Spike.

### Core → vector

| Port | Type | Direction |
| --- | --- | --- |
| `Core::OutVector` → `Vector::InInsts` | `InstRef` | instructions routed to `VECTOR` |
| `Vector::OutCredits` → `Core::InVectorCredits` | `uint32_t` | free command-queue entries |

Vector loads and stores cross this interface like every other vector
instruction.

### Vector → core (LSU)

The vector module runs vector loads and stores through its own pipeline, one
uop per 16-byte vector register, and books the core's LSU for each register
through `VectorLsuInterface` (`common/Interfaces.h`). It is synchronous, like
`MemoryInterface`: the call books the LSU and returns the timing at once.
`Core::getVectorLsu()` returns it and `Vector::setLsu()` connects it.

| `VectorLsuRequest` field | Meaning |
| --- | --- |
| `IsStore` | true for a vector store |
| `Accesses` | the register's bytes, its part of `Inst::Mem` (`splitByRegister()`) |
| `Earliest` | first cycle the LSU may take it: it left the vector LSU reservation station |
| `FirstRegister` | the instruction's first register; its LSU queue entry frees when it starts |
| `LastRegister` | the instruction's last register; the LSU then takes a cycle to load the next instruction |

| `VectorLsuResponse` field | Meaning |
| --- | --- |
| `Start` | cycle the LSU took it (≥ `Earliest`) |
| `Done` | cycle the LSU hands the loaded register back, or reports the store complete (M3's `lsu2rvv` and `storeComplete`) |

Scalar and vector accesses share the LSU's one slot, in the order they book
it. A vector load or store also takes an LSU queue entry at dispatch, as on
M3, so Dispatch needs both a vector and an LSU credit for it. The LSU owns the per-register cost (`lsu_vector_*` parameters); the
vector module owns everything before and after it.

### Core → matrix

Same shape as vector, so the two engines stay interchangeable from the
core's point of view:

| Port | Type | Direction |
| --- | --- | --- |
| `Core::OutMatrix` → `Matrix::InInsts` | `InstRef` | instructions routed to `MATRIX` |
| `Matrix::OutCredits` → `Core::InMatrixCredits` | `uint32_t` | free command-queue entries |

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

**Data never passes through this interface.** Spike executes every load
and store when the instruction is fetched (execute-at-fetch), against its own
copy of memory; the timing model only receives the addresses and sizes
(`Inst::Mem`). The memory module decides *when*, never *what*.

**`access()` answers at once ("booking").** It is a plain function call, not
a message: memory works out when the transfer can start and how long it takes,
reserves those future cycles on its ports (`ResourcePool`), and returns the
result before the call ends. There is no request in flight and no event later.
The execution pools in the core time instructions the same way. So the LSU
knows an instruction's result-ready cycle the moment it calls.

This is simple and fast, and enough while the LSU is the only requester. It
can't express:

- **arbitration:** two requesters in the same cycle are served in call order,
  not by priority;
- **backpressure:** memory can't refuse or delay a request after the call;
- **out-of-order responses:** each answer is fixed when it is booked.

Modeling any of these means replacing the call with a pair of Sparta ports: a
request port, and a response port the memory sends on in a later cycle. That
changes both the requesters and the memory module, so under rule 1 it is an
interface change: one PR that edits this section and both sides together. The
memory owner decides when it is needed, for example when the matrix engine or
instruction fetch becomes a second requester.

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
- **Core:** Reef's special dispatch rules (a branch ends the group, FP and
  CSR dispatch alone, CSRs wait for an empty Rob); the LSU handles one memory instruction at a time.
- **Vector:** the divider's time doesn't depend on its operands; a load or
  store has one uop per register its accesses touch, whatever its EMUL; only
  unit-stride e32 loads and stores at LMUL 1 and 4 are measured.
- **Matrix:** everything. The ISA encoding, an opaque-op extension in Spike,
  operands from vector registers (a vector → matrix interface) and a memory
  port (`RequesterKind::Matrix`).
- **Memory:** synchronous booking instead of request/response; one port per
  TCM, no banks; AXI read and write channels not separated; no arbitration
  between requesters; the AXI latency and outstanding limit are assumptions.
  No microbenchmark touches AXI memory yet, because Spike only backs the ITCM and
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

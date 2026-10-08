# Module interfaces

reef-perf is split into five modules so that each can have one owner who
changes it freely, as long as the interfaces in this document hold. This page
is the contract between them. An interface change is a PR that edits this
page and both sides of the interface together.

| Module | Sparta subtree | Code | Owner |
| --- | --- | --- | --- |
| Frontend | `top.frontend` | `frontend/` | TBD |
| Backend | `top.backend` | `backend/` | TBD |
| Vector engine | `top.vector` | `vector/` | TBD |
| Matrix engine | `top.matrix` | `matrix/` | TBD |
| Interconnect / memory | `top.mem` | `mem/` | TBD |

Shared types live in `common/` and belong to everyone: changing them needs
review from every module they touch.

`<module>/` means `src/csrc/include/reef_perf/<module>/`,
`src/csrc/src/<module>/` and `tests/cpp/<module>/`.

## Rules

1. **Modules connect only through public ports and `MemoryInterface`.** Each
   module's header lists its public ports as `k...` constants, e.g.
   `Backend::kOutVector`. `ReefSim::bindTree_()` binds them; nothing else
   reaches into another module's units.
2. **A module wires its own units** in `bind()` and may add, remove or rename
   them freely, as long as its public port constants still point at the right
   ports.
3. **Unit names and pool names are unique across modules.** Sparta's resource
   set is shared, and pools appear unprefixed in the JSON output.
4. **Each module owns its section of `configs/m3.yaml`**
   (`top.<module>.<unit>.params.<name>`). Mark every value with where it came
   from (RTL `file:line`, a measurement, or `ASSUMPTION`).
5. **Each module reports its own statistics** through `Module::pools()` and
   its Sparta counters. `ReefSim` only asks for the headline numbers
   (retired instructions, cycles, stall breakdown, redirects).
6. **Timing-preserving changes pass the golden test unchanged**
   (`tests/python/test_golden.py`). Changes that move timing regenerate it in
   the same commit and say why.

## The instruction object

Every instruction is one `Inst` (`common/inst.hpp`), shared between modules
as an `InstPtr`. Any module may **read** any field. Each field has exactly
one **writer**:

| Fields | Written by |
| --- | --- |
| `seq`, `pc`, `next_pc`, `encoding`, `disasm`, `mnemonic`, `mem`, `vl`, `sew_bytes`, `lmul8`, `cls`, `srcs`, `dsts` | frontend, when the instruction is fetched |
| `fetch_cycle` | frontend |
| `dispatch_cycle`, `retire_cycle` | backend |
| `issue_cycle`, `result_ready_cycle`, `complete_cycle` | the unit that executes the instruction (below) |

Within the frontend, `mem` and `dsts` come straight from Spike's commit log
(exact); `cls` and `srcs` are decoded from the encoding, because Spike does
not log register reads.

`result_ready_cycle` and `complete_cycle` are `Inst::kNever` until their
writer sets them. Readers must treat `kNever` as "not yet": Dispatch stalls a
dependent instruction, and the Rob keeps waiting.

**Who executes what** is decided by `exec_target()` (`common/interfaces.hpp`):

| `ExecTarget` | Unit | Instruction classes |
| --- | --- | --- |
| `SCALAR` | `backend.scalar_exec` | ALU, branch, jump, MUL, DIV, CSR, fence, system, FP, FP divide, unknown |
| `LSU` | `backend.lsu` | every load and store, scalar, FP and vector (one LSU serves both, as on M3) |
| `VECTOR` | `vector.vxu` | `vset*` and vector arithmetic, permutes and vector-to-scalar moves |
| `MATRIX` | `matrix.mxu` | `MATRIX` (nothing decodes to it until the matrix ISA exists) |

### Timing conventions

- A unit that starts an instruction at cycle `start` with latency `L` sets
  `issue_cycle = start`, `result_ready_cycle = start + L - 1` and
  `complete_cycle = start + L`. The `- 1` lets a dependent instruction
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

### Frontend → backend

| Port | Type | Direction |
| --- | --- | --- |
| `Frontend::kOutInsts` → `Backend::kInInsts` | `FetchPacket` | instructions, in program order; `last = true` ends the program |
| `Backend::kOutFetchCredits` → `Frontend::kInCredits` | `std::uint32_t` | free instruction-buffer entries |

Not modelled yet: a redirect/flush signal from the backend. Today the frontend
charges a fixed `redirect_penalty` on every taken branch, because it already
knows the outcome from Spike.

### Backend → vector

| Port | Type | Direction |
| --- | --- | --- |
| `Backend::kOutVector` → `Vector::kInInsts` | `InstPtr` | instructions routed to `VECTOR` |
| `Vector::kOutCredits` → `Backend::kInVectorCredits` | `std::uint32_t` | free command-queue entries |

Vector loads and stores do **not** cross this interface; they go to the
backend's LSU.

### Backend → matrix

Same shape as vector, so the two engines stay interchangeable from the
backend's point of view:

| Port | Type | Direction |
| --- | --- | --- |
| `Backend::kOutMatrix` → `Matrix::kInInsts` | `InstPtr` | instructions routed to `MATRIX` |
| `Matrix::kOutCredits` → `Backend::kInMatrixCredits` | `std::uint32_t` | free command-queue entries |

### Anything → memory

Requesters call `MemoryInterface::access(MemRequest)` and get a
`MemResponse` (`common/interfaces.hpp`):

| `MemRequest` field | Meaning |
| --- | --- |
| `requester` | `IFETCH`, `LSU` or `MATRIX` |
| `accesses` | the bytes accessed (`Inst::mem`, from Spike) |
| `earliest` | first cycle the requester's own port is free |

| `MemResponse` field | Meaning |
| --- | --- |
| `start` | cycle the transfer starts (≥ `earliest`) |
| `occupancy` | cycles the requester's port stays busy |
| `latency` | cycles from `start` until the data is usable or the write is done |

The call is **synchronous**: memory books its own resources and answers at
once ("booking" timing, like the execution pools). The memory owner may
replace it with request/response ports later; that is an interface change.

The memory module routes by the operation's first address, using the memory
map in `common/memory_map.hpp`, the same list Spike is given:

| Region | Default map | Timed by |
| --- | --- | --- |
| ITCM | `0x0`, 8 KB | `mem.tcm` (ITCM port) |
| DTCM | `0x10000`, 32 KB | `mem.tcm` (DTCM port) |
| anything else | — | `mem.axi`: data channel, outstanding limit, fixed black-box latency |

Today only the LSU calls it. Instruction fetch is still free, and the matrix
engine does not touch memory yet.

## Known simplifications, by module

These are the obvious next steps for each owner. None of them changes an
interface unless noted.

- **Frontend:** no branch predictor (M3 predicts backward-taken,
  forward-not-taken); fetch does not go through memory (adds a `IFETCH`
  requester); no redirect/flush signal to the backend (interface change).
- **Backend:** Reef's special dispatch rules (a branch ends the group, FP and
  CSR dispatch alone, CSRs wait for an empty Rob); the LSU has one slot.
- **Vector:** one pool of identical lanes; no per-type units, chaining or
  vector ROB.
- **Matrix:** everything. The ISA encoding, an opaque-op extension in Spike,
  operands from vector registers (a vector → matrix interface) and a memory
  port (`Requester::MATRIX`).
- **Memory:** synchronous booking instead of request/response; one port per
  TCM, no banks; AXI read and write channels not separated; no arbitration
  between requesters; the AXI latency and outstanding limit are assumptions.
  No workload touches AXI memory yet, because Spike only backs the ITCM and
  DTCM regions.

## Adding a unit to a module

1. Write the unit (a `sparta::Unit` with a `ParameterSet` and a static
   `name`) in `<module>/`.
2. Register its factory in the module's `add_factories()`, list it in
   `unit_names()`, and bind its ports in `bind()`.
3. Add its source to `CMakeLists.txt` and its parameters to the module's
   section of `configs/m3.yaml`.
4. If other modules need to reach it, add a public port constant and a line
   to this page.

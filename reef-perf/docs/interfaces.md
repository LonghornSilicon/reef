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
| Matrix engine | `top.matrix` | `matrix/` | Akul Laddha |
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
   `VectorLsuInterface`, `VectorRegisterInterface` and `MemoryInterface`.** Each
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
| `Seq`, `PC`, `NextPC`, `Encoding`, `Disasm`, `Mnemonic`, `Mem`, `VL`, `SEWBytes`, `LMUL8`, `Class`, `Srcs`, `Dsts`, `Rs1Val`, `Rs2Val`, `Mtx` | frontend, when the instruction is fetched |
| `FetchCycle` | frontend |
| `DispatchCycle`, `RetireCycle` | core |
| `IssueCycle`, `ResultReadyCycle`, `CompleteCycle` | the unit that executes the instruction (below) |

Fields and interfaces use the purpose-defined types in `common/Types.h`:
`Cycle` for points in time (durations stay plain integers), `Address` for
byte addresses and PCs, and `RegId` for registers (`Srcs`, `Dsts`): a register
file and an index, with `getDense()` for arrays indexed by register.

Within the frontend, `Mem` and `Dsts` come straight from Spike's commit log
(exact); `Class` and `Srcs` are decoded from the encoding, because Spike does
not log register reads. Zvt (matrix) instructions are the exception: Spike
does not know them, so when `top.frontend.params.zvt` is on (it is off by
default, and on in `configs/machines/m3_vme.yaml`) the frontend runs them
itself (`frontend/Zvt.h`) and
synthesizes `Mem` (unit-stride tile loads and stores), `Dsts` (`rd` of
`msettn`/`msettm`/`msettk`, `vd` of `vtmv.v.t`) and `Mtx`, the matrix
details: op, tile group, `tm`/`tn`/`tk`, element type (docs/matrix.md).

`ResultReadyCycle` and `CompleteCycle` are `Inst::Never` until their
writer sets them. Readers must treat `Never` as "not yet": Dispatch stalls a
dependent instruction, and the Rob keeps waiting.

Dispatch doesn't wait for vector registers written by vector instructions,
as on M3, whose scalar scoreboards cover only x and f; the vector module
tracks those itself, one register at a time. Dispatch does track vector
registers that a matrix instruction (`vtmv.v.t`) writes, so their readers
wait until the matrix module has timed it. Matrix instructions that read
vector registers ask the vector module through `VectorRegisterInterface`.

**Who executes what** is decided by `getExecTarget()` (`common/Interfaces.h`):

| `ExecTarget` | Unit | Instruction classes |
| --- | --- | --- |
| `Scalar` | `core.scalar_exec` | ALU, branch, jump, MUL, DIV, CSR, fence, system, FP, FP divide, unknown |
| `LSU` | `core.lsu` | scalar and FP loads and stores |
| `Vector` | `vector` | every vector instruction: `vset*`, arithmetic, permutes, vector-to-scalar moves, loads and stores. Loads and stores reach `core.lsu` through `VectorLsuInterface`; one LSU serves both, as on M3 |
| `Matrix` | `matrix` | `MATRIX`: Zvt `vtzero`, `vtdiscard`, `vtmv.*`, `vtle*`, `vtse*`, `vtmm*`. Tile loads and stores reach `core.lsu` through `VectorLsuInterface`. The Zvt `mset*` instructions are `VSET` and go to `vector` |

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
| `IsTile` | a 16-byte beat of a Zvt tile load or store, set only by the matrix module (default false). With `lsu_tile_cost` on, the LSU charges it the `lsu_tile_*` cost instead of the vector-register cost; off, or false, it costs a vector register (`docs/lsu-tile-beats.md`) |

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

A tile load or store also takes an LSU queue entry at dispatch, as CoralNPU
sends `VTLOAD`/`VTSTORE` to the LSU (Decode.scala), so Dispatch needs a
matrix and an LSU credit for it. The matrix module books the LSU one 16-byte
beat at a time through `VectorLsuInterface` (the same calls as the vector
module); the LSU frees the queue entry when the first beat starts.

### Vector → matrix (register scoreboard)

`VectorRegisterInterface::getRegisterReady(Index)` returns the first cycle a
reader of `v<Index>` may start, as the vector module recorded it. The matrix
module times each instruction in the Tick phase of the cycle it arrives, so
every vector instruction dispatched in the same cycle (delivered in
PortUpdate) has already been booked. The vector module implements it; the
matrix module reads it. One known gap: a younger vector instruction
dispatched in the same cycle that overwrites a register an older matrix
instruction reads makes the matrix instruction wait for the younger write.

### Matrix → vector (register scoreboard)

`vtmv.v.t` writes a vector register from the matrix module.
`VectorRegisterInterface::setRegisterReady(Index, Ready)` records that write
in the vector module's scoreboard (the later of it and what is already
there), so a vector instruction that reads the register waits for it inside
the vector module, as CoralNPU's RVV backend resolves the dependence. The
matrix module calls it in the Tick phase of the cycle it times the `vtmv`;
the vector module implements it, and vector-only programs never call it.

Dispatch keeps the rest of the rule. A reader of a matrix-written vector
register may not dispatch in the writer's own cycle, because the vector
module would book it in PortUpdate before the matrix module records the
write. A second matrix writer of the same register (`vtmv.v.t` after
`vtmv.v.t`) does not wait at all, because the matrix module runs them in
order. A vector writer after a matrix writer still waits for the matrix
result (WAW), as before.

Until 2026-11-07 Dispatch held every reader until the matrix result was
ready. That kept the vector module untouched, but stalled the in-order core
behind each `vtmv` → `vse32` pair, which the RTL does not do: `gemm_tile`'s
read-back rows took 13 cycles each against the RTL's 7.

### Anything → memory

Requesters call `MemoryInterface::access(MemRequest)` and get a
`MemResponse` (`common/Interfaces.h`):

| `MemRequest` field | Meaning |
| --- | --- |
| `Requester` | `IFetch`, `LSU`, `Matrix`, `AxiSlave` (host or DMA) or `Debug`; sets the TCM arbiter priority |
| `Accesses` | the bytes accessed (`Inst::Mem`, from Spike) |
| `Earliest` | first cycle the requester's own port is free |
| `OnFinishMoved` | optional; called with the new finish cycle when a higher-priority requester later takes one of this transfer's TCM cycles |

`MemAccess::Value` holds the value a store writes, from Spike. The memory module uses it for the DMA's registers (`memory/Dma.h`).

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

- **arbitration, except at the TCMs:** two requesters in the same cycle are
  served in call order. The TCMs (`memory/Tcm.h`) arbitrate by M3's fixed
  priority instead: the core books as before and takes any cycle a
  lower-priority transfer (AXI slave, debug) already holds; that transfer
  moves to its next free cycle and hears its new finish cycle through
  `OnFinishMoved`;
- **backpressure:** memory can't refuse a request after the call;
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
engine reaches memory through the LSU, as CoralNPU's does; a dedicated
matrix memory port (`RequesterKind::Matrix`) is not modeled yet.

## Matrix-engine edits to other modules (for review)

The matrix engine needed small changes outside `matrix/`. This section lists
every one, so each owner can review the part in their module. They share
three properties:

- **Off for M3.** Each fires only for a Zvt instruction, a request from the
  matrix module, or a parameter that defaults off. Zvt itself is off unless
  `top.frontend.params.zvt` is true ([`configs/machines/m3_vme.yaml`](../configs/machines/m3_vme.yaml)); with
  it off a Zvt word traps in Spike, as on the M3 RTL.
- **Golden unchanged.** [`tests/golden/m3_results.json`](../tests/golden/m3_results.json) is the same as on the
  branch base (`5393b2f`), so no M3 cycle or stall count moved.
- **Additive.** No existing behavior was removed. Where a rule changed
  (Dispatch, below), it is a rule that only ever applied to matrix writers.

Branch `akul/matrix_engine`, against `yuvesh-kernel-integration` (`5393b2f`).

### Frontend ([`frontend/`](../model/frontend/))

| File | Change | Fires when | Verified by |
| --- | --- | --- | --- |
| [`Fetch.h`](../model/frontend/Fetch.h), [`Fetch.cpp`](../model/frontend/Fetch.cpp) | Parameter `zvt` (default false), passed to `FuncSim` in `setFuncSim`. | Always read; it changes nothing while false. | `ZvtTest.InstDecodeIgnoresZvtWhenOff`, `test_zvt_traps_without_vme_config` |
| [`FuncSim.h`](../model/frontend/FuncSim.h), [`FuncSim.cpp`](../model/frontend/FuncSim.cpp) | With `zvt` on, `next()` looks for a Zvt encoding before Spike; a Zvt word runs in `executeZvt` (mset* configuration and `rd` writes, traps where the RTL traps, the shapes the RTL retires as no-ops, operand values for the timing model). A standard `vset*` also clears mtype (`RvvFrontEnd.sv:205`). | Only with `zvt` on, and only for Zvt words (and the mtype clear on `vset*`). | Matrix microbenchmarks (`EXPECTED_OPS`), `ZvtTest` |
| [`InstDecode.h`](../model/frontend/InstDecode.h), [`InstDecode.cpp`](../model/frontend/InstDecode.cpp) | `decodeInst(Inst &, bool Zvt = false)`: with Zvt on, a Zvt word gets class `MATRIX` (`VSET` for mset*) and its sources, per the RTL's uop operand offsets. | Only with `zvt` on. | `ZvtTest.InstDecodeClassifiesZvt` |
| [`Zvt.h`](../model/frontend/Zvt.h), [`Zvt.cpp`](../model/frontend/Zvt.cpp) (new) | Zvt encodings (spec v0.4 at CoralNPU `8a5e1b31`) and the mset* semantics. | Called only from the two rows above. | `ZvtTest` |

mset* are vector configuration on CoralNPU (the RVV front end runs them), so
they reach the vector module as `VSET` instructions. The vector module's
code is unchanged for them.

### Core ([`core/`](../model/core/))

| File | Change | Fires when | Verified by |
| --- | --- | --- | --- |
| [`Dispatch.cpp`](../model/core/Dispatch.cpp) | A tile load or store (`vtle*`, `vtse*`) also needs an LSU credit, as CoralNPU sends it to the LSU at dispatch; the LSU frees it when the first beat starts ("Core → matrix"). | A `MATRIX` instruction whose `Mtx.Op` is a tile load or store. | `tile_*` microbenchmarks |
| [`Dispatch.cpp`](../model/core/Dispatch.cpp) | Vector registers written by a matrix instruction (`vtmv.v.t`): a reader waits only for the writer's own cycle, a second matrix writer does not wait, a vector writer still waits for the matrix result ("Matrix → vector"). Vector registers written by vector instructions are still not tracked in Dispatch, as before. | Only when a vector register's last writer is a matrix instruction. | `test_vtmv_readers_do_not_stall_dispatch`, `gemm_tile` |
| [`Lsu.h`](../model/core/Lsu.h), [`Lsu.cpp`](../model/core/Lsu.cpp) | `lsu_tile_cost` (default false) and `lsu_tile_inst_overhead`, `lsu_tile_beat_overhead`, `lsu_tile_switch_cycles`: a tile beat's own cost in place of the vector-register cost. Off in `m3.yaml` and `m3_vme.yaml`; on only in `matrix_diff`'s `REFERENCE_CORE_PARAMS` (`docs/lsu-tile-beats.md`). | Only for a request with `IsTile` and `lsu_tile_cost` true. | `test_tile_beats_take_their_own_lsu_cost` |

[`configs/machines/m3_vme.yaml`](../configs/machines/m3_vme.yaml) sets `top.core.lsu.params.lsu_tile_*` with
`lsu_tile_cost: false`. `m3.yaml`'s core section is unchanged.

Open items for the core owner are in [`lsu-tile-beats.md`](lsu-tile-beats.md), "What the core
owner is asked to do": which LSU and ROB Reef has, the vector load path, and
the `vtdiscard` interlock.

### Vector ([`vector/`](../model/vector/))

| File | Change | Fires when | Verified by |
| --- | --- | --- | --- |
| [`Vector.h`](../model/vector/Vector.h) | `Vector` implements `VectorRegisterInterface`: `getRegisterReady` returns its scoreboard entry (the matrix module waits for vector writers of its operands) and `setRegisterReady` records a matrix write (the later of it and the entry). | Called only by the matrix module. | `gemm_tile`, matrix microbenchmarks |

### Common ([`common/`](../model/common/))

| File | Change | Fires when | Verified by |
| --- | --- | --- | --- |
| [`Inst.h`](../model/common/Inst.h), [`Inst.cpp`](../model/common/Inst.cpp) | `MatrixOp`, `MatrixType`, `MatrixInfo`, `Inst::Mtx` (default `MatrixOp::None`), `getMatrixOpName`. | Filled in only for `MATRIX` instructions. | `InstTest`, matrix microbenchmarks |
| [`Interfaces.h`](../model/common/Interfaces.h) | New `VectorRegisterInterface` (`getRegisterReady`, `setRegisterReady`); new `VectorLsuRequest::IsTile`, default false. | Only the matrix module calls the first or sets the second. | As above |

### Top level and build

| File | Change | Fires when |
| --- | --- | --- |
| [`ReefSim.h`](../model/ReefSim.h), [`ReefSim.cpp`](../model/ReefSim.cpp) | Binds the matrix module to the core's LSU (`setLsu`) and to the vector module's scoreboard (`setVectorRegisters`), and writes its statistics as the last JSON object, `"matrix"`. | Wiring always; the matrix module does nothing without Zvt instructions. The JSON block is new; existing keys are unchanged. |
| [`CMakeLists.txt`](../CMakeLists.txt), [`tests/cpp/CMakeLists.txt`](../tests/cpp/CMakeLists.txt) | Build `frontend/Zvt.cpp`, `matrix/MatrixFabric.cpp`, `ZvtTest` and `MatrixFabricTest`. | Build only. |
| [`inference-engine/coral/tools/setup_sim.sh`](../../inference-engine/coral/tools/setup_sim.sh) | Compiles `frontend/Zvt.cpp` into `coral_sim` with `FuncSim.cpp` and `InstDecode.cpp`, which now call it. The script lists reef-perf sources by hand, so without it the Coral kernels' `golden` CI job fails to link. | Build only; `coral_sim` keeps Zvt off. |
| [`configs/machines/m3.yaml`](../configs/machines/m3.yaml) | The matrix module's own section (`top.matrix.params`), rule 4. | Matrix only. |

### Frontend → memory (fetch)

Each fetch group books its 16-byte ITCM line through
`MemoryInterface::access()` with `RequesterKind::IFetch`, in the cycle it is
fetched. Fetch ignores the answer and keeps its own timing
(`fetch_interval`). The booking marks the cycle as core traffic, so the host
and the DMA (AXI slave) only get the cycles fetch leaves free. An LSU access to
the ITCM never waits for fetch, as in the RTL. Fetch outside the ITCM (over
AXI) is not modeled.

### Memory → frontend (boot)

`ReefSim::bindTree_()` connects them once, before the run. It is a call, not
a port:

1. `Memory::setBootImage()` gets the ELF's loaded segments
   (`FuncSim::getLoadSegments()`) and counts the bytes that fall in the ITCM.
2. The memory module's host (`memory/Host.h`) turns them into boot cycles:
   `host_setup_cycles` + (bytes + 4) x `host_cycles_per_byte`, or 0 with
   `host_boot` off.
3. `Fetch::setStartCycle()` holds the first fetch until then.

The memory module owns the boot timing; the frontend only waits for it.

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
- **Matrix:** see docs/matrix.md, "Open cases". The main ones: Zvt uops do
  not compete with vector uops for the RVV dispatch they share on CoralNPU;
  no dedicated memory port; the core's `vtdiscard` interlock is not modeled.
  The Coral engine is correlated against the VME RTL (docs/matrix.md,
  "Correlating it").
- **Memory:** synchronous booking instead of request/response; one port per
  TCM, no banks; AXI read and write channels not separated; no arbitration
  between requesters other than at the TCMs (`memory/Tcm.h`); the DMA (`memory/Dma.h`) is
  the only `AxiSlave` requester, and nothing drives `Debug`; the AXI latency and
  outstanding limit are assumptions.
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

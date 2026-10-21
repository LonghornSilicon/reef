# Frontend v0 spec (CoralNPU M3 baseline)

Status: **draft v0**, for review with the leads. Owner: Frontend.

This describes the frontend of the CoralNPU M3 scalar core as a set of black
boxes: what each one does, every signal that crosses its boundary, and the
hardware parameters we are confident in. It is written from the RTL, not from
the M3 docs, which `deviations.md` shows are often stale.

**Source of truth.** CoralNPU at tag `M3-2026-04-27`
(`~/coralnpu-m3`). Paths below are relative to `hdl/chisel/src/coralnpu/`
unless they start with another directory. Every claim has a `file:line`.

**Confidence labels.**

- **code**: read directly from the RTL.
- **derived**: follows from the code, but no one has measured it yet.
- **open**: not known; needs a measurement or a question to the RTL owners.

## 1. Which frontend M3 actually uses

The default `Parameters` describe a different machine from the one M3 ships.
The product core is built with these overrides (`BUILD:572-578`, used by
`rvv_core_mini_axi` and `rvv_core_mini_verification_axi`, `BUILD:600-624`):

| Flag | Default (`Parameters.scala`) | M3 product | Effect |
| --- | --- | --- | --- |
| `enableFetchL0` | `true` (`:104`) | **`False`** | Uses `UncachedFetch`, not the L0-cached `Fetch` (`scalar/SCore.scala:57`) |
| `fetchDataBits` | `256` (`:109`) | **`128`** | 4 instruction slots per fetch, not 8 |
| `lsuDataBits` | `256` | `128` | (backend) |
| `enableRvv`, `enableFloat` | `false` | **`True`** | RVV and F decode/dispatch paths exist |

So everything below is `scalar/UncachedFetch.scala` with
`fetchInstrSlots = 128 / 32 = 4` (`Parameters.scala:110-115`).
`scalar/Fetch.scala` (L0 cache, return-address prediction through the link
port) is **not** in M3's product core.

## 2. Block diagram

```
                 boot PC (CSR 0x4)      debug_pc       iflush (fence.i, from LSU)
                        |                  |                    |
                        v                  v                    v
  BRU taken x4 ---> +-----------------------------------------------+
  (redirect)        | FetchControl: PC, redirect/flush, fetch gate, |---> fetch fault --> FaultManager
                    |   fault latch, Predecode + static predictor   |
                    +-----------------------------------------------+
                          | fetchAddr (Decoupled)    ^ fetchData (Valid, 4 insts)
                          v                          |
                    +-------------+   IBus (128 b)   +-------------------------+    +--------------------+
                    |   Fetcher   |----------------->| core IBus mux           |--->| ITCM arbiter / AXI |
                    +-------------+<-----------------| (LSU ITCM loads win)    |    | (see FE7)          |
                                                     +-------------------------+    +--------------------+
                                                              ^ LSU loads from ITCM addresses (.rodata)
                          | bufferRequest (4 x FetchInstruction, nValid)
                          v
                    +-----------------------------+
                    | InstructionBuffer (8 entry) |  nSpace --> FetchControl
                    +-----------------------------+
                          | 4 lanes, Decoupled FetchInstruction {addr, inst, brchFwd}
                          v
                    +-----------------------------------------------+
  scoreboards, ---->| DispatchV2: 4 x DecodeInstruction, hazards,   |--> ALU x4, BRU x4, MLU, DVU,
  queue capacities, |   interlocks, in-order try-dispatch, routing  |    LSU x4, CSR, Float, RVV x4,
  ROB space/empty   +-----------------------------------------------+    regfile read/mark, ROB, faults
```

Whether **dispatch** belongs to the frontend is an open ownership question:
reef-perf puts `dispatch` in the `core` module
(`reef-perf/docs/interfaces.md`). This spec covers it because the frontend
task list includes "instruction dispatch".

## 3. Black boxes

### FE1. FetchControl: PC generation and fetch gating

`scalar/UncachedFetch.scala:73-206`

**Function.**

- Holds the fetch PC. On leaving reset it loads the boot PC from CSR input 0,
  word-aligned (`:176-178`). The boot PC comes from `CoreAxiCSR` register
  `0x4` (`CoreAxiCSR.scala:103,153`). (code)
- Next-PC priority: reset > `iflush` > branch redirect > predecoded next PC
  (`:177-185`). (code)
- Starts a new fetch only when none of these hold (`:194-198`): still in reset;
  **response data is valid this cycle** (so one request in flight, and a new
  request at most every other cycle); a redirect or flush is happening this
  cycle; the instruction buffer would not have room for a full fetch after
  this cycle's push (`nSpace < nValid + 4`, `:192`); or a fetch fault is
  latched. (code)
- After a redirect or flush, discards responses until a new fetch can start
  (`:157-159,171,202`). (code)
- Latches an instruction-fetch fault and stops fetching until a branch
  redirect clears it (`:161-166`). (code)

**Interface.**

| Signal | Dir | Width | Meaning | Source |
| --- | --- | --- | --- | --- |
| `csr.value(0)` | in | 32 | Boot PC (only read when leaving reset) | `:178` |
| `iflush` | in | Valid(32) | Flush to `pcNext` (from `fence.i`) | `:77,179` |
| `branch` | in | Valid(32) | Redirect target (OR of the 4 BRUs, first valid wins) | `:78,210-214` |
| `fetchData` | in | Valid(FetchResponse) | Response from the Fetcher | `:79` |
| `fetchAddr` | out | Decoupled(32) | Address of the next fetch | `:82,204-205` |
| `bufferRequest` | out | 4 x FetchInstruction + `nValid` | Predecoded instructions to the buffer | `:83,155,173` |
| `bufferSpaces` | in | 4 | Free entries in the instruction buffer | `:84` |
| `fetchFault` | out | Valid(32) | Faulting fetch address | `:75,165` |
| `linkPort` | in | Valid(32) | **Unused** in `UncachedFetch` (only the L0 `Fetch` uses it, `scalar/Fetch.scala:324-330`) | |

### FE2. Predecode and static branch predictor

`scalar/UncachedFetch.scala:87-151` (inside FetchControl, combinational on
the fetch response)

**Function.**

- Aligns the 16-byte response to the fetch PC: slots before the PC's position
  in the block are dropped (`startElem`, `:107-120`). A fetch at a PC that is
  not 16-byte aligned delivers fewer than 4 instructions. (code)
- Predicts each slot (`PredictJump`, `:87-100`) (code):
  - `jal`: predicted **taken**.
  - Conditional branch with a **negative** offset (bit 31 set): predicted
    **taken** (`bxx`, `:91-92`; funct3 `01x` is excluded).
  - Conditional branch with a positive offset: not predicted (falls through).
  - `jalr` (including `ret`): **never** predicted. No return-address stack.
- The first predicted-taken slot ends the group; later slots are dropped
  (`:121-131`). The next fetch PC is that slot's target, else the next 16-byte
  block (`:133-134`). (code)
- Marks predicted-taken slots with `brchFwd = 1`, which travels with the
  instruction to the BRU (`:143`). (code)

**Interface.** Input: `FetchResponse {addr 32, inst 4 x 32, fault}`. Output:
`PredecodeOutput {insts 4 x FetchInstruction, count 4 b, nextPc 32}`
(`:21-31`).

### FE3. Fetcher: instruction-bus transaction

`scalar/UncachedFetch.scala:44-71`

**Function.** Drives the IBus with the 16-byte-aligned address (`:56`), passes
`ready` back, and registers the address and fault so the response arrives the
cycle after the request is accepted (`:60-66`). Splits `rdata` into 4 x 32-bit
slots (`:67-70`). (code)

**IBus interface** (`Interfaces.scala:51-60`):

| Signal | Dir | Width | Meaning |
| --- | --- | --- | --- |
| `valid` | out | 1 | Request; `addr` must stay stable until `ready` |
| `ready` | in | 1 | Accept |
| `addr` | out | 32 | 16-byte aligned |
| `rdata` | in | 128 | Read data, the cycle after acceptance |
| `fault` | in | Valid(FaultInfo {write, addr 32, epc 32}) | Bus fault |

**Memory behind it:** see FE7. In short, fetch shares the core's instruction
bus with LSU loads from the ITCM, which win (`scalar/SCore.scala:484-497`);
the bus then goes either to the ITCM, where the core has top priority, or to
AXI through `IBus2Axi` (`CoreAxi.scala:162-184`). (code)

### FE4. Instruction buffer

`common/InstructionBuffer.scala:36-75`, instantiated at
`scalar/UncachedFetch.scala:234-240`

**Function.** In-order circular buffer of `FetchInstruction`s.

- Capacity: **8** (`window = fetchInstrSlots * 2`, `UncachedFetch.scala:234`). (code)
- Up to 4 in per cycle (`DecoupledVectorIO`, `nValid` / `nReady`), up to 4
  out per cycle as 4 `Decoupled` lanes; lanes must fire in order
  (`:63-71`). (code)
- An instruction becomes visible at the output **one full cycle after it is
  enqueued** (`:34`). (code)
- Flushed by `iflush`, a BRU redirect, or `debug_pc` (`UncachedFetch.scala:239`). (code)

**Interface out of the fetch unit** (`Interfaces.scala:62-84`):
`inst.lanes`: 4 x Decoupled(`FetchInstruction {addr 32, inst 32, brchFwd 1}`)
to dispatch; `pc` (address at the head of the buffer, `:242-244`) to CSR
input 12 (`scalar/SCore.scala:193`); `fault` to the FaultManager.

### FE5. Decoders (4 lanes)

`scalar/Decode.scala:29-214` (bundle and helpers), `DecodeInstruction`
object at `:826`; one decoder per lane at `:303-306`.

**Function.** Combinational decode of each lane's instruction into
`DecodedInstruction`: one-hot op flags for RV32I, M, Zbb, system ops,
`fence.i`/`flushat`/`flushall`, plus an optional RVV compressed instruction
and an optional `FloatInstruction` (`:29-129`); immediates (`imm12`, `imm20`,
`immjal`, `immbr`, `immcsr`, `immst`); register-use predicates
(`readsRs1/2`, `rs1Set`, `rs2Set`, float/RVV f-register reads and writes,
`:200-213`). (code)

Instruction classes that matter for dispatch:

| Predicate | Includes | Source |
| --- | --- | --- |
| `isJump` | `jal`, `jalr`, `ebreak`, `ecall`, `mpause`, `mret` | `:173-176` |
| `isFency` | `fence.i`, `ebreak`, `wfi`, `mpause`, `flushat`, `flushall` | `:163` |
| `forceSlot0Only` | `isFency`, CSR ops, every F instruction, RVV ops that read or write an f-register | `:167-169` |
| `isLsu` | scalar, FP and RVV loads/stores, `flushat`, `flushall` | `:153-160` |

### FE6. Dispatch control (DispatchV2)

`scalar/Decode.scala:301-824`

**Function.** Each cycle, decides which of the 4 lane instructions dispatch,
**in order**: a lane dispatches only if every earlier lane dispatched this
cycle (`lastReady` chain, `:527-530,726-735`). (code)

A lane can dispatch only if all of these hold (`canDispatch`, `:499-522`):

| Rule | Effect | Source |
| --- | --- | --- |
| Not halted, not interlocked | `interlock` = BRU0 holding an `ebreak/ecall/mpause/mret`/fault, or an LSU flush | `:500-501`, `scalar/SCore.scala:137`, `Bru.scala:228-231` |
| No earlier jump or fence in the group | A jump/fence ends the group | `:313-318,503` |
| No earlier **conditional branch** in the group | At most one conditional branch per cycle; nothing after it | `:324-327,508` (D2) |
| No scalar RAW / WAW | Against the forwarded scoreboard (`comb`) and earlier lanes | `:331-363` |
| Address operands use the **registered** scoreboard | Load/store base, `jalr` source and store data cannot use a value written the cycle before | `:347-351` (D4) |
| No float RAW / WAW | | `:367-388` |
| Fence waits for memory idle | | `:396` |
| Slot-0-only ops dispatch alone from lane 0 | | `:400-406` (D3) |
| RVV config valid for vector loads/stores; `vstart == 0` where required | | `:411-444` |
| Queue space | LSU queue (`lsuQueueCapacity`, 3 b), RVV queue (`rvvQueueCapacity`, 4 b), counted across lanes | `:448-463` (D5) |
| `undef` only in lane 0 | | `:468-471` |
| Retirement buffer has a free entry for this lane, no trap pending | | `:517-518` |
| **CSR ops wait for an empty retirement buffer** | `csrr mcycle` drains the pipeline | `:519` (M2) |
| Single-step, `mpause`: lane 0 and core idle | | `:478-494` |

**Routing** (one command per lane per unit, `:533-731`):

| Unit | Port | Payload | Lanes | Source |
| --- | --- | --- | --- | --- |
| ALU | `alu` x4, Valid | `AluCmd {addr 5, op}` | 4 | `:533-570`, `Alu.scala` |
| BRU | `bru` x4, Valid | `BruCmd {fwd, op, pc 32, target 32, link 5, inst 32}` | 4 | `:572-607`, `Bru.scala:44-51` |
| MLU | `mlu` x4, Decoupled | `MluCmd {addr 5, op}` | 4 ports into **one** MLU (arbiter) | `:610-620`, `Mlu.scala:68` |
| DVU | `dvu` x4, Decoupled | `DvuCmd {addr 5, op}` | **lane 0 only** (`ready` is 0 on lanes 1-3) | `:622-632`, `scalar/SCore.scala:262-264` |
| LSU | `lsu` x4, Decoupled | `LsuCmd {store, addr 5, op, pc 32, elemWidth 3, nfields 3, umop 5}` | 4 | `:634-676`, `Lsu.scala` |
| CSR | `csr`, Valid | `CsrCmd {addr 5, index 12, rs1 5, op}` | lane 0 | `:678-701` |
| Float | `float`, Decoupled | `FloatInstruction` (opcode, rs1-3, rd, rm, inst, pc, flags) | lane 0 | `:710-715` |
| RVV | `rvv` x4, Decoupled | `RvvCompressedInstruction {pc 32, opcode, bits 25}` | 4 | `:703-708`, `rvv/RvvDecode.scala` |

Also: register-file read addresses and immediate overrides per lane
(`rs1Read/rs2Read/rs1Set/rs2Set`, `:753-770`), scoreboard marks
(`rdMark` x4, `rdMark_flt`, `rvvRdMark` x4, `:772-808`), LSU base-address
bus read with a zero-offset bypass (`:810-822`), and per-lane `branch` /
`jump` flags to the retirement buffer (`:317,325`).

**Faults raised at dispatch:** misaligned `jal` / branch / `jalr` targets
(`:598-605`), invalid CSR address (`:698`), `undef` (`:470-471`), RVV
`vstart != 0` (`:738-751`). Sent to the FaultManager with each lane's PC and
instruction bits (`scalar/SCore.scala:140-153`).

**Inputs from the rest of the core** (`Decode.scala:216-298`): `halted`,
`interlock`, `lsuActive`, `scoreboard.regd/comb` (32 b each), `fscoreboard`
(32 b), `branchTaken`, `jalrTarget` x4, `lsuQueueCapacity` (3 b),
`rvvState` (config valid, `vstart`), `rvvIdle`, `rvvQueueCapacity` (4 b),
`csrFrm` (3 b), `retirement_buffer_nSpace` (5 b), `_empty`,
`_trap_pending`, `single_step`, `debug_mode`.

### FE7. Instruction memory boundary (ITCM)

The ITCM is **not** part of the frontend: in the RTL it is built in the SoC
wrapper `CoreAxi.scala`, outside `SCore`, and in reef-perf it belongs to the
memory module (`mem.tcm`). This section describes the boundary the frontend
fetches across.

**Structure** (code):

| Property | Value | Source |
| --- | --- | --- |
| Address, size | `0x0`, 8 KB (highmem build: 1 MB) | `Parameters.scala:43,50`, `BUILD:608` |
| Organisation | 512 rows x 128 bits, one SRAM macro (`Sram_Nx128`), byte-masked writes | `CoreAxi.scala:146-150`, `TCM.scala:21-43` |
| Ports | One SRAM port, shared through an arbiter | `CoreAxi.scala:152-160` |
| Read latency | 1 cycle: data the cycle after the request | `CoreAxi.scala:180-183` |
| Faults | The ITCM path never faults; only the AXI path does | `CoreAxi.scala:185` |

**Arbitration happens at two levels** (code):

1. **Inside the core** (`scalar/SCore.scala:484-497`): fetch and the LSU share
   the core's single instruction bus. When the LSU has an ITCM load, it takes
   the bus and fetch sees `ready = 0`:

   ```scala
   io.ibus.valid := Mux(lsu.io.ibus.valid, lsu.io.ibus.valid, fetch.io.ibus.valid)
   fetch.io.ibus.ready := Mux(lsu.io.ibus.valid, false.B, io.ibus.ready)
   ```

   Read data is broadcast to both (`:496-497`).
2. **At the ITCM** (`CoreAxi.scala:141-143,159-172,213-223`,
   `Fabric.scala:24-56`): a `FabricArbiter` with three sources and **fixed
   priority, lowest index wins**; lower-priority sources get `fabricBusy`
   back-pressure:

   | Source | Requester | Priority |
   | --- | --- | --- |
   | 0 | Core instruction bus (fetch, or an LSU ITCM load) | highest |
   | 1 | AXI slave through `FabricMux`: external masters, e.g. a host loading the program, or a DMA | middle |
   | 2 | Debug module (`dm.io.itcm`) | lowest |

   The core always gets `ready = 1` for ITCM addresses (`CoreAxi.scala:184`).
   Addresses outside the ITCM go to `IBus2Axi`, which shares the AXI master's
   read-address channel with the LSU's external loads (`CoreAxi.scala:246-250`).

So **external masters and the debugger never stall fetch; LSU loads from the
ITCM do.** The DTCM is a separate SRAM with its own three-source arbiter of
the same shape (`CoreAxi.scala:187-223`), so data accesses to the DTCM do not
touch the ITCM.

**Who reads the ITCM besides fetch.** CoralNPU's standard linker template puts
`.rodata` (constant tables, string literals, `switch` jump tables, `const`
globals, `.data.rel.ro`) in the ITCM, which it marks `rx`
(`toolchain/coralnpu_tcm.ld.tpl:5-8,48-56`). Loads of those constants go
through the LSU's instruction-bus port (`scalar/Lsu.scala:913-928`) and stall
fetch while they run. (code) The reef-perf microbenchmarks keep all data in the
DTCM (`microbenchmarks/link.ld`), so the B01 measurements never see this.

**Writes.** A core store to an ITCM address raises a store fault
(`ibusFault`, `write = 1`, the store's PC as `epc`, `scalar/Lsu.scala:969-983`),
reported as a memory fault (`scalar/SCore.scala:154`). Only an external master
(source 1) or the debug module (source 2) can write the ITCM. (code)

**`fence.i` and self-modifying code.** `fence.i` decodes to an LSU `FENCEI`
op (`scalar/Decode.scala:646`); like every fence it waits for the LSU to be
idle (`:396`), then the LSU's flush tells fetch to empty the instruction
buffer and restart at `pcNext` (`scalar/SCore.scala:108-111`,
`UncachedFetch.scala:179,239`). M3 has no instruction cache
(`CoreAxi.scala:120-122`). (code) Therefore (derived):

- Code in the ITCM cannot modify itself (the core cannot store there).
- Code in external memory can: store the new code, `fence.i`, then jump.
  Without `fence.i`, up to 8 already-buffered instructions and one fetch in
  flight may still be stale.
- An external master changing the ITCM under a running core is not tracked;
  software must `fence.i` (or redirect) to drop fetched instructions.

## 4. Redirect protocol (frontend and BRU together)

| Case | Fetch prediction | BRU action | Source |
| --- | --- | --- | --- |
| `jal` | taken (`brchFwd = 1`) | no redirect | `Bru.scala:202-204` |
| Backward branch, taken | taken | no redirect | `:205-210` |
| Backward branch, not taken | taken | **redirect to `pc + 4`** | `:142,205-210` |
| Forward branch, not taken | not taken | no redirect | |
| Forward branch, taken | not taken | **redirect to target** | |
| `jalr` / `ret` | never predicted | **always redirect** to `rs1 + imm & ~1` | `:143,204` |
| `ecall` / `mret` / `wfi` / interrupt / fault (lane 0 BRU) | n/a | redirect to `mtvec` / `mepc` / `pc + 4` | `:128-140,175-188,221` |

The BRU registers its command, so the redirect (`taken.valid`) appears the
cycle after dispatch (`Bru.scala:114-147,202`). Fetch then blocks a new fetch
that cycle, loads the new PC, fetches the next cycle, and the response is
visible in the buffer a cycle after it arrives. (derived) That suggests the
first correct-path instruction dispatches about 3 cycles after the branch's
execute cycle; **the actual mispredict and `jalr` penalties are open (U1)**.

A correctly predicted taken branch causes no redirect, but it ends its fetch
group and the target fetch can start mid-block, so it costs fetch bandwidth.
(derived)

## 5. Hardware parameters we are confident in

| Parameter | Value | Source |
| --- | --- | --- |
| Instruction / PC width | 32 / 32 | `Parameters.scala:71-72` |
| Dispatch lanes | 4 | `Parameters.scala:73` |
| Fetch width | 128 bits = 4 instructions, 16-byte aligned | `BUILD:573`, `UncachedFetch.scala:56` |
| Fetches in flight | 1; a new request at most every other cycle | `UncachedFetch.scala:195` (D7) |
| Sustained fetch rate | at most 4 instructions per 2 cycles (about 2 IPC) | derived from the line above; RTL `alu_indep` measures 1.5 IPC with its loop branch |
| Instruction buffer | 8 entries, 4 in / 4 out per cycle, 1-cycle enqueue-to-visible | `UncachedFetch.scala:234`, `InstructionBuffer.scala:34` |
| Fetch start condition | at least `nValid + 4` free buffer entries | `UncachedFetch.scala:192` |
| Branch prediction | static: `jal` and backward conditional taken; `jalr` never | `UncachedFetch.scala:87-100` |
| Return-address prediction | none | `UncachedFetch.scala` (`linkPort` unused) |
| Conditional branches per dispatch cycle | at most 1, and it ends the group | `Decode.scala:324-327` |
| Slot-0-only ops | CSR, F, fences, `mpause`, `flushat/all`, f-register RVV ops | `Decode.scala:167-169` |
| Divider dispatch | lane 0 only | `scalar/SCore.scala:262-264` |
| Retirement buffer | 8 entries; CSRs wait for it to be empty | `Parameters.scala:95`, `Decode.scala:519` |
| Boot PC | `CoreAxiCSR` register `0x4`, word-aligned | `CoreAxiCSR.scala:103,153`, `UncachedFetch.scala:178` |
| ITCM | `0x0`, 8 KB, 512 x 128-bit single-port SRAM, 1-cycle read | `Parameters.scala:43`, `CoreAxi.scala:146-183`, `TCM.scala:21-43` |
| ITCM arbitration | core instruction bus > AXI slave > debug module (fixed priority) | `CoreAxi.scala:141-143,213-223`, `Fabric.scala:24-56` |
| Core instruction bus | LSU loads from the ITCM take priority over fetch | `scalar/SCore.scala:484-497` |
| Core stores to the ITCM | fault | `scalar/Lsu.scala:969-983` |

## 6. Open questions

1. **Penalties (U1):** cycles lost per mispredict, per `jalr`, per correctly
   predicted taken branch, per `fence.i`. Measure with the tests below.
2. **ITCM contention:** answered in part by FE7. The ITCM arbiter never
   stalls fetch (the core has top priority), but LSU loads from the ITCM do.
   Still open: how many cycles each such load costs fetch.
3. **Fetch from AXI:** latency when code is outside the ITCM (`IBus2Axi`).
4. **Ownership:** is dispatch frontend or core? In reef-perf it is in the
   `core` module. The 8-entry instruction buffer is in the frontend
   (`top.frontend.params.ibuf_entries`), as in the RTL.
5. **Interfaces missing from reef-perf** (`interfaces.md`, "Known
   simplifications"): a redirect/flush signal from the core to the
   frontend, and instruction fetch through the memory module (`IFETCH`).

## 7. How reef-perf's frontend differs today

| Behavior | M3 RTL | reef-perf (`model/frontend/Fetch.cpp`) |
| --- | --- | --- |
| Fetch rate | one 4-instruction fetch every 2 cycles | `fetch_interval: 2` (B04) |
| Alignment | 16-byte blocks; partial block after a mid-block target | `aligned_blocks: true` (B06) |
| Prediction | static (above) | `predictor: static_btfn` (B05): predicted-taken jumps free, mispredicts 2 cycles (measured) |
| Redirect | from the BRU, one cycle after the branch dispatches | penalty counted from the branch's fetch cycle; no core-to-frontend signal |
| Instruction buffer | 8 entries in the fetch unit | `ibuf_entries: 8` in the frontend |

The tickets that close these gaps are B04 (fetch rate), B05 (prediction), B06
(alignment), and B07-B10 (dispatch rules).

## 8. Stress tests

Each test is a `.S` microbenchmark in `reef-perf/microbenchmarks/`, using
`common.inc`
(`START`, `ROI_BEGIN/END`, `END`), with at least 1000 loop iterations so the
loop dominates. Each runs on the M3 RTL (B01 flow) and the model, and goes
into the `correlation` report. The instruction counts must match on both sides.

The `csrr mcycle` markers drain the pipeline (M2), so tests measure
**steady-state** behavior per iteration, not single events.

### Fetch (FE1, FE3, FE4)

| Test | Stresses | Construction | Expected on M3 |
| --- | --- | --- | --- |
| `fetch_straight` | Peak fetch rate | Long unrolled block (e.g. 64) of independent ALU ops, one loop branch | About 2 IPC (fetch-bound), `ibuf_empty` stalls |
| `fetch_align_{0,1,2,3}` | Partial first block | Same tight loop, start padded by 0-3 `nop`s within a 16-byte block | Cycles per iteration rise with misalignment |
| `fetch_ibuf_fill` | Buffer back-pressure, `nValid + 4` rule | A `div` chain followed by many independent ALU ops | Fetch stops while the buffer is near full; then a 4-wide dispatch burst |
| `fetch_fencei` | `iflush` path (FE7) | `fence.i` inside a loop | Waits for the LSU to be idle, then empties the buffer and refetches; the cost per iteration is open |
| `fetch_rodata_conflict` | LSU loads from the ITCM stall fetch (FE7) | Loop loading from a constant table placed in the ITCM, as CoralNPU's standard linker script does for `.rodata` | Fetch loses cycles on each load; compare with the same loop reading from the DTCM |

### Branch prediction and redirect (FE2, section 4)

| Test | Stresses | Construction | Expected on M3 |
| --- | --- | --- | --- |
| `branch_tight` (exists) | Predicted-taken backward branch | 2-instruction loop | About 1 IPC (RTL: 10005 cycles / 10000 insts) |
| `branch_fwd_not_taken` | Correct not-taken | Forward branch that never fires, inside a loop | No redirect; near `fetch_straight` |
| `branch_fwd_taken` | Mispredict | Forward branch that always fires | One redirect per iteration: the mispredict penalty |
| `branch_bwd_not_taken` | Mispredict the other way | Backward branch that never fires | Redirect to `pc + 4` per iteration |
| `branch_alternate` | Static predictor limits | Forward branch taken every other iteration | Half the iterations mispredict |
| `jal_chain` | Predicted-taken unconditional | Chain of `jal`s to the next block | No redirects; only group truncation |
| `jalr_chain` | Never-predicted indirect | Chain of `jalr`s (and a `call`/`ret` pair) | A redirect per `jalr` |
| `branch_density_{1..4}` | One branch per group | A not-taken branch every k instructions | Dispatch drops as branches end groups |
| `branch_two_per_block` | Predecode truncation | Two backward branches in the same 16-byte block | The first predicted-taken ends the group |

### Decode and dispatch (FE5, FE6)

| Test | Stresses | Construction | Expected on M3 |
| --- | --- | --- | --- |
| `branch_group` (exists) | Branch ends the group | `[alu, beq, alu, alu]` | About 2 instructions per dispatch cycle |
| `disp_csr_drain` | CSR waits for an empty ROB | `div` then `csrr` (not `mcycle`) in a loop | Each CSR waits for the divider |
| `disp_slot0` | Slot-0-only ops | Interleave `fadd.s` / `csrr` / `fence` with ALU ops | Each special op dispatches alone (`fp_mix` exists for FP) |
| `addr_forward` (exists) | `regd` for addresses | `addi` producing a load base | One extra cycle per pair |
| `disp_store_data_regd` | `regd` for store data | ALU result stored the next instruction | Extra cycle per pair |
| `disp_jalr_regd` | `regd` for `jalr` source | Compute target, then `jalr` | Extra cycle |
| `disp_waw` | WAW stalls | Independent ops writing the same `rd` | Serialised by WAW |
| `disp_lsu_queue` | LSU queue capacity | Bursts of more than 4 independent loads | Stalls on queue space |
| `disp_rvv_queue` | RVV queue capacity | Bursts of independent vector ALU ops | Stalls on queue space |
| `disp_rob_full` | Retirement buffer space | One `div` then more than 8 independent ops | Dispatch stops at 8 in flight |
| `disp_div_lane0` | DVU lane-0 rule | `div` after 1-3 ALU ops in a group | The `div` waits to be in lane 0 |
| `disp_mlu_single` | One multiplier | 4 independent `mul`s per group | At most one `mul` accepted per cycle (open) |
| `disp_vset_config` | RVV config interlock | `vsetvli` then a vector load in the same group | The load cannot dispatch in the `vsetvli`'s group; how long it waits is open |

Run order: start with the mispredict and alignment tests, because they
calibrate the parameters of B05 and B06 and answer open question 1.

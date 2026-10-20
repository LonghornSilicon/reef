# RvvCoreMiniAxi M3 scalar core: claim checklist

Source checked: `google-coral/coralnpu` at tag `M3-2026-04-27` (commit `72d700e`).
Paths are relative to `hdl/chisel/src/`. `D` = the cycle an instruction is dispatched (leaves the instruction buffer).

Status key:
- **SRC ✓**: the source matches Orhan's claim. It still needs a waveform check.
- **SRC ✗**: the source disagrees with the claim, or the claim is incomplete. Fix it.
- **NEW**: missing from Orhan's drafts. Add it.
- **WAVE** / **BENCH**: fill these in with what you see in sim.
- **ALEX ✓**: Alex checked it personally, using the review packet (`deliverable/review_packet.md`). **Only Alex writes this mark.** The agent never sets it.

Verification levels (every row has exactly one): **human** = Alex signed off · **sim+wave** = cycle counts plus a scripted waveform check · **sim** = cycle counts only · **src** = source reading only.

## Status at a glance (2026-10-19)

| Level | Rows |
|---|---|
| **human** (15) | 0.3, 1.5, 1.7, 3.4, 3.6, 3.10, 3.14, 3.19; §4 BRU mispredict, BRU correct, DVU, Load, Store; 5.1, 5.2 |
| **sim+wave** | 1.9 (via 3.20), 2.2, 2.3, 3.1 (indirect), 3.3, 3.5, 3.7, 3.11, 3.16, 3.17, 3.18, 3.20; §4 ALU, CSR, MLU, FPU (sim value, see D4; R5 agreed for sim only), FPU div/sqrt, FP→int, fmv, flw/fsw, RVV handoff; 5.3 |
| **sim** | 0.2, 0.4, 1.8, 1.10, 3.2, 3.8, 3.9, 3.12, 3.15; §4 LSU opQueue, line-crossing |
| **src** only | 0.1, 1.1, 1.2, 1.3, 1.4, 1.6, 2.1, 3.13; §4 BRU link write; 5.4 |

**Alex's sign-off: all nine agreed (2026-10-18/19)** (`deliverable/review_packet.md` has Alex's notes per item; the rows moved to **human**):

| Review item | Checklist rows | Alex |
|---|---|---|
| R1 LSU 1 load / 4 cycles, 1 store / 2 | 3.19, §4 Load, Store | [x] 2026-10-18 |
| R2 mispredict target at D+4 | §4 BRU mispredict | [x] 2026-10-18 |
| R3 BTFN prediction | 1.7, §4 BRU correct | [x] 2026-10-18 |
| R4 conditional branch ends the group | 3.4 | [x] 2026-10-18 |
| R5 FP/CSR dispatch alone in lane 0 | 3.6, §4 FPU | [x] 2026-10-19 (FPU timing: sim behaviour; hardware numbers per D4) |
| R6 DIV latency vs dividend | §4 DVU | [x] 2026-10-19 |
| R7 ROB-full stall | 3.10, 0.3, 5.1 | [x] 2026-10-19 |
| R8 retire timing | 5.2 | [x] 2026-10-19 |
| R9 random check: same-cycle forwarding | 3.14 | [x] 2026-10-19 |

**D4 (decided 2026-10-19):** the diagrams use the hardware FPU numbers, D+4 and 1 per 5 (the DC and Yosys-slang netlists have the 3 FMA stages); the Verilator sim shows D+1, 1 per 2. See `notes/discrepancies.md`.

Work top to bottom. It follows an instruction's path through the core.

---

## 0. Build facts (check once)

| # | Claim | Source | Status | WAVE/BENCH |
|---|---|---|---|---|
| 0.1 | Gen flags: `enableFetchL0=False` (uses `UncachedFetch`), 128-bit fetch and LSU, RVV and float on, 8 KB ITCM, 32 KB DTCM | `coralnpu/BUILD` ~L570-608, `scalar/SCore.scala:57` | SRC ✓ | **src**. Consistent with sim (16-byte fetch lines in 1.5, float and RVV ops execute in B5/TRACE); TCM sizes not checked. |
| 0.2 | Toolchain `-march=rv32imf_zve32f_zicsr_zifencei_zbb` | `toolchain/cc_toolchain_config.bzl:163` | toolchain only | Read `misa` in sim to confirm the hardware <br>**sim** (2026-10-17): `misa` = 0x40201120 -> RV32, extensions F I M V. Zbb is not a misa bit, but all 18 Zbb ops execute at 1 cycle/inst (b9_alu_zbb). |
| 0.3 | Retirement buffer is the **mini** variant: 8 entries, no data payload (because `enableVerification=false`) | `coralnpu/Parameters.scala:76,83,95`, `SCore.scala:64` | NEW | **human** (ALEX ✓ R7, 2026-10-19; was sim+wave): 8 entries. b7_rob.vcd ~1697-1732 ns: `retirement_buffer.io_nSpace` = 0 after div + 7 addi, dispatch held until the div writes back, then 8 retire in one cycle (`io_nRetired` = 8). |
| 0.4 | No `cycle` CSR (0xC00), so **`rdcycle` will fault**. Time with `csrr x, mcycle` (0xB00). `minstret` (0xB02) counts ROB retirements. | `scalar/Csr.scala` CsrAddress enum, L289-290, L505-519 | NEW, affects the harness | **sim**: `csrr mcycle` works in every bench; no harness used rdcycle. |

## 1. Fetch (F1 / F2)

| # | Orhan's claim | Source | Status | WAVE/BENCH |
|---|---|---|---|---|
| 1.1 | `FetchControl.pc` is a register, read in F1 | `scalar/UncachedFetch.scala:176-187` | SRC ✓ | **src** |
| 1.2 | PC is aligned to a 16-byte line | `UncachedFetch.scala:56` (done combinationally in `Fetcher`, not as a separate step) | SRC ✓ | **src** (consistent with `io_ibus_addr` stepping 0x10 in 1.5) |
| 1.3 | Registers at the end of F1: SRAM read address, `Fetcher.ibusCmd`, `SRAM.readIssued`, `CoreAxi.inItcmReg` | `UncachedFetch.scala:61`, `coralnpu/SRAM.scala:44-48`, `coralnpu/CoreAxi.scala:182`. The address register is inside the SRAM macro (`Sram_Nx128`). | SRC ✓ | **src** |
| 1.4 | "ITCM is always ready" | `CoreAxi.scala:184` returns `ready = true` for ITCM addresses. The core is arbiter source 0, which has the highest priority (`Fabric.scala:34-42`). | SRC ✓ (but AXI-slave ITCM access starves while the core fetches) | **src** |
| 1.5 | **"No new request this cycle" in F2** | `UncachedFetch.scala:195`: `blockNewFetch` includes `io.fetchData.valid` | SRC ✓, **big consequence**: at most one 16-byte line every 2 cycles, so the front end peaks at 2 IPC | **human** (Alex read the waveform, 2026-10-13). **BENCH ✓ (sim)**: predicted IPC ≈ 2.0, measured marginal IPC = 2.000 (slope 0.500 cyc/inst over N=64→256); N=16/64/256 body = 7/31/127 cyc (IPC 2.29/2.07/2.02), 3 identical reps. See `results/fetch_ipc.md`. **WAVE ✓ (read by Alex, 2026-10-13)**: `io_ibus_valid` period = 2 cycles (2500 ps) in steady state; `io_ibus_valid == !blockNewFetch` on all 854 sampled edges (scripted check); `ibusCmd_valid` rises 1 cycle after each request and is the cycle `blockNewFetch` is high due to line 195; `io_ibus_addr` advances 0x10 per request (reads 0 on idle cycles, `ForceZero`, L199); all four lanes fire together every 2nd cycle. Pattern ends at the line holding `csrr a1,mcycle` (0x320, ~465 ns). Signals: `RvvCoreMiniAxi.core.score.fetch.{fetcher,ctrl,instructionBuffer}.*`, `...score.fetch.io_inst_lanes_*`, `...score.dispatch.io_retirement_buffer_empty`. |
| 1.6 | Predecode shifts by PC offset and writes ≤4 instructions to the IBuf | `UncachedFetch.scala:102-151` | SRC ✓ | **src** |
| 1.7 | "predict taken" | `UncachedFetch.scala:87-100` | **SRC ✗, incomplete**: static backward-taken/forward-not-taken (BTFN). `jal` is always predicted taken. Conditional branches are predicted taken only if the offset is negative (`inst(31)`). **`jalr` is never predicted.** | BENCH: see §6 <br>**human** (ALEX ✓ R3, 2026-10-18; was sim+wave) (B1, TRACE): backward-taken loop 2.0 cycles/iter (no penalty); forward taken 4.0 cycles/unit and backward not-taken 4.0 (mispredict); TRACE: `bru_N.io_taken_valid` + IBuf flush at D+1, target D+4; jalr (even with rs1 ready long before) always redirects, D+4. Offset-0 self-branch counts as forward (D7). |
| 1.8 | A predicted jump truncates the rest of the line and next PC = target | `UncachedFetch.scala:121-134` | NEW | BENCH: branch at line slot 0 vs slot 3 <br>**sim**: jal unit = 2.0 cycles (one fetch slot, no redirect; TRACE target at D+2). Truncation itself not isolated in the wave. |
| 1.9 | New fetch only if IBuf space ≥ (instructions being written + 4) | `UncachedFetch.scala:192` | NEW | **sim+wave** via 3.20: `blockNewFetch=1` with `nSpace=3` outside a data-return cycle (dep_chain_alu.vcd 742.5-753.75 ns). |
| 1.10 | Fetch starting mid-line returns fewer than 4 instructions (`validsIn`) | `UncachedFetch.scala:118-120` | NEW | BENCH: branch to target at offset +4/+8/+12 <br>**sim**: jal to next-line offset 0/+4/+8/+12 = 4.0 cycles per 2-line unit for all k: a partial line still costs a full 2-cycle fetch slot. |

## 2. Instruction buffer

| # | Claim | Source | Status | WAVE/BENCH |
|---|---|---|---|---|
| 2.1 | 8 × {pc, inst, brchFwd} + pointers, count | `UncachedFetch.scala:234-236` (window = 2×4), `common/InstructionBuffer.scala` | SRC ✓ | **src** |
| 2.2 | Written at the end of F2, readable the next cycle | `common/InstructionBuffer.scala:34-35` comment | SRC ✓ | WAVE <br>**sim+wave** (TRACE): F2 = F1+1, IBuf (first cycle visible to dispatch) = F1+2 for every class; in a free stream D = IBuf. |
| 2.3 | Flushed by a mispredict, `fence.i`, or debug | `UncachedFetch.scala:239` | SRC ✓ | **sim+wave** (TRACE): `instructionBuffer.io_flush` at D+1 for br_mp_fwd/br_mp_bwd/jalr and for fence.i. |

## 3. Decode / dispatch (D)

| # | Claim | Source | Status | WAVE/BENCH |
|---|---|---|---|---|
| 3.1 | Decode × 4 lanes, strictly in order: the first lane that can't go stops all later lanes | `scalar/Decode.scala:527-535,733-735` (`lastReady` chain) | SRC ✓ | **sim+wave** (indirect): later lanes never dispatch past a held lane, e.g. b8_x0.vcd cycle 1202 (lanes 0-2 go, lane 3 held) and the lane-by-lane pattern in 3.18. |
| 3.2 | Scoreboard RAW/WAW | `Decode.scala:330-363` | SRC ✓ | **sim**: RAW stalls measured in dep_chain (3.3, 3.14) and B2/B4 chains; WAW via 3.17. Hazard terms are not named signals. |
| 3.3 | **The load/store base register, store data, and `jalr` rs1 use the *registered* scoreboard (no bypass)** | `Decode.scala:341-351`, `scalar/Regfile.scala:192-196` (bus address comes from raw `rdata`) | NEW | **sim+wave**. **SIM ✓ (2026-10-13)**: producer→consumer dispatch distance is **+1 cycle for an ALU consumer, +2 for store data (rs2) and +2 for load base (rs1)**. Store data: dep_chain_sw.vcd 1122.5→1135 ns (d=2) and fetch_ipc waves (`sw s1` at +2 after `sub`, 467.5 ns empty cycle); load base: dep_chain_lw.vcd 742.5→757.5 ns (d=0, `lw` held one cycle, `ready=0000`). Cause (registered scoreboard) still SRC-inferred, hazard term not a named signal. **Not visible as throughput for lw/sw**: the LSU is the bottleneck (3.19). See `results/dep_chain.md`. |
| 3.4 | **A conditional branch ends the dispatch group.** Nothing in later lanes dispatches that cycle. (The comment says ALU ops could; the code blocks everything.) | `Decode.scala:324-327,508` | NEW | BENCH: `beq` (not taken) in lane 0 + 3 `addi` <br>**human** (ALEX ✓ R4, 2026-10-18; was sim+wave): 4 not-taken bne per line = 1.000 cycle/branch (0.5 if they packed); bne in lane 0 + 3 addi = 2.0, bne in lane 3 = 2.0 (B1). TRACE br_nt_fwd: fall-through (same line) dispatches D+1. |
| 3.5 | `jal`/`jalr`/`ecall`/`mret`/`ebreak`/`mpause` and fences end the group | `Decode.scala:173-176,313,503` | NEW | **sim+wave**: jal/jalr end the group (TRACE: target never in the same cycle); fence.i ends it (next inst D+4). **Correction (D6): plain `fence` is not fency, does not end the group**; the next x0-reading op still splits off (3.17b). |
| 3.6 | **Slot-0-only *and dispatched alone***: CSR, fences, **all float ops including `flw`/`fsw`**, RVV ops that read or write f-regs | `Decode.scala:167-169,400-406` | **SRC ✗**: the drafts say "lane 0 only"; it's actually lane 0, with nothing else dispatched that cycle | BENCH: `fadd` + 3 `addi` <br>**human** (ALEX ✓ R5, 2026-10-19; was sim+wave) (TRACE): CSR, FLW, FSW, FADD/FMUL/FMADD/FDIV/FSQRT, feq, fcvt, fmv.* all dispatch at D=3, one cycle after the lane-0 pre addi (D=2), and the following addi waits to D+1: lane 0 and alone. fence: lane 0 only. Throughput side: see 3.16 / D4. |
| 3.7 | DVU: lane 0 only, but later lanes can dispatch with it | `SCore.scala:256-264` | SRC ✓ | BENCH: `addi; div` (div waits a cycle to reach lane 0) <br>**sim+wave** (TRACE div_*): pre addi in lane 0 at D=2, div waits to D=3 (lane 0), the following addi dispatches in the same cycle as the div (later lanes allowed). |
| 3.8 | MLU: one per cycle (4→1 arbiter, lowest lane wins) | `scalar/Mlu.scala:67-76` | SRC ✓ | BENCH: 4 independent `mul` in a row <br>**sim**: 4 independent mul per line = 4.0 cycles/line (1 mul/cycle); 2 mul + 2 addi = 2.0 (B2). |
| 3.9 | LSU: up to (opQueue free space) memory ops per cycle | `Decode.scala:459-463`, `scalar/Lsu.scala:858-863` | NEW | **sim**: bursts of 1-8 independent lw add exactly 4 cycles each (B4 burst*); queue-full stall on the 5th not isolated in a wave. F1 (2026-10-19): in dep_chain_lw the 8-entry ROB fills before the queue (`io_queueCapacity` floors at 1/2/3/3 for d=0..3), so that bench never reaches capacity 0 either; see D3. |
| 3.10 | **ROB: lane i dispatches only if ROB free entries > i. Dispatch stops while a trap is pending.** | `Decode.scala:517-518` | NEW (the ROB is missing from the diagram) | BENCH: `div` + 8 independent `addi`; expect a stall once the ROB fills <br>**human** (ALEX ✓ R7, 2026-10-19; was sim+wave): B7 T(K)-T(0) = 0 for K<=7 addi behind a long div, +2 at K=8, +17 at K=40 (40 addi with no ROB limit would hide under the 34-cycle div). b7_rob.vcd: `io_nSpace`=0, lanes holding 0x680.. for 31 cycles until `dvu_rd`. |
| 3.11 | **CSR instructions wait until the ROB is empty** (and the FPU is idle) | `Decode.scala:519,693` | NEW, **this makes `csrr mcycle` a serializing fence**. Calibrate harness overhead. | **sim+wave**: csrr stream 2.0 cycles; `div; csrr` 36 vs div alone 35 (B6); TRACE csr D=4 = after the pre addi retired (ROB empty). With FPU: `fadd; csrr` 3.0 (FPU per D4). |
| 3.12 | `fence`/flush wait for the LSU to be idle | `Decode.scala:396` | NEW | **Corrected (D6), sim**: plain `fence` does NOT wait (`lw; fence` = 4.0 = `lw` alone). fence.i waits for LSU idle: `lw; fence.i` 9, `sw; fence.i` 7, fence.i alone 4 (B10). |
| 3.13 | "No flip-flops in D". The operand registers are `Regfile.readDataBits`/`readDataReady`, not in the units. | `Regfile.scala:117-186` | SRC ✓ (name the register correctly in the diagram) | **src** |
| 3.14 | Same-cycle forwarding: a write-back in cycle X is visible to an instruction dispatching in cycle X | `Regfile.scala:111-112` (`comb` scoreboard), `165-174` (`rwdata` bypass) | SRC ✓ | **human** (ALEX ✓ R9, 2026-10-19; was sim). **BENCH ✓ (sim)**: `addi a2,a2,1` x16 / x64 = **1.000 cycles/instruction** (predicted 1.0), identical in all three dep_chain ELFs. `results/dep_chain.md` |
| 3.15 | Regfile write ports: 0-3 = lane i (ALU / BRU link / CSR on 0 / RVV rd), **4 = arbiter {MLU > DVU > RVV async > FPU scalar-rd > debug}**, 5 = LSU | `SCore.scala:288-302,403-422` | SRC ✓ for ports; **NEW**: port 4 is shared and MLU has priority (`dvu.rd.ready := !mlu.rd.valid`, L259) | BENCH: div finishing while muls stream <br>**sim**: `div(N=2) + 6 mul` = 9 cycles/unit vs `div + 6 addi` = 5 (B3 dmul/dadd): the div write-back waits behind the mul stream on port 4 (MLU priority). Not isolated in a wave. |
| 3.16 | Float scoreboard has **no** same-cycle bypass, so consumers dispatch the cycle *after* write-back | `scalar/FRegfile.scala:36-42` | NEW | BENCH: dependent `fadd` chain <br>**sim+wave** (TRACE): fadd f-write D+1 (sim, D4) -> dependent FP op D+2; flw f-port 1 at D+4 -> consumer D+5: never same-cycle. Dependent fadd chain 2.0 cycles (sim, D4). With 3 pipe regs (F2 experiment build): write D+4, dependent FP op D+5, chain 5.0. |
| 3.17 | **rd=x0 instructions are not free to pack, in two ways.** (a) WAW: four `nop` (= `addi x0,x0,0`) in one group dispatch one per cycle because the same-group `rdScoreboard`/`scoreboardScan` does not special-case x0. (b) RAW: an rd=x0 instruction also blocks later same-group lanes that *read* x0 (`rs1=x0`, e.g. `li a0,1` = `addi a0,zero,1`), because `readScoreboardComb` includes x0 and the same-group scan has bit 0 set. So `nop; li a0,1` in one group should serialize. Across cycles x0 never blocks: `Regfile.scala:104-108` forces scoreboard bit 0 to 0, so this is a same-group effect only. | `Decode.scala:337-344` (`rdScoreboard`, `scoreboardScan`), `:353-357` (`readScoreboardComb`), `:359-363` (RAW/WAW), `Regfile.scala:104-112` | NEW. (a) and (b) **sim+wave** (B8, see next column). | BENCH: 4 `nop`; 4 `addi` distinct rd; `nop; li a0,1; li a1,1; li a2,1`; `li x0..` mix. **Don't use nop as filler in timing benches.** <br>**sim+wave** (B8, D5): (a) 4 nop = 4.0 cycles/line, 4 `addi x0,s1,1` = 4.0, 4 addi distinct rd = 2.0; (b) b8_x0.vcd cycle 1202: `li; nop; li` - the li after the nop is held out of the group; prefilled `nop; li x6` = +1 cycle, control `nop; addi rd,s1` = +0. |
| 3.18 | Observed in fetch_ipc waves: `csrr a1,mcycle` at 0x320 dispatched with `retirement_buffer_empty = 1` (no wait). The following `sub` (RAW on a1) and `sw` (RAW on s1) then dispatched one at a time. | `Decode.scala:359-361,519` | **SIM ✓** for lane-by-lane dispatch pattern at 465.0 / 466.25 / 467.5 / 468.75 ns; RAW as the cause is SRC-inferred | Refines 3.11: the ROB-empty wait only costs cycles if something long is still in flight. **See 3.3 for the `sw`**: its extra cycle at 467.5 ns is the registered-scoreboard delay for store data (+2 after the producer instead of +1), not a longer ALU latency |
| 3.19 | **LSU sustains at most one load per 4 cycles and one store per 2 cycles** (steady state, independent ops, distinct rd). This is slower than both the fetch bound and the dependency; it swallows the 3.3 +1. | `Lsu.scala:835` (4-entry `opQueue`), `:858-862` (req ready from queue space), `:1056,1071` (single slot, dequeue only when idle), `Decode.scala:459-463` (`lsuInterlock`) | **NEW, SRC ✗ vs the drafts if they assume 1 mem op/cycle.** Loads **SIM ✓** (lsu dispatch pulses every 4 cycles, `io_queueCapacity` refills 1 slot per 4 cycles, dep_chain_lw.vcd 903.75-918.75 ns); stores SIM ✓ (`io_dbus_valid & io_dbus_write` every 2nd cycle). **Derived from the slot FSM and verified in sim**: load = request, data back, writeback, 1 idle/dequeue cycle = 4; store = 1 bus cycle + 1 idle/dequeue = 2 (`Lsu.scala:391-395,931-934,993-995,1017,1056,1071`; dep_chain_lw.vcd 900-925 ns, dep_chain_sw.vcd 450-463 ns). | BENCH ✓ `dep_chain_lw`: 4.00 cycles/unit for all d, DEP and IND. `dep_chain_sw` IND d=0,1: 2.00. TODO: load-to-use latency; ITCM/ebus targets <br>**human** (ALEX ✓ R1, 2026-10-18; was sim+wave) |
| 3.20 | **Dispatch inefficiency feeds back into fetch bandwidth.** A group that leaves instructions in the buffer when the next line lands makes `insufficientBuffer` (`nSpace >= nValid + 4`) hold the next request one cycle; the line period goes 2 -> 2,3,2,3. So the front end is not a fixed 2 IPC once dispatch isn't draining 4 per fire. | `UncachedFetch.scala:192,194-198` | **NEW**, **SIM ✓**: dep_chain_alu.vcd 742.5-753.75 ns (`blockNewFetch=1`, `nSpace=3` outside a data-return cycle); dep_chain_sw.vcd 1122.5-1135 ns | BENCH ✓ ALU d=0 DEP 1.25 cycles/unit (source-only 1.0), SW d=2 DEP 2.5 (2.0). `notes/discrepancies.md` D1 |

## 4. Execute: per-unit timing (from Orhan's timing draft)

| Row | Orhan | Source says | Status | BENCH |
|---|---|---|---|---|
| ALU ×4 | D+1: op regs + `readDataBits` → port i | `scalar/Alu.scala:80-97`: `valid`/`addr`/`op` registered, result combinational, write in D+1 | SRC ✓ | latency 1, throughput 4/cycle (but fetch caps at 2) <br>**sim+wave** (TRACE): `alu_N.valid` D+1, write port = lane (port 1 for lane 1), retire D+1 (same cycle as WB). Chain 1.0 for all 21 RV32I + 18 Zbb ops (B9). |
| BRU mispredict | D+1 resolve, D+2 F1, D+3 F2, D+4 target dispatches, 3 cycles lost | `scalar/Bru.scala:114-147,202-212`; fetch blocked in the redirect cycle (`UncachedFetch.scala:196`) | SRC ✓ | BENCH: forward-taken, backward-not-taken, `jalr` <br>**human** (ALEX ✓ R2, 2026-10-18; was sim+wave): redirect (`io_taken_valid`) D+1, IBuf flush D+1, target dispatch D+4 (TRACE br_mp_fwd/br_mp_bwd/jalr/jalr_rdy); 4.0 cycles/unit (B1). |
| BRU correct | not drawn | Correctly predicted taken: no redirect (`taken = actual ≠ fwd`); cost is only the fetch-line truncation | NEW | BENCH: tight backward loop <br>**human** (ALEX ✓ R3, 2026-10-18; was sim+wave): no `io_taken_valid` for predicted-taken bwd and jal (TRACE); loop 2.0 cycles/iter, jal 2.0 (B1). |
| BRU link write | not drawn | `jal`/`jalr` rd written D+1 on lane port i (`Bru.scala:223-225`) | NEW | **src**: not traced (benches use rd=x0). |
| CSR | D+1 `Csr.req`, RMW → port 0 | `scalar/Csr.scala:241-242,628` | SRC ✓, but add 3.6 and 3.11 | **sim+wave**: `csr.req_valid` D+1, `io_rd_valid` D+1 -> port 0, retire D+1 (TRACE); stream 2.0 (B6). |
| MLU | D+1 `stage2In` multiply, D+2 `stage3In` → port 4 | `Mlu.scala:76,101,116` | SRC ✓ | dependent `mul` chain: 1 per 3 cycles? <br>**sim+wave**: stage2Input D+1, stage3Input + port 4 at D+2, retire D+2 (TRACE). mul chain 2.0, mul->ori->mul 3.0, all mulh* 3.0 (B2). |
| DVU | D+1 normalize, then 1 bit/cycle, "32 − clz iterations" | `scalar/Dvu.scala:90-142`. Write-back at **D + 2 + N**, where N = 32 − `Clz1(\|rs1\|)` and `Clz1` = clz − 1 (0 if the MSB is set). So N = 2…32 (2 for a dividend of 0 or 1; **32 for divide-by-zero**). Clz is on the **dividend**. Next `div` accepted at D+3+N. | **SRC ✗, likely off by one**. Measure it. | BENCH: dividends 1, 2^8, 2^16, 2^30, 2^31, −1, divisor 0 <br>**human** (ALEX ✓ R6, 2026-10-19; was sim+wave): `dvu.active` D+1, `compute` D+2, write-back D+2+N with N = 32 - Clz1(|a|): dividend 1 -> D+4, 2^30 / div-by-0 -> D+34 (TRACE). **sim**: next div at D+3+N for 1, 0, -1 (5), 256 (13), 65536 (21), 2^30, -2^31, 2^31 unsigned, b=0 (35); divu/rem/remu same (B3). The draft's '32 - clz' is off by one: it is 32 - (clz - 1). |
| Load | D+1 opQueue, D+2 slot, D+3 readFired, D+4 → port 5, slot next D+5 | `Lsu.scala:835,876-880,903-1018,1056-1080`. (SRAM address is driven in D+2; `readFired` is the register at the end of D+2.) | SRC ✓ (minor label shift). **Consequence: only 1 load per 4 cycles** | BENCH: 8 independent `lw`; expect ~4 cycles each. Load-use: dependent `addi` dispatches at D+4. <br>**human** (ALEX ✓ R1, 2026-10-18; was sim+wave) (TRACE lw): opQueue D+1, dbus request D+2, readFired D+3, port 5 at D+4, retire D+4; dependent ALU D+4, pointer chase 5.0, lw->add->lw 6.0 (B4). |
| Store | D+1 opQueue, D+2 bus write, D+3 storeComplete | `Lsu.scala:931-934,1000-1006`; ROB sees it one cycle later via `Pipe` (`RetirementBuffer.scala:71`) | SRC ✓. **Consequence: 1 store per 2 cycles** | BENCH: 8 `sw`; load after store to same address <br>**human** (ALEX ✓ R1, 2026-10-18; was sim+wave) (TRACE sw): opQueue D+1, dbus write + storeComplete D+2, retire D+3. 1 store/2 cycles incl. sb/sh (B4). |
| LSU opQueue = 4 | | `Lsu.scala:835` | SRC ✓ | BENCH: 5+ loads in a row; dispatch stalls on the 5th <br>**sim**: each extra load in a burst adds 4 (B4); the 5th-load dispatch stall itself not isolated. |
| Line-crossing / misaligned | not drawn | multiple transactions (`targetAddress` loop) | NEW | BENCH: `lw` at line offset 14 <br>**sim**: lw misaligned inside a line = 4 cycles (no penalty), lw crossing a 16-byte line = 5, sw crossing = 3 (B4 misaligned). No trap. |
| FPU | lane 0, one in flight, D+1 instQueue, D+2..D+4 fpnew regs, → f-port 0 | `float/FloatCore.scala:222-225` (queue depth 1, ready only when empty), fpnew `PipeRegs 3` (L112-118), write at L336 | SRC ✓ for add/mul/fma. **Next FP op can't dispatch until D+5.** | BENCH: independent `fadd` stream; expect 1 per 5 cycles (see §8 B5) <br>**sim+wave, see D4** (ALEX ✓ R5 2026-10-19 for the sim behavior only; the diagrams use the hardware numbers, decided 2026-10-19): in the Verilator model fpnew FP32 has 0 pipe regs: instQueue D+1, fpnew in/out D+1, f-port 0 D+1, queue empty D+2 -> 1 FP op per 2 cycles (B5). Source intent (3 regs): D+4 / 1 per 5, which is the hardware (D4). <br>**F2 (2026-10-19, agent):** cause confirmed (Verilator 5.032/5.052 put fpnew's `'{default: 'd3}` into the last format slot of each packed row, so FP32 = 0; slang gives 3). On the experiment build with `PipeRegs` written row by row (M3 + `bench/d4_repro/floatcore_piperegs_row_by_row.patch`): fpnew accept D+1, write **D+4**, queue empty / next FP op **D+5**, fadd/fmul/fmadd **5.0** = source prediction exact (sim+wave on that build; patch and `regen_d4exp.sh` in `bench/d4_repro/`). Synthesis: the Drop 0 DC (269 flops) and Yosys-slang (278) netlists have the 3 FP32 FMA stages (Aaron, 2026-10-19), so only the Verilator sim has 0. D4 in `notes/discrepancies.md`. |
| FPU div/sqrt | not drawn | iterative fpnew DIVSQRT unit (TH32 or PULP, set by `floatPulpDivsqrt`) | NEW | BENCH: `fdiv.s`, `fsqrt.s` <br>**sim+wave**: fdiv write D+5 (next FP op D+6), fsqrt write D+18 (D+19) (TRACE, B5). May also change with D4. <br>**F2 (2026-10-19):** it does: with 3 regs (experiment build) fdiv write D+8 (9 cycles), fsqrt D+21 (22), i.e. +3 each. |
| FPU → int rd (`feq`/`flt`/`fle`/`fclass`/`fcvt.w.s`) | not drawn | extra `Queue(2)` (L352), then port-4 arbiter (behind MLU/DVU) | NEW | BENCH <br>**sim+wave**: Queue(2) enq D+1, port 4 at D+2, int consumer D+2 (TRACE). With 3 pipe regs it would be D+5 (D4). <br>**F2 (2026-10-19):** measured on the experiment build: Queue(2) enq D+4, port 4 at **D+5**, feq/fcvt stream 5.0, fcvt→addi 6. |
| `fmv.x.w` / `fmv.w.x` | not drawn | bypass fpnew, complete in D+1 (L300-302, 336-350) | NEW | BENCH <br>**sim+wave**: fmv.w.x f-port 0 at D+1; fmv.x.w port 4 at D+2 (via the Queue(2)); both free the FPU at D+2 -> 2.0 cycles/op (TRACE, B5). Not affected by D4. |
| `flw`/`fsw` | not drawn | go through the LSU (`LsuOp.FLOAT`); load writes f-port 1; dispatched alone (3.6) | NEW | BENCH <br>**sim+wave**: lane 0 alone (D=3 in TRACE), flw f-port 1 at D+4, fsw storeComplete D+2 / retire D+3; LSU-bound 4.0 / 2.0 (B5). |
| RVV handoff | D+1 `RvvFrontEnd` `inst_q`, `rvvFrsReg` → CQ | `SCore.scala:446-453`, `verilog/rvv/design/RvvFrontEnd.sv:75-77` | SRC ✓ (names exist). Boundary with Pranav. | **sim+wave** (TRACE vadd.vv): dispatched in lane 1 with the pre addi (D), `RvvFrontEnd.valid_inst_q` and `cmd_valid_o` at D+1. ROB retire observed at D+8 (Pranav's side, not analysed). |

## 5. Retire (missing from both drafts)

| # | Claim | Source | Status | WAVE/BENCH |
|---|---|---|---|---|
| 5.1 | 8-entry in-order retirement buffer, enqueued at dispatch | `coralnpu/RetirementBuffer.scala:73-80` | NEW | **human** (ALEX ✓ R7, 2026-10-19; was sim+wave): 8 entries (B7, see 0.3). |
| 5.2 | An entry retires when its write-back is seen ("data done"). Control-flow entries also wait until the **next instruction has dispatched** ("cf done"). | `RetirementBuffer.scala:330-416` | NEW | WAVE: when does an ALU op retire, D+1 or D+2? <br>**human** (ALEX ✓ R8, 2026-10-19 (ALU/load retire; cf-done part stays sim+wave); was sim+wave) (TRACE): ALU, MUL, CSR, load retire in the same cycle as their write-back (ALU D+1). Control flow: not-taken branch retires D+1 (next instruction dispatched D+1), mispredict retires D+4 (when the target dispatches) - the 'cf done' rule. |
| 5.3 | Stores retire on `storeComplete` (+1 cycle `Pipe`) | `RetirementBuffer.scala:71` | NEW | **sim+wave** (TRACE sw/fsw): storeComplete D+2, retire D+3. |
| 5.4 | `minstret` increments by `nRetired` | `SCore.scala:188` | NEW | **src**: not checked. |

## 6. Dependency microbenchmark matrix (minimum set)

**Data:** producer {ALU, MUL, DIV, LW, FADD, FLW, CSR} × consumer {ALU, store-data, load-base, BR, MUL, FADD} × distance 0..5.
**Structural:**
- 2 MULs in one group
- DIV + DIV
- MUL + DIV write-back collision on port 4
- 5 loads (opQueue full)
- ROB full behind a DIV
- FP op + anything
**Control:**
- `beq` not-taken fwd
- `beq` taken fwd (mispredict)
- `beq` taken bwd (predicted)
- `beq` not-taken bwd (mispredict)
- `jal`
- `jalr`
- branch target at line offsets 0/4/8/12
- branch in lanes 0..3

**Coverage (2026-10-17):** control set: all done (B1, TRACE). Structural: all done (B2, B3, B4, B7, B5/TRACE for FP + anything); the port-4 MUL/DIV collision and the 5th-load stall are cycle counts only. Data: only part of the matrix ran: {ALU, LW} producers × {ALU, store data, load base} consumers (dep_chain, B4), plus MUL, DIV, FADD, FLW and CSR producers with an ALU or same-unit consumer (B2, B3, B5, B6, TRACE). The full producer × consumer × distance 0..5 sweep was not run. Not needed for the diagrams, so it is left for Samuel's pass if needed.

## 8. Deliverable sprint: remaining benches (run all in one batch)

Predictions are from source. Measure, compare, and let the sim win. "D" = dispatch cycle.

| ID | Bench | Rows | Source prediction |
|---|---|---|---|
| B1 | **Branches** | 1.7, 1.8, 1.10, 3.4, 3.5, BRU rows | <ul><li>Not-taken forward `bne`: no redirect, but nothing else dispatches in its group (3.4).</li><li>Backward taken (loop): predicted at predecode, so no BRU redirect. The only cost is the rest of the line being dropped.</li><li>Mispredicts (forward taken, backward not-taken) and **every `jalr`**: target dispatches at D+4, so 3 bubbles.</li><li>`jal`: predicted, no redirect.</li><li>`jalr` rs1 uses the registered scoreboard (+1 vs ALU).</li><li>Target at line offset +4/+8/+12 gives only 3/2/1 instructions from the first line.</li></ul> |
| B2 | **MUL** | 3.8, 3.15, MLU row | <ul><li>Write-back on port 4 at D+2, so a dependent ALU op can dispatch at D+2. A `mul` chain costs 2 cycles/mul.</li><li>Independent: 1 `mul`/cycle. Two `mul`s in one group: the second waits a cycle.</li><li>Run all four `mulh*` variants.</li></ul> |
| B3 | **DIV** | 3.7, 3.15, DVU row | <ul><li>Write-back at D+2+N, where N = 32 − max(clz(\|a\|)−1, 0) for a≠0. a ∈ {0,1} gives N=2; b=0 gives N=32.</li><li>Next `div` accepted at D+3+N.</li><li>Lane 0 only: `addi; div` makes the `div` wait a cycle.</li><li>A `mul` finishing in the same cycle as a `div` delays the `div` write-back (port 4, MLU has priority).</li><li>Sweep a = 1, 2^8, 2^16, 2^30, 2^31, −1, and b = 0.</li></ul> |
| B4 | **LSU latency / structure** | 3.9, 3.19, Load/Store rows | <ul><li>Load-to-use: `lw` write-back at D+4, so a dependent ALU op dispatches at D+4.</li><li>5+ loads back to back: the 5th waits for queue space (4 entries).</li><li>Load after a store to the same address: serialized through the single slot.</li><li>`lw` crossing a 16-byte line: 2 bus transactions.</li><li>Byte/half variants (`lb`/`lbu`/`lh`/`lhu`/`sb`/`sh`) should match `lw`/`sw`.</li></ul> |
| B5 | **FPU** | 3.6, 3.16, FPU rows | <ul><li>Every float op, `flw` and `fsw` included, dispatches alone in lane 0.</li><li>`fadd`/`fmul`/`fmadd`: f-reg write at D+4. The next FP op can't dispatch until **D+5**, so an independent stream runs at **1 per 5 cycles** (the 1-in-4 written earlier was wrong). A dependent chain is also 5.</li><li>`feq`/`flt`/`fcvt.w.s`/`fclass` (FP → int): +1 cycle from `Queue(2)`, so port 4 at D+5.</li><li>`fmv.x.w`: int write-back around D+2. `fmv.w.x`: f write at D+1.</li><li>`flw`: f-port 1 at D+4.</li><li>`fdiv`/`fsqrt`: iterative, so measure them.</li></ul> |
| B6 | **CSR** | 3.6, 3.11, CSR row | <ul><li>`csrr`/`csrw` dispatch alone in lane 0. Result at D+1 on port 0.</li><li>`csrr` right behind a long `div`: waits for the ROB to empty.</li></ul> |
| B7 | **ROB full** | 3.10, 5.1 | `div` (large N), then 12 independent `addi`: dispatch stalls once 8 entries are in flight, and resumes when the `div` retires. |
| B8 | **x0 / nop** | 3.17 | <ul><li>4× `nop`: 4 cycles to dispatch.</li><li>4× `addi` with distinct rd: 1 cycle.</li><li>`nop; li a0,1; li a1,1; li a2,1`: serialized.</li></ul> |
| B9 | **ALU class sweep** | ALU row, 0.2 | <ul><li>Every RV32I ALU op, every Zbb op, `lui`, `auipc`: a dependent chain runs at 1 cycle/inst.</li><li>Read `misa` and decode the ISA letters.</li></ul> |
| B10 | **System** | 3.5, 3.12 | <ul><li>`fence` waits for the LSU to go idle.</li><li>`fence.i`: flush, then refetch. Measure the cost.</li><li>`ecall`→`mret` round trip: optional, if time allows.</li></ul> |
| TRACE | **Single-instruction trace** (waves) | all of §4, §5 | <ul><li>One isolated instance per class gives the F1/F2/D/stage/WB/retire cycle numbers for `deliverable/timing.json` (schema in its `schema` field; produced by `bench/collect.py`).</li><li>Settles 5.2: when does an ALU op retire?</li></ul> |

**Sprint results (2026-10-17, agent; levels per row above):** all of B1-B10 + TRACE written, run in one `bazel test //tests/cocotb/scalar_timing:all` (17 targets, 17/17 pass after calibration). Source predictions matched exactly for B2, B3, B4, B9 and most of B1/B6/B7/B8/B10. Mismatches: B5 FPU (D4, Verilator elaborates 0 fpnew pipe regs: 1 FP op per 2 cycles, not 5), B8 prefilled diffs (D5, bench model), plain `fence` (D6), and one bench bug (D7, fixed). Per-bench tables: `results/b*.md`. Timeline per class: `deliverable/timing.json`. Alex signed R1-R9 on 2026-10-18/19; see `deliverable/review_packet.md` and the levels above.

Correction to §4 FPU row: the earlier "expect 1 per 4 cycles" contradicts its own "next FP op can't dispatch until D+5". The source says **1 per 5**: the queue entry is freed at D+4 (`inst.fire` when `out_valid`), and `io.inst.ready` (`count === 0`) is seen at D+5 (`FloatCore.scala:222-225,356`).

## 7. Open questions (ask Orhan / Pranav)

- Does "retire" mean ROB dequeue (`nRetired`) or register write-back?
- What is cycle 0 for the timing diagrams (F1 = ITCM request)?
- Does the RVV boundary stop at the `RvvFrontEnd` enqueue? Does scalar own `vset*` timing?
- **D4 (Alex + Pranav):** which FPU numbers go in the diagrams: sim (write D+1, 1 per 2) or RTL intent (write D+4, 1 per 5)? Can a second simulator or the synthesized netlist confirm the pipe-register count?
- Still src only, if anyone needs them measured: fetch structure 1.1-1.4 and 1.6, `jal`/`jalr` link write, `minstret` (5.4), `ecall`/`mret`, LSU timing to ITCM / external bus.

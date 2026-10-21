# RvvCoreMiniAxi (M3) scalar core: timing summary for Peter / Orhan

Source: `google-coral/coralnpu` tag `M3-2026-04-27` (72d700e), Verilator model. All numbers are in cycles; at Reef's 50 MHz target 1 cycle = 20 ns. The sim clock is CoralNPU's default 1.25 ns (800 MHz), which changes no cycle count: the whole suite gives identical results at 20 ns (checked 2026-10-20).
File:line paths are relative to `hdl/chisel/src/coralnpu/`. D = dispatch cycle. Full data: `deliverable/timing.json`
(31 instruction classes, 31 pipeline structures), per-bench tables in `results/`, mismatches in `notes/discrepancies.md`.

**Who verified what.** Rows marked `human` were checked by Alex by hand in the waveform: **1.5** fetch (2026-10-13)
and the review packet **R1-R9**, all agreed 2026-10-18/19 (`review_packet.md` has his notes): BTFN prediction,
CSR/FP dispatched alone in lane 0, branch mispredict D+4, DVU latency, load and store throughput, the 8-entry ROB
stall, retire timing of an ALU op and a load, and same-cycle forwarding. R5 was agreed for the **sim** FPU behaviour
only; the FPU timing is the hardware's (D4: the netlists have the 3 FMA stages the sim lacks). Every other row was verified by the agent: `sim+wave` = cycle-count
bench plus a scripted VCD scan, `sim` = cycle counts only, `src` = read from source only. **Samuel will re-run and
verify everything.**

## Orhan's claims: verdicts

| Claim (draft) | Verdict | Level | Evidence |
|---|---|---|---|
| 1.5 F2: no new fetch request | **confirmed**: one 16-byte line per 2 cycles, front end peaks at 2 IPC | human | fetch_ipc; `UncachedFetch.scala:195` |
| 1.7 "predict taken" | **corrected**: static BTFN. Backward cond. taken and `jal` predicted; forward cond. predicted not-taken; `jalr` never predicted | human (R3) | B1 loop2 2.0, ft 4.0, bnt_bwd 4.0; TRACE; `UncachedFetch.scala:87-100` |
| Lane rules: CSR / FP "lane 0 only" | **corrected**: lane 0 **and dispatched alone** (CSR, all float incl. flw/fsw, fence.i) | human (R5) | TRACE (D one cycle after lane-0 addi, next addi D+1); `Decode.scala:167-169,400-406` |
| DVU lane 0 only | confirmed (later lanes may dispatch in the same cycle) | sim+wave | TRACE div_*; `SCore.scala:256-264` |
| MLU one per cycle | confirmed | sim | B2 mul_ind4 4.0/line; `Mlu.scala:67-76` |
| ALU D+1 → port i | confirmed; retires D+1 | sim+wave | TRACE alu; B9 (39 ops, chain 1.0); `Alu.scala:79-97` |
| BRU mispredict: target at D+4 | confirmed (also every `jalr`; the `jalr` case is sim+wave) | human (R2, R3) | TRACE br_mp_*, jalr; B1 ft/bnt_bwd 4.0; `Bru.scala:114-147` |
| CSR D+1, port 0 | confirmed; but waits for ROB empty + FPU ready | sim+wave | TRACE csr; B6; `Csr.scala:241`, `Decode.scala:519,693` |
| MLU D+1 / D+2 → port 4 | confirmed | sim+wave | TRACE mul; B2 chain 2.0; `Mlu.scala:76,101` |
| DVU "32 − clz iterations" | **corrected**: WB at D+2+N, N = 32 − (clz−1) (MSB set: 32; dividend 0/1: 2; ÷0: 32); next div D+3+N | human (R6; ÷0 case sim+wave) | TRACE div; B3 (12 dividend/op cases exact); `Dvu.scala:90-142` |
| Load D+1 queue … D+4 port 5 | confirmed (D+2 = bus request, D+3 = `readFired`) | human (R1, R8) | TRACE lw; B4 chase 5.0; `Lsu.scala:835,879-880,1017` |
| Store D+1 queue, D+2 write, D+3 complete | **corrected**: storeComplete at D+2, ROB retire D+3 | sim+wave | TRACE sw; `Lsu.scala:1005`, `RetirementBuffer.scala:71` |
| LSU opQueue = 4 | confirmed (each extra load in a burst adds 4) | sim | B4 burst1-8 |
| FPU D+1 queue, D+2-D+4 fpnew, write D+4 | **confirmed for the hardware**: the Drop 0 DC and Yosys-slang netlists have fpnew's 3 FMA stages (Aaron), and with them the FPU measures **write D+4, 1 FP op / 5 cycles, exactly as drafted**; fdiv 9, fsqrt 22. Only the Verilator model differs: it builds 0 regs for FP32 (`'{default: 'd3}` on a packed array), so the sim shows write D+1, 1 / 2 (D4) | sim+wave (both builds) | B5, TRACE on M3 and on the row-by-row build; `FloatCore.scala:112-119,222-225`; `notes/discrepancies.md` D4 |
| RVV handoff D+1 `inst_q` → CQ | confirmed (`valid_inst_q`, `cmd_valid_o` at D+1) | sim+wave | TRACE rvv; `RvvFrontEnd.sv:75-77` |
| "No flip-flops in D" | confirmed; operand regs are `Regfile.readDataBits/Ready` | src | `Regfile.scala:117-118` |
| Fences end the group / wait for LSU | **corrected (D6)**: only `fence.i` (and flushes). Plain `fence` neither waits nor ends the group | sim | B10 `lw;fence` 4.0 = `lw`; `Decode.scala:163,396,723` |
| *(missing)* ROB, retire | **new**: 8-entry ROB, dispatch stalls when full; entries retire in the WB cycle; control flow also waits for the next dispatch | human (R7, R8; the control-flow wait is sim+wave) | B7, TRACE; `Decode.scala:517`, `RetirementBuffer.scala:330-416` |

## Top surprises (things a diagram reader would get wrong)
1. **Fetch peaks at 2 IPC**, not 4: one 16-byte line per 2 cycles (`human`). Any dispatch leftover when a line lands costs a fetch cycle (2,3,2,3 pattern, 3.20).
2. **LSU sustains 1 load per 4 cycles and 1 store per 2** (single slot, `Lsu.scala:1056,1071`; `human`). sb/sh/lb/lh same. A line-crossing lw costs 5, sw 3.
3. **BTFN** (`human`): backward loops are free, forward-taken branches cost 3 bubbles (4.0 cycles). `jal` costs one fetch slot (2.0). Every `jalr` mispredicts (D+4).
4. **A conditional branch ends its dispatch group** (`human`): 4 not-taken branches need 4 cycles.
5. **CSR and all FP ops dispatch alone in lane 0** (`human`); CSRs also wait for an empty ROB (`csrr mcycle` is a fence, behind a long div it waits ~35 cycles).
6. **ROB = 8 entries** (`human`): behind a 34-cycle div only 7 more instructions dispatch. 40 independent addi that could hide under the div cost +17 cycles.
7. **Store data and load/jalr base use the registered scoreboard**: consumer at +2 instead of +1 (pointer chase = 5 cycles/load).
8. **rd = x0 is not free**: 4 nops take 4 cycles; an instruction reading x0 can't share a group with an earlier rd=x0 one (incl. `fence`).
9. **The Verilator model's FPU is faster than the hardware (D4)**: Verilator builds fpnew's `'{default: 'd3}` (a packed array) as 0 FP32 pipeline registers, so sim shows 1 FP op per 2 cycles. The synthesized Drop 0 netlists (DC, Yosys-slang) have the 3 stages: D+4, 1 per 5, which is what the diagram uses. A row-by-row `PipeRegs` fix (works in Verilator, slang and DC) is planned after Reef PR #15. Any cycle counts with scalar FP from the upstream CoralNPU Verilator sim (e.g. reef-perf's `fp_chain`) stay optimistic even after Reef's fix, because that sim doesn't use Reef's RTL; they need a build with the fix.

## Still SRC-only / not measured
Fetch structure rows 1.1-1.4, 1.6, 1.9 (no direct check; 1.9 seen indirectly in 3.20); `jal`/`jalr` link write (benches use rd=x0);
`minstret` (5.4); the 5th-load queue-full stall and the MUL/DIV port-4 collision as wave events (both only as cycle counts);
`ecall`/`mret` round trip; LSU timing to ITCM / external bus targets; anything inside the RVV engine (Pranav);
Verilator FPU numbers until the `PipeRegs` fix PR lands (the hardware numbers come from the row-by-row rebuild).

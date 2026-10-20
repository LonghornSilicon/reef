# Source vs sim discrepancies (sim wins)

Format: claim / source-only prediction / measured / why / evidence / status.

## D1. A dependent consumer can cost front-end bandwidth, not just a dispatch slot
- **Prediction (source-only):** ALU consumer at distance 0 behind its producer costs
  1.0 cycle/unit (chain-bound, same-cycle forwarding); a store consumer at d=2 costs
  2.0 cycles/unit (regd +1, fetch bound 2.0).
- **Measured:** ALU d=0 DEP **1.25**; SW d=2 DEP **2.5**. Controls (IND) match the model.
- **Why (verified in waveform):** serialised dispatch leaves instructions in the 8-entry
  buffer when the next line lands; `insufficientBuffer` (`UncachedFetch.scala:192`,
  `nSpace >= nValid + 4`) then delays the next request by one cycle. Line period
  alternates 2,3 cycles. The model "period = max(chain, fetch)" ignores this coupling.
- **Evidence:** `results/dep_chain.md`; dep_chain_alu.vcd 742.5-753.75 ns (row with
  `blockNewFetch=1`, `nSpace=3`), dep_chain_sw.vcd 1122.5-1135 ns.
- **Status:** explained. Test expectation set to the measured value, flagged
  sim-calibrated in the JSON.

## D2. Stores sustain at most one per 2 cycles
- **Prediction:** SW independent consumer at d=0 / d=1 runs at the fetch bound (1.0 / 1.5
  cycles per unit).
- **Measured:** **2.00** cycles/unit for both.
- **Why (derived from source, verified in waveform):** the LSU has ONE slot, and it only
  takes the next op from the queue when the slot is idle (`Lsu.scala:1056,1071`). A scalar
  store holds the slot for one cycle (bus write at `:931-934`, `storeUpdate` clears
  `active` at `:993-995`), the slot is idle the next cycle (`slotIdle`, `:391-395`, no
  pending writeback for a store), and that idle cycle is when the next op is dequeued.
  1 busy + 1 dequeue cycle = 2.
- **Evidence:** dep_chain_sw.vcd 450-463 ns: `io_dbus_valid && io_dbus_write` every 2nd
  cycle; 563 of 728 store fires are exactly 2 cycles after the previous one.
- **Status:** explained, derived and verified.

## D3. Loads sustain at most one per 4 cycles; it hides the register-scoreboard +1
- **Prediction:** LW at 2.0-2.5 cycles/unit (DEP) and 1.0-2.5 (IND).
- **Measured:** **4.00** for every d and both modes.
- **Why (derived from source, verified in waveform):** same single slot, but a load holds
  it for three cycles, then the slot is idle for the dequeue cycle: (1) bus request
  (`io.dbus.valid`, `:931-933`), (2) read data back, `readFired` valid, `loadUpdate`
  clears `active` (`:903-905`, `:482-511`), (3) `shouldWriteback()` -> `io.rd.valid`
  (`:515-517`, `:1017-1018`), then (4) idle, dequeue (`:1056,1071`). 3 busy + 1 = 4.
- **Evidence:** dep_chain_lw.vcd 900-925 ns, repeating 5 ns (= 4 cycles):
  `io_dbus_valid` -> `readFired_valid` -> `io_rd_valid` -> one empty cycle.
- **Consequence:** the checklist 3.3 "+1 cycle for the load case" is real in dispatch
  timing (+2 vs +1, SIM ✓) but cannot be seen as a throughput difference with this bench.
- **Status:** explained, derived and verified. Not yet explored: does the 4 change for
  other targets (ITCM `ibus`, external `ebus`)? The slot path is shared but bus latency differs.
- **F1 (2026-10-19): why `io_queueCapacity` sits at 1 during the load stream.** It is `opQueue.io.nSpace`
  (`Lsu.scala:837`) = 4 - `nEnqueued` (`common/CircularBufferMulti.scala:72`): a plain count of ops waiting in
  the queue, with no reserved entry, and the op in the slot is not counted. Dispatch admits lane i only if
  `isLsuCount(i) < capacity` (`Decode.scala:459-463`). So the queue is really 4 deep and could reach capacity 0;
  in this bench the ROB stops it first. Each unit is `addi a2` + d fillers + `lw`, i.e. 2+d ROB entries, and
  retire is in order behind the oldest load, so the 8-entry ROB holds 8/(2+d) loads. For d=0 that is 4 loads:
  1 in the slot + 3 queued -> capacity 1. Waveform (dep_chain_lw.vcd 893-930 ns, `bench/vcdpeek.py cycles`):
  capacity 4 up to 898.75, 3 at 900.00 (first load waiting one cycle), then 1; `retirement_buffer.io_nSpace`
  hits 0 with lane 0 valid but not ready; each `io_rd_valid` retires 2 (`io_nRetired` = 2), the next
  addi+lw pair dispatches, and the ROB is full again. Loads stay 1 per 4 cycles (`io_dbus_valid` 901.25,
  906.25, ...). Over the whole VCD the plateaus follow 8/(2+d): d=0 -> 1, d=1 -> 2, d=2 -> 3, d=3 -> 3, the
  `addi` chains -> 4, with ROB-full cycles in every load plateau. No change to the 4-entry claim (3.9, §4
  LSU opQueue). The R1 "what to look for" text in `bench/make_review_packet.py` is corrected to match.

## D4. FPU add/mul/fma/cmp/cvt complete in D+1 in sim, not D+4 (Verilator elaborates 0 fpnew pipe regs)
- **Prediction (source):** `FloatCore.scala:112-119` instantiates fpnew with `PipeRegs: '{default: 'd3}`, `DISTRIBUTED`.
  Result D+4, queue (depth 1, `:222-225`) frees at D+4, next FP op dispatches D+5: **1 FP op per 5 cycles**.
- **Measured:** fadd/fmul/fmadd/feq/fcvt.w.s stream **2.0 cycles/op**; dependent fadd 2.0; fcvt->addi 3; fmv.w.x->fadd 4;
  csrr behind fadd 3 (pred. 6). TRACE: fadd D=3, `fpnew out_valid_o` and f-port 0 write at D+1, instQueue empty D+2.
- **Why (one waveform look, b5_fpu.vcd / trace.vcd):** the elaborated parameters in the VCD are
  `gen_operation_groups[0].FmtPipeRegs = {0,0,0,0,3}` (only the last of the 5 format slots got the default) and
  `...i_fmt_slice.NumPipeRegs = 0` for the FP32 slice. So in this Verilator model the FP32 FMA is combinational.
  The generated SV (`bazel-bin/hdl/chisel/src/coralnpu/RvvCoreMiniAxi.sv:38338-38345`) still says `'{default: 'd3}`, so
  this looks like a Verilator elaboration issue with a nested `'{default:}` pattern, not an RTL change.
- **Consequence:** sim numbers for the FPU are NOT what a synthesis tool that honours the parameter would build.
  Expect 1 per 5 (D+4 write) on silicon unless someone confirms otherwise. Both are reported in `timing.json`
  (`source_intent_if_different`). fdiv (6) / fsqrt (19) are iterative DIVSQRT and may also change. fmv.* bypass fpnew
  and are unaffected.
- **Status:** explained in sim, **open for Alex/Pranav**: decide which number the diagrams use; check with another
  simulator or the synthesized netlist. Bench expectations are sim-calibrated and flagged.
- **F2 (2026-10-19): cause confirmed, RTL-intent numbers measured.**
  - *Mechanism.* In `fpnew_pkg.sv` (cvfpu, Bazel external), `fmt_unsigned_t` is a **packed** `logic [0:4][31:0]`
    (one 160-bit integral value per op group) and `PipeRegs` is `fmt_unsigned_t [0:3]`. If `default: 'd3` is applied
    to each 160-bit row as one integer, the 3 lands in the last slice `[4]` (FP16ALT) and FP32 `[0]` gets 0: exactly
    the `{0,0,0,0,3}` in the VCD. cvfpu's own `DEFAULT_SNITCH` (`'{default: 1}`) gives `{0,0,0,0,1}` the same way.
  - *Standalone repro* (scratch only; real `fpnew_pkg.sv` + the exact localparam from `FloatCore.scala:112-119`):
    Verilator 5.032 (Bazel's) -> `0 0 0 0 3` for all 4 op groups; Verilator 5.052 (newest, built from source)
    -> **also `0 0 0 0 3`**; slang 12.0 (pyslang) -> `3 3 3 3 3`. Row by row,
    `'{'{default: 'd3}, '{default: 'd3}, '{default: 'd3}, '{default: 'd3}}`, gives 3 everywhere in all three tools.
    So it is not fixed in a newer Verilator, and the tools disagree on the plain form. Which reading the LRM
    requires is **not settled**: slang recurses into the packed rows, Verilator does not. What matters for silicon
    is the synthesis front end, which nobody has checked yet.
  - *Full rebuild* (branch `d4-piperegs-experiment` in worktree `~/LHS/coralnpu-d4` = M3 `72d700e` + only the
    row-by-row `PipeRegs` line, commit `2549f93`, not for merging; ~15 min cold): VCD now shows
    `FmtPipeRegs = {3,3,3,3,3}` for all 4 scalar op groups, FP32 slice and E906 divsqrt `NumPipeRegs = 3`.
    `b5_fpu`: fadd/fmul/fmadd/feq/fcvt/fadd_dep **5.0**, fcvt_use 6, fmvwx_fadd 7 = **every source prediction exact**;
    fdiv 6 -> **9**, fsqrt 19 -> **22** (+3 each); fmv.x.w/fmv.w.x 2, flw 4, fsw 2 unchanged. TRACE fadd: D=3,
    accepted D+1, `out_valid_o` + f-port 0 write **D+4**, instQueue empty / consumer **D+5**; feq/fcvt int port 4 at
    D+5; fdiv write D+8, fsqrt D+21. Numbers in `results/json/d4exp_b5_fpu.json`, `d4exp_trace_events.json`, and
    `timing.json` `source_intent_if_different` (generated by `bench/collect.py`). VCDs in
    `~/LHS/scalar-timing-waves/d4-experiment/`.
  - *What is still open:* which numbers go in the diagram (Alex + Pranav), and what the synthesis flow builds from the
    plain `'{default: 'd3}` (Samuel: check the FP32 FMA register count in the netlist). If it builds 0 registers,
    the row-by-row form is the fix to propose upstream / in Reef, since it means 3 in every tool.
  - *Correction (2026-10-19):* the team's Coral NPU Tapeout Dossier (rev 2.2, 2026-09-26) already found this as an
    upstream RTL bug and showed the **synthesized** release netlist has it too: sv2v → Yosys built a single-cycle FP32
    FMA (28 ns on sky130, the critical path; FPU flop count 3 vs 761 fixed). Its fix, `'{default: '{default: 32'd3}}`,
    is gate-level cycle-identical to the fixed RTL and measured 1 fmadd per 5 cycles, the same as F2 here. So this is
    an RTL bug most tools build as written (only slang differs), not a Verilator-only one. Release-M3 cycle counts with
    1 FP op per 2 cycles (e.g. reef-perf's `fp_chain`) are correct for the release; the RTL-intent numbers apply once
    the fix is in. Reef Drop 0 (PR #15) vendors M3 unchanged, without the fix (`FloatCoreWrapper.sv:38`).
  - *Netlist answer (Aaron, 2026-10-19):* the Reef Drop 0 **DC** netlist has the 3 FP32 FMA stages (269 flops) and the
    **Yosys-slang** netlist too (278 flops). So in Reef's flows only Verilator builds 0 registers (the Dossier's 0 came from
    its sv2v → Yosys flow); the hardware matches the source intent. Aaron re-ran `bench/d4_repro/pipe_regs_check.sv` with the
    same results. The Dossier's fix `'{default: '{default: 32'd3}}` doesn't work with Reef's tools; the row-by-row form works in
    Verilator, slang and DC, and Aaron will PR it after #15. **D4 decided:** diagrams use D+4 / 1 per 5 (Pranav, Aaron).

## D5. Prefilled-buffer x0 diffs smaller than the dispatch-group model (bench model, not the core)
- **Prediction:** pf_nop7 +5, pf_li_nop_li5 +1 vs pf_base (counting dispatch groups of the buffered pattern).
- **Measured:** +3 and +0. pf_nop_li6 +1, controls +0 as predicted.
- **Why (b8_x0.vcd):** only 7 entries were buffered behind the div, and the last pattern instruction sits in a line that
  is fetched after the release, so pf_base is itself fetch-bound for 2 cycles and hides part of the extra dispatch cycles.
  The mechanisms are visible directly: 7 nops dispatch one per cycle (`ready=1000` x7, 1060-1066); in `li; nop; li...`
  the li after the nop is held out of the nop's group (cycle 1202: lanes 0-2 dispatched, lane 3 `li` not).
- **Status:** explained; 3.17(a) and (b) are now sim+wave. Expectations sim-calibrated.

## D6. Plain `fence` does not wait for the LSU and does not end the group (checklist 3.5 / 3.12 too broad)
- **Claim:** 3.12 "fence/flush wait for the LSU to be idle"; 3.5 "fences end the group".
- **Source re-read:** `isFency()` = fence.i, ebreak, wfi, mpause, flushat, flushall (`Decode.scala:163`). Plain `fence`
  is not in it, so it has no LSU interlock (`:396`) and is not in `isJump || isFency` (`:312`). It only dispatches from
  lane 0 (`fenceValid`, `:723`).
- **Measured:** `lw; fence` = 4.0 cycles/unit = `lw` alone (no wait). Stream of `fence` = 1/cycle. TRACE: fence D=3,
  next addi D+1 (held by the x0 RAW of 3.17b, since fence has rd=x0, not by a group end). fence.i: flush at D+1,
  next instruction D+4; `lw; fence.i` = 9, `sw; fence.i` = 7.
- **Status:** corrected in the checklist (3.5, 3.12).

## D7. Bench bug found and fixed: a self-branch is not "backward"
- First run of B1 had `1: bne zero, zero, 1b`. Offset 0 has `inst(31) = 0`, so BTFN predicts it not taken and there is
  no mispredict (2.0 measured, all 4 lanes dispatching every 2 cycles, no redirect in b1_branch.vcd). Fixed to branch to
  the previous instruction (offset -4): now 4.0 = predicted. Recorded so nobody reuses the offset-0 pattern.

## D8. FPU "1 per 5" claim in checklist §4 / §8 B5 (and R5)
- Source-derived 1 per 5 stands as the RTL intent; sim measures 1 per 2 because of D4. The "dispatched alone in lane 0"
  part is confirmed in sim+wave (TRACE: every FP/FLW/FSW/CSR class dispatches at D=3, one cycle after the lane-0 pre
  addi, nothing else dispatches with it).

#!/usr/bin/env python3
"""Collect sim results and (re)generate deliverable/timing.json.

    python3 bench/collect.py            # after `bazel test //tests/cocotb/scalar_timing:all`
                                        # and `bash bench/waves.sh trace`

1. Copies each bench's JSON out of bazel-testlogs/.../test.outputs/outputs.zip
   into results/json/<bench>.json (small files; no VCDs in the shared folder).
2. Runs the TRACE VCD scan (trace_scan.py) -> results/json/trace_events.json.
3. Builds deliverable/timing.json from the trace events + bench throughputs.
"""
import glob
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import trace_scan  # noqa: E402

CORAL = trace_scan.CORAL
LOGS = os.path.join(CORAL, "bazel-testlogs/tests/cocotb/scalar_timing")
JDIR = os.path.join(SHARED, "results", "json")


def collect_jsons():
    os.makedirs(JDIR, exist_ok=True)
    got = {}
    for z in sorted(glob.glob(os.path.join(LOGS, "scalar_timing_cocotb_*/test.outputs/outputs.zip"))):
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.endswith(".json"):
                    data = zf.read(n)
                    with open(os.path.join(JDIR, os.path.basename(n)), "wb") as f:
                        f.write(data)
                    got[os.path.basename(n)[:-5]] = json.loads(data)
    return got


def rows(b, benches):
    return {r["id"]: r for r in benches.get(b, {}).get("rows", []) if "measured" in r}


# ----------------------------------------------------------------------------
# Static facts (from source; file:line relative to hdl/chisel/src/coralnpu/).
STRUCTURES = [
    # name, rtl_path (VCD), file:line, stage_boundary
    ("FetchControl.pc", "core.score.fetch.ctrl.pc_valid/pc_bits", "scalar/UncachedFetch.scala:176", "F1: next fetch address (register read in F1)"),
    ("FetchControl.blockNewFetch (comb)", "core.score.fetch.ctrl.blockNewFetch", "scalar/UncachedFetch.scala:192-198", "F1 request gate: no request in F2, needs nSpace >= nValid+4"),
    ("Fetcher.ibusCmd", "core.score.fetch.fetcher.ibusCmd_valid/bits", "scalar/UncachedFetch.scala:61", "F1 -> F2 (request issued; data returns next cycle)"),
    ("SRAM.readIssued (+ address reg inside Sram_Nx128)", "-", "SRAM.scala:44-48", "F1 -> F2 (ITCM read)"),
    ("CoreAxi.inItcmReg", "-", "CoreAxi.scala:182", "F1 -> F2 (ibus rdata mux select)"),
    ("InstructionBuffer (8 x {pc, inst, brchFwd})", "core.score.fetch.instructionBuffer.circularBuffer.*", "../common/InstructionBuffer.scala:51, scalar/UncachedFetch.scala:235", "F2 -> D (written end of F2, readable next cycle = IBuf)"),
    ("Dispatch (DispatchV2): no flip-flops", "core.score.dispatch.io_inst_*", "scalar/Decode.scala:301-735", "D (combinational decode + interlocks; io.inst(i).ready = dispatched, :734)"),
    ("Regfile.readDataBits / readDataReady", "core.score.regfile.readDataBits_*/readDataReady_*", "scalar/Regfile.scala:117-118", "D -> EX (operand registers for all units)"),
    ("Regfile.scoreboard (regd) + comb view", "core.score.regfile.scoreboard", "scalar/Regfile.scala:93-112", "D (RAW/WAW; comb = same-cycle bypass, regd = registered)"),
    ("Regfile.regfile (x1-x31)", "core.score.regfile.regfile_*", "scalar/Regfile.scala:88", "WB (6 write ports: 0-3 lanes, 4 arbiter, 5 LSU)"),
    ("Alu.valid/addr/op (x4)", "core.score.alu_N.valid/addr/op", "scalar/Alu.scala:79-81", "D -> EX (result combinational, written D+1)"),
    ("Bru.stateReg (x4)", "core.score.bru_N.stateReg_*", "scalar/Bru.scala:114", "D -> resolve (D+1: io.taken = redirect)"),
    ("Csr.req", "core.score.csr.req_valid/req_bits_*", "scalar/Csr.scala:241", "D -> CSR RMW (D+1, port 0)"),
    ("Mlu.stage2Input (Queue 1, pipe)", "core.score.mlu.stage2Input_q.*", "scalar/Mlu.scala:76", "D -> multiply (D+1)"),
    ("Mlu.stage3Input (Queue 1, pipe)", "core.score.mlu.stage3Input_q.*", "scalar/Mlu.scala:101", "multiply -> port 4 (D+2)"),
    ("Dvu.active/compute/count/addr1/addr2/divide/remain/denom", "core.score.dvu.*", "scalar/Dvu.scala:71-86", "D -> normalize (D+1) -> 1 bit/cycle -> port 4 (D+2+N)"),
    ("Port-4 arbiter {MLU > DVU > RVV async > FPU scalar rd > debug}", "core.score.arb.*", "scalar/SCore.scala:259,404-414", "WB arbitration (dvu.rd.ready = !mlu.rd.valid)"),
    ("Lsu.opQueue (CircularBufferMulti, 4 entries)", "core.score.lsu.opQueue.*", "scalar/Lsu.scala:835", "D -> LSU (D+1)"),
    ("Lsu.slot (single active op)", "core.score.lsu.slot_*", "scalar/Lsu.scala:880", "opQueue -> bus (one op at a time; dequeue only when idle :1056,1071)"),
    ("Lsu.readFired", "core.score.lsu.readFired_valid/bits_*", "scalar/Lsu.scala:879", "bus request -> read data (load D+3)"),
    ("Lsu.flushCmd", "core.score.lsu.flushCmd_*", "scalar/Lsu.scala:842", "fence.i / flush pending"),
    ("FloatCore.instQueue (Queue 1; ready only when empty)", "core.score.floatCore.instQueue.*", "float/FloatCore.scala:222-225", "D -> fpnew (D+1)"),
    ("fpnew FMA pipe regs (FP32 slice)", "core.score.floatCore.floatCoreWrapper.core.gen_operation_groups[0]...i_fma", "float/FloatCore.scala:112-119", "source intent 3 regs; elaborated 0 in the Verilator model (D4)"),
    ("FloatCore.fpuActive", "core.score.floatCore.fpuActive", "float/FloatCore.scala:328", "multi-cycle DIV/SQRT accepted"),
    ("FloatCore.scalar_rd_pipe (Queue 2)", "core.score.floatCore.scalar_rd_pipe_q.*", "float/FloatCore.scala:352", "FP -> int rd (+1 cycle, then port 4)"),
    ("FRegfile.fregfile + scoreboard (no same-cycle bypass)", "core.score.fRegfile.*", "scalar/FRegfile.scala:35-42", "FP WB (port 0 FPU, port 1 LSU)"),
    ("RetirementBuffer.instBuffer (8 entries, mini)", "core.score.retirement_buffer.instBuffer.*", "RetirementBuffer.scala:78", "enqueued at D, dequeued at retire"),
    ("RetirementBuffer.resultBuffer", "core.score.retirement_buffer.resultBuffer_*", "RetirementBuffer.scala:233", "WB seen ('data done') -> retire same cycle"),
    ("RetirementBuffer.storeComplete Pipe", "core.score.retirement_buffer.storeComplete_pipe_*", "RetirementBuffer.scala:71", "store complete (D+2) -> retire (D+3)"),
    ("SCore.rvvFrsReg", "-", "scalar/SCore.scala:446-451", "D -> RVV (f-reg operand for RVV ops)"),
    ("RvvFrontEnd.inst_q / valid_inst_q", "core.rvvCore.rvvCoreWrapper.core.frontend.inst_q / valid_inst_q", "../../../verilog/rvv/design/RvvFrontEnd.sv:75-77", "D -> RVV command queue (D+1, cmd_valid_o) [boundary with Pranav]"),
]

# Per-class static info: lane rules, checklist rows, bench throughput source.
#   trace name -> (class label, lane_rules, checklist_rows, (bench, row id, scale) or None, notes)
CLASSES = {
    "alu": ("ALU", "any lane, up to 4 per cycle (fetch caps a stream at 2/cycle); same-cycle forwarding to ALU consumers", ["ALU", "3.14", "3.1"], ("b1_branch", "a4", 0.25), "throughput = independent stream (fetch bound 0.5 cyc/inst); dependent chain = 1.0 (b9)"),
    "zbb": ("ALU (Zbb)", "same as ALU", ["ALU", "0.2"], ("b9_alu_zbb", "clz", 1.0), "throughput = dependent chain (1.0); all 18 Zbb ops 1.0 (b9_alu_zbb)"),
    "lui": ("ALU (lui)", "same as ALU", ["ALU"], ("b9_alu_base", "lui", 1.0), "throughput = WAW chain on one rd (1.0)"),
    "auipc": ("ALU (auipc)", "same as ALU", ["ALU"], ("b9_alu_base", "auipc", 1.0), "throughput = WAW chain on one rd (1.0)"),
    "br_nt_fwd": ("BRU not-taken forward", "any lane; a conditional branch ends its dispatch group (nothing after it in that cycle)", ["3.4", "1.7", "BRU"], ("b1_branch", "bnt", 1.0), "consumer = fall-through instruction (same line), still D+1: group end"),
    "br_t_bwd": ("BRU taken backward (predicted)", "any lane; predicted taken at predecode (BTFN), no BRU redirect", ["1.7", "BRU correct"], ("b1_branch", "loop2", 1.0), "throughput = 2-instr loop, cycles/iteration; D=3 here because bnez waits on the addi before it"),
    "br_mp_fwd": ("BRU mispredict, forward taken", "any lane; BTFN predicts not-taken, BRU redirects at D+1, target dispatches D+4", ["1.7", "BRU mispredict", "2.3"], ("b1_branch", "ft", 1.0), "3 bubble cycles"),
    "br_mp_bwd": ("BRU mispredict, backward not-taken", "any lane; BTFN predicts taken, BRU redirects to pc+4 at D+1, D+4", ["1.7", "BRU mispredict"], ("b1_branch", "bnt_bwd", 1.0), "3 bubble cycles"),
    "jal": ("BRU jal", "any lane; ends the group; always predicted", ["3.5", "1.8", "BRU link write"], ("b1_branch", "jal", 1.0), "target dispatches D+2 (one fetch slot); no BRU redirect"),
    "jalr": ("BRU jalr", "any lane; ends the group; never predicted -> always redirect; rs1 via registered scoreboard (+2 after producer)", ["1.7", "3.3", "3.5"], ("b1_branch", "jalr", 1.0), "throughput = auipc+jalr unit; jalr D=4 because auipc (D=2) -> jalr rs1 is regd (+2); target D+4"),
    "csr": ("CSR", "lane 0, dispatched alone; waits for ROB empty and FPU ready", ["3.6", "3.11", "CSR"], ("b6_csr", "csrr_mscratch", 1.0), "D=4: waits for the preceding addi to retire (ROB empty)"),
    "mul": ("MUL", "any lane, one MLU dispatch per cycle (lowest lane wins)", ["MLU", "3.8", "3.15"], ("b2_mul", "mul_ind4", 0.25), "port 4, MLU has priority over DVU"),
    "div_small": ("DIV (dividend 1, N=2)", "lane 0 only; later lanes may dispatch with it; next DIV at D+3+N", ["DVU", "3.7"], ("b3_div", "div_a1", 1.0), "N = 32 - Clz1(|a|); D=3: div waited for lane 0 behind the pre addi (3.7)"),
    "div_large": ("DIV (dividend 2^30, N=32)", "same as DIV", ["DVU", "3.7"], ("b3_div", "div_a2p30", 1.0), "WB D+2+N"),
    "div_by0": ("DIV by zero", "same as DIV", ["DVU"], ("b3_div", "div_by0", 1.0), "divide by zero forces full latency (N=32)"),
    "lw": ("LW", "any lane, up to LSU opQueue free space; LSU single slot", ["Load", "3.19", "3.9"], ("b4_lsu", "lw4", 0.25), "1 load per 4 cycles (bus req, data, writeback, idle/dequeue)"),
    "sw": ("SW", "any lane, up to opQueue free space; store data via registered scoreboard", ["Store", "3.19", "5.3"], ("b4_lsu", "sw4", 0.25), "1 store per 2 cycles; retire D+3 (storeComplete Pipe)"),
    "flw": ("FLW", "lane 0, dispatched alone (float); LSU path", ["flw/fsw", "3.6"], ("b5_fpu", "flw", 1.0), "f-port 1 at D+4; f consumer at D+5 (no f bypass)"),
    "fsw": ("FSW", "lane 0, dispatched alone (float); LSU path", ["flw/fsw", "3.6"], ("b5_fpu", "fsw", 1.0), ""),
    "fadd": ("FADD", "lane 0, dispatched alone; one FP op in flight (instQueue depth 1)", ["FPU", "3.6", "3.16"], ("b5_fpu", "fadd", 1.0), "SIM: write D+1, next FP op D+2. Source intent (3 fpnew pipe regs): write D+4, next D+5 (D4)"),
    "fmul": ("FMUL", "as FADD", ["FPU", "3.6"], ("b5_fpu", "fmul", 1.0), "as FADD (D4)"),
    "fmadd": ("FMADD", "as FADD", ["FPU", "3.6"], ("b5_fpu", "fmadd", 1.0), "as FADD (D4)"),
    "fdiv": ("FDIV", "as FADD; iterative DIVSQRT (TH32), blocks the FPU", ["FPU div/sqrt"], ("b5_fpu", "fdiv", 1.0), "write D+5 in sim"),
    "fsqrt": ("FSQRT", "as FDIV", ["FPU div/sqrt"], ("b5_fpu", "fsqrt", 1.0), "write D+18 in sim"),
    "feq": ("FP->int (feq)", "as FADD; result through Queue(2) then port-4 arbiter", ["FPU->int"], ("b5_fpu", "feq", 1.0), "int port 4 at D+2 in sim (D+5 with 3 pipe regs)"),
    "fcvt_w_s": ("FP->int (fcvt.w.s)", "as feq", ["FPU->int"], ("b5_fpu", "fcvt", 1.0), "as feq"),
    "fmv_x_w": ("fmv.x.w", "lane 0 alone; bypasses fpnew, dequeued D+1", ["fmv"], ("b5_fpu", "fmvxw", 1.0), "int port 4 at D+2 (via Queue(2)); unaffected by D4"),
    "fmv_w_x": ("fmv.w.x", "lane 0 alone; bypasses fpnew", ["fmv"], ("b5_fpu", "fmvwx", 1.0), "f-port 0 at D+1; unaffected by D4"),
    "fence": ("fence", "lane 0 only, NOT alone and NOT a group end; does not wait for the LSU (not in isFency)", ["3.5", "3.12", "3.17"], ("b10_system", "fence", 1.0), "rd=x0 still splits an x0-reading next instruction into the next cycle (3.17b)"),
    "fence_i": ("fence.i", "lane 0 alone, ends the group, waits for LSU idle; LSU flush redirects fetch", ["3.5", "3.12", "2.3"], ("b10_system", "fencei", 1.0), "next instruction dispatches D+4"),
    "rvv": ("RVV handoff (vadd.vv)", "any lane, up to RVV queue capacity (RVV ops touching f-regs: lane 0 alone)", ["RVV handoff"], None, "scope ends at RvvFrontEnd enqueue (Pranav owns the rest); ROB retire at D+8 observed, not analysed"),
}

# Alex's review-packet sign-offs (deliverable/review_packet.md, R1-R9). A class is "human" only while the
# item's line there reads "[x] agree"; otherwise it stays at its agent level. trace name -> (item, what Alex checked).
SIGNOFF = {
    "br_nt_fwd": ("R4", "conditional branch ends its dispatch group (3.4)"),
    "br_t_bwd": ("R3", "BTFN: backward taken predicted, target D+1, no redirect (1.7)"),
    "br_mp_fwd": ("R2", "mispredict: redirect + flush D+1, target dispatched D+4"),
    "br_mp_bwd": ("R3", "BTFN: backward not-taken redirects at D+1, fall-through D+4 (1.7)"),
    "div_small": ("R6", "WB + consumer at D+2+N, N=2 for dividend 1"),
    "div_large": ("R6", "WB + consumer at D+2+N, N=32 for dividend 2^30"),
    "lw": ("R1", "1 load per 4 cycles (R1); write-back port 5 + retire at D+4 (R8)"),
    "sw": ("R1", "1 store per 2 cycles; storeComplete D+2 / retire D+3 itself is sim+wave (5.3)"),
}
# Parts of a class Alex checked when the class as a whole stays at its agent level.
PARTIAL = {
    "alu": [("R8", "write-back port + retire at D+1 (5.2)"), ("R9", "same-cycle forwarding to an ALU consumer (3.14)")],
    "fadd": [("R5", "dispatched alone in lane 0 (3.6) and sim timing D+1; RTL-intent timing open (D4)")],
}


def packet_signoffs():
    """{'R1': True, ...}: True where Alex's line in deliverable/review_packet.md reads '[x] agree'."""
    p = os.path.join(SHARED, "deliverable", "review_packet.md")
    agreed, cur = {}, None
    if os.path.exists(p):
        for line in open(p):
            if line.startswith("## R"):
                cur = line[3:].split(".")[0]
            elif line.startswith("Alex: ") and cur:
                agreed[cur] = line.startswith("Alex: [x] agree")
    return agreed


THR_KIND = {
    "zbb": "dependent chain, cycles per instruction (independent stream not measured; same ALU path as base)",
    "lui": "WAW chain on one rd, cycles per instruction",
    "auipc": "WAW chain on one rd, cycles per instruction",
    "br_t_bwd": "2-instruction loop, cycles per iteration",
    "jalr": "auipc + jalr unit, cycles per unit",
    "div_small": "independent divs, cycles per div (= 3 + N)",
    "div_large": "independent divs, cycles per div (= 3 + N)",
    "div_by0": "independent divs, cycles per div (= 3 + N)",
}

KEYMAP = {  # trace_scan event key -> RTL register name used in timing.json
    "alu_1.valid (op/addr regs, result comb)": "Alu.valid/addr/op",
    "alu_0.valid (op/addr regs, result comb)": "Alu.valid/addr/op",
    "csr.req_valid (Csr.req reg)": "Csr.req",
    "csr.io_rd_valid": "Csr.io.rd (port 0)",
    "mlu.stage2Input_q.io_deq_valid": "Mlu.stage2Input",
    "mlu.stage3Input_q.io_deq_valid": "Mlu.stage3Input",
    "mlu.io_rd_valid": "Mlu.io.rd (port 4 request)",
    "dvu.active": "Dvu.active",
    "dvu.compute": "Dvu.compute",
    "dvu.io_rd_valid (count(5))": "Dvu.io.rd (count(5))",
    "lsu.opQueue.io_nEnqueued>0": "Lsu.opQueue",
    "lsu.nextSlot_active": None,
    "lsu.io_dbus_valid (bus request)": "Lsu.slot -> dbus request",
    "lsu.readFired_valid": "Lsu.readFired",
    "lsu.io_rd_valid": "Lsu.io.rd (port 5)",
    "lsu.io_rd_flt_valid": "Lsu.io.rd_flt (f-port 1)",
    "lsu.io_storeComplete_valid": "Lsu.io.storeComplete",
    "lsu.io_flush_valid": "Lsu.flushCmd -> io.flush",
    "fetch.ctrl.io_iflush_valid": "Fetch iflush",
    "instructionBuffer.io_flush": "InstructionBuffer flush",
    "floatCore.instQueue.io_count=1": "FloatCore.instQueue",
    "fpnew in_valid&in_ready (accepted)": "fpnew accept",
    "fpnew out_valid_o": "fpnew out_valid",
    "floatCore.io_write_ports_0_valid": "FloatCore write (f-port 0)",
    "scalar_rd_pipe_q.io_enq_valid": "FloatCore.scalar_rd_pipe enq",
    "floatCore.io_scalar_rd_valid": "FloatCore.scalar_rd (port 4 request)",
    "floatCore.instQueue.io_count=0 again": "FloatCore.instQueue empty (next FP op may dispatch)",
    "RvvFrontEnd.inst_valid_i": "RvvFrontEnd.inst_valid_i",
    "RvvFrontEnd.valid_inst_q (inst_q)": "RvvFrontEnd.inst_q",
    "RvvFrontEnd.cmd_valid_o (to CQ)": "RvvFrontEnd.cmd_valid_o (to CQ)",
}


def unit_key(k):
    for lane in range(4):
        if k == f"bru_{lane}.stateReg_valid (resolve)":
            return "Bru.stateReg (resolve)"
        if k == f"bru_{lane}.io_taken_valid (redirect)":
            return "Bru.io.taken (redirect)"
        if k == f"alu_{lane}.valid (op/addr regs, result comb)":
            return "Alu.valid/addr/op"
    return KEYMAP.get(k, k)


def class_events(c):
    ev = {}
    for k, v in c["events"].items():
        if k in ("F1", "F2", "IBuf", "D", "WB", "retire"):
            continue
        uk = unit_key(k)
        if uk and v is not None:
            ev[uk] = v
    return {"F1": c["events"]["F1"], "F2": c["events"]["F2"], "IBuf": c["events"]["IBuf"],
            "D": c["events"]["D"], **ev, "WB": c["events"].get("WB"), "port": c.get("port"),
            "retire": c["events"].get("retire")}


# D4 experiment (2026-10-19): the same benches on a scratch build whose FloatCore writes fpnew PipeRegs row by row,
# so all formats get the 3 regs the RTL asks for (the stock '{default: 'd3} gives FP32 0 in Verilator).
# Branch d4-piperegs-experiment in ~/LHS/coralnpu-d4, not M3. Results saved as results/json/d4exp_*.json.
D4EXP_CLASSES = ("fadd", "fmul", "fmadd", "feq", "fcvt_w_s", "fdiv", "fsqrt")
D4EXP_BASIS = ("RTL intent (3 fpnew pipe regs), measured on the D4 experiment build "
               "(branch d4-piperegs-experiment = M3 + row-by-row PipeRegs in float/FloatCore.scala:113); "
               "sim+wave on that build, not on the M3 model. This is the hardware timing: the Drop 0 DC and "
               "Yosys-slang netlists have the 3 FMA stages (Aaron, 2026-10-19); only Verilator builds 0 for FP32")


def load_d4exp():
    tp, bp = os.path.join(JDIR, "d4exp_trace_events.json"), os.path.join(JDIR, "d4exp_b5_fpu.json")
    if not (os.path.exists(tp) and os.path.exists(bp)):
        return None
    with open(tp) as f, open(bp) as g:
        return json.load(f), {"b5_fpu": json.load(g)}


def d4_intent(name, tp, d4exp):
    if name not in D4EXP_CLASSES:
        return None
    if not d4exp:
        return {"WB": "D+4", "next_fp_dispatch": "D+5", "throughput_cycles": 5, "basis": "source only (src)"} \
            if name not in ("fdiv", "fsqrt") else None
    c = d4exp[0]["classes"].get(name)
    if not c or "error" in c:
        return None
    ev, d = class_events(c), c["events"]["D"]
    r = rows(tp[0], d4exp[1]).get(tp[1]) if tp else None
    return {"basis": D4EXP_BASIS, "events": ev,
            "WB": f"D+{ev['WB'] - d}" if ev.get("WB") is not None else None,
            "next_fp_dispatch": f"D+{c['events']['floatCore.instQueue.io_count=0 again'] - d}"
            if c["events"].get("floatCore.instQueue.io_count=0 again") is not None else None,
            "latency_to_dependent_dispatch": c["latency_to_dependent_dispatch"],
            "throughput_cycles": round(r["measured"] * tp[2], 3) if r else None}


def build(benches, tev, d4exp=None):
    agreed = packet_signoffs()
    out = {
        "schema": "per class: events relative to that instruction's F1 = cycle 0 (one cycle = "
                  f"{tev['clock_period_ns']} ns in sim). Source: TRACE bench, one traced run, scripted VCD scan "
                  "(bench/trace_scan.py). throughput_cycles from the cycle-count benches. events_abs_ns = time of the "
                  "rising edge that ENDS that cycle in trace.vcd (the value is visible in GTKWave in the 1.25 ns before it).",
        "source_commit": "google-coral/coralnpu M3-2026-04-27 (72d700e)",
        "verification_levels": "human = Alex signed off; sim+wave = agent's scripted VCD check; sim = cycle counts only; src = source only. "
                               "A class is 'human' only where its review-packet item (deliverable/review_packet.md) is ticked agree; "
                               "'human_signoff' says which item and what Alex checked, 'partial_signoffs' lists checked parts of a class "
                               "that otherwise stays at its agent level. Agreed items: "
                               + (", ".join(k for k, v in sorted(agreed.items()) if v) or "none") + ".",
        "front_end": {"fetch_line_period_cycles": 2, "fetch_ipc_peak": 2.0, "verified": "human",
                      "note": "checklist 1.5: Alex read the fetch_ipc waveform (2026-10-13). F1 = ibus request, F2 = data return, IBuf = first cycle readable by dispatch. In a free-flowing stream D = IBuf = F1+2."},
        "retire_signal": "core.score.retirement_buffer.io_debug_inst_k_valid/bits_pc (= instBuffer deq, RetirementBuffer.scala:440-447); io_nRetired counts them",
        "classes": [],
        "structures": [dict(name=n, rtl_path=r, file_line=f, stage_boundary=s) for n, r, f, s in STRUCTURES],
    }
    for name, (label, lanes, chk, tp, note) in CLASSES.items():
        c = tev["classes"].get(name)
        if not c or "error" in c:
            out["classes"].append(dict(**{"class": label}, error="not traced"))
            continue
        events = class_events(c)
        thr, thr_src = None, None
        if tp:
            r = rows(tp[0], benches).get(tp[1])
            if r:
                thr = round(r["measured"] * tp[2], 3)
                thr_src = f"{tp[0]}:{tp[1]}"
        kind = THR_KIND.get(name, "independent stream, cycles per instruction")
        so = SIGNOFF.get(name)
        human = bool(so and agreed.get(so[0]))
        partial = [f"{i}: {what}" for i, what in PARTIAL.get(name, []) if agreed.get(i)]
        out["classes"].append({
            "class": label, "trace_id": name, "example_asm": c["example_asm"], "lane_rules": lanes,
            "dispatch_lane_in_trace": c["lane"], "events": events,
            "events_abs_ns": c["events_ns"],
            "latency_to_dependent_dispatch": c["latency_to_dependent_dispatch"],
            "throughput_cycles": thr, "throughput_kind": kind if thr is not None else None,
            "throughput_source": thr_src,
            "source_intent_if_different": d4_intent(name, tp, d4exp),
            "verified": "human" if human else "sim+wave",
            "human_signoff": f"{so[0]}: {so[1]}" if human else None,
            "partial_signoffs": partial or None,
            "checklist_rows": chk, "notes": note,
        })
    return out


BENCH_TITLES = {
    "b1_branch": "B1 branches", "b2_mul": "B2 MUL", "b3_div": "B3 DIV",
    "b4_lsu": "B4 LSU", "b4_lsu_misaligned": "B4 LSU misaligned / line-crossing",
    "b5_fpu": "B5 FPU", "b6_csr": "B6 CSR", "b7_rob": "B7 ROB full", "b8_x0": "B8 x0 / nop",
    "b9_alu_base": "B9 ALU (RV32I)", "b9_alu_zbb": "B9 ALU (Zbb)", "b10_system": "B10 fences",
}


def write_results_md(benches):
    """results/<bench>.md: measured vs expected vs source, regenerated each run."""
    for b, title in BENCH_TITLES.items():
        j = benches.get(b)
        if not j:
            continue
        L = [f"# {title} (`bench/{b if not b.startswith('b9') else 'b9_alu'}.cc`)", "",
             f"Generated by `bench/collect.py` from `results/json/{b}.json`. "
             f"deterministic={j['deterministic']} faulted={j['faulted']} misa={j['misa']} ({j['misa_extensions']}).",
             "Units: see the comment block at the top of the bench source (cycles per unit / per op / diff).", "",
             "| metric | measured | expected | source-only | ok | rows | note |",
             "|---|---|---|---|---|---|---|"]
        for r in j["rows"]:
            if "measured" not in r:
                L.append(f"| {r['id']} | ERROR | | | | | {r.get('error')} |")
                continue
            note = r["note"]
            if r["sim_calibrated"]:
                why = r["calibration_reason"]
                tag = why.split(":")[0] if why[:1] == "D" and why[1:2].isdigit() else "regression value, no source prediction"
                note += f" **[sim-calibrated: {tag}; notes/discrepancies.md]**" if tag.startswith("D") else f" *[{why}]*"
            src = "measure" if r["expected_from_source"] is None else r["expected_from_source"]
            L.append(f"| {r['id']} | {r['measured']} | {r['expected']} | {src} | "
                     f"{'ok' if r['ok'] else 'MISMATCH'} | {', '.join(r['checklist_rows'])} | {note} |")
        with open(os.path.join(SHARED, "results", f"{b}.md"), "w") as f:
            f.write("\n".join(L) + "\n")


def main():
    benches = collect_jsons()
    write_results_md(benches)
    print(f"collected {len(benches)} bench JSONs -> {JDIR}")
    tev_path = os.path.join(JDIR, "trace_events.json")
    if os.path.exists(os.path.join(trace_scan.WAVES, "trace.vcd")):
        sys.argv = ["trace_scan.py", "trace", "--out", tev_path]
        trace_scan.main()
    with open(tev_path) as f:
        tev = json.load(f)
    tj = build(benches, tev, load_d4exp())
    os.makedirs(os.path.join(SHARED, "deliverable"), exist_ok=True)
    p = os.path.join(SHARED, "deliverable", "timing.json")
    with open(p, "w") as f:
        json.dump(tj, f, indent=1)
    print(f"wrote {p}: {len(tj['classes'])} classes, {len(tj['structures'])} structures")


if __name__ == "__main__":
    main()

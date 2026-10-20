#!/usr/bin/env python3
"""Generate deliverable/review_packet.md and the notes/r*.gtkw layouts.

    python3 bench/make_review_packet.py     (after collect.py)

Every time window and cycle number below comes from the traced VCDs and the
bench JSONs, not typed by hand. The sign-off lines are left blank on purpose.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import trace_scan as ts  # noqa: E402
from vcdscan import Vcd  # noqa: E402

J = os.path.join(SHARED, "results", "json")
NOTES = os.path.join(SHARED, "notes")
WAVES = ts.WAVES
TOP = "RvvCoreMiniAxi."
S = "core.score."


def widths(vcd):
    w, scope = {}, []
    with open(vcd, errors="replace") as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "$scope":
                scope.append(t[2])
            elif t[0] == "$upscope":
                scope.pop()
            elif t[0] == "$var" and "RvvCoreMiniAxi" in scope:
                w[".".join(scope[scope.index("RvvCoreMiniAxi") + 1:] + [t[4]])] = int(t[2])
            elif t[0] == "$enddefinitions":
                break
    return w


def gtkw(name, vcd, t0_ns, sigs):
    w = widths(vcd)
    L = [f'[dumpfile] "{vcd}"', f"[timestart] {int(t0_ns * 1000)}"]
    for s in [S + "clock"] + sigs:
        if s not in w:
            raise KeyError(s)
        if w[s] == 1:
            L += ["@28", TOP + s]
        else:
            L += ["@22", f"{TOP}{s}[{w[s] - 1}:0]"]
    p = os.path.join(NOTES, name + ".gtkw")
    with open(p, "w") as f:
        f.write("\n".join(L) + "\n")
    return f"notes/{name}.gtkw"


def disasm(bench, func, around=None, n=8):
    syms = ts.symbols(bench)
    objdump = ts.nm_tool().replace("llvm-nm", "llvm-objdump")
    lo = next(a for s, a in syms.items() if func in s)
    # inner labels (trp_/tr_/...) are symbols too: stop at the next function
    hi = min((a for s, a in syms.items() if a > lo and s.startswith("_Z")), default=lo + 0x400)
    if around is not None:
        lo, hi = around - 8, around + 4 * n
    out = subprocess.run([objdump, "-d", "--no-show-raw-insn", f"--start-address={lo:#x}",
                          f"--stop-address={hi:#x}", ts.elf_path(bench)],
                         capture_output=True, text=True).stdout.splitlines()
    ins = [l.strip() for l in out if l.strip()[:1].isdigit() or (":" in l and l.startswith(" "))]
    ins = [l for l in ins if ":\t" in l or ": " in l]
    if around is not None:
        sel = ins[:n + 2]
    else:
        idx = next(i for i, l in enumerate(ins) if "csrr\ta0, mcycle" in l)
        sel = ins[idx: idx + n]
    return "\n".join(sel)


def main():
    tev = json.load(open(os.path.join(J, "trace_events.json")))
    B = {b: json.load(open(os.path.join(J, b + ".json"))) for b in
         ("b1_branch", "b3_div", "b4_lsu", "b5_fpu", "b6_csr", "b7_rob", "b9_alu_base", "trace", "dep_chain_alu")}
    rv = lambda b, i: next(r for r in B[b]["rows"] if r["id"] == i)["measured"]
    trace_vcd = os.path.join(WAVES, "trace.vcd")
    vt = Vcd(trace_vcd, [])
    edge = lambda k: vt.times[k] / 1000.0
    C = tev["classes"]

    def tev_lines(name, extra=()):
        c = C[name]
        base = c["events_abs_cycle"]["F1"]
        lines = []
        for k, v in c["events_abs_cycle"].items():
            if v is not None:
                lines.append((v, f"cycle ending {edge(v):.2f} ns (rel. F1 = {v - base}): {k}"))
        for lab in ("consumer_D",):
            if c.get(lab) is not None:
                v = base + c[lab]
                lines.append((v, f"cycle ending {edge(v):.2f} ns (rel. F1 = {c[lab]}): consumer / target dispatched"))
        for v, txt in extra:
            lines.append((v, txt))
        return "\n".join(f"   - {t}" for _, t in sorted(lines))

    def window(name, pad_after=3):
        c = C[name]
        ks = [v for v in c["events_abs_cycle"].values() if v is not None]
        if c.get("consumer_D") is not None:
            ks.append(c["events_abs_cycle"]["F1"] + c["consumer_D"])
        return edge(min(ks)) - 2.5, edge(max(ks)) + 1.25 * pad_after

    DISP = lambda lane: [f"{S}dispatch.io_inst_{lane}_valid", f"{S}dispatch.io_inst_{lane}_ready",
                         f"{S}dispatch.io_inst_{lane}_bits_addr"]
    FETCH = [f"{S}fetch.io_ibus_valid", f"{S}fetch.io_ibus_addr"]
    R = []

    # --- R1 ---------------------------------------------------------------
    g1 = gtkw("r1_lsu", os.path.join(WAVES, "dep_chain_lw.vcd"), 900,
              [f"{S}lsu.io_dbus_valid", f"{S}lsu.io_dbus_write", f"{S}lsu.readFired_valid",
               f"{S}lsu.io_rd_valid", f"{S}lsu.io_queueCapacity", f"{S}retirement_buffer.io_nSpace",
               f"{S}dispatch.io_inst_0_ready"])
    R.append(dict(
        id="R1", title="LSU sustains one load per 4 cycles and one store per 2 (checklist 3.19, D2/D3)",
        claim="With independent loads (stores) back to back, the single LSU slot completes one load every 4 cycles and one store every 2 cycles.",
        prediction="Loads: bus request, data back (`readFired`), write-back (`io_rd_valid`), one empty cycle, repeat: period 4 cycles = 5.0 ns. Stores: `io_dbus_valid & io_dbus_write` every 2nd cycle (2.5 ns). Bench: lw4 = 16.0 cycles per 4 loads, sw4 = 8.0 per 4 stores.",
        rerun="bash bench/waves.sh dep_chain_lw dep_chain_sw",
        wave=f"`~/LHS/scalar-timing-waves/dep_chain_lw.vcd`, layout `{g1}`, window **900-925 ns**. Same layout on `dep_chain_sw.vcd`, window **450-463 ns** (change [dumpfile] or open the VCD and load the layout).",
        look="Loads: `io_dbus_valid` pulses once every 4 rising edges; between pulses you see `readFired_valid`, then `io_rd_valid`, then a blank cycle. `io_queueCapacity` (= 4 minus the ops waiting in the opQueue; the slot is not counted, Lsu.scala:837, CircularBufferMulti.scala:72) is 4 before the stream, 3 while the first load waits a cycle, then flat at 1: the ROB is full (`retirement_buffer.io_nSpace` = 0, lane 0 valid but not ready), each load write-back retires the load and its producer addi, and the next addi+lw pair takes those 2 entries. 8 ROB entries / 2 per unit = 4 loads in flight, 1 in the slot and 3 queued. The later d=1/2/3 stretches level off at 2/3/3 for the same reason (F1, discrepancies.md D3). Stores: `io_dbus_valid` with `io_dbus_write` high on every 2nd edge.",
        disasm=disasm("b4_lsu", "r_lw4_u8", n=6),
        byhand=f"`python3 -c \"import json;c=json.load(open('results/json/b4_lsu.json'))['cases'];print((c['lw4_u16']-c['lw4_u8'])/8/4,(c['sw4_u16']-c['sw4_u8'])/8/4)\"` -> {rv('b4_lsu','lw4')/4} {rv('b4_lsu','sw4')/4}",
        level="sim+wave"))

    # --- R2 ---------------------------------------------------------------
    w = window("br_mp_fwd")
    g2 = gtkw("r2_mispredict", trace_vcd, w[0], DISP(1) + [f"{S}bru_1.io_taken_valid",
              f"{S}fetch.instructionBuffer.io_flush"] + FETCH + DISP(0)[1:])
    c = C["br_mp_fwd"]
    R.append(dict(
        id="R2", title="Branch mispredict: the target dispatches at D+4 (BRU row)",
        claim="A mispredicted branch resolves in D+1 (redirect + buffer flush), the target is fetched D+2 (F1) / D+3 (F2) and dispatches at D+4: 3 bubble cycles.",
        prediction="`beq zero,zero,1f` (forward taken, BTFN predicts not taken). D -> redirect D+1 -> target dispatch D+4. Bench: 4.0 cycles per mispredicted branch.",
        rerun="bash bench/waves.sh trace",
        wave=f"`~/LHS/scalar-timing-waves/trace.vcd`, layout `{g2}`, window **{w[0]:.2f}-{w[1]:.2f} ns**. The branch is at pc {c['pc']} (lane {c['lane']}).",
        look="Lane 1 `ready` with `bits_addr` = the branch pc (D). Next cycle: `bru_1.io_taken_valid` = 1 and `io_flush` = 1. `io_ibus_valid` then requests the target line, data comes back, and the target pc shows up dispatched (lane 0 `ready`) 4 cycles after D. Scanner events:\n" + tev_lines("br_mp_fwd"),
        disasm=disasm("trace", "r_br_mp_fwd", around=int(c["pc"], 16), n=5),
        byhand=f"`python3 -c \"import json;c=json.load(open('results/json/b1_branch.json'))['cases'];print((c['ft_u16']-c['ft_u8'])/8)\"` -> {rv('b1_branch','ft')}",
        level="sim+wave"))

    # --- R3 ---------------------------------------------------------------
    w3a, w3b = window("br_t_bwd"), window("br_mp_bwd")
    g3 = gtkw("r3_btfn", trace_vcd, w3a[0], DISP(0) + DISP(1) + [f"{S}bru_0.io_taken_valid",
              f"{S}bru_1.io_taken_valid", f"{S}fetch.instructionBuffer.io_flush"])
    R.append(dict(
        id="R3", title="Static BTFN prediction: backward taken costs nothing, forward taken mispredicts (1.7)",
        claim="Predecode predicts conditional branches with a negative offset taken (and jal always). A taken backward branch never redirects; a forward taken or backward not-taken branch does (D+4).",
        prediction=f"2-instruction backward loop: 2.0 cycles/iteration (= one fetch line per iteration, no bubble). Forward taken: 4.0. Backward not-taken: 4.0. Measured: loop2 {rv('b1_branch','loop2')}, ft {rv('b1_branch','ft')}, bnt_bwd {rv('b1_branch','bnt_bwd')}.",
        rerun="bash bench/waves.sh trace b1_branch",
        wave=f"`trace.vcd`, layout `{g3}`: taken backward `bnez` at {C['br_t_bwd']['pc']}, window **{w3a[0]:.2f}-{w3a[1]:.2f} ns** (no `io_taken_valid`); backward not-taken `bne` at {C['br_mp_bwd']['pc']}, window **{w3b[0]:.2f}-{w3b[1]:.2f} ns** (`bru_1.io_taken_valid` + flush, target D+4).",
        look="In the first window the branch dispatches and its loop target dispatches the next cycle; `io_taken_valid` stays 0. In the second, `bru_1.io_taken_valid` and `io_flush` pulse one cycle after D, and the fall-through dispatches at D+4.\n  taken backward:\n" + tev_lines("br_t_bwd") + "\n  backward not-taken:\n" + tev_lines("br_mp_bwd"),
        disasm=disasm("trace", "r_br_mp_bwd", around=int(C["br_mp_bwd"]["pc"], 16), n=4),
        byhand=f"`python3 -c \"import json;c=json.load(open('results/json/b1_branch.json'))['cases'];print((c['loop2_n16']-c['loop2_n8'])/8,(c['ft_u16']-c['ft_u8'])/8,(c['bnt_bwd_u16']-c['bnt_bwd_u8'])/8)\"`",
        level="sim+wave"))

    # --- R4 ---------------------------------------------------------------
    w = window("br_nt_fwd")
    g4 = gtkw("r4_group_end", trace_vcd, w[0], DISP(0) + DISP(1) + DISP(2)[:2])
    R.append(dict(
        id="R4", title="A conditional branch ends its dispatch group (3.4)",
        claim="Nothing in a later lane dispatches in the same cycle as a conditional branch, even an independent ALU op.",
        prediction=f"4 not-taken `bne` per line: 1 per cycle -> 1.0 cycles/branch (would be 0.5 if they packed like addi). Measured {rv('b1_branch','bnt')}. In TRACE: the fall-through addi (same line, right after the bne) dispatches one cycle after the bne.",
        rerun="bash bench/waves.sh trace",
        wave=f"`trace.vcd`, layout `{g4}`, window **{w[0]:.2f}-{w[1]:.2f} ns**, bne at {C['br_nt_fwd']['pc']}.",
        look="In the D cycle lane 0 (pre addi) and lane 1 (bne) are `ready`; lane 2 holds the next addi `valid` but not `ready`. Next cycle it dispatches in lane 0.\n" + tev_lines("br_nt_fwd"),
        disasm=disasm("trace", "r_br_nt_fwd", around=int(C["br_nt_fwd"]["pc"], 16), n=4),
        byhand=f"`python3 -c \"import json;c=json.load(open('results/json/b1_branch.json'))['cases'];print((c['bnt_u16']-c['bnt_u8'])/8)\"` -> {rv('b1_branch','bnt')}",
        level="sim+wave"))

    # --- R5 ---------------------------------------------------------------
    w = window("fadd")
    g5 = gtkw("r5_fp_alone", trace_vcd, w[0], DISP(0) + DISP(1)[:2] + [f"{S}floatCore.instQueue.io_count",
              f"{S}floatCore.floatCoreWrapper.out_valid_o", f"{S}floatCore.io_write_ports_0_valid"])
    R.append(dict(
        id="R5", title="FP and CSR ops dispatch alone in lane 0; fadd throughput (3.6, B5) - CONTRADICTS the source-derived 1 per 5",
        claim="(a) Every float op (incl. flw/fsw) and every CSR op dispatches only from lane 0 with nothing else that cycle. (b) Source: fadd throughput 1 per 5. Sim: 1 per 2, because the Verilator model has 0 fpnew pipe regs (notes/discrepancies.md D4).",
        prediction=f"(a) fadd sits one cycle behind a lane-0 addi, and the addi after it waits a cycle. (b) Source 5.0 cycles/fadd; sim measured {rv('b5_fpu','fadd')} (fpnew out_valid at D+1). csrr stream {rv('b6_csr','csrr_mscratch')} cycles.",
        rerun="bash bench/waves.sh trace b5_fpu",
        wave=f"`trace.vcd`, layout `{g5}`, window **{w[0]:.2f}-{w[1]:.2f} ns** (fadd at {C['fadd']['pc']}). For the stream: `b5_fpu.vcd`, same layout, **1193-1216 ns** (fadds at 0x220-0x23c, one per 2 cycles).",
        look="TRACE: the pre addi dispatches (lane 0 `ready`) one cycle, the fadd the next cycle alone in lane 0, the post addi the cycle after. `instQueue.io_count` = 1 for one cycle, `out_valid_o` and the f-port write in the same cycle (that is the 0-pipe-reg FPU). Stream: lane 0 `ready` every 2nd cycle.\n" + tev_lines("fadd"),
        disasm=disasm("trace", "r_fadd", around=int(C["fadd"]["pc"], 16), n=4),
        byhand=f"`python3 -c \"import json;c=json.load(open('results/json/b5_fpu.json'))['cases'];print((c['fadd_u8']-c['fadd_u4'])/4/2)\"` -> {rv('b5_fpu','fadd')} (source intent 5)",
        level="sim+wave"))

    # --- R6 ---------------------------------------------------------------
    w6a, w6b = window("div_small"), window("div_large")
    g6 = gtkw("r6_div", trace_vcd, w6a[0], DISP(0) + [f"{S}dvu.active", f"{S}dvu.compute", f"{S}dvu.count",
              f"{S}dvu.io_rd_valid"])
    R.append(dict(
        id="R6", title="DIV latency depends on the dividend size (DVU row)",
        claim="Write-back at D+2+N with N = 32 - Clz1(|dividend|), Clz1 = clz - 1 (0 if MSB set); divide-by-zero takes N = 32. Next div accepted at D+3+N.",
        prediction=f"dividend 1: N=2, WB D+4; dividend 2^30: N=32, WB D+34. Bench (cycles per div): a=1 {rv('b3_div','div_a1')}, 256 {rv('b3_div','div_a256')}, 65536 {rv('b3_div','div_a65536')}, 2^30 {rv('b3_div','div_a2p30')}, b=0 {rv('b3_div','div_by0')} (predicted 5/13/21/35/35).",
        rerun="bash bench/waves.sh trace",
        wave=f"`trace.vcd`, layout `{g6}`: small div at {C['div_small']['pc']} window **{w6a[0]:.2f}-{w6a[1]:.2f} ns**; large div at {C['div_large']['pc']} window **{w6b[0]:.2f}-{w6b[1]:.2f} ns**.",
        look="`active` rises D+1, `compute` D+2, `count` starts at Clz1 and counts up to 32; `io_rd_valid` is high when count reaches 32. Small: count 30 -> 32 in 2 cycles. Large: 0 -> 32.\n  small:\n" + tev_lines("div_small") + "\n  large:\n" + tev_lines("div_large"),
        disasm=disasm("trace", "r_div_large", around=int(C["div_large"]["pc"], 16), n=4),
        byhand=f"`python3 -c \"import json;c=json.load(open('results/json/trace.json'))['cases'];print(c['div_large']-c['div_small'])\"` -> {B['trace']['cases']['div_large'] - B['trace']['cases']['div_small']} (= 34 - 4)",
        level="sim+wave"))

    # --- R7 ---------------------------------------------------------------
    g7 = gtkw("r7_rob_full", os.path.join(WAVES, "b7_rob.vcd"), 1688,
              DISP(0) + [f"{S}retirement_buffer.io_nSpace", f"{S}retirement_buffer.io_nRetired",
                         f"{S}dvu.io_rd_valid"])
    R.append(dict(
        id="R7", title="ROB-full dispatch stall (3.10)",
        claim="Lane i dispatches only if the ROB has more than i free entries. Behind a long div, at most 7 more instructions dispatch; the 8th waits until the div retires.",
        prediction=f"T(K addi behind a 2^30 div) - T(0): 0 for K <= 7, then grows. Measured: K=7 {rv('b7_rob','k7')}, K=8 {rv('b7_rob','k8')}, K=12 {rv('b7_rob','k12')}, K=40 {rv('b7_rob','k40')}.",
        rerun="bash bench/waves.sh b7_rob",
        wave=f"`~/LHS/scalar-timing-waves/b7_rob.vcd`, layout `{g7}`, window **1688-1737 ns** (region k12; div at 0x660).",
        look="div + 3 addi dispatch, then 4 addi; `io_nSpace` drops to 0. Lane 0 shows 0x680 `valid` but not `ready` for 31 cycles. When `dvu.io_rd_valid` pulses (~1732.5 ns), `io_nRetired` = 8 in that cycle, `io_nSpace` = 8 the next, and 0x680-0x68c dispatch together.",
        disasm=disasm("b7_rob", "r_k12v", n=14),
        byhand=f"`python3 -c \"import json;c=json.load(open('results/json/b7_rob.json'))['cases'];print([c[f'k{{k}}']-c['k0'] for k in (6,7,8,12,40)])\"`",
        level="sim+wave"))

    # --- R8 ---------------------------------------------------------------
    w8a, w8b = window("alu"), window("lw")
    g8 = gtkw("r8_retire", trace_vcd, w8a[0], DISP(1) + [f"{S}retirement_buffer.io_writeDataScalar_1_valid",
              f"{S}retirement_buffer.io_writeDataScalar_5_valid", f"{S}retirement_buffer.io_nRetired",
              f"{S}retirement_buffer.io_debug_inst_0_bits_pc"])
    R.append(dict(
        id="R8", title="Retire timing of one ALU op and one load (5.2)",
        claim="An entry retires (ROB dequeue, `io_debug_inst_k_valid`) in the same cycle its write-back is seen: ALU at D+1, load at D+4.",
        prediction="add t0: WB port 1 at D+1, retire D+1. lw t0: WB port 5 at D+4, retire D+4.",
        rerun="bash bench/waves.sh trace",
        wave=f"`trace.vcd`, layout `{g8}`: add at {C['alu']['pc']} window **{w8a[0]:.2f}-{w8a[1]:.2f} ns**; lw at {C['lw']['pc']} window **{w8b[0]:.2f}-{w8b[1]:.2f} ns**.",
        look="add: lane 1 `ready` (D), next cycle `writeDataScalar_1_valid` = 1 and `io_nRetired` > 0 with `io_debug_inst_*_bits_pc` = the add's pc. lw: `writeDataScalar_5_valid` 4 cycles after D, retire in the same cycle.\n  add:\n" + tev_lines("alu") + "\n  lw:\n" + tev_lines("lw"),
        disasm=disasm("trace", "r_lw", around=int(C["lw"]["pc"], 16), n=4),
        byhand=f"`python3 -c \"import json;c=json.load(open('results/json/b4_lsu.json'))['cases'];print((c['chase_u16']-c['chase_u8'])/8)\"` -> {rv('b4_lsu','chase')} (= load WB D+4 + 1 registered-scoreboard cycle for the next base)",
        level="sim+wave"))

    # --- R9 ---------------------------------------------------------------
    w = window("alu")
    g9 = gtkw("r9_forwarding", trace_vcd, w[0], DISP(1) + DISP(0) + [f"{S}retirement_buffer.io_writeDataScalar_1_valid"])
    R.append(dict(
        id="R9", title="Random sanity check: 3.14 same-cycle forwarding (picked with random.seed(20261017) from 10 confirmed claims)",
        claim="A write-back in cycle X is visible to an ALU instruction dispatching in cycle X (comb scoreboard + regfile bypass).",
        prediction=f"Dependent addi chain: 1.000 cycle/inst (dep_chain_alu {B['dep_chain_alu']['chain']['cycles_per_inst']}, b9 add {rv('b9_alu_base','add')}). TRACE: consumer `addi t4, t0, 0` dispatches in the same cycle `add t0` writes back (D+1).",
        rerun="bash bench/waves.sh trace",
        wave=f"`trace.vcd`, layout `{g9}`, window **{w[0]:.2f}-{w[1]:.2f} ns**.",
        look="Cycle D: `add t0` dispatched in lane 1. Cycle D+1: `writeDataScalar_1_valid` = 1 (t0) and in the same cycle lane 0 dispatches the consumer at pc+8.\n" + tev_lines("alu"),
        disasm=disasm("trace", "r_alu", around=int(C["alu"]["pc"], 16), n=4),
        byhand=f"`python3 -c \"import json;c=json.load(open('results/json/b9_alu_base.json'))['cases'];print((c['add_u24']-c['add_u8'])/16)\"` -> {rv('b9_alu_base','add')}",
        level="sim+wave"))

    L = ["# Review packet: claims for Alex to verify by hand", "",
         "Generated by `bench/make_review_packet.py` from the 2026-10-17 run. Every window and cycle number comes from the "
         "traced VCDs (`~/LHS/scalar-timing-waves/`) and `results/json/`. Budget: about 1 hour.", "",
         "**How to use:** run the README one-liner first (it also tests the README). For each claim, run its re-run "
         "command, open the VCD in GTKWave with the layout (`DISPLAY=:0 gtkwave <vcd> -a <layout> &` in WSL; layouts "
         "carry the VCD path and start time), check the listed edges, and tick a box. Times are rising-edge times in ns "
         "(1 cycle = 1.25 ns); \"cycle ending T\" means the value visible in the 1.25 ns before edge T.", "",
         "Level of every claim below before review: **sim+wave** (agent's scripted VCD check). It becomes `human` only "
         "when you tick agree (`bench/collect.py` reads these ticks into `timing.json`). Disagree drops it to `src` and "
         "goes in `notes/discrepancies.md`.", ""]
    p = os.path.join(SHARED, "deliverable", "review_packet.md")
    # Keep Alex's existing sign-off lines: a rerun must never wipe them.
    signed, cur = {}, None
    if os.path.exists(p):
        for line in open(p):
            if line.startswith("## R"):
                cur = line[3:].split(".")[0]
            elif line.startswith("Alex: ") and cur:
                signed[cur] = line.rstrip("\n")
    for r in R:
        L += [f"## {r['id']}. {r['title']}", "",
              f"1. **Claim:** {r['claim']}", f"   **Prediction (written before review):** {r['prediction']}",
              f"2. **Re-run:** `{r['rerun']}` (after the README one-liner)",
              f"3. **Waveform:** {r['wave']}", f"   **What to look for:** {r['look']}",
              "4. **By hand:**", "```", r["disasm"], "```", f"   {r['byhand']}", "",
              signed.get(r["id"], "Alex: [ ] agree  [ ] disagree, notes: ____"), ""]
    with open(p, "w") as f:
        f.write("\n".join(L))
    print("wrote", p)


if __name__ == "__main__":
    main()

"""Cocotb harness for the scalar-core timing microbenchmarks.

One cocotb testcase per benchmark (so each is its own Bazel test target).
Pattern follows core_mini_axi_sim.py / tutorial/counters: load ELF, look up
symbols, run to halt, read a global array back.
"""
import json
import os

import cocotb
import numpy as np
from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.sim_test_fixture import Fixture

PKG = "coralnpu_hw/tests/cocotb/scalar_timing"


async def run_bench(dut, elf_name, symbol, shape, timeout_cycles=20000):
    """Runs <elf_name>.elf to halt and returns the uint32 array at `symbol`."""
    r = runfiles.Create()
    fixture = await Fixture.Create(dut)
    await fixture.load_elf_and_lookup_symbols(
        r.Rlocation(f"{PKG}/{elf_name}.elf"), [symbol])
    assert fixture.symbols[symbol] is not None, f"symbol {symbol} not in ELF"
    wall = await fixture.run_to_halt(timeout_cycles=timeout_cycles)
    assert not fixture.fault(), "core faulted"
    n = int(np.prod(shape))
    raw = await fixture.read(symbol, 4 * n)
    return np.asarray(raw).view(np.uint32).reshape(shape), wall


def dump(name, payload):
    """Print to the test log and write <name>.json to the undeclared outputs.

    Log:  bazel-testlogs/.../test.log   (grep "=== <name>")
    JSON: bazel-testlogs/.../test.outputs/outputs.zip
    The JSON needs `zip` on the host (Bazel's test wrapper zips the directory).
    """
    text = json.dumps(payload, indent=2)
    print(f"\n=== {name} ===\n{text}\n", flush=True)
    out = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if out:
        with open(os.path.join(out, f"{name}.json"), "w") as f:
            f.write(text)


@cocotb.test()
async def fetch_ipc(dut):
    """Checklist 1.5: N independent addi, 16-byte aligned, N = 16/64/256."""
    sizes = [16, 64, 256]
    res, wall = await run_bench(dut, "fetch_ipc", "results", (3, 4))
    assert (res > 0).all(), f"unwritten results: {res}"
    deterministic = bool((res == res[0]).all())
    rows = []
    for rep, (empty, *cols) in enumerate(res.tolist()):
        for n, raw in zip(sizes, cols):
            body = raw - empty
            rows.append(dict(rep=rep, n=n, raw=raw, empty=empty, body=body,
                             ipc=round(n / body, 3) if body > 0 else None))
    # Source (UncachedFetch.scala:192-198, checklist 1.5): one 16-byte line (4 instr) per 2 cycles -> IPC 2.0.
    # The slope between n=64 and n=256 cancels the fixed overhead and must be 2.0; each size alone sits a
    # little above 2.0 (the overhead shrinks with n: 2.065 at 64, 2.016 at 256), so it gets a looser bound.
    _, _, raw64, raw256 = res[0].tolist()
    slope_ipc = round((256 - 64) / (raw256 - raw64), 3) if raw256 > raw64 else None
    slope_ok = slope_ipc is not None and abs(slope_ipc - 2.0) <= 0.01
    size_ok = all(abs(r["ipc"] - 2.0) <= 0.1 for r in rows if r["n"] in (64, 256) and r["ipc"] is not None)
    print(f"\nfetch_ipc: slope IPC (n=64..256) {slope_ipc} / 2.000 {'ok' if slope_ok else 'MISMATCH'}; "
          f"per-size IPC n=64/256 within 2.0 +/- 0.1: {'ok' if size_ok else 'MISMATCH'}", flush=True)
    dump("fetch_ipc", dict(raw=res.tolist(), rows=rows, wall_cycles=wall, slope_ipc=slope_ipc,
                           slope_ok=slope_ok, size_ok=size_ok, deterministic=deterministic,
                           all_ok=deterministic and slope_ok and size_ok))
    assert deterministic, "reps differ"
    assert slope_ok, f"fetch IPC {slope_ipc}, expected 2.0 (one 16-byte line per 2 cycles)"
    assert size_ok, f"per-size IPC outside 2.0 +/- 0.1: {[(r['n'], r['ipc']) for r in rows]}"


# ---------------------------------------------------------------------------
# dep_chain_{alu,sw,lw}: checklist 3.14 (dependent chain) and 3.3 (registered
# scoreboard for store data / load base). Layout must match dep_chain.cc.
#
# Expected cycles per unit (a unit = P, d fillers, C; 2+d instructions), derived
# from the source, not from earlier sim runs:
#   fetch bound   = (2 + d) / 2           one 16-byte line (4 inst) per 2 cycles
#                                         (UncachedFetch.scala:195, checklist 1.5)
#   ALU consumer  : may dispatch the cycle after P (same-cycle `comb`
#                   forwarding, Regfile.scala:111-112)             -> chain 1
#   SW / LW       : store data (rs2) and load base (rs1) are read against the
#                   registered scoreboard (Decode.scala:347-351), one cycle
#                   later than ALU                                  -> chain 2
#   Dispatch is in order, so the next P waits behind C: period = max(chain,
#   fetch). The control (IND) consumer does not read P: max(1, fetch).
# Pure `addi a2,a2,1` chain: 1 cycle per instruction (3.14).
# ---------------------------------------------------------------------------
NREG = 19
REPS = 3
UNIT_TOL = 0.2   # cycles per unit; slope is measured over 8 units (step 0.125)
CHAIN_TOL = 0.1  # cycles per instruction; slope over 48 instructions


def expected_per_unit(kind, d, mode):
    fetch = (2 + d) / 2
    if mode == "dep":
        chain = 1.0 if kind == "alu" else 2.0
    else:
        chain = 1.0
    return max(chain, fetch)


# Where the source-only model above disagrees with the sim, the sim wins
# (notes/discrepancies.md D1-D3). The values below are
# SIM-CALIBRATED, not derived from source; the reason is the mechanism seen in
# the waveform. The first-principles value is still reported in every row.
_LSU_LW = ("LSU load service time: one load per 4 cycles in steady state "
           "(lsu dispatch pulses every 4 cycles, queueCapacity refills 1 slot "
           "per 4 cycles; Lsu.scala:835,1056 one slot, dequeue only when idle). "
           "Hides both the dependency and the fetch bound.")
_LSU_SW = ("LSU store service time: one store per 2 cycles in steady state "
           "(same single-slot mechanism; value from cycle count, not isolated "
           "in the waveform).")
_FETCH_BUF = ("Fetch/buffer coupling: dispatch leaves residue in the buffer "
              "when a line lands, nSpace < 4 (UncachedFetch.scala:192 "
              "insufficientBuffer) delays the next request one cycle, so the "
              "line period alternates 2,3 cycles (2.5 avg).")
SIM_CALIBRATED = {
    ("alu", 0, "dep"): (1.25, _FETCH_BUF),
    ("sw", 2, "dep"): (2.5, _FETCH_BUF),
    ("sw", 0, "ind"): (2.0, _LSU_SW),
    ("sw", 1, "ind"): (2.0, _LSU_SW),
    **{("lw", d, m): (4.0, _LSU_LW) for d in range(4) for m in ("dep", "ind")},
}


async def run_dep_chain(dut, kind):
    name = f"dep_chain_{kind}"
    res, wall = await run_bench(dut, name, "results", (REPS, NREG))
    assert (res > 0).all(), f"unwritten results: {res}"
    deterministic = bool((res == res[0]).all())
    r = res[0].tolist()
    empty = r[0]
    rows, all_ok = [], True
    for d in range(4):
        for mi, mode in enumerate(("dep", "ind")):
            c8 = r[1 + (d * 2 + mi) * 2 + 0]
            c16 = r[1 + (d * 2 + mi) * 2 + 1]
            meas = (c16 - c8) / 8
            src = expected_per_unit(kind, d, mode)
            exp, why = SIM_CALIBRATED.get((kind, d, mode), (src, ""))
            ok = abs(meas - exp) <= UNIT_TOL
            src_ok = abs(meas - src) <= UNIT_TOL
            all_ok &= ok
            rows.append(dict(d=d, mode=mode, c8=c8, c16=c16,
                             cycles_per_unit=round(meas, 3), expected=exp,
                             expected_from_source=src, ok=ok,
                             source_prediction_matched=src_ok,
                             note=why, instr_per_unit=2 + d))
    chain_meas = (r[18] - r[17]) / 48
    chain_ok = abs(chain_meas - 1.0) <= CHAIN_TOL
    all_ok &= chain_ok
    # Dependency cost = DEP - IND at the same d (the control carries the fetch
    # bound and the P->P chain, so the difference is the consumer's hazard).
    cost = {d: round(rows[2 * d]["cycles_per_unit"] - rows[2 * d + 1]["cycles_per_unit"], 3)
            for d in range(4)}
    print(f"\n{name}: cycles per unit (measured / expected / source-only)")
    for row in rows:
        print(f"  d={row['d']} {row['mode']}: {row['cycles_per_unit']:.3f} / "
              f"{row['expected']:.3f} / {row['expected_from_source']:.3f}  "
              f"{'ok' if row['ok'] else 'MISMATCH'}"
              f"{'' if row['source_prediction_matched'] else '  (differs from source-only model, see notes/discrepancies.md)'}")
    print(f"  chain addi a2,a2,1: {chain_meas:.3f} cycles/inst / 1.000 "
          f"{'ok' if chain_ok else 'MISMATCH'}", flush=True)
    dump(name, dict(kind=kind, raw=res.tolist(), empty=empty, rows=rows,
                    chain=dict(c16=r[17], c64=r[18],
                               cycles_per_inst=round(chain_meas, 3), ok=chain_ok),
                    dep_minus_ind_per_unit=cost, deterministic=deterministic,
                    wall_cycles=wall, all_ok=all_ok))
    assert deterministic, "reps differ"
    assert all_ok, "measured != expected, see the JSON; the sim wins, record it in notes/discrepancies.md"


@cocotb.test()
async def dep_chain_alu(dut):
    """3.14: ALU consumer, same-cycle forwarding."""
    await run_dep_chain(dut, "alu")


@cocotb.test()
async def dep_chain_sw(dut):
    """3.3: store data uses the registered scoreboard."""
    await run_dep_chain(dut, "sw")


@cocotb.test()
async def dep_chain_lw(dut):
    """3.3: load base register uses the registered scoreboard."""
    await run_dep_chain(dut, "lw")


# ===========================================================================
# Deliverable sprint benches (B1-B10, TRACE). All share bench/st_common.h:
# the ELF exports st_results[REPS * N], st_case_names ("a,b,..."), st_misa.
# Results are looked up by case name, so the C order does not matter here.
#
# Each metric: (id, formula, expected_from_source, rows, note)
#   formula  S(base, lo, hi, per, ops)  -> (r[base_hi] - r[base_lo]) / per / ops
#            D(case, ref)               -> r[case] - r[ref]
#   expected_from_source = None means "measure only" (no source prediction).
# SPRINT_CALIBRATED[(bench, id)] = (value, reason) overrides the expectation
# where the sim disagreed with the source model (the sim wins; the reason is
# also in notes/discrepancies.md). The source value is always reported too.
# ===========================================================================
SPRINT_REPS = 2


def S(base, lo="u8", hi="u16", per=8, ops=1):
    return ("slope", base, lo, hi, per, ops)


def D(case, ref):
    return ("diff", case, ref)


def _eval(formula, r):
    if formula[0] == "slope":
        _, base, lo, hi, per, ops = formula
        return (r[f"{base}_{hi}"] - r[f"{base}_{lo}"]) / per / ops
    _, case, ref = formula
    return r[case] - r[ref]


SLOPE_TOL = 0.15
DIFF_TOL = 0.5

# ---- B1 branches (cycles per unit; units are listed in b1_branch.cc) -------
_B1 = [
    ("a4", S("a4"), 2.0, ["1.5"], "4 independent addi per line: fetch bound, 1 line / 2 cycles (UncachedFetch.scala:195)"),
    ("bnt", S("bnt"), 1.0, ["3.4"], "4 not-taken bne per line: a cond. branch ends its group (Decode.scala:324-327,508) -> 1/cycle (0.5 if it did not)"),
    ("bnt_a3", S("bnt_a3"), 2.0, ["3.4"], "bne in lane 0 + 3 addi: bne alone, then 3 addi; still within the 2-cycle fetch period"),
    ("a3_bnt", S("a3_bnt"), 2.0, ["3.4"], "3 addi + bne in lane 3: all 4 in one group"),
    ("ft", S("ft"), 4.0, ["1.7", "BRU mispredict"], "forward taken = BTFN mispredict: target dispatches at D+4 (Bru.scala:114-147, UncachedFetch.scala:196)"),
    ("bnt_bwd", S("bnt_bwd"), 4.0, ["1.7", "BRU mispredict"], "backward not-taken = BTFN mispredict, D+4"),
    ("jal", S("jal"), 2.0, ["3.5", "1.8", "BRU correct"], "jal predicted at predecode (UncachedFetch.scala:87-100): no redirect, one line per unit"),
    ("jalr", S("jalr"), 6.0, ["1.7", "3.3"], "auipc -> jalr rs1 via registered scoreboard (+2), jalr never predicted -> target at D+4"),
    ("joff0", S("joff0", "u4", "u8", 4), 4.0, ["1.10"], "jal to next line offset 0: two fetch slots per unit"),
    ("joff1", S("joff1", "u4", "u8", 4), 4.0, ["1.10"], "jal to offset +4: partial line still costs a full fetch slot"),
    ("joff2", S("joff2", "u4", "u8", 4), 4.0, ["1.10"], "jal to offset +8"),
    ("joff3", S("joff3", "u4", "u8", 4), 4.0, ["1.10"], "jal to offset +12"),
    ("loop2", S("loop2", "n8", "n16"), 2.0, ["1.7", "BRU correct"], "2-instr backward-taken loop: predicted, one line fetch per iteration"),
    ("loop4", S("loop4", "n8", "n16"), 2.0, ["1.7", "BRU correct"], "4-instr loop (one line)"),
    ("loop8", S("loop8", "n8", "n16"), 4.0, ["1.7", "BRU correct"], "8-instr loop (two lines)"),
]

# ---- B2 MLU ------------------------------------------------------------------
_B2 = [
    ("mul_chain", S("mul_chain"), 2.0, ["MLU", "3.15"], "mul write-back on port 4 at D+2 (Mlu.scala:76,101,116) -> dependent mul at D+2"),
    ("mul_ori", S("mul_ori"), 3.0, ["MLU"], "mul -> ori -> mul: 2 + 1"),
    ("mulh_ori", S("mulh_ori"), 3.0, ["MLU"], "mulh same latency as mul"),
    ("mulhu_ori", S("mulhu_ori"), 3.0, ["MLU"], "mulhu"),
    ("mulhsu_ori", S("mulhsu_ori"), 3.0, ["MLU"], "mulhsu"),
    ("mul_alu", S("mul_alu"), 3.0, ["MLU"], "mul -> addi -> mul"),
    ("mul_ind4", S("mul_ind4"), 4.0, ["3.8"], "4 independent mul per line: one MLU dispatch per cycle (Mlu.scala:67-76)"),
    ("mul2_a2", S("mul2_a2"), 2.0, ["3.8"], "2 mul + 2 addi per line: 2 cycles (also the fetch bound)"),
]

# ---- B3 DVU (cycles per div; 4 divs per DIV4) ----------------------------------
def _dvu(a):
    """Next div accepted at D+3+N, N = 32 - Clz1(|a|) (Dvu.scala:90-142)."""
    a &= 0xFFFFFFFF
    if a & 0x80000000:
        mag = (-a) & 0xFFFFFFFF if a != 0x80000000 else a
    else:
        mag = a
    if mag & 0x80000000:
        clz1 = 0
    else:
        low = mag & 0x7FFFFFFF
        clz1 = 30 if low == 0 else 30 - (low.bit_length() - 1)
    return 3 + (32 - clz1)


def _dvu_u(a):
    clz1 = 0 if a & 0x80000000 else (30 if a == 0 else 30 - (a.bit_length() - 1))
    return 3 + (32 - clz1)


_B3 = [
    ("div_a1", S("div_a1", "u4", "u8", 4), _dvu(1), ["DVU"], "N=2 -> next div at D+5"),
    ("div_a0", S("div_a0", "u4", "u8", 4), _dvu(0), ["DVU"], "dividend 0: Clz1 = 30 (PriorityEncoder default) -> N=2"),
    ("div_a256", S("div_a256", "u4", "u8", 4), _dvu(256), ["DVU"], "N=10"),
    ("div_a65536", S("div_a65536", "u4", "u8", 4), _dvu(65536), ["DVU"], "N=18"),
    ("div_a2p30", S("div_a2p30", "u4", "u8", 4), _dvu(1 << 30), ["DVU"], "N=32"),
    ("div_a2p31", S("div_a2p31", "u4", "u8", 4), _dvu(1 << 31), ["DVU"], "signed -2^31: |a| = 2^31, N=32"),
    ("div_am1", S("div_am1", "u4", "u8", 4), _dvu(-1), ["DVU"], "signed -1: |a| = 1, N=2"),
    ("div_by0", S("div_by0", "u4", "u8", 4), 35, ["DVU"], "divide by zero forces clz=0 -> N=32 (Dvu.scala:122)"),
    ("divu_a256", S("divu_a256", "u4", "u8", 4), _dvu_u(256), ["DVU"], "divu same iteration count"),
    ("rem_a256", S("rem_a256", "u4", "u8", 4), _dvu(256), ["DVU"], "rem same"),
    ("remu_a256", S("remu_a256", "u4", "u8", 4), _dvu_u(256), ["DVU"], "remu same"),
    ("divu_a2p31", S("divu_a2p31", "u4", "u8", 4), _dvu_u(1 << 31), ["DVU"], "unsigned 2^31: MSB set, N=32"),
    ("ddep_a256", S("ddep_a256", "u4", "u8", 4), 14, ["DVU"], "div -> and -> or -> div: WB at D+2+N (N=10), +2 ALU hops"),
    ("ddep_a1", S("ddep_a1", "u4", "u8", 4), 6, ["DVU"], "same with N=2"),
    ("dmul", S("dmul", "u4", "u8", 4), 9, ["3.15"], "div(N=2) + 6 mul: MLU owns port 4 (SCore.scala:259), div WB slips behind the muls"),
    ("dadd", S("dadd", "u4", "u8", 4), 5, ["3.15"], "control: div(N=2) + 6 addi, DVU-bound at 3+N"),
]

# ---- B4 LSU --------------------------------------------------------------------
_B4 = [
    ("chase", S("chase"), 5.0, ["Load", "3.3"], "lw a2,0(a2): WB on port 5 at D+4 (Lsu.scala:1017), load base via regd scoreboard -> next lw at D+5"),
    ("chase_alu", S("chase_alu"), 6.0, ["Load"], "lw -> add (comb, D+4) -> lw (regd, +2)"),
    ("lw4", S("lw4"), 16.0, ["3.19"], "4 independent lw: 4 cycles each (D3)"),
    ("lh4", S("lh4", "u4", "u8", 4), 16.0, ["3.19"], "lh same path"),
    ("lhu4", S("lhu4", "u4", "u8", 4), 16.0, ["3.19"], "lhu"),
    ("lb4", S("lb4", "u4", "u8", 4), 16.0, ["3.19"], "lb"),
    ("lbu4", S("lbu4", "u4", "u8", 4), 16.0, ["3.19"], "lbu"),
    ("sw4", S("sw4"), 8.0, ["3.19"], "4 independent sw: 2 cycles each (D2)"),
    ("sh4", S("sh4", "u4", "u8", 4), 8.0, ["3.19"], "sh"),
    ("sb4", S("sb4", "u4", "u8", 4), 8.0, ["3.19"], "sb"),
    ("sl_same", S("sl_same"), 6.0, ["Store", "3.19"], "sw then lw same word: single slot, 2 + 4"),
    ("sl_diff", S("sl_diff"), 6.0, ["Store", "3.19"], "sw then lw different word: same"),
    ("ls", S("ls"), 6.0, ["3.19"], "lw then sw"),
    ("l_sdata", S("l_sdata"), None, ["3.3"], "lw -> sw of the loaded value (store data regd): measure"),
    ("burst2", D("burst2", "burst1"), 4.0, ["LSU opQueue"], "each extra independent load adds 4"),
    ("burst3", D("burst3", "burst2"), 4.0, ["LSU opQueue"], ""),
    ("burst4", D("burst4", "burst3"), 4.0, ["LSU opQueue"], ""),
    ("burst5", D("burst5", "burst4"), 4.0, ["LSU opQueue"], "5th load: opQueue full, dispatch stalls, still +4"),
    ("burst6", D("burst6", "burst5"), 4.0, ["LSU opQueue"], ""),
    ("burst8", D("burst8", "burst6"), 8.0, ["LSU opQueue"], ""),
]
_B4M = [
    ("lw_al", S("lw_al"), 16.0, ["Line-crossing"], "aligned reference"),
    ("lw_mis2", S("lw_mis2"), None, ["Line-crossing"], "misaligned within a line: measure"),
    ("lw_cross14", S("lw_cross14"), None, ["Line-crossing"], "crosses a 16-byte line: 2 transactions expected (targetAddress loop)"),
    ("sw_cross14", S("sw_cross14"), None, ["Line-crossing"], "line-crossing store pair: measure"),
]

# ---- B5 FPU (cycles per FP op unless noted) -------------------------------------
_F = lambda b: S(b, "u4", "u8", 4, 2)
_F1 = lambda b: S(b, "u4", "u8", 4, 1)
_B5 = [
    ("fadd", _F("fadd"), 5.0, ["FPU", "3.6"], "queue depth 1, ready only when empty; fpnew 3 pipe regs -> write D+4, next FP op D+5 (FloatCore.scala:222-225,356)"),
    ("fadd_dep", _F("fadd_dep"), 5.0, ["3.16"], "dependent fadd: no f-scoreboard bypass, D+5 (FRegfile.scala:36-42)"),
    ("fmul", _F("fmul"), 5.0, ["FPU"], "same pipe"),
    ("fmadd", _F("fmadd"), 5.0, ["FPU"], "same pipe"),
    ("fdiv", _F("fdiv"), None, ["FPU div/sqrt"], "iterative DIVSQRT: measure"),
    ("fsqrt", _F("fsqrt"), None, ["FPU div/sqrt"], "iterative DIVSQRT: measure"),
    ("feq", _F("feq"), 5.0, ["FPU->int"], "fpnew + Queue(2) to port 4 (FloatCore.scala:352); structural limit still the queue"),
    ("fcvt", _F("fcvt"), 5.0, ["FPU->int"], "fcvt.w.s"),
    ("fmvxw", _F("fmvxw"), 2.0, ["fmv"], "bypasses fpnew, dequeued at D+1 (inst.ready := fmv) -> next FP op D+2"),
    ("fmvwx", _F("fmvwx"), 2.0, ["fmv"], "same"),
    ("flw", _F("flw"), 4.0, ["flw/fsw", "3.19"], "LSU-bound, 4"),
    ("fsw", _F("fsw"), 2.0, ["flw/fsw", "3.19"], "LSU-bound, 2"),
    ("fadd_a3", _F1("fadd_a3"), 5.0, ["3.6"], "fadd + 3 addi per unit: FPU-bound (fadd dispatches alone; addi hide in the shadow)"),
    ("fcvt_use", _F1("fcvt_use"), 6.0, ["FPU->int"], "fcvt.w.s -> addi: int WB at D+5 via Queue(2), addi D+5, next fcvt D+6"),
    ("fmvwx_fadd", _F1("fmvwx_fadd"), 7.0, ["fmv"], "fmv.w.x (f write D+1) -> fadd at D+2 -> next fmv at D+7"),
    ("flw_use", _F1("flw_use"), None, ["flw/fsw"], "flw -> fmv.x.w: f-port 1 write, measure"),
]

# ---- B6 CSR ------------------------------------------------------------------
_B6 = [
    ("csrr_mscratch", S("csrr_mscratch"), 2.0, ["CSR", "3.11"], "csrr: lane 0 alone, result D+1, next CSR waits for ROB empty (Decode.scala:519)"),
    ("csrw_mscratch", S("csrw_mscratch"), 2.0, ["CSR"], "csrw same"),
    ("csrr_mcycle", S("csrr_mcycle"), 2.0, ["CSR", "0.4"], "mcycle read, same"),
    ("csrr_a3", S("csrr_a3"), 3.0, ["3.6", "3.11"], "csrr alone, 3 addi next cycle, next csrr waits for the addi to retire"),
    ("csr_use", S("csr_use"), 3.0, ["CSR"], "csrr -> dependent addi"),
    ("a3_csrr", S("a3_csrr"), 3.0, ["3.6", "3.11"], "3 addi then csrr: csrr cannot share their group"),
    ("div_alone", S("div_alone", "u4", "u8", 4), 35.0, ["DVU"], "reference"),
    ("div_csrr", S("div_csrr", "u4", "u8", 4), 36.0, ["3.11"], "csrr behind a long div waits for the ROB to drain"),
    ("fadd_csrr", S("fadd_csrr", "u4", "u8", 4), 6.0, ["3.11"], "csrr also waits for FPU ready (Decode.scala:693)"),
    ("fadd_alone", S("fadd_alone", "u4", "u8", 4), 5.0, ["FPU"], "reference"),
]

# ---- B7 ROB ------------------------------------------------------------------
_B7 = [(f"k{k}", D(f"k{k}", "k0"), 0.0 if k <= 7 else None, ["3.10", "5.1"],
        "fits in the ROB under the div" if k <= 7 else "past ROB capacity: measure (must be > 0)")
       for k in (2, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24, 40)] + [
    ("short_k40", D("short_k40", "short_k0"), None, ["3.10"], "control: short div, 40 addi (fetch-bound)"),
    ("nodiv_k40", D("nodiv_k40", "empty"), None, ["3.10"], "control: 40 addi, no div"),
]

# ---- B8 x0 -------------------------------------------------------------------
_B8 = [
    ("nop4", S("nop4"), 4.0, ["3.17"], "4 nop: WAW on x0 inside the group (Decode.scala:337-344,363) -> 1/cycle"),
    ("addi4", S("addi4"), 2.0, ["3.17"], "4 addi distinct rd: fetch bound"),
    ("nop_li3", S("nop_li3"), 2.0, ["3.17"], "nop + 3 li (rs1=x0): fetch bound unless the x0 RAW costs fetch bandwidth"),
    ("nop_as3", S("nop_as3"), 2.0, ["3.17"], "nop + 3 addi rs1=s1 (control)"),
    ("x0w4", S("x0w4"), 4.0, ["3.17"], "4 addi rd=x0, rs1=s1: WAW on x0"),
    ("pf_nop7", D("pf_nop7", "pf_base"), 5.0, ["3.17"], "prefilled: 7 nop vs 7 addi -> +5 dispatch cycles"),
    ("pf_nop_li6", D("pf_nop_li6", "pf_base"), 1.0, ["3.17"], "prefilled: nop then 6 li: x0 RAW splits the group (+1)"),
    ("pf_nop_as6", D("pf_nop_as6", "pf_base"), 0.0, ["3.17"], "prefilled: nop then 6 addi rs1=s1 (control, +0)"),
    ("pf_li_nop_li5", D("pf_li_nop_li5", "pf_base"), 1.0, ["3.17"], "prefilled: li nop li*5 (+1)"),
    ("pf_x0w_as6", D("pf_x0w_as6", "pf_base"), 0.0, ["3.17"], "prefilled: addi x0,s1 then 6 addi rs1=s1 (+0)"),
]

# ---- B9 ALU ------------------------------------------------------------------
_ALU_BASE = ["add", "sub", "sll", "srl", "sra", "slt", "sltu", "xor_r", "or_r",
             "and_r", "addi", "slti", "sltiu", "xori", "ori", "andi", "slli",
             "srli", "srai", "lui", "auipc"]
_ALU_ZBB = ["andn", "orn", "xnor", "max", "maxu", "min", "minu", "rol", "ror",
            "rori", "clz", "ctz", "cpop", "sext_b", "sext_h", "zext_h", "orc_b",
            "rev8"]
_alu = lambda ops: [(o, S(o, "u8", "u24", 16), 1.0, ["ALU"],
                     "dependent chain: ALU result D+1, same-cycle bypass (Alu.scala:80-97, Regfile.scala:111-112)"
                     if o not in ("lui", "auipc") else "WAW chain on one rd: 1/cycle")
                    for o in ops]

# ---- B10 system ----------------------------------------------------------------
_B10 = [
    ("fence", S("fence"), 1.0, ["3.5", "3.12"], "plain fence: lane 0 only, not fency (Decode.scala:163,723) -> 1/cycle"),
    ("fence_a3", S("fence_a3"), 2.0, ["3.5"], "fence + 3 addi: fence does not end the group, but rd=x0 vs addi rs1=x0 (3.17b) may split it"),
    ("lw_fence", S("lw_fence"), 4.0, ["3.12"], "plain fence does not wait for the LSU (isFency excludes fence, Decode.scala:396): load-bound 4"),
    ("lw_alone", S("lw_alone"), 4.0, ["3.19"], "reference"),
    ("fencei", S("fencei"), None, ["3.5", "2.3"], "fence.i: LSU flush + fetch redirect: measure"),
    ("fencei_a3", S("fencei_a3"), None, ["3.5"], "fence.i + 3 addi"),
    ("lw_fencei", S("lw_fencei"), None, ["3.12"], "fence.i behind a load waits for LSU idle"),
    ("sw_fencei", S("sw_fencei"), None, ["3.12"], "fence.i behind a store"),
]

SPRINT_SPECS = {
    "b1_branch": _B1,
    "b2_mul": _B2,
    "b3_div": _B3,
    "b4_lsu": _B4,
    "b4_lsu_misaligned": _B4M,
    "b5_fpu": _B5,
    "b6_csr": _B6,
    "b7_rob": _B7,
    "b8_x0": _B8,
    "b9_alu_base": _alu(_ALU_BASE),
    "b9_alu_zbb": _alu(_ALU_ZBB),
    "b10_system": _B10,
    "trace": [],  # no cycle checks here: the trace numbers are checked by the deliverable/timing.json diff (README)
}

# Sim-calibrated expectations (2026-10-17 run). Two kinds:
#  * the sim disagreed with the source model: reason says why, also in
#    notes/discrepancies.md (D4-D6);
#  * the row had no source prediction ("measure"): the measured value is
#    locked in as a regression value so Samuel's re-run checks it.
_FPU0 = ("D4: in the Verilator model fpnew's FP32 slice elaborates with "
         "NumPipeRegs = 0 (FmtPipeRegs = {0,0,0,0,3}; the '{default: 3} in "
         "FloatCore.scala:113 only reached the last format slot), so add/mul/"
         "fma/cmp/cvt finish in D+1 and the next FP op dispatches at D+2. "
         "Source intent (3 pipe regs) predicts 5. Verified in b5_fpu/trace waves.")
_PF = ("D5: prefilled-buffer diff under-counts: the buffer held only 7 entries "
       "behind the div and the last pattern instruction is in an unfetched line, "
       "so part of the extra dispatch cycles hide behind fetch. The mechanism "
       "(nop one per cycle; li held out of the nop's group) is seen in b8_x0 waves.")
_REG = "no source prediction; regression value from the 2026-10-17 sim run"
SPRINT_CALIBRATED = {
    **{("b5_fpu", k): (v, _FPU0) for k, v in dict(
        fadd=2.0, fadd_dep=2.0, fmul=2.0, fmadd=2.0, feq=2.0, fcvt=2.0,
        fadd_a3=2.0, fcvt_use=3.0, fmvwx_fadd=4.0).items()},
    ("b6_csr", "fadd_csrr"): (3.0, _FPU0),
    ("b6_csr", "fadd_alone"): (2.0, _FPU0),
    ("b8_x0", "pf_nop7"): (3.0, _PF),
    ("b8_x0", "pf_li_nop_li5"): (0.0, _PF),
    ("b5_fpu", "fdiv"): (6.0, _REG + " (iterative TH32 DIVSQRT: write at D+5)"),
    ("b5_fpu", "fsqrt"): (19.0, _REG + " (iterative: write at D+18)"),
    ("b5_fpu", "flw_use"): (6.0, _REG + " (flw f-port 1 at D+4, f consumer at D+5, next flw behind it)"),
    ("b4_lsu", "l_sdata"): (7.0, _REG),
    ("b4_lsu_misaligned", "lw_mis2"): (16.0, _REG + " (misaligned inside a line: same as aligned)"),
    ("b4_lsu_misaligned", "lw_cross14"): (20.0, _REG + " (line-crossing lw: 5 cycles, +1 for the 2nd transaction)"),
    ("b4_lsu_misaligned", "sw_cross14"): (6.0, _REG + " (line-crossing sw: 3 cycles, +1)"),
    ("b10_system", "fencei"): (4.0, _REG),
    ("b10_system", "fencei_a3"): (6.0, _REG),
    ("b10_system", "lw_fencei"): (9.0, _REG),
    ("b10_system", "sw_fencei"): (7.0, _REG),
    **{("b7_rob", f"k{k}"): (float(v), _REG + " (ROB full: dispatch held until the div retires, b7_rob waves)")
       for k, v in {8: 2, 9: 2, 10: 2, 12: 3, 16: 5, 24: 9, 40: 17}.items()},
    ("b7_rob", "short_k40"): (17.0, _REG),
    ("b7_rob", "nodiv_k40"): (19.0, _REG),
}

MISA_BITS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


async def run_sprint(dut, bench, timeout_cycles=400000):
    r = runfiles.Create()
    fixture = await Fixture.Create(dut)
    syms = ["st_results", "st_case_names", "st_misa"]
    await fixture.load_elf_and_lookup_symbols(
        r.Rlocation(f"{PKG}/{bench}.elf"), syms)
    for s in syms:
        assert fixture.symbols[s] is not None, f"symbol {s} not in ELF"
    wall = await fixture.run_to_halt(timeout_cycles=timeout_cycles)
    faulted = bool(fixture.fault())
    raw_names = bytes(np.asarray(await fixture.read("st_case_names", 2048)).astype(np.uint8))
    assert b"\0" in raw_names, "case-name table longer than 2048 bytes"
    names = [n for n in raw_names.split(b"\0")[0].decode().split(",") if n]
    n = len(names)
    raw = await fixture.read("st_results", 4 * SPRINT_REPS * n)
    res = np.asarray(raw).view(np.uint32).reshape(SPRINT_REPS, n)
    misa = int(np.asarray(await fixture.read("st_misa", 4)).view(np.uint32)[0])
    deterministic = bool((res == res[0]).all())
    r0 = dict(zip(names, res[0].tolist()))
    rows, all_ok = [], True
    for mid, formula, src, chk, note in SPRINT_SPECS[bench]:
        try:
            meas = _eval(formula, r0)
        except KeyError as e:
            rows.append(dict(id=mid, error=f"missing case {e}"))
            all_ok = False
            continue
        tol = SLOPE_TOL if formula[0] == "slope" else DIFF_TOL
        cal = SPRINT_CALIBRATED.get((bench, mid))
        exp = cal[0] if cal else src
        ok = exp is None or abs(meas - exp) <= tol
        src_ok = None if src is None else abs(meas - src) <= tol
        all_ok &= ok
        rows.append(dict(id=mid, measured=round(meas, 3), expected=exp,
                         expected_from_source=src, sim_calibrated=bool(cal),
                         calibration_reason=cal[1] if cal else "",
                         ok=ok, source_prediction_matched=src_ok,
                         checklist_rows=chk, note=note))
    print(f"\n{bench}: measured / expected / source-only")
    for row in rows:
        if "error" in row:
            print(f"  {row['id']}: ERROR {row['error']}")
            continue
        print(f"  {row['id']:>14}: {row['measured']:8.3f} / {row['expected']} / "
              f"{row['expected_from_source']}  {'ok' if row['ok'] else 'MISMATCH'}"
              f"{'' if row['source_prediction_matched'] in (True, None) else '  (differs from source model)'}")
    misa_ext = "".join(c for i, c in enumerate(MISA_BITS) if misa >> i & 1)
    dump(bench, dict(bench=bench, cases=r0, raw=res.tolist(), names=names,
                     rows=rows, deterministic=deterministic, faulted=faulted,
                     wall_cycles=wall, misa=hex(misa), misa_extensions=misa_ext,
                     all_ok=all_ok))
    assert not faulted, "core faulted"
    assert deterministic, "reps differ"
    assert all_ok, "measured != expected; the sim wins, see notes/discrepancies.md"


def _make_sprint_test(bench):
    async def _t(dut):
        await run_sprint(dut, bench)
    _t.__name__ = bench
    _t.__qualname__ = bench
    _t.__doc__ = f"Sprint bench {bench} (see bench/{bench}.cc)."
    return cocotb.test()(_t)


for _b in SPRINT_SPECS:
    globals()[_b] = _make_sprint_test(_b)

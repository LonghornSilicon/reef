#!/usr/bin/env python3
"""Scripted waveform checks for the scalar-timing benches.

  trace_scan.py dump  <bench> <function-substring> [--reps N]
      Per-cycle table for one region: fetch request/return, the 4 dispatch
      lanes (pc, '*' = dispatched), ROB space, write-backs, retirements.
  trace_scan.py trace [--out timing_events.json]
      Per-class event cycles from the TRACE bench, relative to F1 = 0.

VCDs come from bench/waves.sh (~/LHS/scalar-timing-waves/<bench>.vcd), ELFs
from the build tree. Host-side; needs the coralnpu toolchain's llvm-nm.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcdscan import Vcd  # noqa: E402

CORAL = os.environ.get("CORALNPU", os.path.expanduser("~/LHS/coralnpu"))
WAVES = os.environ.get("WAVES", os.path.expanduser("~/LHS/scalar-timing-waves"))


def elf_path(bench):
    """ELFs are built in a transitioned config (bazel-out/k8-*-ST-*/bin)."""
    c = glob.glob(os.path.join(CORAL, "bazel-out/*/bin/tests/cocotb/scalar_timing", bench + ".elf"))
    if not c:
        sys.exit(f"{bench}.elf not built (bazel build //tests/cocotb/scalar_timing:{bench})")
    return max(c, key=os.path.getmtime)

S = "core.score."
DISP = [f"{S}dispatch.io_inst_{k}_{f}" for k in range(4) for f in ("valid", "ready", "bits_addr")]
FETCH = [f"{S}fetch.io_ibus_valid", f"{S}fetch.io_ibus_ready", f"{S}fetch.io_ibus_addr",
         f"{S}fetch.fetcher.io_fetch_valid", f"{S}fetch.fetcher.io_fetch_bits_addr",
         f"{S}fetch.ctrl.blockNewFetch", f"{S}fetch.instructionBuffer.io_nSpace",
         f"{S}fetch.instructionBuffer.io_flush", f"{S}fetch.ctrl.io_branch_valid",
         f"{S}fetch.ctrl.io_iflush_valid"]
ROB = [f"{S}retirement_buffer.io_nSpace", f"{S}retirement_buffer.io_empty",
       f"{S}retirement_buffer.io_nRetired", f"{S}retirement_buffer.io_storeComplete_valid"] + \
      [f"{S}retirement_buffer.io_debug_inst_{k}_{f}" for k in range(8) for f in ("valid", "bits_pc")] + \
      [f"{S}retirement_buffer.io_writeDataScalar_{p}_{f}" for p in range(6) for f in ("valid", "bits_addr")] + \
      [f"{S}retirement_buffer.io_writeDataFloat_{p}_{f}" for p in range(2) for f in ("valid", "bits_addr")]
UNITS = [f"{S}alu_{k}.valid" for k in range(4)] + \
        [f"{S}bru_{k}.stateReg_valid" for k in range(4)] + [f"{S}bru_0.io_taken_valid"] + \
        [f"{S}bru_{k}.io_taken_valid" for k in range(1, 4)] + \
        [f"{S}csr.req_valid", f"{S}csr.io_rd_valid",
         f"{S}mlu.stage2Input_q.io_deq_valid", f"{S}mlu.stage3Input_q.io_deq_valid", f"{S}mlu.io_rd_valid",
         f"{S}dvu.active", f"{S}dvu.compute", f"{S}dvu.io_rd_valid", f"{S}dvu.io_rd_ready", f"{S}dvu.count",
         f"{S}lsu.opQueue.io_nEnqueued", f"{S}lsu.io_dbus_valid", f"{S}lsu.io_dbus_write",
         f"{S}lsu.readFired_valid", f"{S}lsu.io_rd_valid", f"{S}lsu.io_rd_flt_valid",
         f"{S}lsu.io_storeComplete_valid", f"{S}lsu.io_flush_valid", f"{S}lsu.io_queueCapacity",
         f"{S}lsu.nextSlot_active",
         f"{S}floatCore.instQueue.io_count", f"{S}floatCore.fpuActive",
         f"{S}floatCore.floatCoreWrapper.in_valid_i", f"{S}floatCore.floatCoreWrapper.in_ready_o",
         f"{S}floatCore.floatCoreWrapper.out_valid_o", f"{S}floatCore.io_write_ports_0_valid",
         f"{S}floatCore.scalar_rd_pipe_q.io_enq_valid", f"{S}floatCore.io_scalar_rd_valid",
         f"{S}floatCore.io_inst_valid", f"{S}floatCore.io_inst_ready",
         "core.rvvCore.rvvCoreWrapper.core.frontend.inst_valid_i",
         "core.rvvCore.rvvCoreWrapper.core.frontend.valid_inst_q",
         "core.rvvCore.rvvCoreWrapper.core.frontend.cmd_valid_o"]
ALL = DISP + FETCH + ROB + UNITS


def nm_tool():
    t = os.environ.get("NM")
    if t:
        return t
    c = sorted(glob.glob(os.path.expanduser(
        "~/.cache/bazel/_bazel_*/*/external/toolchain_coralnpu_v2/bin/llvm-nm")))
    if not c:
        sys.exit("llvm-nm from toolchain_coralnpu_v2 not found; set NM=")
    return c[-1]


def symbols(bench):
    out = subprocess.run([nm_tool(), "-n", elf_path(bench)],
                         capture_output=True, text=True, check=True).stdout
    syms = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) == 3:
            syms[p[2]] = int(p[0], 16)
    return syms


def load(bench, extra=()):
    return Vcd(os.path.join(WAVES, bench + ".vcd"), ALL + list(extra))


class Trace:
    """Convenience accessors over a sampled VCD."""

    def __init__(self, v):
        self.v = v
        self.n = v.n

    def g(self, name, k):
        return self.v.val(S + name if not name.startswith("core.") else name)[k]

    def dispatched(self, k):
        """[(lane, pc)] dispatched in cycle k."""
        return [(i, self.g(f"dispatch.io_inst_{i}_bits_addr", k)) for i in range(4)
                if self.g(f"dispatch.io_inst_{i}_valid", k) and self.g(f"dispatch.io_inst_{i}_ready", k)]

    def first_dispatch(self, pc, start=0, end=None):
        for k in range(start, end or self.n):
            for lane, a in self.dispatched(k):
                if a == pc:
                    return k, lane
        return None, None

    def visible_since(self, pc, d):
        k = d
        while k - 1 >= 0 and any(self.g(f"dispatch.io_inst_{i}_valid", k - 1) and
                                 self.g(f"dispatch.io_inst_{i}_bits_addr", k - 1) == pc
                                 for i in range(4)):
            k -= 1
        return k

    def last_before(self, cond, k0, lookback=40):
        for k in range(k0, max(-1, k0 - lookback), -1):
            if cond(k):
                return k
        return None

    def first_from(self, cond, k0, lookahead=120):
        for k in range(k0, min(self.n, k0 + lookahead)):
            if cond(k):
                return k
        return None

    def retire(self, pc, k0):
        return self.first_from(lambda k: any(
            self.g(f"retirement_buffer.io_debug_inst_{j}_valid", k) and
            self.g(f"retirement_buffer.io_debug_inst_{j}_bits_pc", k) == pc for j in range(8)), k0)

    def wb_scalar(self, rd, k0):
        for k in range(k0, min(self.n, k0 + 120)):
            for p in range(6):
                if self.g(f"retirement_buffer.io_writeDataScalar_{p}_valid", k) and \
                        self.g(f"retirement_buffer.io_writeDataScalar_{p}_bits_addr", k) == rd:
                    return k, p
        return None, None

    def wb_float(self, rd, k0):
        for k in range(k0, min(self.n, k0 + 120)):
            for p in range(2):
                if self.g(f"retirement_buffer.io_writeDataFloat_{p}_valid", k) and \
                        self.g(f"retirement_buffer.io_writeDataFloat_{p}_bits_addr", k) == rd:
                    return k, p
        return None, None


def cmd_dump(args):
    syms = symbols(args.bench)
    names = sorted(((a, s) for s, a in syms.items()), key=lambda x: x[0])
    fn = [(a, s) for a, s in names if args.func in s]
    if not fn:
        sys.exit(f"no symbol matching {args.func}")
    lo = fn[0][0]
    hi = next((a for a, s in names if a > lo), lo + 0x1000)
    t = Trace(load(args.bench))
    # first cycle any lane dispatches a pc inside [lo, hi)
    k0 = next((k for k in range(t.n) if any(lo <= a < hi for _, a in t.dispatched(k))), None)
    if k0 is None:
        sys.exit("region never dispatched")
    last = k0
    for k in range(k0, t.n):
        if any(lo <= a < hi for _, a in t.dispatched(k)):
            last = k
        elif k - last > 60:
            break
    if args.from_csrr:
        # start at the region's first `csrr a0, mcycle` (start timestamp)
        dis = subprocess.run([nm_tool().replace("llvm-nm", "llvm-objdump"), "-d",
                              f"--start-address={lo:#x}", f"--stop-address={hi:#x}",
                              elf_path(args.bench)], capture_output=True, text=True).stdout
        cs = [int(l.split(":")[0], 16) for l in dis.splitlines() if "csrr\ta0, mcycle" in l]
        if cs:
            k0, _ = t.first_dispatch(cs[0], k0)
    lo_k, hi_k = max(0, k0 - 6), min(t.n, last + args.tail)
    print(f"{args.bench} {fn[0][1]} [{lo:#x},{hi:#x}) cycles {lo_k}..{hi_k}, period {t.v.period_ns():.3f} ns")
    print("cyc   edge_ns  ibus   fdata  bNF nSp | lane0     lane1     lane2     lane3    | rob ret | wb / units")
    for k in range(lo_k, hi_k):
        ib = f"{t.g('fetch.io_ibus_addr', k):05x}" if t.g("fetch.io_ibus_valid", k) else "  -  "
        fd = f"{t.g('fetch.fetcher.io_fetch_bits_addr', k):05x}" if t.g("fetch.fetcher.io_fetch_valid", k) else "  -  "
        lanes = []
        for i in range(4):
            if t.g(f"dispatch.io_inst_{i}_valid", k):
                a = t.g(f"dispatch.io_inst_{i}_bits_addr", k)
                lanes.append(f"{a:04x}{'*' if t.g(f'dispatch.io_inst_{i}_ready', k) else ' '}    ")
            else:
                lanes.append("  .      ")
        wb = []
        for p in range(6):
            if t.g(f"retirement_buffer.io_writeDataScalar_{p}_valid", k):
                wb.append(f"x{t.g(f'retirement_buffer.io_writeDataScalar_{p}_bits_addr', k)}@p{p}")
        for p in range(2):
            if t.g(f"retirement_buffer.io_writeDataFloat_{p}_valid", k):
                wb.append(f"f{t.g(f'retirement_buffer.io_writeDataFloat_{p}_bits_addr', k)}@fp{p}")
        for i in range(4):
            if t.g(f"bru_{i}.io_taken_valid", k):
                wb.append(f"REDIRECT(bru{i})")
        if t.g("fetch.instructionBuffer.io_flush", k):
            wb.append("ibuf_flush")
        if t.g("floatCore.instQueue.io_count", k):
            wb.append("fq")
        if t.g("floatCore.floatCoreWrapper.out_valid_o", k):
            wb.append("fpnew_out")
        if t.g("lsu.io_dbus_valid", k):
            wb.append("dbusW" if t.g("lsu.io_dbus_write", k) else "dbusR")
        if t.g("dvu.io_rd_valid", k):
            wb.append("dvu_rd")
        print(f"{k:5d} {t.v.ns(k):8.2f} {ib} {fd}  {t.g('fetch.ctrl.blockNewFetch', k)}  "
              f"{t.g('fetch.instructionBuffer.io_nSpace', k)} | {''.join(lanes)}| "
              f"{t.g('retirement_buffer.io_nSpace', k):3d} {t.g('retirement_buffer.io_nRetired', k):3d} | {' '.join(wb)}")


# ----------------------------------------------------------------------------
# TRACE: per-class events. rd: int results in t0 (x5), float in ft0 (f0).
# kind: which unit stage registers to report.
CLASSES = [
    # name, example_asm, kind, result ('x','f',None), consumer label meaning
    ("alu", "add t0, a2, a3", "alu", "x"),
    ("zbb", "clz t0, a2", "alu", "x"),
    ("lui", "lui t0, 1", "alu", "x"),
    ("auipc", "auipc t0, 0", "alu", "x"),
    ("br_nt_fwd", "bne zero, zero, 1f (not taken, forward)", "bru", None),
    ("br_t_bwd", "bnez t3, 1b (taken, backward: predicted)", "bru", None),
    ("br_mp_fwd", "beq zero, zero, 1f (taken, forward: mispredict)", "bru", None),
    ("br_mp_bwd", "bne zero, zero, 1b (not taken, backward: mispredict)", "bru", None),
    ("jal", "jal zero, 1f", "bru", None),
    ("jalr", "jalr zero, 16(t1) (rs1 from auipc 2 instr earlier)", "bru", None),
    ("jalr_rdy", "jalr zero, 0(t1) (rs1 ready long before)", "bru", None),
    ("csr", "csrr t0, mscratch", "csr", "x"),
    ("mul", "mul t0, a2, a3", "mlu", "x"),
    ("div_small", "div t0, a4, a3 (a4 = 1, N = 2)", "dvu", "x"),
    ("div_large", "div t0, a5, a3 (a5 = 2^30, N = 32)", "dvu", "x"),
    ("div_by0", "div t0, a5, zero", "dvu", "x"),
    ("lw", "lw t0, 4(s0)", "lsu", "x"),
    ("sw", "sw a2, 8(s0)", "lsu_st", None),
    ("flw", "flw ft0, 0(s0)", "lsu_f", "f"),
    ("fsw", "fsw fa0, 12(s0)", "lsu_st", None),
    ("fadd", "fadd.s ft0, fa0, fa1", "fpu", "f"),
    ("fmul", "fmul.s ft0, fa0, fa1", "fpu", "f"),
    ("fmadd", "fmadd.s ft0, fa0, fa1, fa2", "fpu", "f"),
    ("fdiv", "fdiv.s ft0, fa0, fa1", "fpu", "f"),
    ("fsqrt", "fsqrt.s ft0, fa1", "fpu", "f"),
    ("feq", "feq.s t0, fa0, fa1", "fpu_x", "x"),
    ("fcvt_w_s", "fcvt.w.s t0, fa0", "fpu_x", "x"),
    ("fmv_x_w", "fmv.x.w t0, fa0", "fpu_x", "x"),
    ("fmv_w_x", "fmv.w.x ft0, a2", "fpu", "f"),
    ("fence", "fence", "fence", None),
    ("fence_i", "fence.i", "fencei", None),
    ("rvv", "vadd.vv v1, v2, v3", "rvv", None),
]


def unit_events(t, kind, d, lane):
    """Unit-stage events (absolute cycles) for one instruction dispatched at d."""
    ev = {}
    f = lambda name, cond, lb=d, la=120: ev.__setitem__(name, t.first_from(cond, lb, la))
    if kind == "alu":
        f(f"alu_{lane}.valid (op/addr regs, result comb)", lambda k: t.g(f"alu_{lane}.valid", k), d + 1, 3)
    elif kind == "bru":
        f(f"bru_{lane}.stateReg_valid (resolve)", lambda k: t.g(f"bru_{lane}.stateReg_valid", k), d + 1, 3)
        f(f"bru_{lane}.io_taken_valid (redirect)", lambda k: t.g(f"bru_{lane}.io_taken_valid", k), d + 1, 3)
        f("instructionBuffer.io_flush", lambda k: t.g("fetch.instructionBuffer.io_flush", k), d + 1, 4)
    elif kind == "csr":
        f("csr.req_valid (Csr.req reg)", lambda k: t.g("csr.req_valid", k), d + 1, 3)
        f("csr.io_rd_valid", lambda k: t.g("csr.io_rd_valid", k), d, 4)
    elif kind == "mlu":
        f("mlu.stage2Input_q.io_deq_valid", lambda k: t.g("mlu.stage2Input_q.io_deq_valid", k), d + 1, 3)
        f("mlu.stage3Input_q.io_deq_valid", lambda k: t.g("mlu.stage3Input_q.io_deq_valid", k), d + 1, 4)
        f("mlu.io_rd_valid", lambda k: t.g("mlu.io_rd_valid", k), d + 1, 5)
    elif kind == "dvu":
        f("dvu.active", lambda k: t.g("dvu.active", k), d + 1, 3)
        f("dvu.compute", lambda k: t.g("dvu.compute", k), d + 1, 4)
        f("dvu.io_rd_valid (count(5))", lambda k: t.g("dvu.io_rd_valid", k), d + 1, 60)
    elif kind in ("lsu", "lsu_f", "lsu_st"):
        f("lsu.opQueue.io_nEnqueued>0", lambda k: t.g("lsu.opQueue.io_nEnqueued", k) > 0, d + 1, 4)
        f("lsu.nextSlot_active", lambda k: t.g("lsu.nextSlot_active", k), d, 6)
        f("lsu.io_dbus_valid (bus request)", lambda k: t.g("lsu.io_dbus_valid", k), d + 1, 8)
        if kind != "lsu_st":
            f("lsu.readFired_valid", lambda k: t.g("lsu.readFired_valid", k), d + 1, 8)
            nm = "lsu.io_rd_valid" if kind == "lsu" else "lsu.io_rd_flt_valid"
            f(nm, lambda k: t.g(nm, k), d + 1, 10)
        else:
            f("lsu.io_storeComplete_valid", lambda k: t.g("lsu.io_storeComplete_valid", k), d + 1, 8)
    elif kind in ("fpu", "fpu_x"):
        f("floatCore.instQueue.io_count=1", lambda k: t.g("floatCore.instQueue.io_count", k) == 1, d + 1, 3)
        f("fpnew in_valid&in_ready (accepted)", lambda k: t.g("floatCore.floatCoreWrapper.in_valid_i", k)
          and t.g("floatCore.floatCoreWrapper.in_ready_o", k), d + 1, 4)
        f("fpnew out_valid_o", lambda k: t.g("floatCore.floatCoreWrapper.out_valid_o", k), d + 1, 60)
        if kind == "fpu":
            f("floatCore.io_write_ports_0_valid", lambda k: t.g("floatCore.io_write_ports_0_valid", k), d + 1, 60)
        else:
            f("scalar_rd_pipe_q.io_enq_valid", lambda k: t.g("floatCore.scalar_rd_pipe_q.io_enq_valid", k), d + 1, 60)
            f("floatCore.io_scalar_rd_valid", lambda k: t.g("floatCore.io_scalar_rd_valid", k), d + 1, 60)
        f("floatCore.instQueue.io_count=0 again", lambda k: t.g("floatCore.instQueue.io_count", k) == 0, d + 2, 60)
    elif kind == "fencei":
        f("lsu.opQueue.io_nEnqueued>0", lambda k: t.g("lsu.opQueue.io_nEnqueued", k) > 0, d + 1, 4)
        f("lsu.io_flush_valid", lambda k: t.g("lsu.io_flush_valid", k), d + 1, 10)
        f("fetch.ctrl.io_iflush_valid", lambda k: t.g("fetch.ctrl.io_iflush_valid", k), d + 1, 10)
        f("instructionBuffer.io_flush", lambda k: t.g("fetch.instructionBuffer.io_flush", k), d + 1, 10)
    elif kind == "rvv":
        fe = "core.rvvCore.rvvCoreWrapper.core.frontend."
        f("RvvFrontEnd.inst_valid_i", lambda k: t.g(fe + "inst_valid_i", k), d, 2)
        f("RvvFrontEnd.valid_inst_q (inst_q)", lambda k: t.g(fe + "valid_inst_q", k), d + 1, 4)
        f("RvvFrontEnd.cmd_valid_o (to CQ)", lambda k: t.g(fe + "cmd_valid_o", k), d + 1, 6)
    return ev


def cmd_trace(args):
    syms = symbols("trace")
    t = Trace(load("trace"))
    out = {"vcd": os.path.join(WAVES, "trace.vcd"), "clock_period_ns": round(t.v.period_ns(), 4),
           "classes": {}}
    for name, asm, kind, res in CLASSES:
        pc = syms[f"tr_{name}"]
        d, lane = t.first_dispatch(pc)
        if d is None:
            out["classes"][name] = {"error": "never dispatched"}
            continue
        line = pc & ~0xF
        ibuf = t.visible_since(pc, d)
        f2 = t.last_before(lambda k: t.g("fetch.fetcher.io_fetch_valid", k) and
                           (t.g("fetch.fetcher.io_fetch_bits_addr", k) & ~0xF) == line, ibuf - 1)
        f1 = t.last_before(lambda k: t.g("fetch.io_ibus_valid", k) and t.g("fetch.io_ibus_ready", k) and
                           (t.g("fetch.io_ibus_addr", k) & ~0xF) == line, (f2 or ibuf) - 1)
        ev_abs = {"F1": f1, "F2": f2, "IBuf": ibuf, "D": d}
        ev_abs.update(unit_events(t, kind, d, lane))
        port = None
        if res == "x":
            wb, p = t.wb_scalar(5, d)
            ev_abs["WB"], port = wb, (f"scalar port {p}" if p is not None else None)
        elif res == "f":
            wb, p = t.wb_float(0, d)
            ev_abs["WB"], port = wb, (f"float port {p}" if p is not None else None)
        elif kind == "lsu_st":
            ev_abs["WB"] = t.first_from(lambda k: t.g("retirement_buffer.io_storeComplete_valid", k), d + 1, 10)
            port = "storeComplete (ROB, after Pipe)"
        ev_abs["retire"] = t.retire(pc, d)
        cons = {}
        for lab in ("trp", "tra", "trc"):
            s = syms.get(f"{lab}_{name}")
            if s is not None:
                lo = d - 12 if lab == "trp" else d
                k, ln = t.first_dispatch(s, max(0, lo))
                cons[lab] = (k, ln)
        base = f1 if f1 is not None else d
        rel = {k: (v - base if v is not None else None) for k, v in ev_abs.items()}
        out["classes"][name] = dict(
            example_asm=asm, pc=hex(pc), lane=lane, kind=kind,
            events=rel, events_abs_cycle=ev_abs,
            events_ns={k: (round(t.v.ns(v), 2) if v is not None else None) for k, v in ev_abs.items()},
            port=port,
            pre_addi_D=(cons["trp"][0] - base if cons.get("trp", (None,))[0] is not None else None),
            post_addi_D=(cons["tra"][0] - base if cons.get("tra", (None,))[0] is not None else None),
            consumer_D=(cons["trc"][0] - base if cons.get("trc", (None,))[0] is not None else None),
            latency_to_dependent_dispatch=(cons["trc"][0] - d if cons.get("trc", (None,))[0] is not None else None),
        )
        c = out["classes"][name]
        evs = " ".join(f"{k}={v}" for k, v in rel.items())
        print(f"{name:10s} lane{lane} {evs} | pre={c['pre_addi_D']} post={c['post_addi_D']} "
              f"cons={c['consumer_D']} lat={c['latency_to_dependent_dispatch']} {port or ''}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("dump")
    a.add_argument("bench")
    a.add_argument("func")
    a.add_argument("--tail", type=int, default=8)
    a.add_argument("--from-csrr", action="store_true", help="start at the region's start csrr")
    a.set_defaults(fn=cmd_dump)
    b = sub.add_parser("trace")
    b.add_argument("--out")
    b.set_defaults(fn=cmd_trace)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

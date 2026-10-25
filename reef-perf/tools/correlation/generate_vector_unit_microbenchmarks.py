"""Write the vector functional unit microbenchmarks (roadmap B15).

Writes one dependent chain and one independent stream per M3 vector
functional unit type, plus LMUL=4 streams, as
microbenchmarks/vector_units/<name>.S. Every file has the same shape: a
setup that fills the source registers, then 250 iterations of a loop of 6
or 8 vector instructions. SPECS lists what differs between files.

The hand-written files in that directory (vmixed_lmul, vdiv_m8_indep,
vdiv_m8_then_vadd_m8) don't fit this shape and aren't generated.

Usage (from reef-perf/):

    python3 tools/correlation/generate_vector_unit_microbenchmarks.py
    python3 tools/correlation/generate_vector_unit_microbenchmarks.py \
        --out /tmp/vector_units
"""

import argparse
import sys
from pathlib import Path

ITERATIONS = 250
"""Loop iterations in the ROI."""

# Vector register setups, run once before the ROI.
INT_M1 = """  vsetivli zero, 4, e32, m1, ta, ma
  vmv.v.i  v1, 1
  vmv.v.i  v2, 1
  vmv.v.i  v3, 0"""
FP_M1 = """  vsetivli zero, 4, e32, m1, ta, ma
  li       t0, 0x3f800000   # 1.0f
  vmv.v.x  v1, t0
  vmv.v.x  v2, t0
  vmv.v.i  v3, 0            # 0.0f"""
E16_M1 = """  vsetivli zero, 8, e16, m1, ta, ma
  vmv.v.i  v1, 1
  vmv.v.i  v2, 1"""
INT_M4 = """  vsetivli zero, 16, e32, m4, ta, ma
  vmv.v.i  v0, 1
  vmv.v.i  v4, 1"""
E16_M4 = """  li       t0, 32
  vsetvli  zero, t0, e16, m4, ta, ma
  vmv.v.i  v0, 1
  vmv.v.i  v4, 1"""
# Divides by -1 flip the sign each time, so no two consecutive divides see the
# same operands: M3's dividers reuse the previous result when they do.
DIV_BIG = """  vsetivli zero, 4, e32, m1, ta, ma
  li       t0, 0x7fffffff
  vmv.v.x  v1, t0
  li       t0, 0x7ffffffe
  vmv.v.x  v4, t0
  vmv.v.i  v2, -1"""
DIV_SMALL = """  vsetivli zero, 4, e32, m1, ta, ma
  vmv.v.i  v1, 1
  vmv.v.i  v4, 2
  vmv.v.i  v2, -1"""
FP_DIV = """  vsetivli zero, 4, e32, m1, ta, ma
  li       t0, 0x3f800000   # 1.0f
  vmv.v.x  v1, t0
  li       t0, 0x40000000   # 2.0f
  vmv.v.x  v4, t0
  li       t0, 0xbf800000   # -1.0f
  vmv.v.x  v2, t0"""

M1_DESTS = [f"v{8 + i}" for i in range(8)]
"""Destinations of an LMUL=1 independent stream."""
M4_DESTS = ["v8", "v12", "v16", "v20", "v24", "v28"]
"""Destinations of an LMUL=4 independent stream: one register group each."""


def chain(op: str) -> list[str]:
    """Return 8 copies of an op that reads its own result: a latency chain."""
    return [op] * 8


def stream(op: str, dests: list[str]) -> list[str]:
    """Return an op writing each destination in turn: a throughput stream.

    Args:
        op: The instruction, with {d} where the destination goes.
        dests: The destinations.
    """
    return [op.format(d=d) for d in dests]


def alternating(op: str) -> list[str]:
    """Return 8 independent ops whose dividend alternates between v1 and v4.

    Args:
        op: The instruction, with {d} for the destination and {s} for the
            dividend.
    """
    return [
        op.format(d=d, s="v1" if i % 2 == 0 else "v4")
        for i, d in enumerate(M1_DESTS)
    ]


# name: (what it measures, setup, loop body)
SPECS: dict[str, tuple[str, str, list[str]]] = {
    # ALU reservation station, 2 ALUs.
    "vadd_chain": (
        "ALU add/subtract path latency",
        INT_M1,
        chain("vadd.vv  v1, v1, v2"),
    ),
    "vadd_indep": (
        "ALU throughput",
        INT_M1,
        stream("vadd.vv  {d}, v1, v2", M1_DESTS),
    ),
    "vadd_m4_indep": (
        "ALU throughput, 4 uops per instruction",
        INT_M4,
        stream("vadd.vv  {d}, v0, v4", M4_DESTS),
    ),
    "vand_chain": (
        "ALU latency outside the add/subtract path",
        INT_M1,
        chain("vand.vv  v1, v1, v2"),
    ),
    "vmseq_m4_indep": (
        "compare (ALU) throughput; a compare takes one vector ROB entry",
        INT_M4,
        stream("vmseq.vv {d}, v0, v4", M4_DESTS),
    ),
    # MUL reservation station, 2 multipliers.
    "vmul_chain": ("multiplier latency", INT_M1, chain("vmul.vv  v1, v1, v2")),
    "vmul_indep": (
        "multiplier throughput",
        INT_M1,
        stream("vmul.vv  {d}, v1, v2", M1_DESTS),
    ),
    "vmul_m4_indep": (
        "multiplier throughput, 4 uops per instruction",
        INT_M4,
        stream("vmul.vv  {d}, v0, v4", M4_DESTS),
    ),
    "vmacc_chain": (
        "multiply-accumulate latency through the accumulator",
        INT_M1,
        chain("vmacc.vv v1, v2, v3"),
    ),
    # DIV reservation station, 1 divider; its time grows with the dividend's
    # significant bits.
    "vdiv_chain": (
        "divider latency, 31-bit dividends (x / -1)",
        DIV_BIG,
        chain("vdiv.vv  v1, v1, v2"),
    ),
    "vdiv_indep": (
        "divider throughput, 31-bit dividends",
        DIV_BIG,
        alternating("vdiv.vv  {d}, {s}, v2"),
    ),
    "vdiv_small_chain": (
        "divider latency, 1-bit dividends (x / -1)",
        DIV_SMALL,
        chain("vdiv.vv  v1, v1, v2"),
    ),
    "vdiv_small_indep": (
        "divider throughput, 1- and 2-bit dividends",
        DIV_SMALL,
        alternating("vdiv.vv  {d}, {s}, v2"),
    ),
    # PMTRDT reservation station, 1 permute/reduction unit.
    "vredsum_chain": (
        "reduction latency",
        INT_M1,
        chain("vredsum.vs v1, v2, v1"),
    ),
    "vredsum_indep": (
        "reduction throughput",
        INT_M1,
        stream("vredsum.vs {d}, v2, v1", M1_DESTS),
    ),
    "vredsum_m4_indep": (
        "reduction throughput, 4 uops per instruction; one vector ROB entry",
        INT_M4,
        stream("vredsum.vs {d}, v4, v0", M4_DESTS),
    ),
    "vwredsum_indep": (
        "widening reduction throughput",
        E16_M1,
        stream("vwredsum.vs {d}, v2, v1", M1_DESTS),
    ),
    "vwredsum_m4_indep": (
        "widening reduction throughput, 4 uops per instruction",
        E16_M4,
        stream("vwredsum.vs {d}, v4, v0", M4_DESTS),
    ),
    "vslidedown_chain": (
        "permute latency",
        INT_M1,
        chain("vslidedown.vi v1, v1, 1"),
    ),
    "vslidedown_indep": (
        "permute throughput",
        INT_M1,
        stream("vslidedown.vi {d}, v2, 1", M1_DESTS),
    ),
    "vrgather_chain": (
        "gather latency (vd may not overlap a source, so two registers "
        "alternate)",
        INT_M1,
        ["vrgather.vv v4, v1, v3", "vrgather.vv v1, v4, v3"] * 4,
    ),
    "vcpop_indep": (
        "vcpop.m (permute/reduction unit) to a scalar register",
        INT_M1,
        [f"vcpop.m  a{i}, v2" for i in range(8)],
    ),
    # FMA reservation station, 2 FMA units. 1.0 + 0.0 and 1.0 * 1.0
    # throughout.
    "vfadd_chain": ("FP add latency", FP_M1, chain("vfadd.vv v1, v1, v3")),
    "vfadd_indep": (
        "FP add throughput",
        FP_M1,
        stream("vfadd.vv {d}, v1, v3", M1_DESTS),
    ),
    "vfmacc_chain": (
        "FP multiply-add latency through the accumulator",
        FP_M1,
        chain("vfmacc.vv v1, v2, v3"),
    ),
    # FP divide goes to the DIV reservation station.
    "vfdiv_chain": (
        "FP divide latency (x / -1.0)",
        FP_DIV,
        chain("vfdiv.vv v1, v1, v2"),
    ),
    "vfdiv_indep": (
        "FP divide throughput",
        FP_DIV,
        alternating("vfdiv.vv {d}, {s}, v2"),
    ),
    "vfredusum_chain": (
        "FP reduction latency",
        FP_M1,
        chain("vfredusum.vs v1, v2, v1"),
    ),
    "vfredusum_indep": (
        "FP reduction throughput",
        FP_M1,
        stream("vfredusum.vs {d}, v2, v1", M1_DESTS),
    ),
}
"""What each microbenchmark measures, its setup and its loop body."""


def render(name: str, what: str, setup: str, body: list[str]) -> str:
    """Return the assembly source of one microbenchmark.

    Args:
        name: Its name; "chain" in it makes it a dependent chain.
        what: What it measures, for the header comment.
        setup: Assembly run once before the ROI.
        body: The vector instructions of one loop iteration.
    """
    kind = "Dependent chain" if "chain" in name else "Independent stream"
    mnemonic = body[0].split()[0]
    lines = [
        f"# {kind}: {len(body)} x {mnemonic} per iteration, "
        f"{ITERATIONS} iterations.",
        f"# Measures the vector {what}.",
        '.include "common.inc"',
        "  START",
        setup,
        f"  li       t6, {ITERATIONS}",
        "  ROI_BEGIN",
        "  .balign 16            # loop at slot 0 of a 16-byte fetch block",
        "1:",
        *[f"  {inst}" for inst in body],
        "  addi     t6, t6, -1",
        "  bnez     t6, 1b",
        "  ROI_END",
        "  END",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    """Write every microbenchmark in SPECS."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("microbenchmarks/vector_units"),
        help="directory to write to (default: microbenchmarks/vector_units)",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (what, setup, body) in SPECS.items():
        (args.out / f"{name}.S").write_text(render(name, what, setup, body))
    print(f"wrote {len(SPECS)} files to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

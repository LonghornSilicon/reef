#!/usr/bin/env python3
"""Roll up Yosys `stat -json` area by block for RvvCoreMiniAxi.

yosys-slang (--keep-hierarchy) names every module `<Name>$RvvCoreMiniAxi.<instance path>`,
so the instance path tells us which block a module belongs to. A module without a path
inherits the block of the instance that contains it.

Local area is recomputed from num_cells_by_type and the liberty cell areas, because the
JSON `area` field of a non-leaf module already includes its submodules.

Usage: area_report.py stat.json --liberty foo.lib --nand2 NAND2_X1 [--top reef_top]
                      [--tree-depth 7 --tree-min 1.0]

With --top reef_top, the core sits under `u_core.` and reef_top's own blocks are reported as
"Reef shell".
"""
import argparse
import json
import re
import sys
from collections import defaultdict

TOP = "RvvCoreMiniAxi"

# First matching prefix wins, so list longer prefixes first.
BLOCKS = [
    ("core.rvvCore", "RVV vector unit"),
    ("core.score.floatCore", "Scalar FPU"),
    ("core.score", "Scalar core (no FPU)"),
    ("core", "Core glue"),
    ("itcm", "TCM logic (macros excluded)"),
    ("itcmWrapper", "TCM logic (macros excluded)"),
    ("itcmArbiter", "TCM logic (macros excluded)"),
    ("dtcm", "TCM logic (macros excluded)"),
    ("dtcmWrapper", "TCM logic (macros excluded)"),
    ("dtcmArbiter", "TCM logic (macros excluded)"),
    ("axiSlave", "AXI / fabric"),
    ("ebus2axi", "AXI / fabric"),
    ("ibus2axi", "AXI / fabric"),
    ("fabricMux", "AXI / fabric"),
    ("readAddrArb", "AXI / fabric"),
    ("dm", "Debug module"),
]
OTHER = "Top-level CSR / clock / reset / misc"
REEF_SHELL = "Reef shell (SPI bridge, DECERR sink, reset sync)"
CORE_INST = "u_core"


def parse_liberty(path):
    """Return {cell: (area, is_sequential)}."""
    text = open(path).read()
    cells = {}
    for chunk in re.split(r"\n\s*cell\s*\(", text)[1:]:
        m = re.match(r'\s*"?([\w$]+)"?\s*\)', chunk)
        a = re.search(r"\barea\s*:\s*([0-9.eE+-]+)", chunk)
        if m and a:
            seq = re.search(r"\n\s*(ff|latch|ff_bank|latch_bank)\s*\(", chunk) is not None
            cells[m.group(1)] = (float(a.group(1)), seq)
    return cells


def inst_path(mod):
    if mod == TOP:
        return ""
    key = "$" + TOP + "."
    return mod.split(key, 1)[1] if key in mod else None


def classify(path):
    if TOP != "RvvCoreMiniAxi":
        if path == CORE_INST:
            return OTHER
        if not path.startswith(CORE_INST + "."):
            return REEF_SHELL
        path = path[len(CORE_INST) + 1:]
    for prefix, block in BLOCKS:
        if path == prefix or path.startswith(prefix + "."):
            return block
    return OTHER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stat_json")
    ap.add_argument("--liberty", required=True)
    ap.add_argument("--nand2", required=True, help="NAND2 X1 cell name for gate equivalents")
    ap.add_argument("--top", default="RvvCoreMiniAxi")
    ap.add_argument("--tree-depth", type=int, default=7)
    ap.add_argument("--tree-min", type=float, default=1.0, help="min %% of total to show in tree")
    args = ap.parse_args()
    global TOP
    TOP = args.top

    lib = parse_liberty(args.liberty)
    nand2 = lib[args.nand2][0]
    # Module keys carry Yosys's leading backslash; cell types that name a module don't.
    mods = {k.lstrip("\\"): v for k, v in json.load(open(args.stat_json))["modules"].items()}

    block_area = defaultdict(float)
    block_seq = defaultdict(float)
    block_cells = defaultdict(int)
    macros = defaultdict(int)
    unknown = defaultdict(int)
    hier = {}  # module -> hierarchical area (memoized)

    def hier_area(mod):
        if mod in hier:
            return hier[mod]
        total = 0.0
        for typ, n in mods[mod]["num_cells_by_type"].items():
            if typ in mods:
                total += n * hier_area(typ)
            elif typ in lib:
                total += n * lib[typ][0]
        hier[mod] = total
        return total

    def walk(mod, block, mult):
        for typ, n in mods[mod]["num_cells_by_type"].items():
            n *= mult
            if typ in mods:
                p = inst_path(typ)
                walk(typ, classify(p) if p is not None else block, n)
            elif typ in lib:
                area, seq = lib[typ]
                block_area[block] += n * area
                block_cells[block] += n
                if seq:
                    block_seq[block] += n * area
            elif typ.split("$")[0] == "Sram":
                macros[typ] += n
            else:
                unknown[typ] += n

    top = TOP
    walk(top, OTHER, 1)
    total = sum(block_area.values())

    order = [b for _, b in BLOCKS] + [OTHER, REEF_SHELL]
    seen = []
    for b in order:
        if b not in seen and b in block_area:
            seen.append(b)

    print(f"## {TOP} area by block\n")
    print(f"Liberty: `{args.liberty.rsplit('/', 1)[-1]}`, NAND2 = `{args.nand2}` "
          f"({nand2:.4g} um^2). Std cells only; TCM SRAM macros are excluded.\n")
    print("| Block | Cells | Area (um^2) | % | kGE (NAND2-eq) | Seq % |")
    print("|---|---:|---:|---:|---:|---:|")
    for b in seen:
        a = block_area[b]
        print(f"| {b} | {block_cells[b]:,} | {a:,.0f} | {100*a/total:.1f} | "
              f"{a/nand2/1000:,.1f} | {100*block_seq[b]/a if a else 0:.0f} |")
    tc = sum(block_cells.values())
    ts = sum(block_seq.values())
    print(f"| **Total** | **{tc:,}** | **{total:,.0f}** | 100 | **{total/nand2/1000:,.1f}** | "
          f"{100*ts/total:.0f} |")

    print("\n### Blackboxed macros (not in totals)\n")
    for typ, n in sorted(macros.items()):
        print(f"- `{typ}` x{n}")
    if not macros:
        print("- none")
    if unknown:
        print("\n### WARNING: cells with no liberty area (not in totals)\n")
        for typ, n in sorted(unknown.items(), key=lambda kv: -kv[1])[:20]:
            print(f"- `{typ}` x{n}")

    print(f"\n### Largest instances (depth <= {args.tree_depth}, >= {args.tree_min}% of total)\n")
    print("| Instance path | Area (um^2) | % |")
    print("|---|---:|---:|")
    rows = []
    for mod in mods:
        p = inst_path(mod)
        if not p or p.count(".") >= args.tree_depth:
            continue
        a = hier_area(mod)
        if 100 * a / total >= args.tree_min:
            rows.append((p, a))
    for p, a in sorted(rows):
        print(f"| `{p}` | {a:,.0f} | {100*a/total:.1f} |")

    # Cross-check against Yosys's own hierarchical number.
    ys = mods[top].get("area")
    if ys is not None and abs(ys - total) > 1e-3 * max(ys, 1):
        print(f"\nWARNING: rollup total {total:,.0f} != yosys top area {ys:,.0f}", file=sys.stderr)


if __name__ == "__main__":
    main()

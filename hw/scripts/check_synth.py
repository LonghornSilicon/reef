#!/usr/bin/env python3
"""Pass/fail check of a Yosys `stat -json` netlist summary (scripts/synth_yosys.sh output), for CI.

Counts every leaf cell through the module hierarchy (yosys-slang --keep-hierarchy keeps one module
per instance path, so a module's num_cells_by_type mixes liberty cells and submodule names) and fails if
  - any leaf is neither a standard cell (--std-prefix) nor an allowed macro (--macro): an unmapped
    Yosys cell ($_DFF_P_, ...) or a black box the filelist left unresolved;
  - an --expect NAME=COUNT does not match (e.g. 5 TCM macros, 2 clock gates);
  - the standard-cell area exceeds --max-area-mm2.
Prints a Markdown summary (CI appends it to the job summary).

Usage: check_synth.py stat.json --top reef_top --std-prefix sky130_fd_sc_hd__ \\
         --macro sram22_512x128m4w8 --expect sram22_512x128m4w8=5 \\
         --expect sky130_fd_sc_hd__sdlclkp_4=2 [--max-area-mm2 8.0]
"""
import argparse
import json
import sys
from collections import Counter


def flatten(modules, name, mult, leaves, stack=()):
    if name in stack:
        sys.exit(f"check_synth: recursive hierarchy at {name}")
    for cell, n in modules[name].get("num_cells_by_type", {}).items():
        if cell in modules:
            flatten(modules, cell, mult * n, leaves, stack + (name,))
        else:
            leaves[cell] += mult * n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stat_json")
    ap.add_argument("--top", required=True)
    ap.add_argument("--std-prefix", required=True, help="standard-cell name prefix, e.g. sky130_fd_sc_hd__")
    ap.add_argument("--macro", action="append", default=[], help="allowed macro cell (repeatable)")
    ap.add_argument("--expect", action="append", default=[], metavar="CELL=N", help="exact leaf count")
    ap.add_argument("--max-area-mm2", type=float, help="fail above this standard-cell area")
    args = ap.parse_args()

    stat = json.load(open(args.stat_json))
    # Module keys are RTLIL ids ("\\name"); cell types in num_cells_by_type are not escaped.
    modules = {k[1:] if k.startswith("\\") else k: v for k, v in stat["modules"].items()}
    top = args.top
    if top not in modules:
        sys.exit(f"check_synth: top {args.top} not in {args.stat_json}")

    leaves = Counter()
    flatten(modules, top, 1, leaves)
    std = {c: n for c, n in leaves.items() if c.startswith(args.std_prefix)}
    macros = {c: n for c, n in leaves.items() if c in args.macro}
    other = {c: n for c, n in leaves.items() if c not in std and c not in macros}
    area_mm2 = stat["design"]["area"] / 1e6
    seq_mm2 = stat["design"].get("sequential_area", 0.0) / 1e6

    errors = []
    if other:
        errors.append("cells that are not standard cells or allowed macros: "
                      + ", ".join(f"{c} x{n}" for c, n in sorted(other.items())))
    for e in args.expect:
        cell, _, want = e.partition("=")
        if leaves.get(cell, 0) != int(want):
            errors.append(f"{cell}: {leaves.get(cell, 0)} instances, expected {want}")
    if args.max_area_mm2 is not None and area_mm2 > args.max_area_mm2:
        errors.append(f"standard-cell area {area_mm2:.3f} mm^2 > budget {args.max_area_mm2:.3f} mm^2")

    print(f"### Synthesis check: `{args.top}`\n")
    print("| | |\n|---|---|")
    print(f"| Standard cells | {sum(std.values()):,} instances, {area_mm2:.3f} mm^2 "
          f"({seq_mm2:.3f} mm^2 sequential){'' if args.max_area_mm2 is None else f', budget {args.max_area_mm2} mm^2'} |")
    for c, n in sorted(macros.items()):
        print(f"| Macro `{c}` | {n} |")
    for e in args.expect:
        cell = e.partition("=")[0]
        if cell not in macros:
            print(f"| `{cell}` | {leaves.get(cell, 0)} |")
    print(f"| Result | {'**FAIL**' if errors else 'PASS'} |\n")
    for err in errors:
        print(f"- {err}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

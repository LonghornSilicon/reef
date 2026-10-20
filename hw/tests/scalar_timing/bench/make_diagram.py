#!/usr/bin/env python3
"""Build the timing-diagram page from deliverable/timing.json.

    python3 bench/make_diagram.py            # -> deliverable/diagram.html

The page (bench/diagram_template.html) is a self-contained HTML artifact: pipeline
block diagram, per-class cycle chart, latency/throughput table, draft verdicts.
FPU classes carry both timings: the M3 model as released and the FPU-fix rebuild
(timing.json `source_intent_if_different`, from bench/collect.py).
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from collect import unit_key  # noqa: E402  (same raw-signal -> event-name mapping as timing.json)

CAT = {"alu": "int", "zbb": "int", "lui": "int", "auipc": "int",
       "br_nt_fwd": "br", "br_t_bwd": "br", "br_mp_fwd": "br", "br_mp_bwd": "br", "jal": "br", "jalr": "br",
       "mul": "md", "div_small": "md", "div_large": "md", "div_by0": "md",
       "lw": "mem", "sw": "mem", "flw": "mem", "fsw": "mem",
       "fadd": "fp", "fmul": "fp", "fmadd": "fp", "fdiv": "fp", "fsqrt": "fp", "feq": "fp", "fcvt_w_s": "fp",
       "fmv_x_w": "fp", "fmv_w_x": "fp",
       "csr": "sys", "fence": "sys", "fence_i": "sys", "rvv": "vec"}
UNIT = {"int": "ALU", "br": "BRU", "md": "MLU", "mem": "LSU", "fp": "FPU", "sys": "CSR", "vec": "RVV queue"}
UNIT_BY_ID = {"div_small": "DVU", "div_large": "DVU", "div_by0": "DVU", "fence": "", "fence_i": "LSU flush"}


def abs_by_event(c):
    out = {}
    for k, v in (c.get("events_abs_ns") or {}).items():
        name = k if k in ("F1", "F2", "IBuf", "D", "WB", "retire") else unit_key(k)
        if name and v is not None:
            out[name] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(SHARED, "deliverable", "diagram.html"),
                    help="default: deliverable/diagram.html (committed; open it in a browser)")
    a = ap.parse_args()
    with open(os.path.join(SHARED, "deliverable", "timing.json"), "rb") as f:
        raw = f.read()
    tj = json.loads(raw)
    with open(os.path.join(HERE, "diagram_template.html")) as f:
        tpl = f.read()
    classes = []
    for c in tj["classes"]:
        if "error" in c:
            continue
        cat = CAT[c["trace_id"]]
        si = c.get("source_intent_if_different") or {}
        intent = None
        if si.get("events"):
            intent = {"events": si["events"], "lat": si.get("latency_to_dependent_dispatch"),
                      "thr": si.get("throughput_cycles")}
        classes.append({
            "id": c["trace_id"], "label": c["class"], "asm": c["example_asm"], "cat": cat,
            "unit": UNIT_BY_ID.get(c["trace_id"], UNIT[cat]),
            "level": c["verified"], "signoff": c.get("human_signoff"), "partial": c.get("partial_signoffs"),
            "lanes": c["lane_rules"], "notes": c.get("notes"), "rows": c.get("checklist_rows") or [],
            "events": c["events"], "abs": abs_by_event(c),
            "lat": c.get("latency_to_dependent_dispatch"), "thr": c.get("throughput_cycles"),
            "thrKind": c.get("throughput_kind"), "thrSrc": c.get("throughput_source"),
            "intent": intent,
        })
    # Stamp the page with the data it was built from, not a date, so regenerating it is reproducible.
    stamp = "timing.json " + hashlib.sha256(raw).hexdigest()[:10]
    data = {"generated": stamp, "source": tj["source_commit"], "classes": classes}
    html = tpl.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    with open(a.out, "w") as f:
        f.write(html)
    print(f"wrote {a.out}: {len(classes)} classes, {sum(1 for c in classes if c['intent'])} with FPU-fix timing, "
          f"{len(html) // 1024} KB")


if __name__ == "__main__":
    main()

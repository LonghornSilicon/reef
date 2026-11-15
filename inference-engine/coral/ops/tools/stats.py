#!/usr/bin/env python3
"""Summarise reef-perf JSON runs of an operator kernel.

  stats.py run.base.json run.fast.json run.ideal.json [--macs N] [--json]

Per run: ROI cycles (first to last mcycle read), the phase breakdown between
mcycle reads, and where on the NPU the time went: busy cycles of each
hardware pool as a % of the ROI. Pool counters cover the whole run (boot +
init + ROI); the boot is small next to a real kernel, but treat % for tiny
kernels with care.

Units (reef-perf pool names):
  alu / mul / div        scalar integer core (vector integer ops also issue here on M3)
  lsu                    load/store unit: scalar + vector loads/stores, Zvt operand loads
  dtcm                   the 32 KB data TCM's port
  matrix_array           the Zvt PE array (MACs)
  matrix_cmd/matrix_ctrl matrix command issue / controller (moves, zero)
  permute_reduce         vector permute / reduction unit
  ext_mem_link           weight flash link (XSPI)
  ext_ram_link (if any)  PSRAM link (activations / KV)
  axi_*                  the DMA's bus
"""
import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def summarise(path, macs=None):
    d = load(path)
    reads = d.get("mcycle_reads", [])
    roi = reads[-1] - reads[0] if len(reads) >= 2 else d["cycles"]
    phases = [reads[i + 1] - reads[i] for i in range(len(reads) - 1)]
    pools = {}
    for name, p in d.get("pools", {}).items():
        busy = p.get("busy_cycles", 0)
        if busy:
            # A pool with count > 1 can be busy more than 1 per cycle; report
            # busy per instance.
            pools[name] = busy / max(1, p.get("count", 1))
    m = d.get("matrix", {})
    out = {
        "file": path,
        "roi_cycles": roi,
        "total_cycles": d["cycles"],
        "instructions": d.get("instructions"),
        "ipc": d.get("ipc"),
        "phases": phases,
        "pool_busy_pct_of_roi": {k: 100.0 * v / roi for k, v in sorted(pools.items(), key=lambda kv: -kv[1])},
        "matrix_useful_macs": m.get("useful_macs"),
        "matrix_mac_utilization_when_busy": m.get("mac_utilization"),
        "matrix_array_busy_cycles": m.get("array_busy_cycles"),
    }
    if macs:
        out["macs_per_cycle"] = macs / roi
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    macs = None
    if "--macs" in sys.argv:
        macs = float(sys.argv[sys.argv.index("--macs") + 1])
        args = [a for a in args if a != sys.argv[sys.argv.index("--macs") + 1]]
    res = [summarise(a, macs) for a in args]
    if "--json" in sys.argv:
        print(json.dumps(res, indent=1))
        return
    for r in res:
        cfg = r["file"].rsplit(".", 2)[-2]
        print(f"[{cfg:5s}] ROI {r['roi_cycles']:>12,} cycles  instr {r['instructions']:,}  "
              f"matrix useful MACs {r['matrix_useful_macs'] or 0:,}"
              + (f"  MAC/cycle {r['macs_per_cycle']:.2f}" if 'macs_per_cycle' in r else ""))
        if len(r["phases"]) > 1:
            print("        phases:", ", ".join(f"{p:,}" for p in r["phases"][:12]),
                  "..." if len(r["phases"]) > 12 else "")
        print("        busy % of ROI:", ", ".join(
            f"{k} {v:.1f}" for k, v in r["pool_busy_pct_of_roi"].items() if v >= 0.5))


if __name__ == "__main__":
    main()

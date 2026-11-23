"""expcmp.py <exp prefix> [maxb]: experiment A or B (int8, both sizes) against the base (Z1) and int4 (S2)."""
import os, sys
import metrics as M
H = os.path.dirname(os.path.abspath(__file__)); R = lambda p: os.path.join(H, "runs", p + ".json")
e = sys.argv[1]; maxb = int(sys.argv[2]) if len(sys.argv) > 2 else 15
for v, i4, nl, hid in (("i8", "S2i4", 8, 512), ("si8", "S2si4", 6, 256)):
    print(f"== {e}{v} vs base Z1{v} (int8) and {i4} (int4)")
    for B in [1, 2, 4, 8] + ([maxb] if v == "si8" else []):
        f = R(f"{e}{v}_b{B}")
        if not os.path.exists(f): print(f"  b{B}: n/a"); continue
        m, b, w4 = M.load(f), M.load(R(f"Z1{v}_b{B}")), M.load(R(f"{i4}_b{B}"))
        t = lambda x: B * 50e6 / x["roi"]
        print(f"  b{B}: {t(m):.2f} tok/s ({(t(m) / t(b) - 1) * 100:+.1f}% vs int8, {(t(m) / t(w4) - 1) * 100:+.1f}% vs int4) | "
              f"flash {m['flash'] / 1e6:.2f} M (int8 {b['flash'] / 1e6:.2f}) | MACs x{m['macs'] / b['macs']:.2f} | "
              f"busy flash {m['flash'] / m['roi'] * 100:.0f}% psram {m['psram'] / m['roi'] * 100:.0f}% lsu {m['busy']['lsu']:.0f}% | top {m['stalls'][0][1]} {m['stalls'][0][0]:.0f}%")
    if os.path.exists(R(f"{e}{v}_ttft32")):
        m, b = M.load(R(f"{e}{v}_ttft32"), 4), M.load(R(f"Z1{v}_ttft32"), 4)
        print(f"  TTFT {m['roi'] / 50e6:.3f} s (int8 {b['roi'] / 50e6:.3f} s)")

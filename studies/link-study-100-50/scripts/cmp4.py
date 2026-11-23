"""cmp4.py <before> <after>: compares a step on the four workloads. before 'base' = the Phase 6 runs."""
import os, sys
import metrics as M
H = os.path.dirname(os.path.abspath(__file__)); R = lambda p: os.path.join(H, "runs", p + ".json")
BASE = {"i8": "s320", "i4": "w4", "si8": "sm", "si4": "smw4"}
NAME = {"i8": "52M int8", "i4": "52M int4", "si8": "small int8", "si4": "small int4"}
a, b = sys.argv[1], sys.argv[2]; maxb = int(sys.argv[3]) if len(sys.argv) > 3 else 15
best = 0
for v in ("i8", "i4", "si8", "si4"):
    pa = BASE[v] if a == "base" else a + v; pb = b + v
    bs = [1, 2, 4, 8] + ([maxb] if v.startswith("s") else [])
    cells = []
    for B in bs:
        fa, fb = R(f"{pa}_b{B}"), R(f"{pb}_b{B}")
        if not (os.path.exists(fa) and os.path.exists(fb)): cells.append(f"b{B} n/a"); continue
        ma, mb = M.load(fa), M.load(fb)
        assert ma["macs"] == mb["macs"], f"MAC count changed {pb}_b{B}"
        va, vb = B * 50e6 / ma["roi"], B * 50e6 / mb["roi"]; g = (vb / va - 1) * 100; best = max(best, g)
        cells.append(f"b{B} {vb:.2f} ({g:+.1f}%)")
    ta, tb = M.load(R(f"{pa}_ttft32"), 4), M.load(R(f"{pb}_ttft32"), 4); g = (ta["roi"] / tb["roi"] - 1) * 100; best = max(best, g)
    print(f"{NAME[v]:11s}: " + " | ".join(cells) + f" | TTFT {tb['roi'] / 50e6:.3f} s ({-g:+.1f}%)")
print(f"largest gain {best:.1f}% -> {'KEEP' if best > 2 else 'reject'}")

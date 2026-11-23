"""Study metrics for one run: speed, link use, matrix use, and where the time goes (the bottleneck evidence)."""
import json, sys
HZ, PEAK = 50e6, 128                      # 50 MHz; 32 PEs x 4 int8 MACs per cycle
LAYER = ["layernorm", "qkv", "attention", "attn_out", "layernorm", "fc1", "fc2"]
def names(reads, passes):                 # kernel name for each interval between mcycle reads
    layers = ((len(reads) - 3) // passes - 1) // len(LAYER)  # 8 (28M) or 6 (small model)
    seq = (["embed"] + LAYER * layers) * passes + ["layernorm", "lm_head"]
    assert len(seq) == len(reads) - 1, (len(seq), len(reads))
    return seq
def load(path, passes=1):
    j = json.load(open(path)); r = j["mcycle_reads"]; roi = r[-1] - r[0]; k = {}
    for i, n in enumerate(names(r, passes)): k[n] = k.get(n, 0) + r[i + 1] - r[i]
    P, c = j["pools"], j["cycles"]; td = j["topdown"]; s = td["slots"]
    stalls = sorted(((v / s * 100, f"{a}.{b}") for a, d in td["backend_bound"].items() for b, v in d.items()), reverse=True)
    return dict(roi=roi, kernels=k, macs=j["matrix"]["useful_macs"],
                flash=P["ext_mem_link"]["busy_cycles"], psram=P["ext_ram_link"]["busy_cycles"],
                busy={n: P[n]["busy_cycles"] / c * 100 for n in ("ext_mem_link", "ext_ram_link", "lsu", "dtcm", "matrix_array", "alu")},
                retiring=td["retiring"] / s * 100, frontend=sum(td["frontend_bound"].values()) / s * 100, stalls=stalls[:4])
def report(m, label, per_token_items=1):
    roi = m["roi"]
    print(f"== {label}: {roi:,} cycles = {roi / HZ:.3f} s; {per_token_items * HZ / roi:.3f} per s; matrix util {m['macs'] / (roi * PEAK) * 100:.2f}%")
    print(f"   links: flash busy {m['flash']:,} cyc ({m['flash'] / roi * 100:.1f}% of run), PSRAM busy {m['psram']:,} ({m['psram'] / roi * 100:.1f}%)")
    print("   units busy %: " + ", ".join(f"{k} {v:.1f}" for k, v in m["busy"].items()))
    print(f"   dispatch slots: retiring {m['retiring']:.1f}%, frontend {m['frontend']:.1f}%, top stalls " + ", ".join(f"{n} {v:.1f}%" for v, n in m["stalls"]))
    print("   kernels: " + ", ".join(f"{n} {v / roi * 100:.1f}%" for n, v in sorted(m["kernels"].items(), key=lambda x: -x[1])))
if __name__ == "__main__":
    d = load(sys.argv[1]); report(d, "decode of token 200, batch 1")
    t = load(sys.argv[2], passes=4); report(t, "time to first token, 32-token prompt (4 passes of 8)", per_token_items=0)

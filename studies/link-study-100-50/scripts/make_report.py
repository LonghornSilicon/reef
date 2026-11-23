"""Builds REPORT.md and report_fig.png from the study runs."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import metrics as M
H = os.path.dirname(os.path.abspath(__file__)); R = lambda p: os.path.join(H, "runs", p + ".json")
BATCHES = [1, 2, 4, 8]
# (step, change, kind, run prefix or None, kept)
STEPS = [
    ("0", "Baseline: M3 core, Joshua's firmware", "", "s0", True),
    ("3.1", "Real multi-story decode", "Firmware", "s1", True),
    ("3.2", "16-entry ROB", "Upstream Coral hardware", "s2", True),
    ("3.3", "Newer LSU (LsuV3)", "Upstream Coral hardware", "s3", True),
    ("3.5", "One pointer per row, running counters", "Firmware", "s4", False),
    ("3.5b", "Alternate register sets for weight loads", "Firmware", "s5", False),
    ("3.6", "Two-pass decode attention", "Firmware", "s6", False),
    ("3.4", "Segment-load weight layout", "Firmware and flash layout", "s7", False),
]
EXTRA = json.load(open(os.path.join(H, "steps_extra.json"))) if os.path.exists(os.path.join(H, "steps_extra.json")) else []
STEPS += [tuple(x) for x in EXTRA]
base = M.load(R("s0_dec200")); F, P = base["flash"], base["psram"]; MAC1 = base["macs"]
def dec(prefix, B):
    p = R(f"{prefix}_b{B}") if prefix != "s0" else (R("s0_dec200") if B == 1 else None)
    if not p or not os.path.exists(p): return None
    m = M.load(p); return dict(tps=B * 50e6 / m["roi"], util=m["macs"] / (m["roi"] * 128) * 100, share=(F + B * P - min(QKV, B * P)) / m["roi"] * 100)
def ttft(prefix):
    p = R(f"{prefix}_ttft32" if prefix != "s0" else "s0_ttft32")
    if not os.path.exists(p): return None
    m = M.load(p, 4); return dict(s=m["roi"] / 50e6, share=m["flash"] / m["roi"] * 100, util=m["macs"] / (m["roi"] * 128) * 100)
ceil = {B: dict(tps=B * 50e6 / max(F, B * P), util=B * MAC1 / (max(F, B * P) * 128) * 100) for B in BATCHES}
turns = {B: dict(tps=B * 50e6 / (F + B * P), util=B * MAC1 / ((F + B * P) * 128) * 100) for B in BATCHES}
QKV = 8 * 3 * (512 * 512 + 8 * 256) / 2  # flash cycles of the qkv weights per pass
dep = {B: dict(tps=B * 50e6 / (F + B * P - min(QKV, B * P)), util=B * MAC1 / ((F + B * P - min(QKV, B * P)) * 128) * 100) for B in BATCHES}
t0 = M.load(R("s0_ttft32"), 4); ceil_ttft = t0["flash"] / 50e6
fmt = lambda d, k, f: (f % d[k]) if d else "n/a"
rows = []
for st, ch, kind, pre, kept in STEPS:
    d = {B: dec(pre, B) for B in BATCHES}; t = ttft(pre)
    rows.append((st, ch, kind, kept, d, t))
L = []
L.append("# Reef at 100 MB/s flash and 50 MB/s PSRAM: what each change does\n")
L.append("Model names: \"52M\" is TinyStories-28M (the published name; about 52M parameters with its 50k-word embedding and output layers). \"Small\" is the 5.8M-parameter model.\n")
L.append("> This report does not make decisions. It shows measured data so the reader can decide.\n")
L.append("## What was measured\n")
L.append("- **Workload:** the 52M model (TinyStories-28M), int8, decode of token 200 with 1, 2, 4 or 8 stories per pass, "
         "and time to first token on a 32-token prompt.")
L.append("- **Fixed:** flash 100 MB/s, PSRAM 50 MB/s, 32-PE Coral matrix array, 32 KB DTCM. Changes are applied one at a time; "
         "each is kept only if it gains more than 2% on some metric.")
L.append("- **The ceiling is computed (not simulated) from the link busy cycles measured in the baseline run:** "
         "the flash needs %.1f M cycles per pass to stream the weights (the same at any batch size), and the PSRAM %.2f M cycles per story "
         "to read and write its KV cache at token 200. **Ceiling** = batch x 50 MHz / (flash + batch x PSRAM - overlap), where overlap = "
         "min(qkv flash time, batch x PSRAM) = min(%.2f M, batch x PSRAM): the two links can work at the same time only while the qkv weights stream in, "
         "because attention needs its head's q, and attn_out, fc1 and fc2 need attention's output. It counts only overlap the design can do (steps 3.16 to 3.20 do it); "
         "a limit that assumed more overlap than the data dependencies allow was dropped from this report as unreachable." % (F / 1e6, P / 1e6, QKV / 1e6))
L.append("- **Reference, limit of the original design** = batch x 50 MHz / (flash + batch x PSRAM): the two links never work at the same time, "
         "which is how the firmware and the single DMA worked up to step 3.15.\n")
if os.path.exists(os.path.join(H, "tiers.md")):
    L.append(open(os.path.join(H, "tiers.md")).read())
L.append("## How the ceilings are calculated\n")
L.append("A formula, not a simulation. The inputs are measured from runs; nothing else is assumed.\n")
L.append("1. **Flash cycles per pass (F):** the flash link's busy cycles in one decode pass, from the run's counters. "
         "It covers the weight bytes (51.3 MB at int8, about 2 bytes a cycle at 100 MB/s) plus each transfer's first-byte latency. "
         "It is the same at every batch size, because one weight pass serves every story in it.")
L.append("2. **PSRAM cycles per story (P):** the PSRAM link's busy cycles in one pass divided by the batch size. "
         "It covers one story's KV cache read and write at token 200 (about 1.84 MB at 50 MB/s) plus burst overhead.")
L.append("3. **qkv window (Q):** the flash cycles of the qkv weights alone, layers x 3 x (hidden / 64) x (hidden x row bytes + 256) / 2, "
         "with row bytes 64 at int8 and 32 at int4. This is the only time the PSRAM can work while the flash works, because attention needs q, "
         "and attn_out, fc1 and fc2 need attention's output.")
L.append("4. **Shortest possible pass (T):** F + batch x P - min(Q, batch x P): the two links' times added, less the part of the PSRAM time that fits inside the qkv window.")
L.append("5. **Ceiling, tokens/s** = batch x 50 MHz / T. **Ceiling, matrix utilization** = MACs per pass / (T x 128), "
         "with 128 the MACs a cycle of 32 PEs. The two show the same thing in different units.")
L.append("6. **Time to first token:** the prefill's flash cycles / 50 MHz (its PSRAM time fits inside the qkv windows).\n")
L.append("For the 28M int8 results below, F = %.1f M, P = %.2f M and Q = %.2f M cycles, from the baseline run. "
         "The Phase 6 workloads use each build's own runs. Measured busy cycles are used instead of byte counts "
         "so that per-transfer latency and burst overhead, which the links really spend, are inside the ceiling.\n" % (F / 1e6, P / 1e6, QKV / 1e6))
L.append("## Results\n")
L.append("Decode is total tokens/s across all stories; the percentage is the share of the ceiling reached.\n")
L.append("| Step | Change | Kind | Kept | Batch 1 | Batch 2 | Batch 4 | Batch 8 | Time to first token |")
L.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
L.append("| | **Ceiling** | | | %.2f | %.2f | %.2f | %.2f | %.2f s |" % tuple([dep[B]["tps"] for B in BATCHES] + [ceil_ttft]))
L.append("| | If about 1.6 MB of on-chip SRAM held prefetched KV (not in the design; not simulated) | | | %.2f | %.2f | %.2f | %.2f | |" % tuple(B * 50e6 / max(F, B * P) for B in BATCHES))
L.append("| | Reference: limit of the original design (links take turns) | | | %.2f | %.2f | %.2f | %.2f | %.2f s |" % tuple([turns[B]["tps"] for B in BATCHES] + [(t0["flash"] + t0["psram"]) / 50e6]))
for st, ch, kind, kept, d, t in rows:
    cells = [fmt(d[B], "tps", "%.2f") + (" (%.0f%%)" % d[B]["share"] if d[B] else "") for B in BATCHES]
    L.append("| %s | %s | %s | %s | %s | %s |" % (st, ch, kind, ("yes" if kept is True else "data only" if kept == "data" else "no"), " | ".join(cells), (("%.2f s (%.0f%%)" % (t["s"], ceil_ttft / t["s"] * 100)) if t else "n/a")))
L.append("\nMatrix engine utilization (MACs done / MACs the 32-PE array can do):\n")
L.append("| Step | Batch 1 | Batch 2 | Batch 4 | Batch 8 | Prefill |")
L.append("| --- | ---: | ---: | ---: | ---: | ---: |")
L.append("| Ceiling | %s | %.1f%% |" % (" | ".join("%.2f%%" % dep[B]["util"] for B in BATCHES), t0["macs"] / (t0["flash"] * 128) * 100))
L.append("| Reference: original design | %s | %.1f%% |" % (" | ".join("%.2f%%" % turns[B]["util"] for B in BATCHES), t0["macs"] / ((t0["flash"] + t0["psram"]) * 128) * 100))
for st, ch, kind, kept, d, t in rows:
    if kept is True: L.append("| %s | %s | %s |" % (st, " | ".join(fmt(d[B], "util", "%.2f%%") for B in BATCHES), fmt(t, "util", "%.1f%%")))
L.append("\n![Utilization and decode speed against batch size](report_fig.png)\n")
L.append("## What the data shows so far\n")
L.append(open(os.path.join(H, "summary.md")).read().strip() + "\n")
L.append("\nDiagnostic and check runs (not steps): DMA setup set to 0 on step 3.9 (runs/d9_nosetup, -1.7% of the b1 token) and flash "
         "first-byte latency set to 0 (runs/d9_nolat, -1.4%), which located the per-transfer cost; step 3.16 with one DMA channel (runs/d316_1ch: "
         "same cycles as 3.14, so 3.16's gain is the overlap); every-PE mapping (step 3.19). Each new simulator build was first checked to run the "
         "previous kept step's programs to the same cycle (runs/v4_* to v10_b1).\n")
if os.path.exists(os.path.join(H, "phase6.md")):
    L.append("\n" + open(os.path.join(H, "phase6.md")).read())
if os.path.exists(os.path.join(H, "open_items.md")):
    L.append("\n" + open(os.path.join(H, "open_items.md")).read())
L.append("\nStep-by-step evidence for every change (what motivated it, what the counters showed after) is in LOG.md.\n")
open(os.path.join(H, "REPORT.md"), "w").write("\n".join(L))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11.5, "axes.edgecolor": "#6b7280", "axes.labelcolor": "#1f2933", "xtick.color": "#6b7280", "ytick.color": "#6b7280"})
fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.2))
cols = ["#9ca3af", "#93c5fd", "#3b82f6", "#1d4ed8", "#1e3a8a", "#d97706", "#059669", "#7c3aed", "#dc2626", "#db2777", "#111827"]
i = 0
FIG_STEPS = {"3.1": "3.1: multi-story decode (start)", "3.3": "3.3: end of tier 1 (real hardware)", "S3": "S3: end of tier 2 (assumed DMA and streams)"}
for st, ch, kind, kept, d, t in rows:
    if kept is not True or st not in FIG_STEPS: continue
    ch = FIG_STEPS[st].split(": ", 1)[1]
    xs = [B for B in BATCHES if d[B]]
    if len(xs) < 2: continue
    c = cols[i % len(cols)]; i += 1
    a.plot(xs, [d[B]["util"] for B in xs], marker="o", lw=2.2, color=c, label=f"{st}: {ch}")
    b.plot(xs, [d[B]["tps"] for B in xs], marker="o", lw=2.2, color=c)
a.plot(BATCHES, [dep[B]["util"] for B in BATCHES], ls=(0, (5, 3)), color="#1f2933", lw=1.4, label="Ceiling")
b.plot(BATCHES, [dep[B]["tps"] for B in BATCHES], ls=(0, (5, 3)), color="#1f2933", lw=1.4)
a.plot(BATCHES, [turns[B]["util"] for B in BATCHES], ls=":", color="#1f2933", lw=1.6, label="Limit of the original design (links take turns)")
b.plot(BATCHES, [turns[B]["tps"] for B in BATCHES], ls=":", color="#1f2933", lw=1.6)
for ax in (a, b):
    ax.set_xscale("log", base=2); ax.set_xticks(BATCHES); ax.set_xticklabels([str(x) for x in BATCHES]); ax.minorticks_off()
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e5e7eb", lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("Stories decoded per pass (batch size)")
a.set_ylabel("Matrix engine utilization"); a.yaxis.set_major_formatter(lambda v, _: f"{v:g}%")
a.set_ylim(0, max(dep[B]["util"] for B in BATCHES) * 1.9)  # room for the legend above the lines
b.set_ylabel("Decode tokens/s, all stories"); b.set_ylim(bottom=0)
a.set_title("Utilization during decode (token 200)", loc="left", fontsize=12); b.set_title("Decode speed (token 200)", loc="left", fontsize=12)
a.legend(frameon=False, fontsize=9.5, loc="upper left", title="Selected kept steps (all steps: tables)", title_fontsize=9.5, alignment="left")
fig.suptitle("Reef at flash 100 MB/s, PSRAM 50 MB/s (52M model, TinyStories-28M, int8, 32-PE array)", x=0.008, ha="left", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=2.5); fig.savefig(os.path.join(H, "report_fig.png"), dpi=200)
print("REPORT.md and report_fig.png written")

"""Builds phase6.md: the Phase 6 experiment tables, from the runs."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import metrics as M
H = os.path.dirname(os.path.abspath(__file__)); R = lambda p: os.path.join(H, "runs", p + ".json")
EXP = [  # label, prefix, layers, hidden, row bytes, batches: the final state (3.20 + S1 + S2 + Z1 + S3)
    ("52M (TinyStories-28M), int8", "S3i8", 8, 512, 64, [1, 2, 4, 8]),
    ("52M, int4 weights", "S3i4", 8, 512, 32, [1, 2, 4, 8]),
    ("Small (5.8M), int8", "S3si8", 6, 256, 64, [1, 2, 4, 8, 15]),
    ("Small (5.8M), int4 weights", "S3si4", 6, 256, 32, [1, 2, 4, 8, 15]),
]
L = ["## Phase 6: other workloads on the final design\n",
     "Each row is the final firmware and hardware (step 3.20 plus the cross-size steps S1, S2, Z1 and S3 below) with the workload changed. The ceiling is the dependency-aware one, "
     "computed from that build's own measured link busy cycles. int4 assumes the weight stream unpacks int4 to int8 at no cost; "
     "the KV cache stays int8. Max batch is the most stories per pass that fit the 32 KB DTCM (15 for the small model; 8 for 52M). Before S1 and S2 (step 3.20 as is), the small model reached 81 to 82% of its ceiling at batch 1; the cross-size hill climb below raised it to 92 to 93%.\n",
     "| Workload | Batch | Tokens/s, all stories | Each story | Share of ceiling | Matrix utilization | Flash busy | PSRAM busy |",
     "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
tt = []; curves = []
for lab, p, nl, h, rb, bs in EXP:
    cur = []
    qkv = nl * 3 * (h // 64) * (h * rb + 256) / 2  # flash cycles of the qkv weights per pass
    for b in bs:
        m = M.load(R(f"{p}_b{b}")); F = m["flash"]; P = m["psram"] / b; roi = m["roi"]
        v = b * 50e6 / roi; dep = b * 50e6 / (F + b * P - min(qkv, b * P))
        cur.append((b, v, dep, m["macs"] / (roi * 128) * 100, b * m["macs"] / b / ((F + b * P - min(qkv, b * P)) * 128) * 100))
        L.append(f"| {lab} | {b} | {v:.2f} | {v / b:.2f} | {v / dep * 100:.0f}% | {m['macs'] / (roi * 128) * 100:.2f}% | {F / roi * 100:.0f}% | {m['psram'] / roi * 100:.0f}% |")
    t = M.load(R(f"{p}_ttft32"), 4); tt.append(f"{lab} {t['roi'] / 50e6:.3f} s")
    curves.append((lab, cur))
L.insert(3, "![Decode speed and utilization for each workload](phase6_fig.png)\n")
L.append("\nTime to first token, 32-token prompt: " + "; ".join(tt) + ".\n")
L.append("Cross-size hill climb (each step written for both sizes, measured on all four workloads, kept if it gains more than 2% on any):\n")
L.append("| Step | Change | 52M int8 | 52M int4 | Small int8 | Small int4 | Result |")
L.append("| --- | --- | --- | --- | --- | --- | --- |")
L.append("| S1 | Bug fix from step 3.16: mark 64 / head size heads ready per finished qkv group (it marked 2, right only for 52M) | 0.0% | 0.0% | +4.4% b1 | +3.8% b1 | Kept |")
L.append("| S2 | Layer norm per token (step 3.12), with ITCM room made by removing dead one-token attention code | +1.7% b1 | +3.1% b1 | +5.8% b1, -0.9% b15 | +8.6% b1, -1.0% b15 | Kept |")
L.append("| S3 | Next linear's weights early (step 3.7 retry) | | | | | Not run: about 270 B more code than the 8 KB ITCM has free |")
L.append("| S4 | Bias-only DTCM weight buffers, so the small model fits batch 16 | +0.2% | +0.4% | b16 +0.9% over b15 | b16 +0.4% over b15 | Rejected: PSRAM-bound at large batch |")
L.append("| Z1 | Code-space pass (see below): frees 976 B of ITCM | -0.1% | -0.1% | +0.2% | +0.6% | Adopted as the base: no measurable cost, makes room for S3 |")
L.append("| S3 | Next linear's weights early, after the code-space pass made room | +1.0% b1 | +1.5% b1 | +2.6% b1 | +0.4% b1, TTFT +1.3% | Kept |")
L.append("| S5 | Decode attention: twice the KV blocks per transfer, three steps in flight | -0.4% b1, +0.5% b8 | +1.8% b1 | +2.5% b1 | +2.7% b1, +0.4% b15 | Dropped: the KV FIFO would grow from 4.2 to 12.3 KB for 2 to 3% at small b1 |\n")
L.append("What limits each workload now: every one is at 90 to 97% of its ceiling. The small model is lowest at batch 1, where fixed costs per pass "
         "matter most; from batch 4 up it waits on the PSRAM. After the code-space pass and S3 the 52M prefill image uses 7,496 of 8,192 B of ITCM (696 B free).\n")
L.append("PSRAM burst length (a part or mode with longer bursts; config only, `ext_ram_max_burst_bytes`; today 93 B), on step 3.20. Decode tokens/s:\n")
L.append("| Workload | Batch | 93 B | 186 B | 372 B | No limit |")
L.append("| --- | ---: | ---: | ---: | ---: | ---: |")
for lab, fam, base in (("52M, int8", "i8", "s320"), ("52M, int4", "i4", "w4")):
    for b in (1, 2, 4, 8):
        v0 = b * 50e6 / M.load(R(f"{base}_b{b}"))["roi"]; cells = [f"{v0:.2f}"]
        for bb in (186, 372, 0):
            v = b * 50e6 / M.load(R(f"pb{bb}_{fam}_b{b}"))["roi"]; cells.append(f"{v:.2f} ({(v / v0 - 1) * 100:+.1f}%)")
        L.append(f"| {lab} | {b} | " + " | ".join(cells) + " |")

L.append("\n## Code space (ITCM) pass\n")
L.append("The 8 KB ITCM must hold the prefill and decode code in one image. The build already optimized for size and dropped unused "
         "functions. Two more changes free 976 B in the 52M prefill image (8,184 to 7,208 B): `-msave-restore` (shared function "
         "entry and exit code, -440 B) and dma() as one function instead of inlined at every call (-388 B). Measured on all four "
         "workloads (step Z1): -0.1% to +0.6%, so no measurable cost. Used as the base for experiments A and B and step S3.\n")

def exp_table(title, intro, prefix):
    L.append(f"\n## {title}\n")
    L.append(intro + "\n")
    L.append("| Workload | Batch | Tokens/s | vs int8 | vs int4 weights | Flash cycles per pass | MACs vs int8 |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for v, i4, lab, bs in (("i8", "S2i4", "52M", [1, 2, 4, 8]), ("si8", "S2si4", "Small", [1, 2, 4, 8, 15])):
        for b in bs:
            m, z, w = M.load(R(f"{prefix}{v}_b{b}")), M.load(R(f"Z1{v}_b{b}")), M.load(R(f"{i4}_b{b}"))
            t = lambda x: b * 50e6 / x["roi"]
            L.append(f"| {lab} | {b} | {t(m):.2f} | {(t(m) / t(z) - 1) * 100:+.0f}% | {(t(m) / t(w) - 1) * 100:+.0f}% | "
                     f"{m['flash'] / 1e6:.2f} M (int8 {z['flash'] / 1e6:.2f} M) | {m['macs'] / z['macs']:.2f} x |")
    tt = []
    for v, lab in (("i8", "52M"), ("si8", "small")):
        f = R(f"{prefix}{v}_ttft32")
        if os.path.exists(f):
            tt.append(f"{lab} {M.load(f, 4)['roi'] / 50e6:.3f} s (int8 {M.load(R(f'Z1{v}_ttft32'), 4)['roi'] / 50e6:.3f} s)")
    L.append("\nTime to first token: " + "; ".join(tt) + ".\n")

exp_table("Experiment A: low-rank weights, computed as (x U) V",
          "Each weight matrix W [in x out] is stored as two factors U [in x r] and V [r x out], about half the bytes "
          "(r = 128 / 192 / 256 at 52M, 64 / 128 / 128 in the small model), and each linear runs as two smaller linears. "
          "Firmware only; no new hardware beyond the kept design. **Accuracy is not known**: the simulator does not compute "
          "values, and a rank-r factorization of trained weights needs an offline check or retraining.", "A")
L.append("At 52M, A runs at the speed of int4 weights (same bytes): the second linear per matrix costs almost nothing. "
         "In the small model the ranks, rounded to multiples of 64, keep 58% of the bytes, so it is 2 to 12% below int4.\n")
L.append("\n### Experiment A on the original one-DMA design (no assumed hardware)\n")
L.append("The same low-rank weights on step 3.3: Joshua's firmware with real multi-story decode, the upstream core (16-entry ROB, newer LSU), "
         "one DMA channel with its original wait-for-everything register, weights through the DTCM. 52M model, int8 factors. "
         "x U goes into a DTCM buffer that is idle at that point (the KV staging buffer, or the layer norm output).\n")
L.append("| Batch | Step 3.3, int8 | A on the same design | Change | Flash cycles per pass | PSRAM cycles per pass |")
L.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
for b in (1, 2, 4, 8):
    a3, a1 = M.load(R(f"s3_b{b}")), M.load(R(f"A1i8_b{b}"))
    t = lambda x: b * 50e6 / x["roi"]
    L.append(f"| {b} | {t(a3):.2f} | {t(a1):.2f} | {(t(a1) / t(a3) - 1) * 100:+.0f}% | {a3['flash'] / 1e6:.2f} M to {a1['flash'] / 1e6:.2f} M | {a1['psram'] / 1e6:.2f} M (unchanged) |")
t3, t1 = M.load(R("s3_ttft32"), 4), M.load(R("A1i8_ttft32"), 4)
L.append(f"\nTime to first token: {t3['roi'] / 50e6:.3f} s to {t1['roi'] / 50e6:.3f} s. The saving does not depend on the assumed hardware: "
         "the flash time halves on either design. The gain is smaller at large batch here because the flash and the PSRAM take turns, "
         "so only the flash half of each pass shrinks. On the final design A reaches 3.35 tokens/s at batch 1 (against 2.71 here).\n")
L.append("\n### Experiment A and A + int4 on the final design\n")
L.append("Same low-rank weights on the final design (S3's next-linear prefetch is off for A), with int8 and with int4 factors. "
         "Tokens/s, all stories; the change is against the same design and precision without A.\n")
L.append("| Workload | Batch | Without A | With A | Change |")
L.append("| --- | ---: | ---: | ---: | ---: |")
for lab, base, a, bs in (("52M int8", "S3i8", "A3i8", (1, 8)), ("52M int4", "S3i4", "A3i4", (1, 8)),
                         ("Small int8", "S3si8", "A3si8", (1, 8, 15)), ("Small int4", "S3si4", "A3si4", (1, 8, 15))):
    for b in bs:
        x, y = M.load(R(f"{base}_b{b}")), M.load(R(f"{a}_b{b}")); t = lambda m: b * 50e6 / m["roi"]
        L.append(f"| {lab} | {b} | {t(x):.2f} | {t(y):.2f} | {(t(y) / t(x) - 1) * 100:+.0f}% |")
L.append("\nA + int4 against plain int8 (52M): 2.9 x at batch 1, 1.6 x at batch 8. Not 4 x: the flash bytes drop to a quarter, "
         "but the PSRAM time per story and the fixed costs per pass do not. Time to first token, 52M: 1.41 s (int8), "
         "0.76 s (A), 0.49 s (A + int4).\n")
exp_table("Experiment B: rebuild each 16x16 weight tile on the matrix engine",
          "Each 16x16 tile is stored as rank-4 factors (16x4 and 4x16, half the bytes). For every tile the matrix engine "
          "multiplies the factors into a spare accumulator; the core reads the 16 rows out, narrows them to int8 and "
          "stores them as a normal weight chunk; then the normal MACs use it. **Accuracy is not known** (as for A).", "B")
L.append("B is 32 to 72% slower than int8. The array is idle (2 to 7% busy) and the flash too (13 to 14%): the core spends "
         "the pass reading rebuilt tiles out of the accumulator (16 reads per tile, about 100 instructions per tile, about "
         "200 K tiles per token at 52M); dispatch waits on a full vector queue 45% of the time. It would need a hardware path "
         "that feeds a rebuilt tile straight back in as the weight operand, which Coral's matrix engine does not have.\n")
open(os.path.join(H, "phase6.md"), "w").write("\n".join(L) + "\n")
print("phase6.md written")

TITLES = ["52M (TinyStories-28M), int8", "52M (TinyStories-28M), int4 weights", "Small model (5.8M parameters), int8", "Small model (5.8M parameters), int4 weights"]
COLS = ["#dc2626", "#7c3aed", "#059669", "#d97706"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#6b7280", "axes.labelcolor": "#1f2933", "xtick.color": "#6b7280", "ytick.color": "#6b7280"})
fig, ax = plt.subplots(2, 4, figsize=(17, 7.6))
for c, ((lab, cur), title, col) in enumerate(zip(curves, TITLES, COLS)):
    bs = [x[0] for x in cur]
    for r, (yi, ci, ylab) in enumerate(((1, 2, "Decode tokens/s, all stories"), (3, 4, "Matrix engine utilization"))):
        a = ax[r][c]
        a.plot(bs, [x[yi] for x in cur], marker="o", lw=2.2, color=col, label="Final design (3.20 + S1 + S2 + Z1 + S3)")
        a.plot(bs, [x[ci] for x in cur], ls=(0, (5, 3)), color="#1f2933", lw=1.4, label="Ceiling")
        if r == 0:
            for b, v, dep, *_ in cur:
                a.annotate(f"{v / dep * 100:.0f}%", (b, v), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=9, color="#6b7280")
            a.set_title(title, loc="left", fontsize=11)
        a.set_xscale("log", base=2); a.set_xticks(bs); a.set_xticklabels([str(b) for b in bs]); a.minorticks_off()
        a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", color="#e5e7eb", lw=0.8); a.set_axisbelow(True); a.set_ylim(bottom=0)
        if r == 1:
            a.yaxis.set_major_formatter(lambda v, _: f"{v:g}%"); a.set_xlabel("Stories decoded per pass (batch size)")
        if c == 0:
            a.set_ylabel(ylab)
ax[0][0].legend(frameon=False, fontsize=9.5, loc="upper left")
fig.suptitle("Final design on four workloads, flash 100 MB/s, PSRAM 50 MB/s, 32-PE array. Percent: share of that workload's ceiling", x=0.008, ha="left", fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=2.0); fig.savefig(os.path.join(H, "phase6_fig.png"), dpi=200)
print("phase6_fig.png written")

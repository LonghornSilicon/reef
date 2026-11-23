"""Builds tiers.md and tiers_fig.png: the kept steps grouped by hardware tier, with experiment A on each tier."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import metrics as M
H = os.path.dirname(os.path.abspath(__file__)); R = lambda p: os.path.join(H, "runs", p + ".json")
# (label, short label, run prefix, tier, what it is)
STEPS = [
    ("Baseline: M3 core, Joshua's firmware", "0", "s0", 1, "Starting point"),
    ("Real multi-story decode", "3.1", "s1", 1, "Firmware: one weight pass serves up to 8 stories"),
    ("16-entry reorder buffer", "3.2", "s2", 1, "Upstream Coral core"),
    ("Newer load/store unit", "3.3", "s3", 1, "Upstream Coral core"),
    ("Per-transfer DMA wait", "3.9", "s9", 2, "Assumed DMA feature: start the next weight transfer before waiting"),
    ("Epilogue one row per token", "3.14", "s10", 2, "Firmware"),
    ("Attention during the qkv weights", "3.16", "s316", 2, "Assumed: second DMA channel; firmware reorders qkv by head pair"),
    ("Weight stream into the matrix engine", "3.17", "s317", 2, "Assumed: FIFO in the engine, MACs read weights from it"),
    ("4 KB stream transfers", "3.18", "s318a", 2, "Assumed: 8.3 KB weight FIFO"),
    ("KV stream into the matrix engine", "3.20", "s320", 2, "Assumed: 4.2 KB KV FIFO"),
    ("Cross-size firmware steps S1, S2, S3", "S1-S3", "S3i8", 2, "Firmware: head-ready fix, layer norm per token, next weights early (with the code-space pass)"),
]
def tps(p, b):
    f = R(f"{p}_b{b}") if p != "s0" else (R("s0_dec200") if b == 1 else None)
    return b * 50e6 / M.load(f)["roi"] if f and os.path.exists(f) else None
rows = [(lab, short, tier, what, tps(p, 1), tps(p, 8)) for lab, short, p, tier, what in STEPS]
A = {1: (tps("A1i8", 1), tps("A1i8", 8)), 2: (tps("Ai8", 1), tps("Ai8", 8))}
def a_ceil(b):  # A's own ceiling, from A's measured link times (tier 2 runs)
    m = M.load(R(f"Ai8_b{b}")); F, P = m["flash"], m["psram"] / b
    q = 8 * 3 * (2 * (512 * 64 + 256) + 8 * (128 * 64 + 256)) / 2  # qkv flash: U [512x128] + V [128x512], 3 matrices, 8 layers
    return b * 50e6 / (F + b * P - min(q, b * P))
ACEIL = (a_ceil(1), a_ceil(8))
CEIL = (1.92, 9.46)
L = ["## The hill climb in one table\n",
     "52M model (TinyStories-28M), int8, decode of token 200, flash 100 MB/s, PSRAM 50 MB/s. Tokens/s, all stories. "
     "Each kept step adds to the one before. **Tier 1** uses only real hardware: the M3 core with two upstream Coral core changes, "
     "and the basic one-channel DMA that the firmware already assumes. **Tier 2** adds assumed hardware (not in the RTL): per-transfer DMA "
     "waits, a second DMA channel, and weight and KV streams into the matrix engine (about 12.5 KB of FIFOs). "
     "Experiment A (low-rank weights, a firmware and weight-format change, accuracy not checked) is shown on top of each tier.\n",
     "| Tier | Step | Change | What it is | Batch 1 | Batch 8 |", "| --- | --- | --- | --- | ---: | ---: |"]
for lab, short, tier, what, b1, b8 in rows:
    L.append(f"| {tier} | {short} | {lab} | {what} | {b1:.2f} | {b8:.2f} |" if b8 else f"| {tier} | {short} | {lab} | {what} | {b1:.2f} | n/a |")
    if short == "3.3":
        L.append(f"| 1 | + A | Low-rank weights on tier 1 | Firmware and weight format | **{A[1][0]:.2f}** | **{A[1][1]:.2f}** |")
L.append(f"| 2 | + A | Low-rank weights on tier 2 | Firmware and weight format | **{A[2][0]:.2f}** | **{A[2][1]:.2f}** |")
L.append(f"| | | Ceiling, int8 weights (from the link times) | | {CEIL[0]:.2f} | {CEIL[1]:.2f} |")
L.append(f"| | | Ceiling with A's weights (half the flash bytes) | | {ACEIL[0]:.2f} | {ACEIL[1]:.2f} |")
L.append("\nThe tier-2 firmware steps (3.14, S1 to S3) were measured on tier 2 only; some would also help tier 1, so tier 1 is a "
         "conservative figure. The ceiling is for int8 weights; A and int4 lower the bytes and so raise their own ceiling.\n")
L.append("![Decode speed step by step, by tier](tiers_fig.png)\n")
L.append(open(os.path.join(H, "changes.md")).read())
open(os.path.join(H, "tiers.md"), "w").write("\n".join(L) + "\n")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#6b7280", "axes.labelcolor": "#1f2933", "xtick.color": "#6b7280", "ytick.color": "#6b7280"})
fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
xs = list(range(len(rows))); split = [r[2] for r in rows].index(2) - 0.5
for ax, k, bname, ceil in ((axes[0], 4, "Batch 1", CEIL[0]), (axes[1], 5, "Batch 8", CEIL[1])):
    pts = [(x, r[k]) for x, r in zip(xs, rows) if r[k]]
    ax.axvspan(split, len(rows) - 0.5 + 1.2, color="#f3f4f6", lw=0)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", lw=2.2, color="#1d4ed8", label="Kept steps")
    ax.axhline(ceil, ls=(0, (5, 3)), color="#1f2933", lw=1.3, label="Ceiling, int8 weights")
    a1x = [r[1] for r in rows].index("3.3"); a2x = len(rows) - 1
    for tier, ax_x in ((1, a1x), (2, a2x)):
        v = A[tier][0 if k == 4 else 1]
        ax.annotate("", xy=(ax_x + 0.35, v), xytext=(ax_x, [r[k] for r in rows][ax_x]), arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.4))
        ax.plot([ax_x + 0.35], [v], marker="*", ms=15, color="#dc2626", ls="", label="+ A (low-rank weights)" if tier == 1 else None)
        ax.text(ax_x + 0.5, v, f"{v:.2f}", va="center", fontsize=10, color="#dc2626")
    ac = ACEIL[0 if k == 4 else 1]
    ax.plot([a2x - 0.15, a2x + 0.85], [ac, ac], ls=(0, (3, 2)), color="#dc2626", lw=1.3, label="Ceiling with A's weights")
    ax.set_xticks(xs); ax.set_xticklabels([r[1] for r in rows], fontsize=10)
    ax.set_xlim(-0.5, len(rows) - 0.5 + 1.2); ax.set_ylim(0, max(ACEIL[0 if k == 4 else 1], ceil) * 1.15)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e5e7eb", lw=0.8); ax.set_axisbelow(True)
    ax.set_title(f"{bname}: decode tokens/s, all stories", loc="left", fontsize=12)
    ax.text(split / 2, ax.get_ylim()[1] * 0.97, "Tier 1: real hardware", ha="center", va="top", fontsize=10, color="#374151")
    ax.text((split + len(rows) - 0.5 + 1.2) / 2, ax.get_ylim()[1] * 0.97, "Tier 2: + assumed DMA and streams", ha="center", va="top", fontsize=10, color="#374151")
    ax.set_xlabel("Kept step")
axes[0].legend(frameon=False, fontsize=9.5, loc="lower right")
fig.suptitle("The hill climb step by step, 52M model, int8, token 200 (shaded: steps that need assumed hardware)", x=0.008, ha="left", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.5); fig.savefig(os.path.join(H, "tiers_fig.png"), dpi=200)
print("tiers.md and tiers_fig.png written")

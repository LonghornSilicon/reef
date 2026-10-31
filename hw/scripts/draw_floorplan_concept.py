#!/usr/bin/env python3
"""Area-proportional concept floorplan of Reef v1 (NOT a P&R layout).

Block areas: Design Compiler sky130hd @10 ns cell areas (results/synth_m3_dc_sky130hd_10ns.md),
drawn at 70% utilization. Reef shell from Yosys (results/synth_reef_top_sky130hd.md).
SRAM: ~8 mm^2 for 40 KB (PD estimate), split by capacity. Die 32 mm^2 drawn square.
Timing caveat: 10 ns is met at tt only; see results/sta_m3_corners_sky130hd.md for ss.
Output: docs/img/reef_floorplan_concept.svg
"""
import math, sys
from xml.sax.saxutils import escape

UTIL = 0.7
S = 100.0                      # px per mm
DIE = math.sqrt(32.0)          # mm, drawn square
RING = (DIE - math.sqrt(DIE**2 - 2.0)) / 2   # 2 mm^2 IO ring
OX, OY = 40, 90                # die origin on canvas
HALO = 0.05                    # mm gap drawn around blocks

INK, INK2, MUTED, SURF = "#1f1f1e", "#55544f", "#8a8984", "#fcfcfb"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
BLUES = ["#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef"]

rvv = [("Vector FMA (FP32)", 0.891), ("Vector control: uop queue, ROB, dispatch, LSU", 0.843),
       ("Vector MUL / MAC", 0.673), ("Vector ALU", 0.540), ("Vector divider", 0.395),
       ("Vector register file", 0.363), ("Permute / reduce", 0.294)]
scalar = [("Scalar other: CSR, dispatch, MUL, DIV", 0.248), ("Scalar LSU", 0.247),
          ("Scalar regfile", 0.168), ("Retirement buffer", 0.111), ("Scalar FPU", 0.100),
          ("Fetch", 0.073)]
uncore = [("AXI slave", 0.042), ("Reef shell (SPI bridge)", 0.036), ("Debug module", 0.012)]
SRAM_MM2 = {"DTCM 32 KB": 6.4, "ITCM 8 KB": 1.6}

def fp(a):  # footprint mm^2 at utilization
    return a / UTIL

def squarify(items, x, y, w, h):
    """items: list of (key, value) with sum(values) == w*h. Returns {key: (x,y,w,h)}."""
    items = sorted(items, key=lambda t: -t[1]); out = {}
    def worst(row, side):
        s = sum(v for _, v in row)
        return max(max(side*side*v/(s*s), (s*s)/(side*side*v)) for _, v in row)
    while items:
        side = min(w, h); row = [items.pop(0)]
        while items and worst(row + [items[0]], side) <= worst(row, side):
            row.append(items.pop(0))
        s = sum(v for _, v in row)
        if w >= h:
            cw = s / h; cy = y
            for k, v in row:
                ch = v / cw; out[k] = (x, cy, cw, ch); cy += ch
            x += cw; w -= cw
        else:
            ch = s / w; cx = x
            for k, v in row:
                cw_ = v / ch; out[k] = (cx, y, cw_, ch); cx += cw_
            y += ch; h -= ch
    return out

def txtcolor(hexc):
    r, g, b = (int(hexc[i:i+2], 16)/255 for i in (1, 3, 5))
    lin = lambda c: c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4
    L = 0.2126*lin(r) + 0.7152*lin(g) + 0.0722*lin(b)
    return "#ffffff" if (1.05/(L+0.05)) >= 4.0 else INK

svg = []
def px(x, y): return OX + x*S, OY + y*S
def rect(x, y, w, h, fill, tip, stroke=SURF, sw=2, extra=""):
    X, Y = px(x, y)
    svg.append(f'<rect x="{X:.1f}" y="{Y:.1f}" width="{w*S:.1f}" height="{h*S:.1f}" rx="3" fill="{fill}" '
               f'stroke="{stroke}" stroke-width="{sw}" {extra}><title>{escape(tip)}</title></rect>')
def text(x, y, s, size=12, color=INK, weight=400, anchor="start"):
    svg.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" font-weight="{weight}" '
               f'text-anchor="{anchor}">{escape(s)}</text>')
SHORT = {"Vector control: uop queue, ROB, dispatch, LSU": "Vector control",
         "Vector register file": "VRF", "Permute / reduce": "Perm/ reduce",
         "Retirement buffer": "Retire buffer", "Scalar other: CSR, dispatch, MUL, DIV": "Scalar other"}
def label_in(r, name, area, fill, min_w=0.44, min_h=0.30):
    x, y, w, h = r; X, Y = px(x, y); c = txtcolor(fill)
    if w >= min_w and h >= min_h:
        words = SHORT.get(name, name)
        lines = []; cur = ""
        for wd in words.split():
            if cur and len(cur) + len(wd) + 1 > max(8, int((w*S-10)/6.2)): lines.append(cur); cur = wd
            else: cur = (cur + " " + wd).strip()
        lines.append(cur)
        need_w = max(len(l) for l in lines) * 6.2 + 10
        need_h = 16 + len(lines) * 14 + 4
        if len(lines) > 3 or need_w > w*S or need_h > h*S:
            return False
        for i, ln in enumerate(lines):
            text(X+6, Y+16+i*14, ln, 11.5, c, 600)
        text(X+6, Y+16+len(lines)*14, f"{area:.2f} mm²", 11, c)
        return True
    return False

W = 1180; H = 760
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif">')
svg.append('<defs><pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" '
           'patternTransform="rotate(45)"><rect width="8" height="8" fill="#f1f0ec"/>'
           '<line x1="0" y1="0" x2="0" y2="8" stroke="#d9d8d2" stroke-width="2"/></pattern></defs>')
svg.append(f'<rect width="{W}" height="{H}" fill="{SURF}"/>')
text(OX, 38, "Reef v1 — concept floorplan (area to scale, not a layout)", 22, INK, 700)
text(OX, 62, "CoralNPU M3 RvvCoreMiniAxi + Reef shell · sky130hd · 32 mm² die · logic from Design Compiler @ 10 ns (tt corner only), drawn at 70% utilization", 13, INK2)

# Die + IO ring
rect(0, 0, DIE, DIE, "#3b3a36", f"IO ring ~2 mm² (drawn to scale: {RING*1000:.0f} µm wide)", stroke="#3b3a36", sw=1)
core = (RING, RING, DIE-2*RING, DIE-2*RING)
rect(*core, "url(#hatch)", "Spare core area, halo and routing", stroke="#3b3a36", sw=0)

# SRAM macros in a row along the bottom edge, logic block top-left
cx, cy, cw, ch = core
sram_h = sum(SRAM_MM2.values()) / cw
x0 = cx
for name, a in SRAM_MM2.items():
    w = a / sram_h
    r = (x0+HALO, cy+ch-sram_h+HALO, w-2*HALO, sram_h-2*HALO)
    rect(*r, AQUA, f"{name} SRAM macro: ~{a:.1f} mm² (PD estimate, node/macro TBD)")
    X, Y = px(r[0], r[1]); text(X+8, Y+20, name, 14, INK, 700); text(X+8, Y+37, "SRAM macro", 12, INK)
    text(X+8, Y+53, f"~{a:.1f} mm² (est.)", 12, INK)
    x0 += w

logic_fp = fp(sum(a for _, a in rvv)) + fp(sum(a for _, a in scalar)) + fp(sum(a for _, a in uncore))
side = math.sqrt(logic_fp)
lx, ly = cx + 0.15, cy + 0.15
groups = squarify([("RVV", fp(sum(a for _, a in rvv))), ("SCORE", fp(sum(a for _, a in scalar))),
                   ("UNCORE", fp(sum(a for _, a in uncore)))], lx, ly, side, side)
callouts = []
def fill_group(key, items, palette):
    gx, gy, gw, gh = groups[key]
    sub = squarify([(n, fp(a)) for n, a in items], gx, gy, gw, gh)
    for i, (n, a) in enumerate(items):
        r = sub[n]; f = palette[i % len(palette)] if isinstance(palette, list) else palette
        rect(*r, f, f"{n}: {a:.3f} mm² cells ({fp(a):.2f} mm² at 70% util.)")
        if not label_in(r, n, a, f):
            callouts.append((r, n, a, f))
fill_group("RVV", rvv, BLUES)
fill_group("SCORE", scalar, [ORANGE])
fill_group("UNCORE", uncore, [VIOLET])
# group outlines
for key, nm in (("RVV", "RVV vector unit"), ("SCORE", "Scalar core")):
    gx, gy, gw, gh = groups[key]; X, Y = px(gx, gy)
    svg.append(f'<rect x="{X:.1f}" y="{Y:.1f}" width="{gw*S:.1f}" height="{gh*S:.1f}" rx="4" fill="none" stroke="{INK}" stroke-width="1.5"/>')
# callouts for small blocks, to the right of the logic block, in block order top to bottom
qx = lx + side + 0.35; last = -1.0
for r, n, a, f in sorted(callouts, key=lambda t: t[0][1] + t[0][3]/2):
    cyb = r[1] + r[3]/2
    ty = max(cyb + 0.04, last + 0.22); last = ty
    X1, Y1 = px(r[0]+r[2]/2, cyb); Xe, _ = px(lx + side + 0.08, 0); X2, Y2 = px(qx, ty)
    svg.append(f'<polyline points="{X1:.1f},{Y1:.1f} {Xe:.1f},{Y1:.1f} {X2-6:.1f},{Y2-4:.1f}" fill="none" stroke="{INK2}" stroke-width="1"/>')
    svg.append(f'<circle cx="{X1:.1f}" cy="{Y1:.1f}" r="2.5" fill="{INK}"/>')
    svg.append(f'<text x="{X2:.1f}" y="{Y2:.1f}" font-size="11.5" fill="{INK}" font-weight="600">{escape(n)}'
               f'<tspan fill="{INK2}" font-weight="400">  {a:.3f} mm²</tspan></text>')
# spare label
sx, sy = px(lx + 0.05, cy + ch - sram_h - 0.55)
spare = (DIE-2*RING)**2 - sum(SRAM_MM2.values()) - logic_fp
text(sx, sy, f"Spare / halo / routing ≈ {spare:.1f} mm²", 15, INK2, 700)
text(sx, sy+18, "(would not exist if PD's ~21 mm² logic estimate is right)", 12, MUTED)

# Pins on the ring
pins_top = ["clk_i", "rst_ni", "test_en_i"]; pins_right = ["spi_sck_i", "spi_csn_i", "spi_mosi_i", "spi_miso_o"]
pins_bottom = ["halted_o", "fault_o", "wfi_o"]
def pad(x, y, w, h, name):
    X, Y = px(x, y); svg.append(f'<rect x="{X:.1f}" y="{Y:.1f}" width="{w*S:.1f}" height="{h*S:.1f}" fill="#c9a227" stroke="#3b3a36" stroke-width="1"><title>pad {name}</title></rect>')
for i, n in enumerate(pins_top):
    x = 1.2 + i*1.4; pad(x, -0.06, 0.18, 0.18, n); X, Y = px(x+0.09, -0.12); text(X, Y, n, 11, INK2, 400, "middle")
for i, n in enumerate(pins_bottom):
    x = 1.2 + i*1.4; pad(x, DIE-0.12, 0.18, 0.18, n); X, Y = px(x+0.09, DIE+0.22); text(X, Y, n, 11, INK2, 400, "middle")
for i, n in enumerate(pins_right):
    y = 1.0 + i*1.1; pad(DIE-0.12, y, 0.18, 0.18, n); X, Y = px(DIE+0.12, y+0.13); text(X, Y, n, 11, INK2)

# Side panel
PX = OX + DIE*S + 120; PY = 110
text(PX, PY, "What is on the chip", 16, INK, 700); PY += 26
legend = [(BLUE, "RVV vector unit (RVV 1.0, VLEN 128, FP32)", f"{sum(a for _, a in rvv):.2f} mm² cells"),
          (ORANGE, "Scalar RISC-V core + scalar FPU", f"{sum(a for _, a in scalar):.2f} mm² cells"),
          (VIOLET, "AXI slave, debug, Reef shell", f"{sum(a for _, a in uncore):.2f} mm² cells"),
          (AQUA, "TCM SRAM macros (ITCM 8 KB + DTCM 32 KB)", "~8 mm² (estimate)"),
          ("#3b3a36", "IO ring (10 signal pads, gold)", "~2 mm²")]
for c, n, v in legend:
    svg.append(f'<rect x="{PX}" y="{PY-11}" width="14" height="14" rx="3" fill="{c}"/>')
    text(PX+22, PY, n, 13, INK, 600); text(PX+22, PY+16, v, 12, INK2); PY += 40
svg.append(f'<rect x="{PX}" y="{PY-11}" width="14" height="14" rx="3" fill="url(#hatch)" stroke="#bdbcb6"/>')
text(PX+22, PY, "Spare core area / halo / routing", 13, INK, 600); text(PX+22, PY+16, f"≈ {spare:.1f} mm²", 12, INK2); PY += 52
text(PX, PY, "Numbers", 16, INK, 700); PY += 24
for ln in [f"Logic cells: ≈{sum(a for _,a in rvv)+sum(a for _,a in scalar)+sum(a for _,a in uncore):.1f} mm² (DC, 10 ns, met at tt only)",
           f"Logic footprint at 70% util.: {logic_fp:.1f} mm²",
           "Die: 32 mm² (drawn square; the real frame may not be)",
           "Host access: SPI -> AXI (ITCM, DTCM, CSR)"]:
    text(PX, PY, ln, 12.5, INK2); PY += 20
PY += 14
text(PX, PY, "Read this as a sketch", 16, INK, 700); PY += 24
for ln in ["• No placement or routing has been done yet;",
           "  block positions are arbitrary, sizes are to scale.",
           "• Timing: 10 ns holds only at tt. At the ss corner",
           "  ~20 ns (50 MHz) closes, before P&R.",
           "• SRAM: ~8 mm² is PD's estimate; sky130 OpenRAM",
           "  macros would be 20 x 0.285 = 5.7 mm² + halo.",
           "• PD's own estimate of the logic is ~21 mm²;",
           "  until that is reconciled, the spare area may not exist.",
           "• Hover a block for its exact area."]:
    text(PX, PY, ln, 12.5, INK2); PY += 19
svg.append('</svg>')
out = sys.argv[1] if len(sys.argv) > 1 else "docs/img/reef_floorplan_concept.svg"
open(out, "w").write("\n".join(svg))
print(out, f"die {DIE:.2f} mm, ring {RING*1000:.0f} um, logic fp {logic_fp:.2f}, spare {spare:.2f}, callouts {len(callouts)}")

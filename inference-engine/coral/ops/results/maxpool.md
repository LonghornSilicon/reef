# Max pool (vector unit, int4)

Kernels: `coral/ops/kernels/maxpool_stem.cc` (3x3 s2 p1), `maxpool_stem2p.cc`
(2-pass variant, for comparison), `maxpool_sppf.cc` (5x5 s1 p2, x3).
Builds and reef-perf JSON/logs: `build/ops/maxpool/<variant>/`.

## Summary

Max pool is a pure data-movement operator: 4 (3x3 separable) or 8 (5x5
separable) int4 compares per output and no MACs. The matrix engine is unused.
On the base PSRAM link (0.25 B/cycle) both shapes are entirely **PSRAM-link
bound**: the link is 99%+ busy. The ResNet-18 stem pool alone costs
2.15 M cycles (43 ms at 50 MHz) because it has to read a 401 KB int4 map. With
the link removed (ideal), both kernels are **LSU bound** (70-88% busy): the
vector ALU sits at about 25% and the permute unit is never used. The int4
max runs directly on packed bytes, using one shift per input byte and 3 ops
per output byte, with no int8 copy. That halves the bytes moved, so int4 is
2.0x faster than int8 when link-bound and 1.4-1.7x faster at ideal. For
SPPF, keeping the 25.6 KB map resident in the 32 KB DTCM across all 3 pools
cuts base cycles from 653 k (streaming each pass) to 433 k when all three
outputs are stored for the concat (1.5x), and to 298 k when only the last
pool's output is stored (2.2x). At fast and ideal all three variants come in
at about 200 k: there the DTCM size only matters when the link is slow.

## Numbers (all from reef-perf runs; ROI cycles)

Element ops = int4 compares (max) per output value. Bytes are int4-packed.

| Kernel | Variant | Compares/out | PSRAM in B | PSRAM out B | DTCM buf B | base | fast | ideal | ideal compares/cyc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stem 112²x64 → 56²x64 | **fused, int4** | 4 (direct: 8) | 401,408 | 100,352 | 21,664 | **2,147,836** | **645,595** | **290,474** | 2.76 |
| | fused, int4, 28-column tiles | 4 | 401,408 | 100,352 | 10,912 | 2,169,110 | 656,196 | 298,568 | 2.69 |
| | fused, int8 (28-col tiles) | 4 | 802,816 | 200,704 | 21,824 | 4,327,757 | 1,298,793 | 414,664 | 1.94 |
| | 2-pass (V rows→DTCM→H), int4 | 4 | 401,408 | 100,352 | 28,896 | 2,149,329 | 649,425 | 474,543 | 1.69 |
| | 2-pass, int8 | 4 | 802,816 | 200,704 | 25,472 | 4,328,886 | 1,303,538 | 619,737 | 1.30 |
| SPPF 3x 5x5 on 20²x128 | **resident, store y1,y2,y3** | 8 (direct: 24) | 25,600 | 76,800 | 28,160 | **432,972** | **207,253** | **201,078** | 6.11 |
| | resident, store y3 only | 8 | 25,600 | 25,600 | 28,160 | 298,390 | 206,233 | 200,058 | 6.14 |
| | streaming each pass | 8 | 76,800 | 76,800 | 10,240 | 652,935 | 216,588 | 198,063 | 6.20 |
| | streaming, column loop not unrolled | 8 | 76,800 | 76,800 | 10,240 | 652,927 | 258,691 | 240,166 | 5.12 |
| | streaming, **direct** 25-tap window | 24 | 76,800 | 76,800 | 10,240 | 949,571 | 903,491 | 884,966 | 1.39 useful (4.17 executed) |
| | streaming, int8 | 8 | 153,600 | 153,600 | 20,480 | 1,303,213 | 410,248 | 331,903 | 3.70 |

Totals: the stem has 200,704 outputs and 802,816 compares. SPPF has
3 x 51,200 = 153,600 outputs and 1,228,800 compares (separable). There are
no flash/weight bytes. The SPPF per-pass phases are 66-72 k cycles at
fast/ideal. At base, the resident pass 1 takes 213 k (load plus the y1
store) and passes 2 and 3 take 107 k each (store only).

## Roofline

| Kernel | AI (compares / ext B) | Balance base (ideal ops/cyc ÷ 0.25) | Balance fast (÷ 1.0) | Bound | base / ideal | fast / ideal |
|---|---:|---:|---:|---|---:|---:|
| Stem int4 | 1.6 | 11.0 | 2.8 | memory (PSRAM) at base and fast | 7.4x | 2.2x |
| SPPF streaming | 8.0 | 24.8 | 6.2 | memory at base; ~compute (LSU) at fast | 3.3x | 1.09x |
| SPPF resident (3 stores) | 12.0 | 24.4 | 6.1 | memory at base; compute (LSU) at fast | 2.15x | 1.03x |
| SPPF resident (1 store) | 24.0 | 24.6 | 6.1 | ≈balanced at base (the first load is not overlapped); compute at fast | 1.49x | 1.03x |
| SPPF direct | 24 executed | — | — | compute (LSU) everywhere | 1.07x | 1.02x |

Link efficiency at base: the stem moves 501,760 B, a 2.007 M-cycle floor,
and runs in 2.148 M (93%). SPPF streaming moves 153,600 B (614 k floor) and
runs in 653 k (94%).

## Where on the NPU (busy % of ROI)

| Variant | base | ideal |
|---|---|---|
| Stem int4 fused | ext_ram_link 99.8, lsu 10, alu 3 | **lsu 75**, alu 23, dtcm 17, permute 0 |
| Stem int8 fused | ext_ram_link 99.8 | **lsu 95**, alu 9 |
| SPPF resident | ext_ram_link 99.3, lsu 41 | **lsu 88**, alu 26, dtcm 21, permute 0 |
| SPPF streaming | ext_ram_link 99.6, lsu 21 | **lsu 70**, alu 27, dtcm 16 (the rest is vector-ROB-full stalls behind load latency) |
| SPPF direct | ext_ram_link 69, lsu 64 | **lsu 69**, alu 21 |

Matrix MAC utilization is 0 (the matrix engine is not used). The bottleneck
is the PSRAM link at base. With the link removed, it is the core LSU (vector
loads cost 4 cycles per 16 B register, stores 3), not the ALU.

## Design notes

- **int4 handled on packed bytes (fused unpack/pack, faster than the
  `unpack_i4_rows`/`pack_i4` helpers).** Pixels are NHWC with the ops.hpp
  row-pair convention per pixel: byte j holds channel j in its low nibble and
  channel j+C/2 in its high nibble.
  - A signed int8 `vmax` of raw bytes gives the correct high nibble.
  - `vmax` of `b<<4` gives the low nibble.
  - Repack: `(max_raw & 0xF0) | (max_lo >>u 4)`.
  - Pad bytes are 0x88 (both nibbles -8 = -inf).
  - Per 32 B of output (64 values), SPPF runs 30 vector instructions by a count of the source code (not measured); int8
    runs 14 per 32 values. That is about the same ALU work per value, at
    half the LSU bytes.
- **Stem:**
  - Input rows stream from PSRAM into a 5-slot DTCM ring. Rows 2y-1, 2y and
    2y+1 are in use while 2y+2 and 2y+3 are in flight (double-buffered).
  - Per output pixel, the kernel loads the 2 new pixels of each of the
    3 rows (one m2 register group per pixel). The vertical and the stride-2
    horizontal max run in registers, with the odd column carried to the next
    output as its left tap. Only the packed result is stored.
  - Output rows are double-buffered and DMA'd back.
  - The 2-pass version (vertical max to DTCM rows, then horizontal) costs
    1.63x more at ideal because it stores and reloads the intermediates.
- **SPPF:** the work unit is one 32 B channel group (m2) of one output row.
  - The 5-row vertical max for a column is computed in registers.
  - The 5-tap horizontal max slides over 5 column maxima held in registers.
    Fully unrolling the 20-column loop removes the window moves (1.21x at
    ideal).
  - Separable vs direct: 8 vs 24 compares and 5 vs 25 loads per output.
    Direct is 4.5x slower at ideal and is compute-bound even at base.
  - **Resident:**
    - Map: 25,600 B plus a 2-row ring of 2,560 B, so 28,160 B of 32,768 B.
    - Load: rows prefetch one row ahead during pass 1.
    - Each pass runs in place. The 2 original rows above the current one are
      copied into the ring as they are read (one extra store per input byte,
      +1.5% at ideal).
    - Each output row is DMA'd out while the next row computes.
  - **Streaming:** each pass goes PSRAM to PSRAM through a 6-row ring (y-2..y+2
    in use, y+3 in flight), with output rows double-buffered.
- **"DTCM size matters":**
  - The int4 SPPF map (25.6 KB) fits the 32 KB DTCM. That saves 2 of the
    3 reloads and gives 1.5x at base (2.2x when only y3 is stored).
  - The int8 map (51.2 KB) does not fit, so int8 must stream: 1.30 M cycles
    at base, 3.0x the int4-resident time.
  - The int8 stem does not fit full-width rows double-buffered (5 x 7.2 KB),
    so it uses 28-column tiles. In int4, those tiles cost 1-3%.

## Extrapolation

- ResNet-18 has 1 stem max pool: 2.15 M cycles = **43 ms base**, 12.9 ms
  fast, 5.8 ms ideal at 50 MHz. That is likely a meaningful share of a
  W4A4 ResNet-18 at base, purely from PSRAM bandwidth.
- YOLO11n/26n has 1 SPPF per model:
  - resident: **8.7 ms base** (6.0 ms if only y3 is stored), 4.1 ms fast;
  - streaming: 13.1 ms base, 4.3 ms fast.
- Fusing the stem pool into the preceding conv's epilogue (so its
  401 KB output never goes to PSRAM) would save about 1.6 M base cycles. This
  is estimated, not run.

## Caveats

- reef-perf moves no data, so outputs are not checked. The nibble-max trick
  was checked by reasoning only.
- SPPF channels = 128 is assumed. It is the ultralytics n-scale SPPF hidden
  width (c_ = 256/2 at P5), but it was not confirmed for YOLO26n.
- Edge rows use duplicate row pointers (max is idempotent). That is about
  6% extra loads on SPPF; exact edge handling would trim it.
- Further LSU savings (not done):
  - Computing 2 output rows per pass would share 4 of the 6 row loads.
    Registers are tight with the lo stream.
  - The stem's extra input-row reads are inherent to stride 2 with a 3-row
    window.
- The direct variant is the rolled generic loop (not unrolled). Only the
  separable SPPF is tuned.

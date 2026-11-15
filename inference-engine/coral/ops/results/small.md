# Small ops: global average pool, 2x nearest upsample, channel concat (int4)

**Summary.** All three ops are pure memory traffic. In every run the matrix engine does 0 MACs. Under `base`, the PSRAM link (`ext_ram_link`) is busy 95-99.9% of the ROI, while the LSU is busy at most 26% and the ALU at most 6%. Base cycles come out within 2-11% of (in+out bytes) / 0.25 B/cycle. That makes these ops 6-25x slower than their compute-only (`ideal`) time on base, and 2-7x slower on `fast`. With compute removed (`ideal`), they are LSU/DTCM-bound on the vector copy or unpack. int4 halves the bytes, so it halves base cycles: 1.9-2.0x against int8. Two cheap fixes:
- **Virtual concat.** The producers write directly into their channel slot of a shared buffer. That costs only +3.5% / +7% on the producers' own stores, against a copy pass that costs 2x the producers' store time. Measured: 2.60M -> 0.044M cycles (40x40) and 5.21M -> 0.18M (80x80) on base.
- **Upsample via the DMA.** The vertical 2x comes from the DMA (each row is written twice). The horizontal 2x is a 3-op nibble duplication on the vector unit. A DMA-only variant is within 5% on base, but it needs channel-pair packing.

## Numbers (all measured in reef-perf unless marked)

Bytes are int4-packed, PSRAM only (no flash weights). Element ops: pool = H·W·C adds; upsample and concat = output elements written.

| Op / shape | PSRAM in B | PSRAM out B | Elem ops | DTCM B | base cyc | fast cyc | ideal cyc | ideal ops/cyc | AI ops/B | base/ideal | fast/ideal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GAP 7x7x512 -> 512 | 12,800¹ | 256 | 25,088 | 9,472 | 58,166 | 20,534 | 9,462 | 2.65 | 1.9 | 6.1x | 2.2x |
| GAP 20x20x256 -> 256 | 51,200 | 128 | 102,400 | 8,832 | 222,989 | 69,005 | 29,021 | 3.53 | 2.0 | 7.7x | 2.4x |
| Upsample 20x20x256 -> 40x40x256 | 51,200 | 204,800 | 409,600 | 15,360 | 1,074,432 | 306,432 | 43,829 | 9.35 | 1.6 | 24.5x | 7.0x |
| Upsample 40x40x128 -> 80x80x128 | 102,400 | 409,600 | 819,200 | 15,360 | 2,148,812 | 612,812 | 86,789 | 9.44 | 1.6 | 24.8x | 7.1x |
| Concat 40x40x(256+128), copy | 307,200 | 307,200 | 614,400 | 24,576 | 2,603,770 | 760,570 | 143,483 | 4.28 | 1.0 | 18.1x | 5.3x |
| Concat 80x80x(128+64), copy | 307,200 | 307,200 | 1,228,800 | 24,576 | 5,207,470 | 1,521,070 | 289,333 | 4.25 | 1.0 | 18.0x | 5.3x |

¹ 49 pixels are padded to 25 pixel pairs, so 12,544 B is the payload.

At 50 MHz, base / fast / ideal times are:

| Op / shape | base | fast | ideal |
|---|---:|---:|---:|
| GAP 7x7 | 1.16 ms | 0.41 ms | 0.19 ms |
| GAP 20x20 | 4.46 ms | 1.38 ms | 0.58 ms |
| Upsample 20->40 | 21.5 ms | 6.1 ms | 0.88 ms |
| Upsample 40->80 | 43.0 ms | 12.3 ms | 1.74 ms |
| Concat 40x40 | 52.1 ms | 15.2 ms | 2.87 ms |
| Concat 80x80 | 104.2 ms | 30.4 ms | 5.79 ms |

**Roofline.** The machine balance is (ideal ops/cyc) / (link B/cyc):
- base PSRAM (0.25 B/cyc): 10.6-14 ops/B for GAP, about 37 for upsample, about 17 for concat.
- fast PSRAM (1 B/cyc): 2.7-3.5 for GAP, 9.4 for upsample, 4.3 for concat.

Every op's arithmetic intensity (AI) is 1-2 ops/B, which is below both balances. All three ops are therefore **memory-bound on the PSRAM link** under base and fast.

**Where on the NPU.** Busy % of the ROI is listed for base, then ideal:

| Run | base | ideal |
|---|---|---|
| GAP 7x7 | ext_ram_link 96.5, lsu 26.5, alu 5.6 | lsu 163, itcm_fetch 68, dtcm 41, alu 35 |
| GAP 20x20 | ext_ram_link 99.0, lsu 11.7, alu 5.1 | lsu 90, alu 39 |
| Upsample (both) | ext_ram_link 99.9, lsu 3.9-4.6 | lsu 97-113, dtcm 27-31, alu 22 |
| Concat (both) | ext_ram_link 99.9, lsu 6 | lsu 108-116, dtcm 29-31 |

Across all runs, matrix_array is 0% and the useful MACs are 0. **Bottleneck: the PSRAM link. Vector unit mostly idle. Matrix engine unused.**

### Variants

| Variant | base | fast | ideal |
|---|---:|---:|---:|
| GAP 7x7x512 **int8** | 111,923 (1.92x int4) | 36,293 | 13,389 |
| GAP 20x20x256 **int8** | 444,515 (1.99x) | 136,547 | 42,515 |
| Upsample 20x20x256 **int8** | 2,146,678 (2.00x) | 610,678 | 69,427 |
| Upsample 40x40x128 **int8** | 4,293,298 (2.00x) | 1,221,298 | 137,187 |
| Concat 40x40 **int8** | 5,207,470 (2.00x) | 1,521,070 | 286,133 |
| Concat 80x80 **int8** | 10,414,870 (2.00x) | 3,042,070 | 577,833 |
| Upsample 20x20x256 **DMA-only** (channel-pair packing, 4 DMAs/pixel, DTCM 5,120 B) | 1,123,601 (+4.6%) | 355,601 | 48,821 |
| Upsample 40x40x128 **DMA-only** | 2,311,161 (+7.6%) | 775,161 | 161,601 |
| Concat 40x40: producers store A, B contiguously (the reference cost they pay anyway) | 1,280,427 | 358,827 | 21,227 |
| Concat 40x40: **virtual concat**, producers store into the channel slots of O | 1,324,829 (**+44,402 = +3.5%**) | 403,229 | 51,229 |
| Concat 80x80: producers contiguous | 2,560,827 | 717,627 | 42,427 |
| Concat 80x80: **virtual concat** | 2,739,229 (**+178,402 = +7.0%**) | 896,029 | 166,429 |
| Concat 40x40 DMA-only PSRAM->PSRAM per segment | 2,673,624 | 830,424 | 38,424 |
| Concat 80x80 DMA-only PSRAM->PSRAM | 5,484,824 | 1,798,424 | 153,624 |

int4 does not cost extra compute here. In `ideal`, int4 is *faster* than int8 (GAP 9.5k vs 13.4k cycles; upsample 43.8k vs 69.4k), because the LSU moves half the bytes. The unpack for GAP is 3 shifts per byte, and the nibble duplication for upsample is 3 ops per output byte. Both are cheaper than the extra loads.

**Virtual concat.** A copy concat costs a full read plus write of both inputs: 2.60M / 5.21M cycles on base. The zero-copy version costs only the overhead of splitting each producer's DMA into per-pixel-pair segments: +44k / +178k cycles, which is 59x / 29x less. That overhead is the 20-cycle DMA setup and the extra PSRAM bursts per segment. A strided (2D) DMA descriptor would remove the overhead entirely (not modeled). PSRAM->PSRAM DMA concat with no DTCM is no better than the copy (2.67M), because it still reads and writes every byte.

**Extrapolation (estimated, sum of measured ops).** I took one upsample and one concat at each YOLO neck scale:

| Concat method | base | fast | ideal |
|---|---:|---:|---:|
| Copy concat | 11.0M cycles = 221 ms | 3.20M = 64 ms | 0.56M = 11 ms |
| Virtual concat | 3.45M cycles = 69 ms | n/a | n/a |

The ResNet-18 head GAP is 1.2 ms on base. The FC 512->1000 that follows is a linear.cc GEMM with 256,000 B of int4 weights. At 1 B/cycle of flash that is about 256k cycles, roughly 4x the GAP (**estimated, not run**).

## Design notes

- **int4 layout.** The ops.hpp row-pair convention is applied to *pixels*. A pixel-pair group of a C-channel map is C bytes: byte c holds channel c of pixel 2p in the low nibble and of pixel 2p+1 in the high nibble. Because of this, a channel offset is always a whole byte (byte CA of each group), so concat needs no nibble shifting for any channel count. The shared `pack_i4` and `unpack_i4_rows` helpers are not used; the kernels use their own inline vector code (below).
- **GAP** (`small_pool.cc`):
  - **Input:** 4 KB chunks, PSRAM to DTCM, double-buffered.
  - **Accumulate:** for each 64-channel block, a 64-lane **int16** accumulator stays in registers. Unpack is `(b<<4)>>4` and `b>>4`, followed by two `vwadd.wv`. int16 is exact for H·W ≤ 4096 (a deviation from the "int32 reduction" in the spec, and it is exact).
  - **Epilogue:** widen to int32, `(sum·recip + 2^19) >> 20` with recip = round(2^20/(H·W)) and the requant scale folded in, clamp to [-8,7], and pack (channel c low, c+C/2 high).
  - **Output:** the C/2 bytes are DMA'd out.
- **Upsample** (`small_upsample.cc`):
  - **Rows:** double-buffered rows. At row y the DMA writes row y-1 twice (output rows 2y-2 and 2y-1) and reads row y+1, while the vector unit expands row y.
  - **Horizontal 2x:** input pair g becomes output pair 2g = `lo|lo<<4` and output pair 2g+1 = `hi|hi>>4`. That is 3 u8m8 ops per output byte, with no permutes.
  - **Can the DMA alone do it?** Vertical: yes. Horizontal: only if a pixel is whole bytes, which needs channel-pair packing (or int8). The DMA-only variant measures +5-8% on base, because many small DMAs add setup and PSRAM burst latency.
- **Concat** (`small_concat.cc`, MODE 0):
  - **Chunks:** 6 KB output chunks. A and B chunks are DMA'd in, and the vector unit copies them into interleaved groups `[A_g | B_g]`.
  - **Double-buffering:** both inputs and the output are double-buffered.
  - **Other modes:** MODE 1/2 model the producers' output stores (contiguous vs into the concat slots). MODE 3 is the DMA-only PSRAM->PSRAM version.

## Caveats

- reef-perf is timing-only, so outputs are not checked. The PSRAM numbers (0.25 B/cyc, 30/16-cycle latency, 100 B bursts) are team assumptions.
- The PSRAM link is shared by reads and writes, and the DMA has one in-order channel. Base cycles therefore track total bytes (in+out).
- Pool busy % includes boot and init. LSU >100% in ideal reflects stats.py's per-instance normalization. Read it as "LSU saturated".
- The virtual-concat rows measure only the store side. The producers' compute is not included, and the comparison is against the producers' contiguous stores.
- GAP 7x7 pads an odd pixel count to 25 pairs.
- int16 accumulation is a deliberate deviation from int32 (it is exact).
- The YOLO-neck and FC totals are estimates built from the measured per-op numbers.
- Files: `coral/ops/kernels/small_{pool,upsample,concat}.cc`. Runs: `build/ops/small/*/` (log: `build/ops/small/all_runs.txt`).

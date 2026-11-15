# Attention, no-mask (bidirectional): YOLO11n / YOLO26n C2PSA at P5

Kernel: `coral/ops/kernels/attn_nomask.cc`. Builds and logs: `build/ops/attn_nomask/<variant>/` (each has `summary.txt`).

## Summary

At the real PSA shape (N=400 tokens, 2 heads, key_dim 32, head_dim 64), the best variant is int4 with K and V kept resident in DTCM. It runs in **2.02M cycles under base (40 ms at 50 MHz), 1.90M under fast and 1.85M under ideal**. That is 16.6 MAC/cycle and **compute-bound**: base is 1.09x ideal.

Streaming K/V from PSRAM again for every 16-query tile multiplies PSRAM traffic by 13x (998 KB instead of 77 KB) and makes the kernel **memory-bound under base**: 4.96M cycles, 2.7x ideal. Under fast it is close to balance at 1.18x ideal.

Of the time under ideal, 55% is matrix work, running at the LSU operand roof (QKᵀ 28.5 MAC/cycle, PV 30.9). The other 43% is vector work: S readout and requant 17%, softmax 19%, epilogue 6%.

Two design choices carry the result:
1. **Exact softmax, no online rescale.** N=400 fits one 6.4 KB int8 S/P buffer per 16 queries. The kernel finds the true max before PV, so P·V runs as a single mac4 sweep over all 400 keys, with no α rescale.
2. **Fused int4 unpack inside the MAC loop.** Unpacking costs nothing: without it, ideal is 2.61–2.66M cycles (+41%).

N=1600 scales as O(N²): 15.8x the cycles for 16x the work. K/V cannot be resident at that size, so it streams and is memory-bound under base.

## Shape

Ultralytics C2PSA: `c = int(c1*0.5)`, and `PSABlock(c, attn_ratio=0.5, num_heads=c//64)`. Attention has `head_dim = c/heads` and `key_dim = head_dim*0.5`. For yolo11n/yolo26n the C2PSA has 1024 channels at width 0.25, so c1 = 256, c = 128 and there are 2 heads.

- **key_dim (Q, K) = 32. head_dim (V) = 64.**
- This comes from the Ultralytics source as I know it. I did not re-fetch it in this session.
- The brief assumed DK = 64, so I also ran DK=64.
- PSA also adds a 3x3 depthwise "pe" conv on V and a proj 1x1 conv. Those are other operators and are not counted here.

Shapes run:
- N = 400 (20x20, a 640-pixel input): all variants.
- N = 1600 (40x40): run with 1 head, and the 2-head figures are 2x that (marked "x2"). This keeps each run under about 50M cycles.

## Ops and bytes (int4-packed)

| Shape | MACs QKᵀ + PV | Softmax: exps (N²·H) | Vector elem-ops* | PSRAM read | PSRAM write | Flash |
|---|---:|---:|---:|---:|---:|---:|
| N=400, DK32, resident | 10.24M + 20.48M = **30.72M** | 320K | ~2.0M | **51.2 KB** (Q 12.8 + K 12.8 + V 25.6) | 25.6 KB | 0 |
| N=400, DK32, streaming | 30.72M | 320K | ~2.0M | **972.8 KB** (Q + 25 tiles x 19.2 KB x 2 heads) | 25.6 KB | 0 |
| N=400, DK64, streaming | 20.48M + 20.48M = 40.96M | 320K | ~2.0M | 1,305.6 KB | 25.6 KB | 0 |
| N=1600, DK32, streaming (x2) | 163.84M + 327.68M = **491.52M** | 5.12M | ~32M | 15.41 MB | 102.4 KB | 0 |

\*Vector element ops:
- Per score: requant and narrow (1), max (1), subtract (1), exp LUT (1), sum (1), store P (1).
- Plus the epilogue: N·DV·H outputs.
- The exp LUT (256 B) is built before the ROI. In a real model it is a one-off 256 B flash load.

## Cycles (each number is from a run)

| Variant | DTCM | base | fast | ideal | MAC/cyc ideal | base / ideal | fast / ideal |
|---|---:|---:|---:|---:|---:|---:|---:|
| **N=400 int4, K/V resident** (fused unpack) | 28.0 KB | **2,016,042** | **1,898,538** | **1,850,117** | 16.6 | 1.09 | 1.03 |
| N=400 int4, K/V streamed per tile (fused) | 10.8 KB | 4,959,532 | 2,215,832 | 1,881,680 | 16.3 | 2.64 | 1.18 |
| N=400 int4 resident, unpack to DTCM (FUSED=0) | 30.0 KB | 2,777,118 | 2,659,614 | 2,611,193 | 11.8 | 1.06 | 1.02 |
| N=400 int4 streamed, unpack to DTCM (FUSED=0) | 12.9 KB | 5,037,988 | 2,831,688 | 2,657,636 | 11.6 | 1.90 | 1.07 |
| N=400 **int8** streamed (int8 resident does not fit: 38.4 KB) | 13.9 KB | 9,197,185 | 3,279,935 | 1,918,233 | 16.0 | 4.79 | 1.71 |
| N=400 int4 resident, online softmax KC=80 (5 chunks) | 26.9 KB | 2,826,653 | 2,709,149 | 2,660,728 | 11.5 | 1.06 | 1.02 |
| N=400 DK=64 streamed (resident needs about 33.6 KB: does not fit) | 13.9 KB | 6,418,870 | 2,658,770 | 2,245,718 | 18.2 | 2.86 | 1.18 |
| N=1600 int4 streamed, KC=800 (2 chunks), 1 head | 21.3 KB | 38,099,046 | 16,847,246 | 14,859,460 | 16.5 | 2.56 | 1.13 |
| N=1600 int4, x2 heads (est.) | | 76.2M | 33.7M | 29.7M | | | |
| N=1600 int8 streamed, 1 head | 24.4 KB | 71,416,472 | 25,037,472 | 15,079,910 | 16.3 | 4.69 | 1.68 |

Time at 50 MHz:
- N=400 resident: 40.3 ms (base), 37.0 ms (ideal). The PSA block appears once per model.
- N=1600 (x2, est.): 1.52 s (base), 674 ms (fast).

## Matrix vs softmax/vector split (`mark()` phases)

N=400, resident, ideal (base adds 166K cycles of exposed K/V load: 8.3%):

| Phase | Cycles | % | Notes |
|---|---:|---:|---|
| QKᵀ mac (mac4_p4, fused int4 K) | 359,100 | 19.4 | 10.24M MAC: 28.5 MAC/cyc |
| PV mac (mac4_p4, fused int4 V) | 663,500 | 35.9 | 20.48M MAC: 30.9 MAC/cyc |
| **Matrix total** | **1,022,600** | **55.3** | at the measured LSU roof (29.5) |
| S readout (vtmv, then vnclip requant to int8) | 322,250 | 17.4 | about 1.0 cyc per score |
| Softmax (max, exp LUT gather, sum, P) | 358,300 | 19.4 | 0.89 exp/cycle; the vluxei8 gather dominates |
| Epilogue (vtmv O, /rowsum, pack int4) | 111,350 | 6.0 | |
| **Vector total** | **791,900** | **42.8** | |
| Q unpack + DMA waits | about 35,000 | 1.9 | |

The split is the same in every config, since the vector work does not change. Under base with streaming, the DMA waits dominate: K wait 19.5% and V wait 43.5% of 4.96M.

For N=1600 under ideal, matrix is 55%, vector 38%, and chunk rescale 3.7%.

## Roofline and where on the NPU

**Machine balance.** The ideal compute is 16.6 MAC/cycle:
- PSRAM base (0.25 B/cycle): 66 MAC/B.
- PSRAM fast (1 B/cycle): 16.6 MAC/B.

**Resident, N=400.** Arithmetic intensity is 30.72M / 76.8 KB = **400 MAC/B**, far above balance, so it is **compute-bound** in both configs.

**Streaming.** Arithmetic intensity is about **31 MAC/B** for N=400 and N=1600 alike, because every query tile re-reads all of K and V.
- Base: **memory-bound on the PSRAM link**, which is 87–88% busy, at 2.6x ideal.
- Fast: about 1.15x ideal, near balance.
- int8 halves the arithmetic intensity to 15 MAC/B and runs 1.85x slower under base.

**Busy units, resident, ideal:**

| Unit | Busy |
|---|---:|
| LSU | 49% |
| itcm_fetch | 26% |
| ALU (vector integer) | 14% |
| DTCM | 11% |
| **matrix_array** | **6.5%** |

MAC utilization is 16.6 / 256 = 6.5% overall, and about 12% during the matrix phases.

**Bottleneck:** the LSU, which feeds Zvt operands through vle8, in the matrix phases. In the vector phases, the issue rate of vtmv readout and the one-element-per-cycle exp-LUT gather. Under base with streaming, the PSRAM link.

## Design notes

**Tiling.** For each head and each 16-query tile (tm = 16):
1. **QKᵀ.** K arrives in 64-key groups, and `mac4_p4` fills 4 accumulators (16 queries x 64 keys). 400 = 6x64 + 16; the 16-key tail uses `mac1_p4`, so no key padding or masking is needed. Each tile column is one key across 16 query lanes. It is read with vtmv, narrowed with `vnclip` (requant shift folded in) to an int8 logit, and stored to S[key][query]. Writing the readout as an inline-asm loop with no vmv4r copy took this phase from 611K to 322K cycles.
2. **Softmax** runs on int8 at e8m4, 4 keys x 16 queries per instruction:
   - max;
   - gap = max − s (0..255);
   - exp via the 256-entry LUT, gathered with `vluxei8`;
   - row sum, widened to u16;
   - P (0..127) written over S in place.

   I also tried a 16-entry in-register `vrgather` LUT plus shift (EXP2=1). It is slower (405K vs 358K cycles), because the permute unit at m4 is costly.
3. **PV.** `mac4_p4(P, V)` accumulates O[16 x 64] in the 4 accumulators over all keys. No rescale is needed because the max is exact.
4. **Epilogue.** One vdivu per tile for 1/rowsum, then O/rowsum, then pack to int4. The O DMA is issued after the next tile's Q wait, so it never blocks.

**Double buffering.**
- Q is prefetched one tile ahead.
- Streaming: K groups (1 KB packed) and V groups (32 keys, 1 KB) are double-buffered in 2 x 1 KB.
- Resident: per head, all of K and V (19.2 KB packed) is loaded once, and every tile reuses it. Overlapping that load with the previous head is not possible because DTCM is full.

**int4.**
- K and V stay packed in DTCM. The operand bytes are packed per k-step: L0 = row0 | row2<<4 and L1 = row1 | row3<<4. So 2 loads plus 6 shifts (vsll/vsra) yield the 4 int8 B rows directly in v16..v22, where the plain mac needs 4 loads from an int8 copy.
- This is a fused unpack (`mac4_p4` / `mac1_p4` in my file). It is the same speed as int8 operands (PV 663K vs 680K cycles), because it issues fewer LSU loads.
- Q is unpacked once per tile with `unpack_i4_rows`. P and S are int8, and the output is int4 packed with `pack_i4`.
- The unfused path (unpack each group into DTCM, then mac4) costs 0.76M extra cycles under ideal (+41%): unpacking K/V once per use, 25 times, is expensive.

**Resident vs streaming.**
- At N=400, DK=32: resident is 2.46x faster under base (2.02M vs 4.96M) and 1.17x under fast. Under ideal they are the same.
- Resident needs 28 KB of DTCM.
- It fits only because the fused unpack removes the 4 KB int8 scratch, and S is int8 (6.4 KB) rather than int32.
- DK=64 resident (25.6 KB of K+V) does not fit, and neither does N=1600.

**What was masking and LLM overhead and is now gone:**
- The causal/local mask, which ran per column in both the max pass and the exp pass: key-position vid/vadd, vmsgeu/vmsltu/vmand, and two vmerge (−inf and 0).
- KV-cache insert and write-back of the newest block.
- **The per-block online-softmax rescale**: α = exp(m_old − m_new), plus a load/mul/shift/add/store of all 64 out_acc rows every 16 keys.
- Measured proxy for the rescale: forcing online softmax with KC=80 keys per chunk costs **+44% (2.66M vs 1.85M ideal)**, of which 25% is the rescale. The LLM kernel rescales every 16 keys.
- The mask cost itself was not isolated in a run.

## Caveats

- reef-perf is timing only, so outputs were not checked. The int4 packing, fused shifts and LUT indexing follow the written layouts, but they are unverified.
- The Q/K/V layouts (tile-blocked, k-step nibble pairing) are assumed to be written by the producing qkv GEMM's epilogue. Its store pattern is free to choose, but that has not been verified.
- **Quantization:**
  - The softmax scale and log2(e) are folded into the int8 logit requant (a shift; a real scale would be a vmul plus a shift, about +1 op per score).
  - P is 0..127, from a 7-bit LUT.
  - Logits are int8, so the LUT step is one logit unit.
- N=1600 2-head figures are 2x a 1-head run (estimated).
- The resident variant's K/V load per head is exposed under base (166K cycles). Splitting it so QKᵀ waits only on K would hide part of it. Not done.
- Phase marks add about 3 csrr per group (negligible).

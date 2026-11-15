# Conv2D (dense, W4A4) on the Reef NPU

Kernel: `coral/ops/kernels/conv2d.cc` (one source, every shape via `-D` macros). Builds and runs: `build/ops/conv2d/<shape>/` (`summary.txt` holds the `run.sh` output). Inner-loop microbenchmark: `coral/ops/kernels/conv2d_micro.cc`.

## Summary

The kernel is an implicit GEMM with no im2col. Activations sit in DTCM as channel-pair planes on a zero-padded, flattened grid, so for every tap the A rows are contiguous 16-byte runs at a shifted pointer. Blocking is 2×2: 32 pixels × 32 output channels in the 4 int32 accumulators. The int4 operands are unpacked inside the asm MAC loop, so int4 never touches DTCM as int8. On the main ResNet 3×3 shapes it reaches **57–60 MAC/cycle in the ideal config** (22–24% of the 256 peak). That is about 2× the 29.5 MAC/cycle int8 `mac4` ceiling in the brief, because int4 halves the bytes the LSU has to load, and the LSU is the limit. The same kernel in int8 runs at 36 MAC/cycle, so **int4 is 1.58× faster at ideal and 1.97× faster at base**. Under `base`, the 3×3 layers with ≥128 input channels are **flash-bound** (1.35–1.4× ideal), because weights are re-streamed once per 32-pixel group. Under `fast` everything is compute-bound within 1–6%. 1×1 convs are limited by the epilogue (20 MAC/cycle ideal) and are PSRAM-bound at base. Estimated ResNet-18 conv total: **38.6 M cycles ≈ 0.77 s at 50 MHz (ideal), 46.4 M cycles ≈ 0.93 s (base)**.

## Results (all full layers, nothing sliced; int4 unless noted)

| | S1: L2 3×3 128→128 @28 | S1 **int8** | S2: L1 3×3 64→64 @56 | S3: 1×1 64→64 @80 | S4: 3×3 **s2** 64→128 @56→28 | S5: L3 3×3 256→256 @14 |
|---|---:|---:|---:|---:|---:|---:|
| MACs (useful) | 115.6 M | 115.6 M | 115.6 M | 26.2 M | 57.8 M | 115.6 M |
| MACs executed (junk cols + 32-px rounding) | 127.4 M | 127.4 M | 120.3 M | 26.2 M | 61.3 M | 132.1 M |
| Weights from flash (B) | 1,991,680 (27× re-read of 73.7 K) | 3,982,336 | 18,944 (resident, once) | 2,560 (resident) | 959,488 (26×) | 2,066,432 (7×) |
| Activations read, PSRAM (B) | 59,264 | 122,624 | 108,224 | 204,800 | 110,336 | 37,120 |
| Activations written, PSRAM (B) | 53,760 | 107,520 | 103,936 | 204,800 | 51,968 | 28,672 |
| DTCM data (B, of 32,768) | 29,904 | 29,904 | 31,440 | 31,412 | 23,760 | 28,880 |
| Cycles, base | 2,801,458 | 5,508,876 | 2,479,019 | 2,089,473 | 2,133,997 | 2,590,321 |
| Cycles, fast | 2,023,788 | 3,346,965 | 2,453,611 | 1,321,820 | 1,296,155 | 1,976,357 |
| Cycles, ideal | 2,007,654 | 3,176,911 | 2,437,861 | 1,301,846 | 1,223,560 | 1,920,159 |
| **MAC/cycle, ideal** (useful) | **57.6** | 36.4 | 47.4 | 20.1 | 47.2 | **60.2** |
| MAC/cycle, base | 41.3 | 21.0 | 46.6 | 12.5 | 27.1 | 44.6 |
| base / fast vs ideal | 1.40× / 1.01× | 1.73× / 1.05× | 1.02× / 1.01× | 1.61× / 1.02× | 1.74× / 1.06× | 1.35× / 1.03× |
| Time at 50 MHz, base | 56 ms | 110 ms | 50 ms | 42 ms | 43 ms | 52 ms |

S1 with the VME RTL's 16-entry ROB (`rob_entries 16`, ideal): 1,683,594 cycles = **68.7 MAC/cycle** (+19%).

### Roofline

| | AI, all external (MAC/B) | AI vs flash | AI vs PSRAM | Balance, base: flash 1 B/c, PSRAM 0.25 B/c | Bound under base | Top busy pools (ideal) | Bottleneck |
|---|---:|---:|---:|---|---|---|---|
| S1 | 54.9 | 58 | 1,023 | 57.6 / 230 MAC/B | **flash** (link 73.5%) | lsu 72%, array 25%, alu 16% | LSU (operand loads), ROB-limited |
| S1 int8 | 27.4 | 29 | 502 | 36.4 / 146 | **flash** (75%) | lsu 90%, array 16% | LSU |
| S2 | 500 | 6,100 | 545 | 47.4 / 190 | compute (PSRAM link 40%) | lsu 71%, array 19% | LSU + per-tap loop overhead (K = 64 per tap) |
| S3 | 63.6 | 10,240 | 64 | 20.1 / 80 | **PSRAM** (88.6%) | lsu 54%, alu 24%, array 8% | epilogue (vtmv + requantize + pack) |
| S4 | 51.4 | 60 | 357 | 47.2 / 189 | flash 48% + PSRAM 36% on the one DMA channel; 32-byte DMAs | lsu 72%, array 20% | LSU; DMA count at base |
| S5 | 54.2 | 56 | 1,750 | 60.2 / 241 | **flash** (82%) | lsu 72%, array 27% | LSU |

The matrix array is 20–27% busy, and its useful-MAC utilization is 22–24% of the 256/cycle peak. On every 3×3 shape the binding unit is the core's LSU: each 16-byte vector register costs 4–6 LSU cycles. The 8-entry ROB stops loads, unpack and vtmm from fully overlapping (52 cycles per k-step against an LSU floor of about 40).

### ResNet-18 conv estimate (int4; **estimated**, built from the measured per-class rates above)

| Layer(s) | MMAC | Ideal Mcyc | Base Mcyc | Source |
|---|---:|---:|---:|---|
| conv1 7×7 s2 3→64 @224 | 118.0 | 6.28 | 6.28 | est.: Cin padded 3→8 (37.5% useful) × s2 rate |
| layer1 4× 3×3 64→64 @56 | 462.4 | 9.75 | 9.92 | measured S2 ×4 |
| layer2.0 3×3 s2 64→128 | 57.8 | 1.22 | 2.13 | measured S4 |
| layer2 3× 3×3 128→128 @28 | 346.8 | 6.02 | 8.40 | measured S1 ×3 |
| layer3.0 3×3 s2 128→256 | 57.8 | 0.96 | 1.30 | est. (S5 rate × geometry) |
| layer3 3× 3×3 256→256 @14 | 346.8 | 5.76 | 7.77 | measured S5 ×3 |
| layer4.0 3×3 s2 256→512 | 57.8 | 1.10 | 1.30 | est. |
| layer4 3× 3×3 512→512 @7 | 346.8 | 6.58 | 7.80 | est. (49/64 tile fill; 1.18 MB weights × 2 groups → flash-bound at base) |
| 3× downsample 1×1 s2 | 19.3 | 0.96 | 1.54 | est. (S3 rate) |
| **Total** | **1,813** | **38.6 (773 ms, 47 MAC/c)** | **46.4 (929 ms, 39 MAC/c)** | |

## Design notes

**Activation layout.** Planes hold channel pairs. Byte j holds pixel j of channel 2c in the low nibble and of channel 2c+1 in the high nibble, which is the shared row-pair convention. Each plane is on a zero-padded grid of width Wg = W + 2·pad, flattened. Output pixel p = y·Wg + x reads input p + ky·Wg + kx for tap (ky,kx). So 16 consecutive outputs read 16 contiguous input bytes per channel, and each tap is a plain GEMM with a shifted base pointer and no im2col. The Wg − W junk columns per row are computed (7% waste at W = 28) and zeroed in the epilogue. Written at offset Wg + 1 of the next layer's grid, they land exactly on its pad columns, so the producer writes the next layer's padding for free. Stride 2 uses 4 phase planes (row parity × column parity); a tap is then contiguous within one phase. The brief's "channel-major per row strip" becomes channel-major per flat-pixel window here: the window slides by 32·NG pixels and is not tied to row boundaries, so there is no per-row rounding waste.

**Tiling and blocking.** One k-step covers 4 input channels: 4 vtmm, 4096 MACs, 2 pixel tiles (tm = 16 each) × 2 output-channel groups of 16. One k-step loads two 32-byte packed A runs (LMUL 2) and one 64-byte packed B run (LMUL 4), against `mac4`'s twenty 16-byte loads. Zvt accepts int8 operand bases v0/1/8/9/16/17/24/25, so one 32-byte register pair feeds tile 0 (vN) and tile 1 (vN+1). The loop holds two register sets and is software-pipelined: one set loads and unpacks while the other set's vtmms run. The 8-entry ROB is the main constraint. Inner-loop history on a 4-group slice of S1 (ideal), in cycles per k-step: 92 for 16-byte loads with a 3-op unpack, unchanged with a 2-op unpack (the loop was ROB-bound), 81 with 32-byte loads, then 52 for the final loop alone (65 including epilogue and DMA bookkeeping).

**int4 handling (fused unpack).** Each packed register pair is unpacked in place with 2 ops: lo = b << 4 and hi = b & 0xF0. Both are 16× the int4 value and exact in int8, so the accumulator holds 256× the true sum (|sum| < 2²⁷ for K ≤ 4608), and the requantizer multiplier absorbs the 256. Inside a vtmm the k-slots run (c0, c2, c1, c3), and A and B use the same order. The unpack costs 8 vector-ALU ops (LMUL 2) per k-step, hidden under the loads. The int8 build has no unpack but needs 6 load instructions per k-step instead of 3. Activations and weights are packed in flash, PSRAM and DTCM; int8 exists only in vector registers.

**Loop order and re-read.** The nesting is strip (NG groups) → pixel group (32 px) → output block (32 ch) → tap / K-chunk (≤ 2 KB of weights per DMA chunk, double-buffered).
- The K reduction has to finish in the 4 accumulators. Spilling them with vtse32/vtle32 costs about 2 K cycles, more than the compute of one chunk. So every (32-pixel, 32-channel) block needs its full K × 32 weights, which is 18 KB packed at K = 1152. That cannot stay resident next to a double-buffered strip.
- So weights are re-streamed per pixel group: 27× for S1 (1.99 MB against 73.7 KB unique), 26× for S4 and 7× for S5. Activations are read about once (1.03× for S1), because each strip DMAs only its new 32·NG bytes per plane and copies the halo from the previous window inside DTCM.
- Flash is chosen as the re-read source because it is 4× faster than PSRAM (1 vs 0.25 B/cycle at base). Weight intensity is 64 MAC per flash byte, about equal to the 58–60 MAC/cycle compute rate, so base is close to balanced (1.35–1.4×) and fast is fully hidden.
- When all of a layer's weights fit next to a 1-group strip (S2: 18 KB, S3: 2 KB), they are loaded once (resident). Without that, S2 would re-read 1.9 MB.

**Double-buffering.** Weight chunks are prefetched one ahead. The next strip's planes and the previous group's output planes (one 32-byte DMA per plane) go out as DMA streams, a few per chunk iteration. Output is double-buffered per group. The input strip is double-buffered when it fits; it is single-buffered for S1-int8 and S5, which stall at strip boundaries.

**Epilogue (hand-written asm).** For each batch of 8 output channels × 16 px: arm the int32 tile mode once, vtmv one column per channel, then vadd bias, vmulh by a Q32 multiplier (shift folded in) and clamp (ReLU gives [0,7]). Channel pairs are nibble-packed, narrowed e32→e16→e8 once per batch (any vsetvli clears mtype), merged with the junk mask and stored. It costs about 1.9 K cycles per 32×32 block, which is 8% of S1 but more than the MACs in a 1×1 conv (S3).

## Caveats

- reef-perf computes no data, so correctness is by construction only. Watch the k-slot permutation (c0, c2, c1, c3), the 256× accumulator scale, and the junk-column masking.
- The loop sets vl = 32/64 with vsetvli and re-arms mtype (msetmtype + msettn) before each vtmm group. The model accepts this. The RTL behaviour of mixed LMUL with Zvt is assumed, not checked.
- MAC loop: about 52 cycles per k-step with ROB 8, 44 with ROB 16 (S1 gets +19% overall). Which ROB Reef has is open.
- 1×1 convs are epilogue-bound (20 MAC/cycle). Fusing a 1×1 into the epilogue of the layer before it, or a cheaper requantize path, would help most there.
- Stride-2 layers expect phase-split input (the producer writes 4 phases). Their output is written on the Wq grid, so the next layer's padding offset needs a small fix. Bytes are unaffected.
- The kernel needs an even number of k-steps per chunk (Cin multiple of 8), so the stem is not run; its estimate assumes Cin padded to 8. Folding kx into channels (K = 7 × 24) would bring it to about 2.7 M cycles (estimated).
- layer4 (512 ch at 7×7) does not fit the current DTCM plan: output buffers are 2 × 8 KB. Its numbers are estimated. Running it needs a single output buffer plus a narrower strip.
- About 13% of S1's ideal time is DMA and loop bookkeeping: a DMA issue costs about 16 cycles and a `dma_wait` about 13 cycles, even when idle. The epilogue is another ~8%.
- `-DABL=1/2/3` (no epilogue / no MAC / neither) are profiling switches only.

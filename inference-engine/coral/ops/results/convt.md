# ConvTranspose2d (DCGAN generator), W4A4 on Zvt

**Summary.** The DCGAN_CelebA generator, confirmed from the repo's `dcgan.py` (nz=100, ngf=64, nc=3; ConvT k4 s1 p0 then 4× ConvT k4 s2 p1, BN+ReLU, Tanh), is five transposed convs with 104.6 M useful MACs. Each stride-2 layer is split into 4 sub-pixel phases. Each phase is a dense 2×2-tap conv, run as an implicit GEMM on Zvt with no zero-multiplies and no im2col. Layer 1 (1×1 input) is a batch-1 GEMV. int4 weights are unpacked **inside the MAC loop, straight into the operand registers** (2 loads + 6 shifts in place of 4 loads, with no separate unpack pass). This made int4 faster than int8 even at the compute ceiling: L4 runs at 1.56 M cycles int4 vs 1.61 M int8 (ideal) and 1.81 M vs 2.16 M (base). The three big layers run at about 23 computed MAC/cycle (about 78% of the 29.5 roof) and are LSU-bound. L1 is weight-stream bound (flash), and L2 and L5 lose to padding and under-use. **Whole generator: 8.40 M cycles base, which is 168 ms, or 5.95 images/s at 50 MHz** (6.44 fast, 6.66 ideal).

## Per-layer results (int4, fused unpack; every number is from a reef-perf run, and every layer was run in full)

| Layer | Shape | Useful MACs | Computed MACs | Flash B (int4) | PSRAM rd B | PSRAM wr B | DTCM B | base cyc | fast cyc | ideal cyc | useful MAC/c (ideal) | computed MAC/c |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | 100→512, 1×1→4×4 (GEMV) | 0.82 M | 0.82 M | 413,696 | 50 | 11,264 | 6,784 | 485,278 | 426,239 | 417,387 | 1.96 | 1.96 |
| L2 | 512→256, 4×4→8×8 | 33.55 M | 67.1 M | 1,050,624 | 11,264 | 8,192 | 27,136 | 3,011,206 | 2,951,302 | 2,909,760 | 11.5 | 23.1 |
| L3 | 256→128, 8×8→16×16 | 33.55 M | 41.9 M | 525,312 | 15,360 | 16,384 | 25,600 | 1,957,656 | 1,861,400 | 1,824,034 | 18.4 | 23.0 |
| L4 | 128→64, 16×16→32×32 | 33.55 M | 35.7 M | 328,192 | 28,928 | 32,768 | 16,384 | 1,808,875 | 1,623,019 | 1,556,382 | 21.6 | 22.9 |
| L5 | 64→3, 32×32→64×64 (+Tanh) | 3.15 M | 17.3 M | 139,392 | 70,784 | 8,192 | 10,880 | 1,137,656 | 900,408 | 797,871 | 3.9 | 21.7 |
| **Generator** | | **104.6 M** | | 2.46 MB | 0.13 MB | 0.08 MB | ≤27 KB | **8,400,671** | **7,762,368** | **7,505,434** | 13.9 | |

"Flash B" counts actual traffic, including weight re-reads per pixel group (L3 ×2, L4 ×5, L5 ×17) and the folded-BN mult/bias. "Computed" counts the MACs the array actually performs, including padded positions or rows (from reef-perf `useful_macs`).

**Whole generator at 50 MHz:** 168.0 ms base (5.95 img/s), 155.2 ms fast (6.44 img/s), 150.1 ms ideal (6.66 img/s).

## Roofline and where the time goes

| Layer | AI (MAC / ext B) | Flash AI | PSRAM AI | base/ideal | fast/ideal | Bound | Top busy pools (ideal) | Matrix util (useful / 256 peak) |
|---|---:|---:|---:|---:|---:|---|---|---:|
| L1 | 1.9 | 2.0 | 72 | 1.16 | 1.02 | **flash (base)**: link 87% busy; LSU when the link is free | lsu 54, itcm 25, alu 11 | 0.8% |
| L2 | 31 | 32 | 1,725 | 1.03 | 1.01 | compute (LSU) | lsu 69, itcm 24, dtcm 21, array 9.0 | 4.5% |
| L3 | 60 | 64 | 1,057 | 1.07 | 1.02 | compute (LSU) | lsu 71, itcm 24, dtcm 21, array 9.0 | 7.2% |
| L4 | 86 | 102 | 544 | 1.16 | 1.04 | compute (LSU); base +16% from unhidden activation loads | lsu 71, itcm 23, dtcm 21, array 8.9 | 8.4% |
| L5 | 14 | 23 | 40 | 1.43 | 1.13 | **PSRAM latency/bandwidth (base)** + under-use | lsu 72, itcm 23, dtcm 22, array 8.5 | 1.5% |

Machine balance, from the ideal useful MAC/c divided by link B/c. Base flash (1 B/c) needs about 22 MAC/B and base PSRAM (0.25 B/c) about 86 MAC/B. L2 to L4 sit above both, so they are compute-bound. L1 (2 MAC/B) is far below flash balance. L5 (40 MAC/B on PSRAM) is below PSRAM balance. **Bottleneck:** the LSU feeding Zvt operands (about 70% busy, 20 or 18 `vle8` per 4 `vtmm`) everywhere except L1, which waits on flash weights at base.

## int4 vs int8 (L4, 128→64, same kernel with `-DINT8=1`)

| Variant | Flash B | PSRAM rd+wr B | DTCM B | base | fast | ideal |
|---|---:|---:|---:|---:|---:|---:|
| int8 | 655,872 | 123,392 | 19,456 | 2,157,161 | 1,773,369 | 1,612,279 |
| int4, unpack pass per chunk (`-DFUSED=0`) | 328,192 | 61,696 | 18,432 | 2,094,363 | 1,908,507 | 1,851,193 |
| **int4, fused in-register unpack** | 328,192 | 61,696 | 16,384 | **1,808,875** | **1,623,019** | **1,556,382** |

The fused unpack also halves L1: 813,534 → 485,278 base and 770,788 → 417,387 ideal. int4 halves the bytes moved, and with the fused unpack it is 16% faster than int8 at base and 3.5% faster at ideal, because the A operand needs 2 loads instead of 4.

## Native int4 (estimated, proxy)

**Hypothetical hardware.** Zvt gets an int4 type: 8 int4 values per PE lane per cycle, so the peak is 512 MAC/cycle. A native-int4 `vtmm` reads the same 16-byte operand registers and holds the array for the same time as an int8 `vtmm`, but covers k=8 instead of k=4. There is no unpack.

**Proxy method (`-DNATIVE4=1` in `convt.cc` and `convt_fc.cc`).** reef-perf cannot model a new type, so I emulated it:
- The int8 MAC path runs directly on the PACKED bytes, with no unpack anywhere: A is the packed weight k-pairs, and B is the packed channel-pair activation planes (or packed z in L1).
- The reduction runs K/2 steps.
- DMA byte counts (int4-packed), the epilogue and the output are unchanged from the int4 version.
- Real MACs = 2× the `useful_macs` reef-perf reports.
- L1 pads K from 100 to 104 (13 steps).
- Builds: `build/ops/convt/n4_l{1..5}`.

| Layer | native-int4 base / fast / ideal (cycles) | real MAC/c ideal, useful (computed) | vs fused int4, base / fast / ideal | Bound |
|---|---|---:|---|---|
| L1 GEMV | 483,847 / 253,104 / 244,252 | 3.4 (3.5) | 1.00× / 1.68× / 1.71× | Flash: 87% busy at base, 85% at fast |
| L2 512→256 | 1,692,305 / 1,632,401 / 1,590,859 | 21.1 (42.2) | 1.78× / 1.81× / 1.83× | LSU (compute); flash 66% busy at base |
| L3 256→128 | 1,182,105 / 1,053,560 / 1,005,413 | 33.4 (41.7) | 1.66× / 1.77× / 1.81× | LSU; flash 47% at base |
| L4 128→64 | 1,159,178 / 966,418 / 886,168 | 37.9 (40.2) | 1.56× / 1.68× / 1.76× | LSU; base +31% (flash 30%, PSRAM 24%) |
| L5 64→3 | 764,074 / 526,826 / 423,489 | 7.4 (40.9) | 1.49× / 1.71× / 1.88× | PSRAM at base (46% busy), plus lane under-use |
| **Generator** | **5,281,509 / 4,432,309 / 4,150,181** | 25.2 (whole model, ideal) | **1.59× / 1.75× / 1.81×** | |

**Whole generator, native int4:** 105.6 ms, or 9.47 img/s (base); 88.6 ms, or 11.28 img/s (fast); 83.0 ms, or 12.05 img/s (ideal). Fused int4 on today's hardware gives 5.95 / 6.44 / 6.66 img/s.

**Takeaways.**
- At the ideal config, native int4 is close to 2× (1.76–1.88×) on the conv layers. The k=8 `vtmm` halves both the operand loads and the matrix instructions, and the activation unpack pass goes away.
- The machine is still LSU-bound: about 40 computed MAC/c, which is 8% of the 512 peak.
- At base, the links eat into the gain. L1 gains nothing (it is purely a flash weight stream), and L4 and L5 expose the unhidden activation loads. The memory system, not the array, sets the base-config ceiling.

## Design notes

- **Phase decomposition (s2, k4, p1).** Output (2m+ry, 2n+rx) uses taps ky∈{1,3} (dy 0, −1) when ry=0 and ky∈{0,2} (dy +1, 0) when ry=1, and likewise in x. Each phase is a 2×2 conv with K = 4·Cin.
- **Zvt orientation.** A (tm = 16 lanes) holds 16 output channels (the weights). B holds 16 output positions per accumulator, so the 4 accumulators cover 64 positions. One pass is 16 och × 64 positions × the full K, so the accumulators never spill. Per k-quad there are 2 A loads (fused int4) and 16 B loads for 4 `vtmm`.
- **Implicit GEMM with no im2col.** Every activation channel plane has row pitch W+1 (one zero column), zero rows above and below, and a leading zero byte. Position p = m·pitch + n. Tap (dy,dx) reads `plane[1 + p + (1+dy)·pitch + dx]`, so every B row is 16 contiguous bytes and the borders hit the zero pad. The cost is that pad positions get computed: 1/(W+1) of them, plus rounding to 16.
- **Dataflow (pixel-group stationary).** Each group of ≤64 positions:
  1. DMA its activation span for all Cin channels from PSRAM (packed). Unpack it once, in place, to int8 `[Cin][S]` in DTCM.
  2. Then loop ry → och block → rx. Stream the (phase, och-block) weight block from flash in 1 KB packed chunks, double-buffered. The chunk stream runs on across phases and groups.
  3. Epilogue: `vtmv` the rows, apply ×mult + bias >> 16 (folded BN), then ReLU (a Tanh LUT via `vluxei8` on L5). Pack channel pairs to int4, then interleave rx=0/1 in a staging row with a stride-2 store.
  4. After rx=1, DMA the finished output-row segments to PSRAM in the next layer's padded layout. This makes each layer's output the next layer's input directly.
- **L1 (GEMV).** tm = 1: A = z (stride 1), B = 64 weight columns (4 och × 16 pixels) per group, 3.2 KB packed plus 32 B params, double-buffered. Each weight is used exactly once at batch 1, so L1 is a pure weight stream. The epilogue scatters each 4×4 map into L2's padded plane (`vsuxei8`).
- **int4 handling.**
  - Weights: packed row pairs (k0/k1, k2/k3 per 16-byte row), unpacked in registers inside the asm MAC loop. This is a custom fused unpack, not `unpack_i4_rows`.
  - Activations: channel-pair planes, unpacked once per group with `unpack_i4_rows`, in place.
  - Outputs: `pack_i4` of channel pairs.
- **Files.**
  - `coral/ops/kernels/convt.cc`: L2–L5; `-DCIN -DCOUT -DHIN -DTANH -DINT8 -DFUSED -DNATIVE4`.
  - `coral/ops/kernels/convt_fc.cc`: L1; `-DFUSED -DNATIVE4`.
  - Builds: `build/ops/convt/{l1f,l2,l3,l4f,l5,l4_i8,l4_unfused,l1_unfused,*_unfused}`.

## Caveats and known inefficiencies (gains marked estimated were not run)

- **L2 is 50% efficient.** Its 4×4 phase grid gives 20 pitch-5 positions, computed as 32. A flipped mapping would avoid this: 16 phase pixels as the tm rows, 64 och as columns, and dx-shifted activation copies. **Estimated** about 1.7 M cycles instead of 2.9 M.
- **L5 under-use: 3 of 16 och lanes** (it is the A/tm side in this orientation, so it is row under-use rather than column under-use), giving 18% useful MACs. Folding all 4 phases × 3 och into 12 of the 16 lanes over a 3×3 union window (K = 576) would cut computed MACs 1.8×. **Estimated** about 0.45 M cycles instead of 0.80 M (ideal).
- **L5 re-reads its 8 KB of weights per group, ×17.** They would fit resident in DTCM.
- **Activation group loads block compute.** They are not double-buffered: there is no DTCM room on L2/L3. This is most of the base/ideal gap on L4 and L5.
- With all of the above fixed, the generator would be about 5.9 M cycles, or about 8.5 img/s (**estimated**).
- Batch 1 throughout. L1 would amortize its 410 KB weight stream across a batch.
- **Fused unpack assumption:** it issues `vsll`/`vsra` under the vtype that `msetmtype` sets (e8, m1, vl=16) between Zvt ops, and the model accepts this. reef-perf is timing-only, so no outputs were checked.
- The quantization scheme (per-channel mult, shift 16, signed int4 clamp after ReLU, so 0..7) is a plausible placeholder, not calibrated.

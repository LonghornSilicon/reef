# Depthwise 3×3 conv (YOLO26n DWConv): W4A4 on the Reef NPU

## Summary

Depthwise 3×3 has K = 9 and no reduction across channels, so it is an operator for the **vector unit, and the matrix engine sits idle**. The best kernel (`dwconv_lane.cc`) puts channels across the vector lanes. Each output takes 9 `vwmacc` (e8→e16). The int4 unpack is fused into the loop. The inner loop is hand-scheduled asm with 4 rolling accumulators.

- **Compute-only (ideal):** 3.9–4.6 MAC/cycle. That is 13–16% of the 29.5 MAC/cycle dense-conv ceiling, and 25–29% of the vector unit's own 16 MAC/cycle widening-MAC peak.
- **Arithmetic intensity:** only 9 MAC per external byte (int4).
- **base config:** every shape is memory-bound on the PSRAM link (99% busy, 1.9–2.2× slower than ideal).
- **fast config:** every shape is compute-bound (within 1–4% of ideal).
- **Compute limit:** vector instruction issue. Each vector op holds one of the core's 8 ROB entries until it retires, which caps the core at about 1 vector instruction per cycle. Only 18 of the roughly 48 instructions per pixel pair are MACs.
- **Matrix engine:** the best mapping I found ("block-Toeplitz") runs at 1.9 MAC/cycle. At most 15% of the issued MACs are useful, the array is 5% busy and the LSU is 52% busy. So it is 2.4× slower than the vector kernel.

Shapes are assumed (YOLO n-scale at 640 px): P3 80×80×64, P4 40×40×128, P5 20×20×256, all stride 1 and pad 1, plus 80×80×64 at stride 2.

## Numbers (int4 weights and activations, all measured)

MACs = H·W·C·9 (outputs × 9). Bytes are int4-packed. Weights come from flash: 9C int4 plus int16 bias and int16 multiplier, so 9C bytes.

| Shape | MACs | PSRAM rd / wr (B) | Flash (B) | DTCM | base cyc | fast cyc | ideal cyc | MAC/cyc (ideal) | base / fast / ideal busy (top) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 80×80×64 s1 | 3,686,400 | 204,800 / 204,800 | 576 | 17.7 KB | 1,739,965 | 804,554 | 794,927 | **4.64** | ext_ram 99.7% / ext_ram 63%, lsu 37%, mul 33% / lsu 37%, mul 33%, alu 14% |
| 40×40×128 s1 | 1,843,200 | 102,400 / 102,400 | 1,152 | 19.8 KB | 871,825 | 430,573 | 420,694 | **4.38** | ext_ram 99.5% / ext_ram 59%, lsu 40% / lsu 41%, mul 32% |
| 20×20×256 s1 | 921,600 | 51,200 / 51,200 | 2,304 | 24.1 KB | 441,093 | 244,436 | 234,053 | **3.94** | ext_ram 98.3% / ext_ram 52%, lsu 47% / lsu 49%, mul 30% |
| 80×80×64 **s2** → 40×40 | 921,600 | 204,800 / 51,200 | 576 | 16.6 KB | 1,106,174 | 508,966 | 500,747 | 1.84 | ext_ram 99% / ext_ram 64% / lsu 28% (intrinsics version, not tuned) |
| 40×40×128 s2 → 20×20 | 460,800 | 102,400 / 25,600 | 1,152 | 17.8 KB | 558,320 | 265,200 | 256,729 | 1.79 | same pattern |

MAC/cycle under base and fast, s1 shapes 1/2/3: base 2.12 / 2.11 / 2.09, fast 4.58 / 4.28 / 3.77.

The matrix engine is unused in every vector kernel (matrix MAC utilization 0%).

### Roofline

| | |
|---|---|
| Arithmetic intensity, int4 | 9.0 MAC/B (each int4 element is read once, written once and used in 9 MACs). int8: 4.5 MAC/B. |
| Machine balance, this kernel (4.6 MAC/cyc) | base PSRAM 0.25 B/cyc → 18.5 MAC/B. fast 1 B/cyc → 4.6 MAC/B. |
| Machine balance, dense-conv roof (29.5) | base → 118 MAC/B. fast → 29.5 MAC/B. |
| Bound | **base: memory-bound (PSRAM link).** The link floor is 1.64M / 0.82M / 0.41M cycles against 1.74M / 0.87M / 0.44M measured. **fast: compute-bound (vector issue).** |
| Bottleneck unit | Vector instruction issue through the 8-entry core ROB. Top-down: about 80% of dispatch slots stall on "vector rob_full". LSU is 37–49% busy and the mul pool 30–33%. |

### Variants (80×80×64 s1, ideal unless noted)

| Variant | Cycles | MAC/cyc | Note |
|---|---:|---:|---|
| **lane, asm, int4** (`dwconv_lane.cc`) | 794,927 | 4.64 | chosen kernel |
| lane, int4, + SiLU LUT (`-DSILU=1`) | 886,803 | 4.16 | +12%: 2 `vluxei8` per pixel pair |
| lane, **int8** activations (`-DINT8=1`) | 912,486 (base 4,319,035 / fast 1,865,141) | 4.04 | optimistic: int16 accumulators, single output buffer |
| lane, int8, 40×40×128 | 479,931 (base 2,178,003 / fast 953,197) | 3.84 | 20×20×256 int8 does not fit the DTCM |
| lane, intrinsics (`dwconv.cc`) | 1,134,295 | 3.25 | same algorithm; the compiler's schedule puts dependent MACs back to back |
| planar: pixels in lanes, `vwmacc.vx` m4→m8, asm (`dwconv_planar.cc`) | 1,867,639 | 1.97 | LSU 56%: 9 shifted, unaligned m4 loads per strip |
| **matrix engine, block-Toeplitz** (`dwconv_mx.cc`) | 1,903,641 (base 1,931,376) | 1.94 | array 5% busy, LSU 52%, ALU 21% |

What int4 buys:
- **base, 80×80×64:** 1.74M cycles at int4 against 4.32M at int8. Bytes alone account for 2×. The rest is the int8 build's single output buffer, which the DTCM forces.
- **Compute:** int4 is also faster on compute, 795K against 912K cycles. The fused unpack (1 load + 2 ALU ops) is cheaper than the 2 loads plus 2 stores of int8.
- **Unpack cost:** 6 ALU ops per 18 MACs, which puts ALU at about 14% busy.

### Matrix-engine version (b)

**Mapping.** For each channel there is one 16×16 tile per 16 output rows × 16 output pixels.
- `A[(dx,i)][m] = w[i−m][dx]` is a banded Toeplitz matrix built offline, 60×16 per channel, stored as int4.
- `B[(dx,i)][n] = in[y0−1+i][x0−1+dx+n]` is the shifted int8 input row.
- K = 3 dx × 20 rows = 60. That is 18 rows padded to whole k-steps, so 15 `vtmm` per tile.
- `mac4` shares A across 4 x-tiles.

**Utilization.**
- Useful MACs are 16·16·9 = 2304 of 15·1024 issued, so **15%**.
- reef-perf counts the zeros as useful, so it reports 24.6M "useful" MACs; the true figure is 3.69M.
- The array is busy 5% of the ROI, which is about 0.8% of the 256 MAC/cycle peak.
- The time goes to operand delivery (20 `vle8` per k-step: LSU 52%) and to the epilogue (16 `vtmv` per tile).

**Block-diagonal alternative (analytic).** K = 16 channels × 9 taps, with `B[(c,t)][n] = w[c][t]·[n == c]`.
- Each B row has one nonzero out of 16 columns, so useful MACs are **1/16 = 6%** of the issued ones.
- A is channel-specific, so it cannot be shared across accumulators: 8 loads per `vtmm`, 64 useful MACs per vtmm, 40 LSU cycles per vtmm.
- That gives about 1.6 MAC/cycle (estimated), worse than Toeplitz.

**Verdict.** The matrix engine is the wrong unit for depthwise. Its best case is 2.4× slower than the vector kernel.

### Extrapolation (estimated: measured MAC/cycle scaled by MACs)

The YOLO11n-style head at n-scale has these depthwise convs:
- Classification branch per scale, first DW: shapes 1–3 above.
- Classification branch per scale, second DW on c3 = 80 channels: 80×80×80, 40×40×80, 20×20×80.
- The C2PSA positional-encoding DW: 20×20×128.

About 13M depthwise MACs in total:

| Config | Cycles | Time at 50 MHz |
|---|---:|---:|
| ideal | ≈ 2.9M | ≈ 58 ms |
| base | ≈ 6.1M | ≈ 120 ms |
| fast | ≈ 3.0M | ≈ 60 ms |

The 3 measured shapes alone (sum of measured runs) take 1.45M ideal and 3.05M base cycles.

For scale: the model's roughly 3.2 GMAC of dense conv at 29.5 MAC/cycle is about 108M cycles. So depthwise is about 0.4% of the MACs but about 3–6% of the time.

## Design notes

**Layout.** The pixel-pair int4 packing applies the shared row-pair convention to 2 neighbouring pixels. A row is W/2 pairs of C bytes. Byte j of pair k holds channel j of pixel 2k (low nibble) and of pixel 2k+1 (high nibble). One 16-byte `vle8` therefore gives 16 channels × 2 pixels, and both pixels use the same 16-lane weight registers. The output uses the same layout, so DW layers chain.

**Inner loop**, per output row, per 16 channels, per input pixel pair:
- 3 `vle8`, one per input row y−1, y, y+1.
- Unpack: `vsll 4` gives lo×16 and `vand 0xF0` gives hi×16. The ×16 is folded into the requant multiplier, and 16·9·64 fits int16.
- 18 `vwmacc.vv` into 4 rolling accumulators. Pair k feeds outputs 2k−1 … 2k+2, so every input byte is loaded once per output row. The 2× unroll renames the accumulators, so there are no register moves.
- The schedule is pipelined one row ahead: the next row's load is issued before the current row's MACs, and the MACs alternate accumulators. A dependent `vwmacc` costs 4 cycles and an independent one about 1 cycle (microbenchmark in `dwconv_ubench.cc`). This schedule took the kernel from 1.07M to 795K cycles.

**Epilogue.**
- The bias is the accumulator's initial value.
- `vmulh` by a per-channel Q16 multiplier, then `vnclip` to int8. The result is out×16, so the int4 output is the saturated top nibble.
- Optional SiLU: a 256-entry int8 Q4 → out×16 LUT applied with `vluxei8`.
- Pack the pixel pair with `vsrl`/`vand`/`vor`, then one `vse8`.

**Dataflow.**
- Input rows go PSRAM → DTCM by DMA into a ring of 4 row buffers (3 in use + 1 prefetching). The ring is the halo, so each row is fetched once.
- Output rows are double-buffered and DMA'd back while the next row computes.
- Padding: zero pairs at the end of each row buffer, and a zeroed ring slot for row −1 and row H.
- The weights (9C int4), the bias and the multiplier are DMA'd once. The weights are unpacked once to int8 and stay resident.

**Why not larger LMUL?** Channels-in-lanes at m2 needs 18 weight registers plus 16 accumulator registers, which does not fit the 32 vector registers.

**Why not pixels in lanes?** The planar variant uses scalar weights and m4/m8 MACs. It does 2.5× more MACs per instruction but is LSU-bound: each output row needs 9 shifted, unaligned loads, and a load costs 4–5 cycles per 16 B register.

**Vector ceiling.** Even a perfect schedule tops out at the mul pool's 16 MAC/cycle for widening int8 MACs. That rate is the same at every LMUL, because cost scales with register count, not vl.

## Caveats

- reef-perf is timing-only, so outputs are unverified. The indexing and sizes are written as for real.
- The stride-2 path is the untuned intrinsics kernel (`dwconv.cc -DSTRIDE=2`). Tuning it like the stride-1 kernel should give roughly 2× (estimated).
- The int8 comparison is optimistic: it uses int16 accumulators where real int8 needs int32, which would roughly double the MAC instructions. It also uses one output buffer, because two int8 rows do not fit.
- The requant uses a per-channel Q16 multiplier with truncation. The bias can absorb a rounding offset.
- The matrix-engine measurement covers the 80×80×64 shape only.
- The extrapolation assumes a YOLO11n-style head (and C2PSA). YOLO26n's exact list of DW layers was not checked.
- No shared headers or tools were changed.

Files:
- Kernels: `coral/ops/kernels/dwconv_lane.cc` (chosen), `dwconv.cc` (intrinsics, stride 1/2), `dwconv_planar.cc`, `dwconv_mx.cc`, `dwconv_ubench.cc`.
- Runs: `build/ops/dwconv/*/`.

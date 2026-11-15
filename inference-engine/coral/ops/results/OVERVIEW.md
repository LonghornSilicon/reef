# Operators on the Reef NPU: design review overview

All numbers come from reef-perf (`yuvesh-kernel-integration` @ b9dc6b6, calibrated Zvt model) at 50 MHz. Weights and activations are int4: stored packed, unpacked to int8 for Zvt, which has no int4 multiply. Per-operator detail is in the other files in this folder.

Memory configs:
- **base:** flash 1 B/cycle (68-cycle latency) + PSRAM 0.25 B/cycle. The team's pessimistic baseline.
- **fast:** flash 2 B/cycle + PSRAM 1 B/cycle.
- **ideal:** unlimited bandwidth, i.e. compute only.

## Slide 1: Operator overview

| Operator | Workloads | Runs on | Typical shape | Cycles base / fast / ideal | Rate (ideal) | Bound at base | Bound at fast |
|---|---|---|---|---|---|---|---|
| Conv2D 3×3 | ResNet-18, YOLO | Matrix engine | 128→128 @28×28 | 2.80M / 2.02M / 2.01M | 57.6 MAC/c | Flash (weight re-reads) | Compute (LSU) |
| Conv2D 1×1 | YOLO, ResNet downsample | Matrix engine | 64→64 @80×80 | 2.09M / 1.32M / 1.30M | 20.1 MAC/c | PSRAM | Compute (epilogue) |
| Transposed conv | DCGAN (whole generator) | Matrix engine | 128→64, 16→32 px | 1.81M / 1.62M / 1.56M | 21.6 MAC/c | Compute (LSU) | Compute (LSU) |
| Depthwise conv 3×3 | YOLO | Vector unit | 80×80×64 | 1.74M / 0.80M / 0.79M | 4.6 MAC/c | PSRAM | Vector issue |
| Attention, no mask | YOLO (PSA) | Matrix 55% + vector 43% | N=400, 2 heads | 2.02M / 1.90M / 1.85M | 16.6 MAC/c | Compute (K/V in DTCM) | Compute |
| Activations (ReLU / SiLU / Tanh / Sigmoid) | All | Vector unit (LUT) | 80×80×64 | Standalone 1.74M / 0.51M / 0.23M; **fused +0.3–0.4 cycles/element** | – | PSRAM (standalone) | PSRAM |
| Max pool 3×3 s2 | ResNet stem | Vector unit | 112²×64 → 56² | 2.15M / 0.65M / 0.29M | 2.8 cmp/c | PSRAM | PSRAM |
| Max pool 5×5 ×3 (SPPF) | YOLO | Vector unit | 20²×128, map kept in DTCM | 0.43M / 0.21M / 0.20M | 6.2 cmp/c | PSRAM | Compute |
| Global avg pool | ResNet-18 | Vector unit | 7×7×512 | 58K / 21K / 9.5K | – | PSRAM | PSRAM |
| Upsample 2× nearest | YOLO neck | Vector + DMA | 20²×256 → 40² | 1.07M / 0.31M / 0.04M | – | PSRAM | PSRAM |
| Concat | YOLO neck | DMA | 40²×(256+128) | Copy 2.60M; **zero-copy +44K** | – | PSRAM | PSRAM |
| Linear (decode GEMV) | TinyStories | Matrix engine | fc1 512→2048, 1 token | 1.20M / 0.71M / 0.68M (int8); **0.66M base with int4** | 1.6 MAC/c | Flash | Compute (LSU) |
| Attention, causal + KV cache | TinyStories | Matrix + vector | ctx 1024, one layer | 4.73M / 1.50M / 1.19M | 0.9 MAC/c | PSRAM (KV cache) | PSRAM |
| LayerNorm, embed | TinyStories | Vector / DMA | 512 features | 37K / 2K | – | Compute / latency | – |

## Slide 2: Roofline

- **Matrix engine peak:** 256 int8 MAC/cycle (64 PEs × 4).
- **Practical roof is set by operand delivery, not the array:**
  - **About 30 MAC/cycle** with the stock loop (16-byte operand loads, as in `linear.cc`).
  - **About 60 MAC/cycle** with the int4 conv loop: wider 32/64-byte loads plus unpack fused into registers.
  - Either way the array is only 12–24% of peak.
- **Machine balance at base:**
  - Flash: about 60 MAC per byte.
  - PSRAM: about 240 MAC per byte (0.25 B/cycle).
- **Where the operators fall:**

  | Group | Arithmetic intensity | Position |
  |---|---|---|
  | Dense 3×3 conv | 50–60 MAC/B | At the flash ridge |
  | Transposed conv (L2–L4) | 30–90 MAC/B | Above the ridge, compute-bound |
  | Attention, K/V in DTCM | 400 MAC/B | Compute-bound |
  | Depthwise conv | 9 MAC/B | Memory-bound |
  | Decode GEMV (LLM) | About 1 MAC/B | Memory-bound |
  | Pool / upsample / concat / standalone activations | 1–2 ops/B | Pure memory traffic |

- **At base, 11 of the 14 operator rows are memory-bound:** 9 on the 0.25 B/cycle PSRAM link and 2 on flash. Only transposed conv, no-mask attention and LayerNorm are compute-bound.
- **At fast,** the convs, depthwise, SPPF and LLM linears become compute-bound. The data-movement ops (pool, upsample, concat, standalone activations) and the KV-cache attention stay memory-bound.

## Slide 3: Where on the NPU (busy % of time, ideal unless noted)

| Operator | Bottleneck unit | Supporting numbers |
|---|---|---|
| Conv2D | LSU (operand loads) | LSU 71%, matrix array 20–27%, vector ALU 16% |
| Transposed conv | LSU | LSU 69–72%, array 9% |
| No-mask attention | LSU (matrix phases), exp-LUT gathers (softmax) | LSU 49%, array 6.5% |
| Depthwise conv | Vector issue: the 8-entry ROB fills with vector ops | LSU 37–49%, vector multiplier 30%, array 0% |
| Activations (fused) | LSU (indexed table loads) | DTCM port 59% with a 256-entry table |
| Pool / upsample / concat / standalone activation | PSRAM link (base) | Link 95–99.9% busy, array 0% |
| LLM linears / lm_head | Flash link (base), LSU (fast) | Flash 89–97% at base |
| LLM attention | PSRAM link (KV cache) | Link 99% at base |

## Slide 4: What it means for the hardware

1. **The matrix array is not the bottleneck; feeding it is.**
   - The LSU caps the array at 12–24% of peak in every matrix operator.
   - Levers:
     - Wider operand loads (the int4 conv loop already got 2×).
     - A tile-load path that skips the vector registers.
     - A bigger ROB: 8 → 16 entries gave Conv2D +19% (68.7 MAC/cycle).
2. **PSRAM bandwidth decides every activation-heavy operator.** At 0.25 B/cycle, pool, upsample, concat, depthwise and standalone activations run 2–25× slower than compute-only.
   - CNN feature maps don't fit the 32 KB DTCM, so they must go through PSRAM.
   - At 1 B/cycle most of them become compute-bound.
3. **DTCM size matters.**
   - YOLO SPPF fits in DTCM only as int4: 1.5× faster than streaming.
   - No-mask attention keeps K/V resident only as int4: 2.5× faster at base.
   - ResNet layer4 does not fit the current DTCM plan.
4. **int4 storage pays off only with the unpack fused into the MAC loop.**
   - Fused: 1.6–2× over int8 (Conv2D 1.97× at base, LLM linears 1.8×, DCGAN 1.16×).
   - Unpacking into DTCM first gives only 1.15× or even loses.
5. **The software rules this implies:**
   - Fuse activations, and concat by writing into the right channel slot.
   - Fold BatchNorm into the weights.
   - Avoid standalone elementwise passes.

## Workload totals (base config)

| Workload | Cycles | Time | Source |
|---|---:|---:|---|
| ResNet-18 convs (1.81 GMAC) | 46.4M | 929 ms | Estimated from measured per-layer rates |
| + stem max pool | 2.15M | 43 ms | Measured |
| DCGAN generator, one image | 8.40M | 168 ms (6 images/s) | Measured, every layer |
| TinyStories decode step, ctx 1024 (int8) | 80.8M | 1.62 s (0.62 tok/s) | Measured |
| Same, with int4 weights | 55.6M | 0.90 tok/s | Estimated |
| YOLO26n | – | – | Per-operator only; no whole-model run |

## Caveats

- reef-perf is timing only: no data is moved and no outputs are checked. Kernels are correct by construction, not by test.
- The YOLO26n shapes (channels, depthwise layer list, PSA dims) are assumed from YOLO11n n-scale. They have not been confirmed against the YOLO26n config. The DCGAN shapes were confirmed from the model's own code.
- The link speeds in base and fast are team assumptions. No PSRAM part has been chosen.
- The indexed-load cost (which drives the LUT activations and softmax) is an unmeasured reef-perf assumption (deviations U6).

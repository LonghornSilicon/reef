# Operators on the Reef NPU: design review overview (native int4)

All numbers come from reef-perf (`yuvesh-kernel-integration` @ b9dc6b6, calibrated Zvt model) at 50 MHz. Weights and activations are int4 throughout.

**Native int4 is an assumption.** Zvt today multiplies only int8, bf16 and fp32. This overview assumes an ISA change that adds an int4 type:
- 8 int4 values per PE lane per cycle, for a **512 MAC/cycle peak**.
- A native-int4 `vtmm` reads the same 16-byte operand registers and takes the same array time as an int8 `vtmm`, but covers k = 8 instead of 4.
- There is no unpack.

**How native int4 was measured (a proxy):** run each kernel's int8 path directly on the packed bytes, with half the reduction steps and no unpack. DMA traffic stays int4-packed. Real MACs = 2× what reef-perf reports. Each kernel has this behind `-DNATIVE4=1`.

**Which numbers are native int4:**
- Matrix-engine operators (Conv2D, transposed conv, attention, linears): the native-int4 estimate.
- Vector-unit operators (depthwise, pools, activations, upsample, concat): unchanged. A native int4 multiply in the matrix engine doesn't help them, and they already work on packed int4 bytes.

**Memory configs:**
- **base:** flash 1 B/cycle (68-cycle latency) + PSRAM 0.25 B/cycle. The team's pessimistic baseline.
- **fast:** flash 2 B/cycle + PSRAM 1 B/cycle.
- **ideal:** unlimited bandwidth, i.e. compute only.

Per-operator detail is in the other files in this folder. Each one has a "Native int4 (estimated, proxy)" section.

## Slide 1: Operator overview

| Operator | Workloads | Runs on | Typical shape | Cycles base / fast / ideal | Rate (ideal) | Today's int4 (unpack), base / ideal | Bound at base |
|---|---|---|---|---|---|---|---|
| Conv2D 3×3 | ResNet-18, YOLO | Matrix engine | 128→128 @28×28 | **2.77M / 1.70M / 1.68M** | 68.8 MAC/c | 2.80M / 2.01M | Flash |
| Conv2D 1×1 | YOLO, ResNet downsample | Matrix engine | 64→64 @80×80 | **2.07M / 1.26M / 1.24M** | 21.1 MAC/c | 2.09M / 1.30M | PSRAM |
| Transposed conv | DCGAN | Matrix engine | 128→64, 16→32 px | **1.16M / 0.97M / 0.89M** | 37.9 MAC/c | 1.81M / 1.56M | Compute (LSU) |
| Attention, no mask | YOLO (PSA) | Matrix 38% + vector 60% | N=400, 2 heads, K/V in DTCM | **1.54M / 1.42M / 1.38M** | 22.3 MAC/c | 2.02M / 1.85M | Compute (vector) |
| Linear (decode GEMV) | TinyStories | Matrix engine | fc1 512→2048, 1 token | **0.65M / 0.40M / 0.36M** | 2.9 MAC/c | 0.66M / 0.59M | Flash |
| Attention, causal + KV cache (int4 KV) | TinyStories | Matrix + vector | ctx 1024, one layer | **2.88M / 1.26M / 1.11M** (estimated) | – | 4.73M / 1.19M (int8 KV) | PSRAM |
| Depthwise conv 3×3 | YOLO | Vector unit | 80×80×64 | 1.74M / 0.80M / 0.79M | 4.6 MAC/c | same | PSRAM |
| Activations (ReLU / SiLU / Tanh / Sigmoid) | All | Vector unit (LUT) | 80×80×64 | Standalone 1.74M / 0.51M / 0.23M; **fused +0.3–0.4 cycles/element** | – | same | PSRAM (standalone) |
| Max pool 3×3 s2 | ResNet stem | Vector unit | 112²×64 → 56² | 2.15M / 0.65M / 0.29M | 2.8 cmp/c | same | PSRAM |
| Max pool 5×5 ×3 (SPPF) | YOLO | Vector unit | 20²×128, map kept in DTCM | 0.43M / 0.21M / 0.20M | 6.2 cmp/c | same | PSRAM |
| Global avg pool | ResNet-18 | Vector unit | 7×7×512 | 58K / 21K / 9.5K | – | same | PSRAM |
| Upsample 2× nearest | YOLO neck | Vector + DMA | 20²×256 → 40² | 1.07M / 0.31M / 0.04M | – | same | PSRAM |
| Concat | YOLO neck | DMA | 40²×(256+128) | Copy 2.60M; **zero-copy +44K** | – | same | PSRAM |
| LayerNorm, embed | TinyStories | Vector / DMA | 512 features | 37K / 2K | – | same | Compute / latency |

## Slide 2: Roofline

- **Matrix engine peak:** 512 int4 MAC/cycle with native int4. Today it's 256 int8.
- **The practical roof is set by operand delivery through the LSU:**
  - About 59 MAC/cycle with the stock 16-byte-load loop (2× int8's 29.5, because each load carries twice the MACs).
  - 69–73 MAC/cycle with the conv loop's wider 32/64-byte loads.
  - That is still only 12–15% of the 512 peak.
- **Machine balance at base:**
  - Flash: about 70 MAC per byte (1 B/cycle).
  - PSRAM: about 280 MAC per byte (0.25 B/cycle).
- **Where the operators fall:**

  | Group | Arithmetic intensity | Position |
  |---|---|---|
  | Dense 3×3 conv | 50–60 MAC/B | Just below the flash ridge: flash-bound at base |
  | Transposed conv | 30–90 MAC/B | Compute-bound |
  | Attention, K/V in DTCM | 400 MAC/B | Compute-bound, now on the vector work |
  | Depthwise conv | 9 MAC/B | Memory-bound |
  | Decode GEMV | About 1 MAC/B | Memory-bound |
  | Pool / upsample / concat / standalone activation | 1–2 ops/B | Pure memory traffic |

- **Native int4 doesn't change any external bytes.** Every operator that was memory-bound at base stays memory-bound, and gains about 1.0×.

## Slide 3: Where on the NPU (busy % of time, ideal unless noted)

| Operator | Bottleneck unit | Supporting numbers |
|---|---|---|
| Conv2D | LSU (operand loads) | LSU 84–90%, matrix array 11–16% |
| Transposed conv | LSU | about 40 computed MAC/cycle, 8% of the 512 peak |
| No-mask attention | Vector work: score readout + exp-LUT softmax | 60% of time; matrix phases 38% |
| Linears (decode) | Flash link at base, LSU at fast | Flash 85–93% at base |
| Depthwise conv | Vector issue: the 8-entry ROB fills with vector ops | Array 0% |
| Activations (fused) | LSU (indexed table loads) | DTCM port 59% with a 256-entry table |
| Pool / upsample / concat / standalone activation | PSRAM link | Link 95–99.9% busy, array 0% |
| LLM attention | PSRAM link (KV cache) | Link 99% at base |

## Slide 4: What native int4 buys, and what it means for the hardware

**Native int4 vs today's int4 (unpack in software):**

| Operator | ideal | base | Why |
|---|---|---|---|
| Conv2D 3×3 | 1.13–1.21× | 1.01–1.13× | The int4 conv loop already loads packed bytes. Native removes only the unpack ops and half the vtmms. Base is flash-bound. |
| Transposed conv | 1.76–1.88× | 1.49–1.78× | Its int4 kernel kept activations unpacked in DTCM. Native halves those operand loads. |
| No-mask attention | 1.35× | 1.31× | The matrix phases halve; the softmax/vector work (60%) now limits it. Assumes the probabilities P are int4 too (16 levels, accuracy unverified); with int8 P it is 1.11×. |
| LLM linears | 1.5–1.8× | 1.02× | Base is flash-bound; halving the bytes was already the whole win. |
| Vector-unit operators | 1.0× | 1.0× | They don't use the matrix engine. |

**Takeaways for the hardware:**
1. **Packing gives most of int4's win; native multiplies give the rest.**
   - int8 → int4 storage with a fused unpack is 1.6–2×.
   - Native int4 adds 1.0–1.8× more on top, mostly when memory is fast.
   - A well-written unpack kernel (Conv2D) captures most of the native benefit.
2. **The LSU is still the limit,** so a 512 MAC/cycle peak goes 85–92% unused. A faster operand path matters more than native int4:
   - wider loads;
   - a tile-load path that bypasses the vector registers;
   - a bigger ROB (16 entries gave +19% on Conv2D int4 and 73.9 MAC/cycle native).
3. **PSRAM bandwidth decides every activation-heavy operator, and native int4 doesn't touch it.** At 0.25 B/cycle, pool, upsample, concat, depthwise and standalone activations run 2–25× slower than compute-only.
4. **With faster matrix math, vector work becomes the next bottleneck in attention:** score readout and the exp lookup.
5. **DTCM size matters.**
   - YOLO SPPF fits in DTCM only as int4: 1.5× faster than streaming.
   - No-mask attention keeps K/V resident only as int4: 2.5× faster at base.
   - ResNet layer4 does not fit the current DTCM plan.
6. **Software rules:**
   - Fuse activations, and concat by writing into the right channel slot.
   - Fold BatchNorm into the weights.
   - Avoid standalone elementwise passes.

## Workload totals

| Workload | Native int4 base / fast / ideal | Today's int4 (base) | int8 (base) | Source |
|---|---|---|---|---|
| ResNet-18 convs (1.81 GMAC) | **888 ms** / – / 662 ms | 929 ms | – | Estimated from measured per-layer rates |
| + stem max pool | 43 ms | 43 ms | 87 ms | Measured |
| DCGAN generator, one image | **105.6 ms (9.5 img/s)** / 88.6 ms (11.3) / 83.0 ms (12.1) | 168 ms (6.0 img/s) | – | Measured, every layer |
| TinyStories decode, ctx 1024 (native int4 + int4 KV) | **1.10 tok/s** / 2.02 / 2.25 | 0.90 tok/s | 0.62 tok/s (measured) | Estimated |
| YOLO26n | – | – | – | Per-operator only; no whole-model run |

## Caveats

- **Native int4 does not exist in Zvt.** Its numbers come from a timing proxy: same instruction and operand costs as int8, at k = 8. The ISA change, accumulator range and any array-timing differences are not modelled.
- **reef-perf is timing only:** no data is moved and no outputs are checked. Kernels are correct by construction, not by test.
- **YOLO26n shapes are assumed from YOLO11n n-scale** (channels, depthwise layer list, PSA dims) and not confirmed. The DCGAN shapes were confirmed from the model's own code.
- **The link speeds in base and fast are team assumptions.** No PSRAM part has been chosen.
- **The indexed-load cost** (which drives the LUT activations and softmax) is an unmeasured reef-perf assumption (deviations U6).

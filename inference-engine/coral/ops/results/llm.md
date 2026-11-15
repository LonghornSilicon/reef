# LLM operators (TinyStories-28M / GPT-Neo): linear, layernorm, attention, embed, lm_head

## Summary

I re-measured the merged int8 firmware kernels (`coral/kernels`, PR #57) under base, fast and ideal, and added an int4 GEMM (`coral/ops/kernels/llm_linear_i4.cc`).

**Decode is weight-streaming.** At one token, every linear does about 1 MAC per weight byte. At base (1 B/cycle flash), every linear and `lm_head` is flash-bound: the link is 89–97% busy. Even at ideal, the decode linears reach only 1.5–1.8 MAC/cycle (0.6–0.7% of the 256 peak), because tm = 1 and the LSU feeds the operands.

**Attention at context 1024 is bound by the PSRAM link.** Layer-0 (global) attention reads 1.06 MB of KV per layer at 0.25 B/cycle, so it takes 4.7 M cycles at base, which is 4× its ideal time.

**A whole decode step at context 1024** takes:

| Config | Cycles | Time | Rate |
|---|---:|---:|---:|
| base | 80.8 M | 1.62 s | 0.62 tok/s |
| fast | 39.6 M | 0.79 s | 1.26 tok/s |
| ideal | 36.6 M | 0.73 s | 1.36 tok/s |

At base, the time splits about evenly into thirds: linears, attention, and `lm_head`.

**int4 weights:** the answer depends on how the unpack is done.
- **Unpack each chunk to DTCM, then `mac4`** (the brief's recipe). This is only 1.17× faster than int8 at base, and 1.6× slower than int8 at fast and ideal. The unpack (about 360 cycles per 1 KB chunk) makes the kernel compute-bound at every config.
- **Fused unpack**, from packed weights straight into the B operand registers inside the asm MAC loop. This is **1.79× faster than int8 at base**. It is still flash-bound, at 85–92% link busy. It is even 8–10% faster than int8 at ideal, because packed B rows halve the LSU operand loads.

So at base, yes: halving the weight bytes beats the unpack cost, but only with the fused unpack. Whole decode step with int4 weights (estimated): 0.90 tok/s at base, 1.43 at fast, 1.49 at ideal. Once the weights are int4, attention's KV traffic becomes the largest term at base (43%).

## 1. Firmware kernels (int8), measured

- **Decode:** `E2E_TOKENS=1`, `E2E_START_POS=1024`.
- **Prefill:** `E2E_TOKENS=8`, position 0.
- **Weight bytes** include the int32 bias trailer.
- **Attention** was run with `psram_kv_e2e.yaml`, so the KV cache is on the PSRAM link.
- **`lm_head`** was also run with `psram_kv_e2e.yaml`; see the caveats for why.
- **MAC utilization** = ideal MAC/cycle ÷ 256.

### Decode, context 1024

| Op | MACs | Weight B | KV B (rd+wr) | base | fast | ideal | MAC/c ideal | util | Top units (base → ideal) | Bound at base / fast |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| embed | 0 | 1,024 | – | 2,091 | 1,579 | 997 | – | – | flash, LSU | latency (tiny) |
| layernorm | 0 (512 el) | 1,024 | – | 36,560 | 36,560 | 36,560 | – | – | LSU, ALU | compute (scalar/LSU) |
| qkv (3×512²) | 786,432 | 792,576 | – | 889,907 | 513,039 | 489,627 | 1.61 | 0.63% | flash 92% → LSU 78% | flash / compute (LSU) |
| attention L0 (global, 1025 keys) | 1,064,960 | – | 1,064,960 + 16,384 | 4,727,680 | 1,496,096 | 1,193,611 | 0.89 | 0.35% | PSRAM 99% → LSU 80% | PSRAM / PSRAM |
| attention L1 (local, 272 keys) | 278,528 | – | 278,528 + 16,384 | 1,289,731 | 417,443 | 324,622 | 0.86 | 0.34% | PSRAM 99% → LSU 88% | PSRAM / PSRAM |
| attn_out (512²) | 262,144 | 264,192 | – | 307,940 | 184,036 | 176,232 | 1.49 | 0.58% | flash 89% → LSU 86% | flash / compute |
| fc1 (512→2048) + GeLU | 1,048,576 | 1,056,768 | – | 1,203,060 | 707,444 | 677,970 | 1.55 | 0.60% | flash 91% → LSU 74% | flash / compute |
| fc2 (2048→512) + res | 1,048,576 | 1,050,624 | – | 1,161,856 | 644,736 | 616,680 | 1.70 | 0.66% | flash 93% → LSU 79% | flash / compute |
| lm_head (512→50304) + argmax | 25,755,648 | 25,956,864 | – | 27,643,631 | 14,882,417 | 14,267,766 | 1.81 | 0.71% | flash 97% → LSU 77% | flash / compute |

### Prefill block (8 tokens, position 0)

| Op | MACs | Weight B | base | fast | ideal | MAC/c ideal | util | Bound at base / fast |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| qkv | 6,291,456 | 792,576 | 898,031 | 539,606 | 537,467 | 11.7 | 4.6% | flash 91% / compute (LSU 71%) |
| attn_out | 2,097,152 | 264,192 | 308,708 | 189,233 | 188,520 | 11.1 | 4.3% | flash 88% / compute |
| fc1 | 8,388,608 | 1,056,768 | 1,206,132 | 729,865 | 727,122 | 11.5 | 4.5% | flash 91% / compute |
| fc2 | 8,388,608 | 1,050,624 | 1,162,624 | 666,513 | 665,832 | 12.6 | 4.9% | flash 93% / compute |
| attention @0 (8 tokens) | 131,072 | – | 162,273 | 115,812 | 114,496 | 1.1 | 0.4% | PSRAM 86% / LSU 89% |
| layernorm | – | 1,024 | 36,560 | 36,560 | 36,560 | – | – | compute |
| embed | – | 8,192 | 16,385 | 12,289 | 7,633 | – | – | flash latency |

### Roofline

**Decode linears:**
- Arithmetic intensity is about 0.99 MAC/B (int8).
- The kernels' ideal rate is about 1.6 MAC/cycle. So the machine balance is 1.6 MAC/B at base (1 B/cycle) and 0.8 MAC/B at fast (2 B/cycle).
- At base they are flash-bound: 1.7–1.9× slower than ideal.
- At fast they are compute-bound (LSU): within 4–7% of ideal.

**Prefill-8 linears:**
- Arithmetic intensity is about 7.9 MAC/B, and the ideal rate is about 11.7 MAC/cycle.
- They are still flash-bound at base (1.66×) and compute-bound at fast (within 0.5%).

**Bottleneck unit:**
- At base, the flash link (or PSRAM for attention).
- Otherwise, the LSU, which feeds the Zvt operands.
- The matrix array is never more than 10% busy.
- tm = 1 in decode wastes the array. The cycles are the same as tm = 8: decode qkv takes 490 K cycles at ideal, against 537 K at tm = 8.

### Whole decode step, context 1024 (measured kernels, README recipe)

The step is embed + 8 × (2·LN + qkv + attn_out + fc1 + fc2) + 4 × attention L0 + 4 × attention L1 + ln_f + lm_head.

| Config | Cycles | ms at 50 MHz | tok/s | Linears | Attention | lm_head |
|---|---:|---:|---:|---:|---:|---:|
| base | 80,838,990 | 1,617 | **0.62** | 35% | 30% | 34% |
| fast | 39,553,712 | 791 | **1.26** | 41% | 19% | 38% |
| ideal | 36,647,287 | 733 | **1.36** | 43% | 17% | 39% |

**Cross-check.** I also ran the whole firmware decode step in one ELF: `e2e.elf` built with `-DE2E_START_POS=1024 -DE2E_NEW_TOKENS=1`, run with `psram_kv_e2e`.

| Config | Whole ELF, measured | Kernel sum | Difference |
|---|---:|---:|---:|
| base | 80,963,935 cycles (0.62 tok/s) | 80.84 M | 0.2% |
| fast | 40,115,643 cycles (1.25 tok/s) | 39.55 M | 1.4% |
| ideal | 38,762,937 cycles (1.29 tok/s) | 36.65 M | 5.8% |

The gap at ideal is time between kernels that the kernel sum does not count.

## 2. int4 GEMM (`llm_linear_i4.cc`), measured

There is one source with three variants, all with the same tiling, DMA, and epilogue:
- **`i8`:** int8 weights and activations (the baseline).
- **`i4`:** packed W4A4. Each 1 KB chunk is unpacked to a 2 KB int8 DTCM buffer with `unpack_i4_rows`, then fed to `mac4`.
- **`i4f`:** packed W4A4 with the fused unpack. 4 packed rows (64 B each) are loaded as m4 groups. The low nibble (`vsll` + `vsra`) gives rows r..r+3, and the high nibble (`vsra`) gives rows r+16..r+19. They go straight into v16–v22 for the vtmm, so there is no DTCM round trip.

Activations are int4 in all int4 variants. They are unpacked once per call and reused by all N/64 column groups. The outputs are requantized and packed back to int4.

The `i8` baseline tracks the firmware within 1–3% (decode fc1: 1,190 K vs 1,203 K at base).

| Shape | tm | Weight B, int8 → int4 | base i8 / i4 / i4f | fast i8 / i4 / i4f | ideal i8 / i4 / i4f | base speedup, i4f vs i8 |
|---|---:|---|---|---|---|---:|
| fc1 512→2048 | 1 | 1,056,768 → 532,480 | 1,190,449 / 1,021,627 / **663,546** | 693,809 / 1,021,627 / 594,969 | 650,382 / 1,021,627 / 594,969 | **1.79×** |
| fc1 512→2048 | 8 | same | 1,193,521 / 1,069,852 / **665,340** | 703,217 / 1,069,852 / 631,323 | 698,574 / 1,069,852 / 631,323 | **1.79×** |
| qkv 3×512² | 1 | 792,576 → 399,360 | 872,043 / 759,282 / **491,772** | 497,259 / 759,282 / 439,569 | 462,999 / 759,282 / 439,569 | **1.77×** |
| qkv 3×512² | 8 | same | 874,371 / 796,146 / **493,500** | 504,795 / 796,146 / 467,220 | 499,176 / 796,146 / 467,220 | **1.77×** |
| fc2 2048→512 | 1 | 1,050,624 → 526,336 | 1,132,503 / 973,115 / 612,106 | 614,359 / 973,115 / 546,193 | 572,218 / 973,115 / 546,193 | 1.85× |
| fc2 2048→512 | 8 | same | 1,133,270 / 1,021,268 / 612,172 | 623,230 / 1,021,268 / 582,547 | 621,370 / 1,021,268 / 582,547 | 1.85× |
| attn_out 512² | 1 | 264,192 → 133,120 | 300,375 / 258,779 / 169,258 | 175,447 / 258,779 / 152,113 | 164,026 / 258,779 / 152,113 | 1.77× |
| attn_out 512² | 8 | same | 301,142 / 270,836 / 169,708 | 178,174 / 270,836 / 161,203 | 176,314 / 270,836 / 161,203 | 1.77× |

**Where the time goes:**
- **`i8` at base:** flash 91–96% busy.
- **`i4`:** compute-bound everywhere (identical cycles across configs). LSU 77–86% busy, flash only 52–58%. The unpack costs about 360 cycles per 1 KB packed chunk (decode fc1: 1.02 M − 0.65 M over 1,024 chunks), about 0.35 cycles per weight. Decode cannot amortize it, because each weight is used once per token.
- **`i4f` at base:** still flash-bound (84–92%). At ideal, LSU is 43–52% busy (against 70–82% for `i8`) and ALU is 13–15%, from the nibble shifts.
- **MAC/cycle at ideal:**

  | Variant | tm = 1 | tm = 8 |
  |---|---:|---:|
  | i8 | 1.6–1.8 | 12–13.5 |
  | i4 | ~1.0 | ~8 |
  | i4f | 1.7–1.9 | 13–14.4 |

**Answer for the slide.** At base (1 B/cycle flash), halving the weight bytes beats the unpack cost:
- only marginally with unpack-to-DTCM: 1.17× on fc1, 1.15× on qkv;
- clearly with the fused unpack: 1.77–1.85×, close to the 2× byte ratio.

At fast, int4 helps only when fused (1.17× on fc1), and the naive unpack loses (0.68×).

### Whole decode step with int4 weights (estimated)

**How the estimate is built:**
- The four linears use the measured `i4f` tm = 1 cycles above.
- `lm_head` is the firmware's measured cycles × the fc1 `i4f`/`i8` ratio for each config. Both have K = 512, so this is an estimate.
- Embed, layernorm, and attention are the firmware's measured int8 numbers.

| Config | Cycles | ms | tok/s | vs int8 | Linears | Attention | lm_head |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 55.6 M | 1,112 | **0.90** | 1.45× | 28% | 43% | 28% |
| fast | 34.9 M | 698 | **1.43** | 1.13× | 40% | 22% | 37% |
| ideal | 33.6 M | 672 | **1.49** | 1.09× | 41% | 18% | 39% |

### KV cache in int4

An int4 KV cache halves the bytes per position: 16 heads × 32 dims × 2 (K and V) × 0.5 B = 512 B per layer per position, against 1 KB in int8. Attention at base is 99% PSRAM-bound, so this should roughly halve it. Estimated, not run: layer 0 drops from 4.7 M to about 2.4 M cycles at base. That would take the int4 decode step to about 43 M cycles, roughly 1.15 tok/s at base, before the unpack cost. That unpack is per use, since each position is read once per token.

## 3. Native int4 (estimated, proxy)

**Assumption.** A hypothetical Zvt int4 type in which each PE lane takes 8 int4 values per cycle, for a peak of 512 MAC/cycle. A native vtmm reads the same 16-byte operand registers and occupies the array for the same time as an int8 vtmm, but covers k = 8 instead of k = 4. There is no unpack.

**Proxy.** reef-perf cannot model a new type, so I emulated it with `llm_linear_i4.cc -DINT4=1 -DNATIVE4=1`:
- The int8 `mac4` runs directly on the packed weight and activation bytes, as if they were int8.
- It takes 4 steps per 1 KB chunk, which is K/2 steps over the reduction.
- DMA is int4-packed, and the epilogue and packed output are the same as in i4f.
- Real MACs = 2 × the useful MACs reef-perf reports.

The cycles below are measured on the proxy. What they represent is an estimate.

| Shape | tm | base i8 / i4f / **n4** | fast i8 / i4f / **n4** | ideal i8 / i4f / **n4** | Real MAC/c, ideal (n4) | n4 vs i8 (base / fast / ideal) |
|---|---:|---|---|---|---:|---|
| fc1 512→2048 | 1 | 1,190,449 / 663,546 / **650,687** | 693,809 / 594,969 / **400,319** | 650,382 / 594,969 / **357,819** | 2.93 | 1.83 / 1.73 / 1.82× |
| fc1 512→2048 | 8 | 1,193,521 / 665,340 / **652,255** | 703,217 / 631,323 / **401,887** | 698,574 / 631,323 / **382,427** | 21.9 | 1.83 / 1.75 / 1.83× |
| qkv 3×512² | 1 | 872,043 / 491,772 / **479,238** | 497,259 / 439,569 / **290,310** | 462,999 / 439,569 / **257,637** | 3.05 | 1.82 / 1.71 / 1.80× |
| qkv 3×512² | 8 | 874,371 / 493,500 / **480,390** | 504,795 / 467,220 / **291,462** | 499,176 / 467,220 / **275,349** | 22.9 | 1.82 / 1.73 / 1.81× |
| fc2 2048→512 | 1 | 1,132,503 / 612,106 / **605,751** | 614,359 / 546,193 / **346,167** | 572,218 / 546,193 / **304,437** | 3.44 | 1.87 / 1.77 / 1.88× |
| fc2 2048→512 | 8 | 1,133,270 / 612,172 / **606,143** | 623,230 / 582,547 / **346,559** | 621,370 / 582,547 / **329,020** | 25.5 | 1.87 / 1.80 / 1.89× |
| attn_out 512² | 1 | 300,375 / 169,258 / **165,687** | 175,447 / 152,113 / **102,711** | 164,026 / 152,113 / **91,699** | 2.86 | 1.81 / 1.71 / 1.79× |
| attn_out 512² | 8 | 301,142 / 169,708 / **166,079** | 178,174 / 161,203 / **103,103** | 176,314 / 161,203 / **97,851** | 21.4 | 1.81 / 1.73 / 1.80× |

**What native int4 adds:**
- **At base** it is still flash-bound (85–93% busy). Native int4 adds only about 2% over the fused unpack (i4f). The byte halving is the whole win, and i4f already gets it.
- **At fast and ideal** the gain is real. Native int4 is 1.5–1.8× faster than i4f, because the LSU work per MAC halves. The kernel is LSU-bound again (65–79%).
- **Utilization at ideal** is 0.6% (tm = 1) and 4.3–5% (tm = 8) of the 512 peak.
- **Fast is now flash-bound:** at 2 B/cycle, n4 is 1.1–1.15× above ideal.

### Whole decode step at context 1024, native int4 (estimated)

**How the estimate is built:**
- The linears use the measured n4 proxy, tm = 1.
- `lm_head` is the firmware's cycles × the fc1 n4/i8 ratio for each config: 15.1 M / 8.6 M / 7.8 M cycles.
- Embed and layernorm stay int8, as measured.
- **int4 KV with native-int4 attention (estimated).** The attention MAC loops are about 14% of the per-block ideal cycles (about 160 of about 1,150 cycles per head-block). Native int4 halves them, so attention compute ≈ 0.93 × the ideal measured. The memory stall (config − ideal) halves with the KV bytes. This gives:
  - L0: 2.88 M / 1.26 M / 1.11 M cycles
  - L1: 0.78 M / 0.35 M / 0.30 M cycles

  (base / fast / ideal.)

| Variant | base tok/s | fast tok/s | ideal tok/s | Cycles (base / fast / ideal) |
|---|---:|---:|---:|---|
| int8 firmware (measured) | 0.62 | 1.26 | 1.36 | 80.8 M / 39.6 M / 36.6 M |
| int4 W, fused unpack (est) | 0.90 | 1.43 | 1.49 | 55.6 M / 34.9 M / 33.6 M |
| native int4 linears + lm_head, int8 KV (est) | 0.91 | 1.92 | 2.21 | 55.0 M / 26.0 M / 22.6 M |
| **native int4 + int4 KV (est)** | **1.10** | **2.02** | **2.25** | 45.6 M / 24.8 M / 22.2 M |

With native int4 and int4 KV, the base step splits about evenly: linears 33%, attention 32%, lm_head 33%. At base, flash and PSRAM bandwidth decide the result. Native int4 compute pays off only once the links are at least at fast.

## Design notes

- **Tiling (same as `coral/kernels/linear.cc`):**
  - N/64 column groups, each with 4 int32 accumulators of 16 columns.
  - K is streamed in chunks of 32 rows, double-buffered by DMA: issue chunk i+1, compute chunk i, then wait.
  - Each `mac4` step does 4 A rows (tokens in lanes, tm = tokens), then 4×4 B rows and 4 vtmm.
  - The epilogue runs `vtmv` column slices, then bias, shift, an optional GeLU or residual, and the store.
- **int4 layout (row-pair convention in `ops.hpp`):**
  - **Weights:** a 32-row chunk packs into 1 KB. Byte j holds row (j/64) in its low nibble and row (j/64)+16 in its high nibble, so `unpack_i4_rows(chunk, buf, 1024)` gives the int8 [row][64] chunk.
  - **Activations [feature][8 tokens]:** 128 B per 32-feature block, features 0–15 in the low nibbles and 16–31 in the high. The epilogue's accumulator pairs (0,1) and (2,3) map exactly onto those feature pairs, so `pack_i4` writes them directly.
  - **Weights are symmetric signed int4.** No zero point is needed, so the firmware's 128·Σact correction is dropped. It is dropped in the `i8` baseline as well, to keep the comparison clean.
- **Bias** stays int32 (256 B per column group).
- **The fused unpack** is my own asm (`mac4_i4`), not the shared helper.
- **DTCM:**

  | Shape | i4 | i4f | i8 |
  |---|---:|---:|---:|
  | fc1 | 19.0 KB | 16.9 KB | 25.4 KB |
  | qkv | 12.8 KB | 10.8 KB | 12.8 KB |
  | fc2 | 31.3 KB (tight) | 29.2 KB | 25.1 KB |

  fc2's int4 activations take 8 KB packed plus 16 KB unpacked; unpacking per chunk would free about 15 KB.

## Caveats

- **`lm_head` address overlap.** The firmware's `lm_head` weights run from about 0x2340_0000 to 0x24A7_3800. They overlap the PSRAM window of `psram_kv.yaml` (0x2400_0000 + 8 MB). Under plain `run.sh`, about 10 MB of `lm_head` streams over the 0.25 B/cycle PSRAM link: 55.1 M cycles at base, against 27.6 M. I ran `lm_head` with `psram_kv_e2e.yaml`, which moves the PSRAM window to the KV cache, so `lm_head` is entirely on flash. Other agents running firmware ELFs should do the same.
- **Pool busy %** covers the whole run, including boot. For tiny ROIs (embed, layernorm) it is above 100% and meaningless. Layernorm always processes all 8 token columns, so its decode and prefill numbers are equal.
- **Attention MACs** are the matrix unit's useful MACs, which include the masked lanes of the tail block.
- **The int4 kernel** treats V like Q and K: feature-major, not token-major as in the firmware's `kValueRows`. Its requant shift and GeLU are placeholders. reef-perf checks timing only, so no outputs were verified.
- **The native int4 numbers come from a proxy.** The proxy runs real int8 vtmm on packed bytes, which is the same operand traffic and array time as the assumed native op. The attention and lm_head numbers there are scaled estimates.
- **Estimates are marked:** the int4 `lm_head` (ratio-scaled), the int4 whole step, and the int4 KV.

# GPT-2 small int8 on Reef (Coral NPU + VME): week-3 kernel analysis

All cycle counts come from Coral's **VME RTL model** (coralnpu
`VmeCoreMiniAxi`, commit pinned in `config/hardware.json`, Verilator via
cocotb, behavioral SRAM model). Raw numbers: `results/results.json`. Model
numbers: `analysis/roofline.json`. Reproduce with the two commands in the
simulations README. Milliseconds assume the **50 MHz clock, which is not yet
confirmed**. Section 7 lists every number that is assumed rather than
measured.

The end-to-end figures include the final LayerNorm and LM head; section 3
also has a vocab-size sweep. Every matmul in a block runs on the matrix
engine (VME) in prefill:
qkv, attn_out, fc1, fc2, and attention's QKᵀ and P·V. Decode's single-row
products run on the vector unit (RVV), which measured faster than a padded
VME tile (see 4). LayerNorm, residual add and GeLU are scalar: they aren't
matmuls, so VME can't run them. Softmax has two versions, float `expf`
(`online_softmax`) and a lookup table (`softmax_lut`). Attention uses the
lookup table; section 1 compares the two, and section 3 shows both
end-to-end.

## 1. What runs, and how well

Every kernel ran on the RTL model and was checked against NumPy:

| Kernel (sim tile) | Cycles | Error vs reference |
| --- | ---: | --- |
| `matmul_vme_s8u8` 16×512×16 | 11,005 | bit-exact |
| `matmul_rvv_s8u8` 16×512×16 | 74,006 | bit-exact |
| `matmul_rvv_s8u8` 1×512×32 (decode) | 9,318 | bit-exact |
| `attention_vme_block` 16q × 64-key block (VME, LUT exp) | ~74.6k (was ~337k with `expf`) | 1.35 LSB vs float at 256 keys; within 1 LSB of serial at 64 keys, 2 at 256 |
| `attention_vme_block` 1q × 64-key block (decode, RVV, LUT exp) | ~9.0k (was ~25k) | 0.76 LSB vs float |
| `attention_flash_chunk` 16q × 64-key block (scalar reference) | ~2.2M | 0.50 LSB; within 1 LSB of serial |
| `attention_serial` 16q×64k (scalar reference) | 1,814,058 | 0.90 LSB vs float |
| `attention_flash` 16q×64k (scalar reference) | 1,980,632 | 0.50 LSB; within 1 LSB of serial |
| `online_softmax` (`expf`), 1024-key row | 572,943 | 0.49 LSB (1/255) |
| `softmax_lut` (lookup-table exp), 1024-key row | **52,674** | 0.49 LSB (1/255) |
| `softmax_lut` 64-key row (vs `online_softmax` 35,606) | 3,307 | 0.46 LSB |
| `softmax_lut_table` (once per layer) | 127,576 | within 0.5 Q15 step |
| `gelu`, 4096 elements | 53,287 | 0.49 LSB |
| `layernorm`, 4×768 | 243,069 | 0.50 LSB |
| `residual_add`, 4096 elements | 241,850 | exact |

**Microbenchmarks (sim):**

- Scalar load-to-use latency: **5 cycles**.
- Vector `vle32` + `vmv.x.s` chain: **10 cycles**.
- VME: **84 cycles per `matmul_step`** (16×16×4 MACs, including its 8
  operand loads) plus **253 cycles per tile**. That makes the measured peak
  **12.2 MACs/cycle (24.4 ops/cycle)**.
- Host→DTCM over the 128-bit `axi_slave`: **15.7 B/cycle** sustained, with
  the core halted.

**Host embedding** (token + position lookup, int8 out): p95 is **0.37 ms at
1024 tokens** and under 1 µs for decode, well under the 20 ms target. This
was measured on the dev Mac (arm64), not an x86 host.

**Accuracy and uint8 probabilities.**

- Moving `attention_serial` from the float softmax to uint8 probabilities
  (scale 1/255) changed its error against the double reference from ±0 to
  ±1 LSB on 4 of the 6 `test_flash` cases.
- The error grows with context: on the same 256-key inputs, the serial
  algorithm is 1.44 LSB from float, and the VME kernel is 1.35. The VME
  kernel also rounds each block's probabilities against that block's
  running max, which adds up to one more step relative to serial (2 LSB at
  256 keys, 1 at 64).
- `run_all.py` therefore checks uint8-probability kernels (serial, VME)
  against float with a 2-LSB tolerance, and float-probability kernels
  (flash, chunked) with 1 LSB.
- This should be re-checked on calibrated activations before committing to
  uint8 P.

### Softmax: two versions

Both versions share one contract: an int32 score row in, uint8
probabilities with scale 1/255 out, and only the visible slice `[lo, hi]`
touched. Both are kept in the tree.

- **`online_softmax` (float `expf`)** is the reference. It makes one pass
  for a running max and normalizer, rescaling the sum when the max grows,
  and a second to normalize. Every score costs two newlib `expf` calls.
- **`softmax_lut` (lookup table)** is what attention uses now. The score
  gap `d = max − s` is an integer, so `exp(−score_scale · d)` is a coarse
  table on d's high byte times a fine table on its low byte: two 256-entry
  Q15 tables, 1 KB in all. That's exact up to Q15 rounding, because
  exp(a + b) = exp(a) · exp(b). It makes three integer passes (max, sum,
  normalize) with one division per row.

| | `online_softmax` (`expf`) | `softmax_lut` (lookup table) |
| --- | ---: | ---: |
| 1024-key row (sim) | 572,943 cycles (560 / score) | **52,674 cycles (51 / score), 10.9×** |
| 64-key row (sim) | 35,606 cycles | **3,307 cycles, 10.8×** |
| Max error vs float (sim, 1024 / 64 keys) | 0.49 / 0.46 LSB | 0.49 / 0.46 LSB |
| Host `test_softmax`, all 6 cases | ≤ 0.50 LSB | identical per case |
| Float ops per score | 2 `expf` + multiplies | none (two table loads, one multiply) |
| Extra DTCM | none | 1 KB table |
| Setup | none | 127,576 cycles to build the table, once per layer |
| Range limit | none | gaps ≥ 65,536 read as 0, so it needs score_scale · 65,536 ≥ 13 (GPT-2: 20.5) |

**Inside `attention_vme_block`.** The running max is always `score_scale`
times an integer score, so the block rescale `exp(m_old − m_new)` comes
from the same table, and the kernel makes no `expf` calls. Measured on the
sim, same inputs:

| | with `expf` | with lookup table |
| --- | ---: | ---: |
| 16q × 64-key prefill block | ~337k cycles | **~74.6k (4.5×)** |
| 1q × 64-key decode block | ~25.4k cycles | **~9.0k (2.8×)** |
| Error vs float (256-key prefill / decode / vs serial) | 1.35 / 0.76 / 0.93 LSB | 1.35 / 0.76 / 0.93 LSB |
| Exp share of the block | up to ~85% (prefill) | small; the scalar loop around it dominates |
| 12 blocks + head: decode | 3.0 s / token | **2.26 s / token (−25%)** |
| 12 blocks + head: prefill 1024 | 12.4 min | **5.8 min (−53%)** |

The two tables cost 12 × 128k cycles = 31 ms per model at 50 MHz. That's
not counted in section 3, and the host could build them instead. The
`expf` version stays as the accuracy reference and for any score scale
too small for the table's range.

## 2. Where each operator sits on the roofline

![prefill](roofline_prefill.png)
![decode](roofline_decode.png)

Each figure has two panels. The left panel plots intensity against
DTCM↔core bytes (ceiling: 16 B/cycle). The right panel plots intensity
against host↔DTCM bytes (ceiling: the measured 15.7 B/cycle). Both panels
share the measured VME peak.

| Operator | Engine | Attained ops/cycle | What bounds it |
| --- | --- | ---: | --- |
| qkv, attn_out, fc1, fc2 (prefill) | VME | 23.6–23.8 | **VME issue rate**: at 97% of the measured peak. They sit at about 15 ops/byte, far right of both bandwidth ridges (~1.5 ops/byte), so the 84-cycle step is the limit, not bandwidth. |
| qkv, attn_out, fc1, fc2 (decode) | RVV | 3.5 | **Vector compute** on a matrix-vector product, at about 1.9 ops/byte. |
| attention (prefill) | VME + scalar LUT softmax | 3.03 | **Scalar per-score loop.** The block's two matmuls take about 12.8k cycles at the measured VME rate (2 × 4 tiles × (253 + 84 × 16)), 17% of the ~74.6k-cycle block. The other ~62k cycles are scalar per-score work: max, `k_zero` fixup, table lookups, the uint8 store, and the float rescale of the 16 × 64 accumulator. |
| attention (decode) | RVV + scalar LUT softmax | 1.78 | About half matmul (two 1×64×64 RVV products, ~4.7k cycles at the measured RVV rate) and half the per-score loop, in a ~9.0k-cycle block. |
| layernorm, residual, gelu | scalar | 0.05–0.10 | **Scalar float per element**: 79, 59 and 13 cycles per element. |

Attention went from 0.12 ops/cycle (scalar) to 0.74 / 0.64 on VME with
`expf`, and to **3.03 (prefill) and 1.78 (decode)** with the lookup-table
exp. That's 25× and 15× over scalar. Exp is no longer the bottleneck: the
scalar per-score loop around it is.

## 3. End-to-end estimate

The estimate is the sum of measured kernel cycles per tile × the number of
tiles, plus host↔DTCM transfer for every tile at the measured rate. Transfers
are not overlapped with compute: there is no DMA, and the core wins the DTCM
arbiter. Tiling is A-stationary for the matmuls, and attention streams
64-key K/V blocks for 16-query blocks.

The full model is 12 blocks plus the final LayerNorm and LM head (768 →
50,257) for one token: only the newest position needs logits. Sampling and
argmax on the host are not counted. Per-operator shares below are for one
block.

| Phase | One block | 12 blocks + head | Attention | Matmuls | LN / residual / GeLU | Host transfer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill 16 | 0.32 s | 4.3 s | 11% | 64% | 26% | 3.5% |
| prefill 64 | 1.3 s | 15.7 s | 11% | 64% | 26% | 3.5% |
| prefill 256 | 5.5 s | 66 s | 18% | 59% | 24% | 3.3% |
| prefill 1024 | 29 s | **5.8 min** | 37% | 45% | 18% | 2.7% |
| decode (context 1024) | 0.15 s | **2.26 s per token** | 25% | 61% | 14% | 7.6% |

The head (final LayerNorm + LM head) costs **494 ms per token**: 439 ms of
RVV compute plus 50 ms streaming the 39 MB weight matrix from the host.
That's 22% of a decoded token.

How we got here, for the 12 blocks + head:

| Attention version | Decode (s/token) | Prefill 1024 |
| --- | ---: | ---: |
| Scalar (`attention_flash_chunk`) | 8.2 | 58 min |
| VME matmuls, `expf` softmax | 3.0 | 12.4 min |
| **VME matmuls, lookup-table softmax** | **2.26 (−25%)** | **5.8 min (−53%)** |

At short prefill, attention is 11% of the block. The matrix engine still
computes the whole 64-key block while only 16 keys are visible, which
wastes most of it (0.24 ops/cycle at 16 tokens, versus 3.03 at 1024).

### Vocab size sensitivity

Only the LM head depends on vocab size. It's a 1 × 768 × V matrix-vector
product per generated token on RVV, so its cost is linear in V. The 12
blocks are unchanged. These figures are extrapolated from the measured
1×512×32 RVV tile and the measured host rate, not separately simulated
(`analysis/vocab_sweep.csv`):

| Vocab | LM-head weights (host traffic) | Head (ms/token) | Decode (ms/token) | Head share of decode | Prefill 1024 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 6.4 MB | 85 | 1,855 | 4.6% | 5.8 min |
| 16,384 | 12.7 MB | 164 | 1,934 | 8.5% | 5.8 min |
| 32,768 | 25.4 MB | 323 | 2,093 | 15.4% | 5.8 min |
| **50,257 (GPT-2)** | 39.0 MB | **494** | **2,264** | **21.8%** | 5.8 min |
| 100,352 | 77.9 MB | 980 | 2,750 | 35.6% | 5.8 min |

- **Each extra 1,000 vocab entries costs about 9.7 ms per decoded token**
  at 50 MHz. About 90% of that is RVV compute and 10% is host transfer:
  every token streams the whole 768 × V weight matrix through DTCM, because
  32 KB can't keep any of it resident.
- **Prefill barely moves** (under 0.1%), because the head runs once while
  the blocks scale with sequence length.
- **The host embedding lookup is insensitive to vocab:** p95 at 1024
  tokens is 0.375–0.43 ms from 8k to 100k (`results.json`,
  `embedding_host`). It gathers one row per token regardless of table
  size; only the table's footprint grows (77 MB at 100k).
- **Tying embedding and LM-head weights** (as GPT-2 does) saves memory, not
  time: the head still streams all V × 768 bytes per token.

## 4. Prefill vs decode

- **Prefill matmuls run on VME at 97% of its measured peak.** On the same
  16×512×16 tile, VME is **6.7× faster than RVV** (11,005 vs 74,006
  cycles).
- **Decode matmuls are matrix-vector (nq = 1), and VME wastes 15 of 16 tile
  rows.** Padded to a VME tile, a 1×512×16 product costs 11,005 cycles for
  8,192 useful MACs: **0.74 useful MACs/cycle, 6% utilization**. RVV does the
  same product in **4,690 cycles (1.75 MACs/cycle), 2.3× faster**. That's why
  `matmul()` sends decode (and decode attention) to RVV.
- **Not yet measured:** VME can be configured for fewer tile rows
  (`msettm`; coralnpu's own test uses a 5-row tile). If a 1-row tile costs
  less than a full one, a VME GEMV path could beat RVV. It deserves one
  microbenchmark.
- **Decode is host-transfer heavier.** Host traffic is 5.3% of decode block
  time versus 1.2% at prefill 1024, because every weight byte crosses the
  host bus once per token, with no reuse across rows.

## 5. Flash vs serial in bytes moved

- **DTCM↔core traffic (derived from the loop structure):** about the same
  for the scalar kernels. Both read a d-byte K column and a d-byte V row per
  visible pair, and both read and write a d-wide accumulator per pair.
  - Serial also writes and reads its score row (int32) and probability row
    (uint8): about 10 extra bytes per pair, roughly 1% more.
  - Flash spends cycles instead: one extra `expf` per key for the rescale.
    It measures **9% slower** than serial on the prefill tile (1.98M vs
    1.81M cycles) and on decode (139,769 vs 128,046).
  - The VME block-flash kernel moves its operands as 128-byte matmul steps
    (about 6 ops/DTCM byte, versus 0.4 for scalar flash), because each K/V
    byte now feeds 16 queries at once.
- **Host↔DTCM traffic is where they differ.**
  - Serial needs one head's whole K and V resident. At n_ctx = 1024 that's
    128 KB, 4× the DTCM, so serial cannot run at full context on this chip.
  - Both block-flash kernels stream K/V instead, at the cost of re-sending
    them for every 16-query block. That's **58.3 MB of host traffic per
    block at prefill 1024**, versus 3.1 MB if K, V, q and out each crossed
    once (19× more).
  - With the lookup-table exp, that's still only about 0.8% (prefill 1024)
    to 6% (decode) of attention kernel time, up from 0.17% and 2% with
    `expf`. It grows as the per-score loop gets faster; see 6c.

## 6. Initial architecture suggestions

These are candidates to test, not conclusions. Each comes with the
measurement it rests on and a first-order sensitivity on the blocks in 3.

a. **A GEMV mode or smaller tile for decode.** This is now the top lever
   for decode.
   - **Evidence:** decode matmuls get 1.75 MACs/cycle on RVV. VME pads to
      6% utilization (0.74 useful MACs/cycle).
   - **Sensitivity:** a mode that delivered VME's measured step rate on one
      row (12.2 MACs/cycle) would make decode matmul kernels about 7×
      faster. Per token, the 12 blocks' matmul kernels (965 ms) plus the LM
      head (439 ms) are 62% of decode, so that's about **−53% per decoded
      token**. The saving grows with vocab size. Check `msettm` with a 1-row
      tile first (4); it may already exist.

b. **Vectorize the per-score softmax loop (software), or add softmax
   support in hardware.** The lookup table took exp from about 85% of an
   attention block to a minor part of the loop; what's left is the scalar
   loop around it.
   - **Evidence:** a 16q × 64-key prefill block is ~74.6k cycles, of which
      the matmuls are ~12.8k. The other ~62k (about 60 cycles per score)
      are scalar integer work plus the float accumulator rescale. Decode
      blocks are about half per-score loop (~4.3k of ~9.0k cycles).
   - **Sensitivity:** if that loop ran at matmul speed, attention would
      shrink by up to ~83% (prefill) and ~48% (decode): about **−31% of
      prefill 1024 and −12% of a decoded token**. RVV has indexed loads
      (`vluxei16`) for the two table lookups, so this is software first;
      hardware exp is no longer worth much on its own. These are upper
      bounds.

c. **Larger DTCM (KV resident) and/or DMA with double buffering.**
   - **Evidence:** streamed attention moves 19× more host bytes than a
      resident KV would (5), and the LM head streams 39 MB per token. All
      host traffic is serialized with compute today: about 8% of a decoded
      token (183 ms) and 2.7% of prefill 1024.
   - **Sensitivity today:** a DMA that fully hid transfers saves at most
      that much (−8% per decoded token). Doubling the host bus saves about
      half of it (≤ −4%).
   - **When it matters:** this grows after (a) and (b). Every compute
      speedup raises the host share, because the bytes stay the same. Once
      decode attention is down to its matmul work, the streamed K/V becomes
      a large share of it. At that point, KV residency (≥128 KB per head)
      or DMA overlap is the main lever.

d. **Separate port per DTCM bank (load VME's A and B in parallel).**
   - **Evidence:** a VME step is 84 cycles for 1,024 MACs, which includes
      four strided A-column loads and four contiguous B-row loads. The VME
      matmuls sit far right of the 16 B/cycle ridge (15 ops/byte, versus a
      ridge at 1.5), so raw DTCM bandwidth isn't the limit. If the 84 cycles
      are mostly serialized load latency, parallel ports would help.
   - **Not yet measured:** how many of the 84 cycles are loads. That needs
      a load-free step microbenchmark before this is worth pursuing.
   - **Sensitivity:** prefill matmuls are 64% of prefill-16 and 45% of
      prefill-1024 block time. Halving the step saves up to about **−32%
      and −22%** respectively.

e. **Vector float for the pointwise kernels** (software plus a build flag,
   not hardware). The RTL enables vector float (`ZVE32F_ON`), but our build
   targets `zve32x`. LayerNorm, residual and GeLU are scalar today: 26% of
   prefill-16, 18% of prefill-1024 and 14% of decode block time. Moving them
   to RVV is the cheapest remaining software win, alongside (b).

## 7. Numbers that are assumed, not measured

| Number | Status / source |
| --- | --- |
| Core clock 50 MHz | PD team plan, **unconfirmed**; every ms figure scales with it |
| DTCM 16 B/cycle (`dbus`), 32 KB, 4 contiguous banks | PD team / coralnpu README; not measured in sim |
| AXI slave width 128 bit | RTL (`flags.bzl` `--lsuDataBits=128`) |
| Host→DTCM 15.7 B/cycle | **Measured**, but with the core halted and using the cocotb AXI driver; contention with a running kernel is not measured |
| Host transfers don't overlap compute | Assumption (no DMA; the host only gets cycles the core leaves free) |
| `ebus` latency | Unknown ("tens of cycles"); not used |
| DTCM↔core byte counts | **Derived** from each kernel's loop structure, not measured |
| Full-model cycles | Extrapolated from measured tiles. The VME k-chunk cost uses the measured k-sweep fit; RVV scales linearly in k; VME attention uses the measured full-block cycles (masked keys still cost work); scalar attention uses the measured cycles per visible pair |
| Final LayerNorm and LM head | Included, for one token per step (newest position only). Sampling/argmax on the host is not counted |
| Vocab sweep | Extrapolated from the measured RVV tile and host rate; only the GPT-2 vocab's tile shapes ran on the sim |
| Host-side work (summing K-split partial sums, orchestration) | Not counted |
| Quantization scales | Placeholders, not calibrated; affect accuracy numbers, not cycles |
| `expf` cost of about 280 cycles | Upper bound inferred from the `online_softmax` measurement; no longer on the attention path |
| Softmax exp tables | Built once per layer (128k cycles measured), not counted in the end-to-end figures |
| Embedding timing | Measured on an arm64 Mac, not the x86 host |
| SRAM timing | Behavioral SRAM model only; no `sram22` macro in coralnpu's flow |

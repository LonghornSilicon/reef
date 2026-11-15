# Activations on int4 maps: ReLU, SiLU, Tanh, Sigmoid (GeLU, LeakyReLU as reference)

Kernel: `coral/ops/kernels/act.cc` (one source; `-DMODE`, `-DMETHOD`, `-DLM`, `-DACT`, `-DMAP_*`). Builds and runs: `build/ops/act/<variant>/` (`builds.txt` lists the flags for each variant, and each directory has a `summary.txt`).

## Summary
Run as a standalone pass over a PSRAM feature map, every activation is purely **memory-bound**. On base the pass takes 4.24 cycles/element and ext_ram_link is 99.8% busy, whatever the activation. The compute part is only 0.27–0.70 cycles/element, so base is 6.1× slower than ideal and fast is 1.8× slower. Fused into a conv/GEMM epilogue, the extra cost is **0.29–0.42 cycles/element for any LUT activation** (m4). **SiLU, Tanh, Sigmoid, GeLU and LeakyReLU all cost exactly the same**: the cycle counts were identical (1,379,015 / 1,097,695) when only the table changed, because only the table contents differ. **ReLU is nearly free.** It costs +0.11 c/elem as a bit trick on packed int4 (m4), and **0 extra instructions** when it is folded into `pack_i4`'s clamp (vmax −8 → vmax 0). The byte-indexed 256-entry LUT (c) needs the fewest instructions: 1 `vluxei8` per 2 elements, with no unpack or pack. In reef-perf it ties with or loses to a 16-entry nibble LUT (b'), because reef-perf charges an indexed load per distinct 16-byte DTCM line it touches. A 256 B table spans 16 lines and a 16 B table spans one. The real RTL cost of indexed loads has not been measured, so the ranking between (b') and (c) is open (see caveats). Fusing is the real answer: it saves 34.7 ms (base) for each 80×80×64 map.

## Methods (all int4 packed 2/byte, row-pair nibble convention)
| # | Method | Ops per 16 B register (32 elems) | Unit |
|---|---|---|---|
| 0 | copy (vle+vse), baseline | – | LSU |
| 1 | ReLU bit trick: `m=((b&0x88)>>3)*15; r=b&~m` | 5 ALU | ALU (+MUL) |
| 2 | (b') nibble unpack (and/srl) → 2× `vluxei8` into 16-entry tables (lo, hi pre-shifted) → or | 3 ALU + 2 idx-ld | LSU |
| 3 | (b) unpack to signed int8 → 2× `vluxei8` into the 256-entry int8 table (the linear.cc `kGelu` path) → pack | 6 ALU + 2 idx-ld | LSU |
| 4 | (c) byte-indexed 256-entry LUT on the packed byte, no unpack | 1 idx-ld | LSU + DTCM port |
| 5 | (d) the tables of method 2 held in vector registers, 2× `vrgather` | 3 ALU + 2 gather | permute_reduce |
| 6 | int8 storage, ReLU = `vmax` | 1 ALU | ALU |
| 7 | int8 storage, `vluxei8` 256-entry (today's GeLU) | 1 idx-ld | LSU |

## Mode 2: fused/epilogue cost (DTCM only, no DMA; 8 KB in → 8 KB out, ×128 passes = 2.1M int4 elems)
Cycles per element under ideal (all three configs are identical, since nothing goes off chip). "+act" = minus the copy baseline: the cost an epilogue pays on top of the load and store it already does.

| Method | m4 c/elem | m4 +act | m1 c/elem | m1 +act | top busy pools (m4) |
|---|---:|---:|---:|---:|---|
| 0 copy int4 | 0.234 | – | 0.344 | – | lsu |
| 1 ReLU bits | 0.344 | **+0.110** | 0.813 | +0.469 | lsu, alu 18%, mul 5% |
| 2 nibble LUT16 (b') | 0.523 | **+0.289** | 0.844 | +0.500 | lsu |
| 3 int8 LUT256 (b) | 0.633 | +0.399 | 1.000 | +0.656 | lsu, alu 15% |
| 4 byte LUT256 (c) | 0.657 | +0.423 | 0.790 | **+0.446** | lsu, dtcm 59% |
| 5 vrgather16 (d) | 0.563 | +0.329 | 0.813 | +0.469 | permute_reduce 55%, lsu 46% |
| 6 int8 ReLU (vmax) | 0.484 | +0.015 | 0.876 | +0.188 | lsu |
| 7 int8 LUT256 (GeLU today) | 1.315 | +0.846 | 1.581 | +0.893 | lsu, dtcm 59% |
| ReLU folded into pack_i4 clamp | – | **0** (same instr count, by construction) | – | 0 | – |

Every activation costs the same as GeLU (ideal ROI, m4):
- (c): ReLU-LUT / GeLU / Tanh / Sigmoid = 1,379,015 cycles each.
- (b'): ReLU-LUT, GeLU, Tanh and Sigmoid each take 1,097,695 cycles, give or take 2.

## Mode 1: standalone pass over a PSRAM map (double-buffered 6 KB chunks, m4)
External bytes are int4-packed and count both the read and the write: 80×80×64 = 204,800 B in plus 204,800 B out, and 160×160×32 = 409,600 B in plus 409,600 B out. There are no weight bytes. DTCM use: 24 KB of buffers plus 320 B of tables (25.2 KB of data in all).

| Map | Method | base | fast | ideal | ideal elem/cycle | base ms @50 MHz |
|---|---|---:|---:|---:|---:|---:|
| 80×80×64 (409,600 elems) | 0 copy | 1,736,640 | 508,047 | 111,548 | 3.67 | 34.7 |
| | 1 ReLU bits | 1,737,088 | 509,839 | 156,348 | 2.62 | 34.7 |
| | 2 LUT16 (b') | 1,737,824 | 512,783 | 229,948 | 1.78 | 34.8 |
| | 3 LUT256 int8 (b) | 1,738,272 | 514,575 | 274,748 | 1.49 | 34.8 |
| | 4 byte LUT (c) | 1,738,352 | 517,685 | 284,994 | 1.44 | 34.8 |
| | 5 vrgather (d) | 1,737,987 | 513,434 | 246,215 | 1.66 | 34.8 |
| | 6 int8 ReLU (2× bytes) | 3,473,211 | 1,015,611 | 229,024 | 1.79 | 69.5 |
| | 7 int8 LUT (2× bytes) | 3,476,617 | 1,027,272 | 569,525 | 0.72 | 69.5 |
| 160×160×32 (819,200 elems) | 0 copy | 3,473,147 | 1,015,547 | 222,624 | 3.68 | 69.5 |
| | 1 ReLU bits | 3,474,043 | 1,016,443 | 312,224 | 2.62 | 69.5 |
| | 2 LUT16 (b') | 3,475,515 | 1,019,026 | 459,424 | 1.78 | 69.5 |
| | 3 LUT256 int8 (b) | 3,476,411 | 1,021,266 | 549,024 | 1.49 | 69.5 |
| | 4 byte LUT (c) | 3,476,617 | 1,027,272 | 569,525 | 1.44 | 69.5 |
| | 5 vrgather (d) | 3,475,838 | 1,019,837 | 491,955 | 1.67 | 69.5 |
| | 6 int8 ReLU | 6,943,472 | 2,028,575 | 457,648 | 1.79 | 138.9 |
| | 7 int8 LUT | 6,945,152 | 2,046,535 | 1,138,644 | 0.72 | 138.9 |

Roofline:
- Arithmetic intensity is 1 element op per external byte (int4: 0.5 B read + 0.5 B written per element). For int8 it is 0.5.
- Machine balance with the byte LUT: 1.44 elem/cycle ÷ 0.25 B/cycle (base) = 5.8 elem/B, and ÷ 1 B/cycle (fast) = 1.44 elem/B.
- So the pass is **memory-bound on the PSRAM link** in both configs. Base is 6.1× slower than ideal (ext_ram_link 99.8% busy), and fast is 1.8× slower (98% busy).
- Base runs at about 0.236 B/cycle, which is the link's 0.25 less burst latency.
- int4 halves the time against int8: 34.7 ms vs 69.5 ms on base for 80×80×64.

Where on the NPU:
- Standalone: ext_ram_link is the bottleneck. LSU is 10–20% busy on base.
- Fused (methods 2–4 and 7): the LSU is the bottleneck because of the indexed loads, and the DTCM port is 59% busy for the 256 B tables.
- vrgather: moves the work to permute_reduce (55%). permute_reduce is not pipelined, at 3 cycles per uop.
- ReLU bit trick: the ALU (18%). The matrix array is idle in every variant.

Extrapolation:
- Standalone: a YOLO-scale SiLU pass costs 34.7 ms per 80×80×64 map and 69.5 ms per 160×160×32 map on base (10–20 ms on fast). That is entirely PSRAM traffic.
- Fused into the producing conv's epilogue: the same 80×80×64 map costs 409,600 × 0.29–0.42 ≈ 118–173k cycles ≈ 2.4–3.5 ms (estimated from the per-element numbers), with zero extra traffic. That is about 10× cheaper on base.
- **Never run activations as a standalone pass.**

## Design notes
- **Stream mode:** 2 input and 2 output 6 KB buffers. Each iteration does `dma_wait`, issues the DMA in for chunk c+1, applies chunk c (in → out), then issues the DMA out for chunk c. The single DMA channel queues the read and the write.
- **Out of place:** buffers are filled with pseudo-random bytes before the ROI. reef-perf runs the vector code functionally (Spike), so indexed-load addresses come from real data, while DMA moves no data. Doing the pass in place would feed already-activated values back in and shrink the set of table lines touched.
- **int4 handling:**
  - (c) never unpacks: the table maps one byte (2 nibbles) to one byte.
  - (b') uses unsigned nibble indices into two 16 B tables, the high one pre-shifted by 4, so there is no sign extension and the pack is one `vor`.
  - (b) is the faithful port of the linear.cc kGelu path, with signed unpack, the 256 B int8 table and pack. It is the slowest LSU variant.
- **Epilogue:**
  - ReLU: change `pack_i4`'s `vmax(x,-8)` to `vmax(x,0)`, which costs nothing.
  - Other activations: put one `vluxei8` on `pack_i4`'s 16-byte result (c), or two 16 B lookups (b').
  - At m1 (the epilogue's natural width), (c) is the cheapest LUT at +0.446 c/elem.
  - At m1, the ReLU bit trick is latency-bound (a chain of 5 dependent ops) at +0.47. Use the clamp instead.
- **Tables:** f(q·0.5) requantized to int4. The output scale is 0.5 for ReLU, Leaky, SiLU and GeLU, 1/7 for Tanh, and 1/16 with zero point −8 for Sigmoid. They are 320 B in DTCM in all. LeakyReLU (DCGAN discriminator) is the same as any other LUT.

## Caveats
- **The cost of an indexed load in reef-perf is a model assumption.** reef-perf charges 3 cycles + the distinct 16 B DTCM lines touched, per register. Indexed and strided accesses are listed as unmeasured in `docs/deviations.md` (U6). That is why the 16 B table (1 line) beats the 256 B table (about 10 lines touched per register on random data).
- If the RTL instead serves indexed loads at one element per cycle, (c) does 16 lookups per 32 elements against 32 for (b') and (b). (c) would then be about 2× cheaper and the clear winner (estimated, not run).
- **Fused costs depend on the data.** Uniform random bytes are close to the worst case. Real post-conv activations cluster near 0 and would touch fewer lines, making them cheaper in the model.
- **Pool busy % covers the whole run,** including the scalar table and fill init. The LSU therefore reads above 100% for short ROIs. Treat it as "saturated".
- The fused numbers are DTCM→DTCM passes. In-register epilogue cost is taken as "minus copy". The ReLU-in-clamp "0" comes from instruction counting, not from a run.
- Only timing is modelled. Outputs are not checked.

# Operator sample kernels: shared brief

Read this whole file before starting. You own ONE operator. Write one sample
kernel for it, run it in reef-perf on typical workload shapes, and report
numbers for a design-review slide on "the operators the NPU needs".

## The machine (Reef NPU = CoralNPU M3 core + Zvt matrix engine + DMA)

- **Core:** RV32IM + Zve32x vector (VLEN 128: 16 int8 lanes at m1, 64 at m4), integer only, no FPU.
- **Zvt matrix engine** (CoralNPU matrix extension v0.4), as modelled by `configs/matrix/coral_vme.yaml`:
  - 64 PEs (8×8), 4 int8 MACs each, so the peak is **256 int8 MAC/cycle**.
  - Output-stationary int32 accumulators: 16 tiles of 16×16.
  - One `vtmm` is C[tm×16] += Aᵀ B, 4 k-steps per instruction (tk = 4). The A rows (tm lanes) come from v8, v10, v12, v14 and the B rows (16 lanes) from v16, v18, v20, v22.
  - The operands reach the vector registers through the core's **LSU** (`vle8`). On the existing kernels, operand delivery limits throughput far more than the PE array does.
  - Types fixed by the ISA: int8→int32, bf16→fp32, fp32→fp32. **There is no int4 multiply.**
- **Memories:**
  - ITCM 8 KB (code).
  - DTCM 32 KB at 0x10000 (all data the core touches, plus stack; leave at least 1 KB).
  - Weight flash at 0x20000000: read only, XSPI link.
  - PSRAM at 0x24000000: read/write, 8 MB, its own link. Use it for activations and feature maps; they do not fit the DTCM.
- **DMA:**
  - `dma(src, dst, len)` starts a copy between external memory and DTCM. `dma_wait()` returns when every started transfer is done.
  - 20 setup cycles, 16 B per beat, one channel.
  - Double-buffer: issue chunk i+1, compute chunk i, wait.
- **Clock:** 50 MHz.

## int4 everywhere (W4A4): REQUIRED

All weights AND activations are int4, stored packed 2 per byte in flash, PSRAM and DTCM. Because Zvt multiplies only int8, unpack to int8 on the vector unit right before the matrix engine (or before vector math), and requantize outputs back to packed int4 before storing them.

Use the shared helpers in `coral/ops/include/ops.hpp`:
- `unpack_i4_rows`
- `pack_i4`
- the row-pair nibble packing convention described there.

You may write a faster fused unpack (for example, unpacking straight into the operand registers inside an asm MAC loop) if it helps. Say so in your report if you do.

Report bytes moved as int4-packed bytes. Note the unpack cost. Where it is cheap to do, ALSO give the int8 version's cycle count (the same kernel without packing) so the slide can show what int4 buys.

## Measured compute ceiling (coral/ops/kernels/peak.cc; use these for the roofline)

Same inner loop as `linear.cc`: `mac4`, operands already in DTCM, no DMA, K=512, N=512.

| Variant | MAC/cycle | Busy units |
|---|---:|---|
| int8, tm = 16 rows | **29.5** | LSU 79%, array 11.5% |
| int8, tm = 8 rows | 14.8 | same cycles as tm=16; the LSU is the limit, not the array |
| int4 weights, unpacked on every use, tm=16 | 17.3 | LSU 83% |

Lessons:
1. **Always fill tm = 16 rows.** Cycles are the same as tm=8, and you get twice the MACs.
2. The practical roof is about 30 MAC/cycle: 11.5% of the 256 peak. Operand loads through the LSU are the limit.
3. **Unpack int4 once per reuse, not per use.** For example, unpack a weight chunk once into DTCM and reuse it across every pixel tile.

## Files and tools (all paths under `/Users/joshv/Projects/lhs/reef/inference-engine`)

- **`coral/ops/include/npu.hpp`:** DMA, Zvt wrappers (`config_int8`, `vtzero`, `mac`, `mac4`, `vtmv`), `mcycle`.
- **`coral/ops/include/ops.hpp`:**
  - Memory map (`kFlash`, `kRam`).
  - `roi_begin` / `roi_end` / `mark`.
  - `matrix_on()`: call it first if you use Zvt.
  - int4 helpers.
- **Reference kernels** (int8, already tuned on this machine): `coral/kernels/linear.cc` (double-buffered weight streaming + Zvt GEMM + epilogue) and `coral/kernels/attention.cc` (online-softmax attention with the exp LUT). Copy patterns from them. Read `coral/kernels/npu.hpp` comments on how `vtmv` and `config_int8` must be used. Note that any `vset*` clears `mtype`.
- **Your kernel:** `coral/ops/kernels/<op>.cc`, one file with `main()`, its own static DTCM buffers, and `roi_begin()` / `roi_end()` around the measured work. Shapes go in `#ifndef` macros so one source builds every shape you test. You may add more `.cc` files named `<op>_*.cc` if you need variants.
- **Build:** `coral/ops/tools/build.sh <kernel> build/ops/<op>/<variant> -DFOO=1 ...` (run from `inference-engine/`; it uses Docker). It prints the ITCM and DTCM use.
- **Run:** `coral/ops/tools/run.sh build/ops/<op>/<variant>/<kernel>.elf` runs 3 memory configs in parallel and prints a summary (`coral/ops/tools/stats.py`):
  - **`base`:** flash 1 B/cycle with 68-cycle latency, plus PSRAM 0.25 B/cycle with 30/16-cycle latency and 100 B bursts. This is the team's pessimistic baseline.
  - **`fast`:** flash 2 B/cycle plus PSRAM 1 B/cycle.
  - **`ideal`:** both links effectively infinite, so compute only. This is the roofline compute ceiling for your code.

  Pool busy % is "where on the NPU" the time goes. Add `--json` to `stats.py` for raw numbers.
- reef-perf moves NO data and computes NO results: timing only. Write the kernel as if it were real (correct indexing, correct sizes), but you cannot check outputs in the simulator. Keep it simple and plausible rather than clever and unverifiable.
- Simulation runs at roughly 1M cycles per second. Keep each run under about 50M cycles. If a full layer is larger, run a representative slice (some output rows or channels) and scale up linearly, saying so.
- Do NOT edit anything outside `coral/ops/kernels/<op>*.cc`, `build/ops/<op>/` and `coral/ops/results/<op>.md`. The shared headers and tools belong to everyone. If one is broken, work around it in your own file and mention it in your report.
- Do not commit, push, or touch git.

## What to measure, for each typical shape (2 or 3 shapes from the named workloads)

1. **Ops:** MACs (or element ops for non-MAC operators).
2. **Bytes, int4-packed:** weight bytes from flash, activation bytes read from PSRAM, activation bytes written to PSRAM.
3. **DTCM footprint** of your kernel's buffers.
4. **Cycles** under base, fast and ideal. Also MAC/cycle (or ops/cycle) under ideal.
5. **Roofline:**
   - Arithmetic intensity = ops ÷ external bytes.
   - Machine balance: the ideal-config ops/cycle ÷ link bytes/cycle.
   - Bound type: compute-bound or memory-bound, and which link.
   - Classify by how far base and fast are from ideal.
6. **Where on the NPU:**
   - The top busy pools from stats (matrix_array, lsu, alu, dtcm, ext links, ...).
   - Matrix MAC utilization.
   - A one-line statement of which unit is the bottleneck.
7. **Full-layer or full-model extrapolation** for the workload where meaningful (for example, "this layer × N layers ≈ X cycles ≈ Y ms at 50 MHz").
8. **int8 comparison** if cheap.

## Report

Write `coral/ops/results/<op>.md` with:
- A one-paragraph summary.
- The table(s) of numbers above.
- The design notes: the tiling and dataflow, what is double-buffered, and how int4 is handled.
- Caveats.

Then reply with the same content, compact, so the coordinator can merge it into slides. Every number must come from a run you did. Mark anything estimated as estimated.

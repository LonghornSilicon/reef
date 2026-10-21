# Kernel results

Cycle counts come from Coral's VME RTL model (`VmeCoreMiniAxi`, Verilator via
cocotb, behavioral SRAM; coralnpu commit pinned in `hardware.json`). Each sim
test checks every call against NumPy, so a logged number is always a
correct result. Milliseconds assume a 50 MHz clock, which is **not
confirmed**.

## Reproduce

From `simulations/`:

    python sim.py build && python sim.py test
    uv run --group sim python analysis/collect_results.py --log "what changed"
    uv run --group sim python analysis/roofline.py
    uv run --group sim python analysis/roofline.py --model analysis/gpt_neo_28m.json

`collect_results.py` writes `out/results.json`. `roofline.py` reads it and
writes the per-operator tables, roofline plots and vocab sweep to `out/`.
Generated files are not committed; this page is the record.

## Metrics in the log

All values are kernel cycles (mcycle around one call), except `vme_step`.
They come from the sim test of each kernel's operator: `test_linear.py`,
`test_attention.py`, `test_pointwise.py` (activation, LayerNorm, residual),
`test_mlp.py` and `test_embedding.py`. The column-to-job map is `METRICS` in
`collect_results.py`.

| Column | What it measures |
|---|---|
| `vme_matmul` | `vme_matmul_u8`, one 16×512×16 prefill tile |
| `rvv_matmul` | `matmul_rvv`, one 1×512×32 decode tile |
| `linear` | `linear` (matmul + zero-point fixup, bias, requantize), 16×512×16 |
| `vme_step` | cycles per 16×16×4 VME `matmul_step` (slope of the k sweep) |
| `attn_prefill` | GPT-2 head (d 64): 16 queries over 256 keys in four 64-key blocks, plus finish |
| `attn_decode` | GPT-2 head (d 64): 1 query over 256 keys (vector-unit path), plus finish |
| `attn_d32_prefill` | GPT-Neo 28M head (d 32), same stream as `attn_prefill` |
| `attn_d32_decode` | GPT-Neo 28M head (d 32), same stream as `attn_decode` |
| `attn_d32_local` | GPT-Neo 28M local layer: 16 queries, 384 keys, 256-key window |
| `lut_table` | `softmax_lut_table`, the once-per-layer exp table build |
| `softmax_row` | `softmax_lut` over one 1024-score decode row |
| `gelu` | `gelu`, 4096 elements (table built untimed) |
| `layernorm` | `layernorm`, 4 rows × 768 |
| `residual` | `residual_add`, 4096 elements |
| `mlp_prefill` | `mlp_block` × 4 (16 rows, d 64, inner 64 in 16-wide chunks) plus `mlp_finish` |
| `mlp_d512_decode` | GPT-Neo 28M width at decode: 1 row, d 512, inner 64 in 8-wide chunks, plus finish |
| `embed_prefill` | `embedding_lookup`: 16 tokens × 512 from a 64-row table in eight 8-row slices |
| `position_prefill` | `add_position_embeddings`: 8 tokens × 512 |

`vs previous` lists every metric that moved by 1% or more since the row above.

## End-to-end estimate (roofline.py)

Measured tiles × tile counts, plus host↔DTCM transfer at the measured 15.7
B/cycle. There is no DMA, so transfers are not overlapped with compute. The
figures include the final LayerNorm and LM head.

| Model | Decode (context 1024) | Prefill 1024 |
|---|---:|---:|
| GPT-2 small, scalar attention (85d6089) | 8.2 s/token | 58 min |
| GPT-2 small, VME attention, `expf` (57df445) | 3.0 s/token | 12.4 min |
| GPT-2 small, VME attention, LUT exp (8f71744) | 2.26 s/token | 5.8 min |
| GPT-2 small, vectorized integer loops (5fb006c+) | 1.93 s/token | 3.9 min |
| GPT-Neo 28M, VME attention, LUT exp (590ad5a) | 0.95 s/token | 2.3 min |
| GPT-Neo 28M, vectorized integer loops (5fb006c+) | 0.76 s/token | 1.3 min |

The estimate charges matmuls at the bare matmul tile cost, so `linear`'s
zero-point fixup, bias and requantization are not in it.

Not measured: the clock, contention between host writes and a running
kernel, DTCM↔core byte counts (derived from loop structure), host-side work
(K-split partial sums, orchestration), and the GPT-Neo head count (16 × 32
is assumed). Quantization scales are placeholders; they affect accuracy,
not cycles.

## Log

Append with `collect_results.py --log "note"` after a sim run, then add a
change note under the table saying what changed, how, why, and the effect.
A `+` after the commit means the run measured uncommitted changes on top
of it; the change that followed is in the next commit.

| date | commit | change | vme_matmul | rvv_matmul | linear | vme_step | attn_prefill | attn_decode | attn_d32_prefill | attn_d32_decode | attn_d32_local | lut_table | softmax_row | gelu | layernorm | residual | mlp_prefill | mlp_d512_decode | embed_prefill | position_prefill | vs previous |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-18 | 85d6089 | Week-3 baseline. Attention is scalar attention_flash_chunk (expf per score), same shapes | 11005 | 9317 |  | 84 | 8608212 | 554812 |  |  |  |  |  | 53287 | 243069 | 241850 |  |  |  |  | first record |
| 2026-10-18 | 57df445 | VME block-flash attention: q.kt and P.v through matmul() on the matrix engine, expf softmax | 11005 | 9317 |  | 84 | 1378744 | 105549 |  |  |  |  |  | 53287 | 243069 | 241850 |  |  |  |  | attn_prefill -84%, attn_decode -81% |
| 2026-10-18 | 8f71744 | LUT exp (softmax_lut_table) replaces expf in VME attention | 11005 | 9317 |  | 84 | 362264 | 40118 |  |  |  | 127576 | 52674 | 53296 | 243078 | 241857 |  |  |  |  | attn_prefill -74%, attn_decode -62% |
| 2026-10-18 | 590ad5a | GPT-Neo 28M d32 attention streams measured (no kernel change) | 11005 | 9317 |  | 84 | 362264 | 40118 | 263044 | 25930 | 290351 | 127576 | 52674 | 53296 | 243078 | 242082 |  |  |  |  | no change |
| 2026-10-20 | 4b30629+ | PR #16 trim: kernels merged into 3 files, 2 ELFs (no kernel changes) | 11002 | 9315 |  | 84 | 358439 | 39934 | 259219 | 25746 | 285563 | 127577 |  | 53286 | 243105 | 241941 |  |  |  |  | attn_prefill -1%, attn_d32_prefill -1%, attn_d32_local -2% |
| 2026-10-21 | 019671f+ | Kernels split one per engine operator; new linear, softmax and streamed mlp kernels | 11003 | 9316 | 67873 | 84 | 362371 | 40128 | 263193 | 25940 | 290533 | 128088 | 52673 | 53296 | 243078 | 241536 | 240474 | 200514 |  |  | attn_prefill +1%, attn_d32_prefill +2%, attn_d32_local +2% |
| 2026-10-21 | a560a2f+ | Embedding kernel (embedding_lookup, add_position_embeddings) for operator/embedding.hpp | 11003 | 9316 | 67873 | 84 | 362371 | 40128 | 263193 | 25940 | 290533 | 128088 | 52673 | 53296 | 243078 | 241536 | 240474 | 200514 | 59371 | 233339 | no change |
| 2026-10-21 | 5fb006c+ | Vectorize every integer loop on the RVV unit; requantize in fixed point | 11003 | 9316 | 24332 | 84 | 230527 | 31665 | 137562 | 17912 | 157043 | 128088 | 13032 | 7040 | 13955 | 7594 | 39690 | 80293 | 6566 | 7634 | linear -64%, attn_prefill -36%, attn_decode -21%, attn_d32_prefill -48%, attn_d32_decode -31%, attn_d32_local -46%, softmax_row -75%, gelu -87%, layernorm -94%, residual -97%, mlp_prefill -83%, mlp_d512_decode -60%, embed_prefill -89%, position_prefill -97% |

## Change notes

One per log row, newest last.

**85d6089: baseline.** The first RTL-sim measurements of every kernel.
Attention was scalar (`attention_flash_chunk`, one `expf` per score),
matmuls ran on the VME (prefill) and RVV (decode), and the pointwise ops
were scalar.

**57df445: attention matmuls on the matrix engine.**
- How: new `vme_attention_block`. Per 64-key block, S = q·kᵀ and P·V go
  through `matmul()` (VME at 16 queries, RVV at decode), and the
  online-softmax rescale runs once per block instead of once per key.
- Why: scalar dot products were nearly all of attention's cycles.
- Effect: attention −84% prefill, −81% decode. GPT-2 decode went from
  8.2 to 3.0 s/token.

**8f71744: lookup-table exp.**
- How: each score is score_scale × an integer, so exp(−scale · gap) for an
  integer gap is coarse[gap >> 8] × fine[gap & 255], read from two
  256-entry Q15 tables built once per layer (`softmax_lut_table`).
- Why: `expf` cost about 280 cycles per score and had become most of
  attention.
- Effect: attention −74% prefill, −62% decode. GPT-2 decode went from 3.0
  to 2.26 s/token.

**590ad5a: GPT-Neo 28M shapes.** Measurement only: d 32 attention streams
and d 512 LayerNorm for the planned model. No kernel change.

**4b30629+: PR #16 trim.** Files merged and superseded kernels removed;
no kernel change. Attention got 1–2% faster, likely because `matmul()`
could inline into attention once both were in one file.

**019671f+: one kernel file per engine operator.**
- What: new `linear` (zero-point fixup, bias, requantize), standalone
  `softmax_lut`, and streamed `mlp_block` / `mlp_finish`, so every kernel
  matches an `operator/*.hpp`.
- Effect: first measurements of these kernels. `linear` cost 6× its
  matmul because its requantization was scalar float. Attention gave back
  the 1–2% (separate files again).

**a560a2f+: embedding kernel.** New `embedding_lookup` (the table streams
through in slices) and `add_position_embeddings`. Other kernels unchanged.

**5fb006c+: vectorized integer loops.**
- What: every integer loop now runs on the RVV unit, 16 lanes at a time:
  `linear`'s fixup and requantization, attention's per-row max, exp,
  probabilities and q sums, softmax, GeLU, LayerNorm, residual add, MLP,
  the embedding row copy and the position add.
- How: table lookups became indexed loads (`vluxei`), and sums and maxes
  became vector reductions. Requantization moved from float to fixed point
  (a Q31 multiply, `vsmul`, then a rounding, saturating narrow, `vnclip`),
  as Coral's own int8 kernels do it. Sums of two scaled int8 terms
  (residual, position, LayerNorm's beta) use fixed-point factors.
  Requantized outputs are within 0.5 + 2⁻⁸ steps of exact (was 0.5).
- Why: scalar loops left 15 of 16 lanes idle, and requantization alone was
  5× the matmul inside `linear`. The vector unit is integer-only (Zve32x),
  so float loops could only vectorize as fixed point.
- Not vectorized: the float parts of attention (running rescale, finish
  divide), LayerNorm's per-row 1/sqrt, and the one-time `expf` / `tanhf`
  table builds. There is no vector float on Zve32x.
- Effect: residual 32×, position add 31×, LayerNorm 17×, embedding lookup
  9×, GeLU 7.6×, MLP prefill 6×, softmax 4×, `linear` 2.8×, attention
  1.3–1.9×. GPT-2 decode went from 2.26 to 1.93 s/token and prefill-1024
  from 5.8 to 3.9 min; GPT-Neo 28M from 0.95 to 0.76 s/token and 2.3 to
  1.3 min.

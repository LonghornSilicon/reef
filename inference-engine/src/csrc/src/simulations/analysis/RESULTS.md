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

## End-to-end estimate (roofline.py, at 590ad5a)

Measured tiles × tile counts, plus host↔DTCM transfer at the measured 15.7
B/cycle. There is no DMA, so transfers are not overlapped with compute. The
figures include the final LayerNorm and LM head.

| Model | Decode (context 1024) | Prefill 1024 |
|---|---:|---:|
| GPT-2 small, scalar attention (85d6089) | 8.2 s/token | 58 min |
| GPT-2 small, VME attention, `expf` (57df445) | 3.0 s/token | 12.4 min |
| GPT-2 small, VME attention, LUT exp (8f71744) | 2.26 s/token | 5.8 min |
| GPT-Neo 28M, VME attention, LUT exp (590ad5a) | 0.95 s/token | 2.3 min |

Not measured: the clock, contention between host writes and a running
kernel, DTCM↔core byte counts (derived from loop structure), host-side work
(K-split partial sums, orchestration), and the GPT-Neo head count (16 × 32
is assumed). Quantization scales are placeholders; they affect accuracy,
not cycles.

## Log

Append with `collect_results.py --log "note"` after a sim run. The note
should say what changed in the kernels, so each row explains its deltas.
A `+` after the commit means the run measured uncommitted changes on top
of it.

| date | commit | change | vme_matmul | rvv_matmul | linear | vme_step | attn_prefill | attn_decode | attn_d32_prefill | attn_d32_decode | attn_d32_local | lut_table | softmax_row | gelu | layernorm | residual | mlp_prefill | mlp_d512_decode | embed_prefill | position_prefill | vs previous |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-18 | 85d6089 | Week-3 baseline. Attention is scalar attention_flash_chunk (expf per score), same shapes | 11005 | 9317 |  | 84 | 8608212 | 554812 |  |  |  |  |  | 53287 | 243069 | 241850 |  |  |  |  | first record |
| 2026-10-18 | 57df445 | VME block-flash attention: q.kt and P.v through matmul() on the matrix engine, expf softmax | 11005 | 9317 |  | 84 | 1378744 | 105549 |  |  |  |  |  | 53287 | 243069 | 241850 |  |  |  |  | attn_prefill -84%, attn_decode -81% |
| 2026-10-18 | 8f71744 | LUT exp (softmax_lut_table) replaces expf in VME attention | 11005 | 9317 |  | 84 | 362264 | 40118 |  |  |  | 127576 | 52674 | 53296 | 243078 | 241857 |  |  |  |  | attn_prefill -74%, attn_decode -62% |
| 2026-10-18 | 590ad5a | GPT-Neo 28M d32 attention streams measured (no kernel change) | 11005 | 9317 |  | 84 | 362264 | 40118 | 263044 | 25930 | 290351 | 127576 | 52674 | 53296 | 243078 | 242082 |  |  |  |  | no change |
| 2026-10-20 | 4b30629+ | PR #16 trim: kernels merged into 3 files, 2 ELFs (no kernel changes) | 11002 | 9315 |  | 84 | 358439 | 39934 | 259219 | 25746 | 285563 | 127577 |  | 53286 | 243105 | 241941 |  |  |  |  | attn_prefill -1%, attn_d32_prefill -1%, attn_d32_local -2% |
| 2026-10-21 | 019671f+ | Kernels split one per engine operator; new linear, softmax and streamed mlp kernels | 11003 | 9316 | 67873 | 84 | 362371 | 40128 | 263193 | 25940 | 290533 | 128088 | 52673 | 53296 | 243078 | 241536 | 240474 | 200514 |  |  | attn_prefill +1%, attn_d32_prefill +2%, attn_d32_local +2% |
| 2026-10-21 | a560a2f+ | Embedding kernel (embedding_lookup, add_position_embeddings) for operator/embedding.hpp | 11003 | 9316 | 67873 | 84 | 362371 | 40128 | 263193 | 25940 | 290533 | 128088 | 52673 | 53296 | 243078 | 241536 | 240474 | 200514 | 59371 | 233339 | no change |

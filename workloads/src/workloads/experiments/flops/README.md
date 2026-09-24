# flops

How many floating-point operations (FLOPs) does TinyStories-Instruct take to read a prompt (prefill) and to generate each new token (decode), at every model size, sequence length and precision, and which operators do they go to? Reef's compute units have to deliver these FLOPs, and the operators that dominate them are the ones worth building hardware for.

## Method

- **Counting:** a forward hook on every operator module records its FLOPs, using a hand-derived formula per operator type (`FORMULAS` in `../tracer.py`). The comment above those formulas gives the convention:
  - a d-wide dot product costs 2d FLOPs, one multiply and one add per term;
  - every other scalar add, multiply, divide, compare or nonlinearity costs one;
  - lookups, reshapes and copies are free.
- **Meta device:** the model is built on PyTorch's meta device, which tracks shapes without storing weights or computing values. Nothing is downloaded. A full run takes about 50 s, most of it tracing every decode step from 1 to 256 tokens for the plot.
- **Phases,** at batch 1:
  - **prefill at length L:** one L-token prompt;
  - **decode at length L:** generating token L after L − 1 cached tokens. It starts at L = 2, since at L = 1 it is the same as prefill.
- **Lengths:** powers of two from 1 to 2,048, the model's position limit.
- **Precisions:** fp32, bf16, int8 and int4, rounding the same operators as story_quality does (`../quantization.py`): every Linear except `lm_head`. The core FLOPs are the same at every precision; quantization adds work, counted in its own rows:

  | precision | arithmetic | extra rows |
  |---|---|---|
  | `fp32`, `bf16` | fp32 or bf16 | none |
  | `int8` | the quantized Linears in int8 | `Quantize`: 4 FLOPs per input element (absolute max, divide, round, clamp); `Dequantize`: 2 per output element (input scale × weight scale) |
  | `int4` | fp32, since only the weights are int4 | `Dequantize`: 1 multiply per weight on every call, expanding it back to full precision |

- **Not counted:** the residual adds and the token-plus-position embedding add, which the hooks can't see. They are elementwise and small.

## Running

From `workloads/`:

```sh
uv run python src/workloads/experiments/flops/main.py                          # all five sizes
uv run python src/workloads/experiments/flops/main.py TinyStories-Instruct-8M  # a subset
```

## Output

The run writes three files to `results/flops/`:

- **`gpt_neo.csv`** (committed): one row per model, precision, phase, length, operator and arithmetic.

  | column | meaning |
  |---|---|
  | `model` | the size, e.g. `TinyStories-Instruct-8M` |
  | `precision` | `fp32`, `bf16`, `int8` or `int4` |
  | `phase` | `prefill` or `decode` |
  | `tokens` | sequence length L |
  | `operator` | the operator class, `Linear (lm_head)` for the output layer, or `Quantize` / `Dequantize` |
  | `arithmetic` | the number type the operator computes in |
  | `calls` | how many times it ran |
  | `flops` | FLOPs summed over those calls |

- **`prefill_decode.png`** (gitignored): per size, the int8 FLOPs to reach each length from 1 to 256 tokens. Dashed is prefill, one L-token prompt in a single pass. Solid is decode, generating tokens 1 to L one at a time and adding up every step. The CSV has every precision up to 2,048 tokens.
- **`operators.png`** (gitignored): per size, the int8 FLOPs of decoding token L, for L from 2 to 256, with one line per operator. The FLOPs axis is logarithmic, labeled at every power of ten, so the output layer and the small operators (GELU, LayerNorm, the int8 rounding), thousands of times apart, are all readable. A line that stays within 12% of a larger one at every length is drawn dotted on top, so it doesn't hide. That always happens to Quantize, whose FLOPs equal GELU's exactly (4 per element over 9 hidden widths of Linear input, against 9 per element over GELU's 4-hidden-wide input), and to Linear at 28M, 2% below `lm_head`.

## Prefill vs decode

FLOPs to decode one token, fp32:

| model | token 2 | token 256 | token 2,048 |
|---|---|---|---|
| TinyStories-Instruct-1M | 7.3 M | 8.0 M | 13.0 M |
| TinyStories-Instruct-3M | 16.1 M | 17.3 M | 26.0 M |
| TinyStories-Instruct-8M | 38.4 M | 40.7 M | 56.8 M |
| TinyStories-Instruct-28M | 102 M | 106 M | 137 M |
| TinyStories-Instruct-33M | 134 M | 137 M | 160 M |

- **A token costs about 2 × the model's matmul weights, plus attention.** Every Linear does one multiply and one add per weight per token, whatever the context. That part is flat across the table. The rest grows with the context.
- **An L-token prefill costs exactly L times decoding token L.** The count gives prefill the full L × L attention score matrix, including the masked half a causal model never uses. A kernel that skipped the masked half would do up to half as much attention work in prefill.
- **Reaching 256 tokens costs about the same either way.** In int8, one 256-token prefill takes 2.05 GFLOPs at 1M and 35.2 GFLOPs at 33M. Decoding the same 256 tokens one at a time takes 1.96 and 34.8 GFLOPs. Prefill does 1–5% more, all of it the masked half of the attention scores and their softmax. Decoding token l scores it against only the l tokens before it, so the steps together score L(L + 1)/2 pairs per head, where prefill scores all L². The difference, `layers × heads × L(L − 1)/2` masked scores at 4 × head_dim + 6 FLOPs each (two dot products, the scale and softmax's 5), matches the gap exactly at every size. The gap is widest on the smallest model, where attention is the largest share. With the FLOPs this close, what separates the two phases on hardware is that decode rereads every weight once per token.
- **Past a few hundred tokens, attention becomes significant.** Going from token 256 to 2,048 adds 17–63% to the cost of a token.

## Operators

- **The output layer (`lm_head`) is the largest single cost.** It multiplies by the 50,257-word vocabulary for every token: 81% of a token's FLOPs at 1M, 63% at 8M and 48% at 28M (token 256). Nothing else in the model scales with the vocabulary.
- **Attention grows from a few percent to a quarter or more.** Its share rises from 2–7% at token 256 to 16–34% at token 2,048. The smaller the model, the larger the share, because attention's cost depends on the number of heads and the context, not the width.
- **Everything else is small.** Softmax, GELU and LayerNorm together stay under 3% at token 256. At token 2,048 Softmax grows with the context, to 10% at 1M and 5% at 3M.

## Quantization overhead

- **int8 adds 0.1–0.3%.** Rounding each input and rescaling each output is cheap next to the matmul it wraps, and it's the same share in prefill and decode.
- **int4 is expensive in decode:** it adds 5% at 1M, 15% at 8M and 24% at 28M (token 256). Expanding a weight costs one multiply, while using it once costs two FLOPs, so for the quantized Linears the expansion adds 50% on top of their matmul work in every decoded token. Small models show less of it only because their unquantized output layer dominates.
- **int4 is nearly free in prefill,** at 0.1% or less at 256 tokens, because one expansion per weight is shared by the whole prompt. On hardware, int4 pays off only if the chip computes on int4 weights directly or expands them for free.

## Tests

`tests/experiments/test_flops.py`:

- **`test_runs_standalone_with_help`:** runs `main.py --help` as a script.
- **`test_committed_csv_matches_a_fresh_trace`:** re-traces 8M at 128 and 2,048 tokens, both phases, every precision, and compares against the CSV.
- **`test_decode_linear_is_constant_and_attention_grows_linearly`:** a decoded token's Linear FLOPs are 2 × weights + biases at every length, and its attention FLOPs are proportional to the length.
- **`test_precision_changes_only_the_overhead`:** all four precisions have the same core FLOPs, and fp32 and bf16 have no overhead.
- **`test_quantization_overhead_matches_the_quantized_linears`:** the int8 and int4 overhead equals the hand count over the quantized Linears, leaving out `lm_head`.

```sh
uv run pytest tests/experiments/test_flops.py tests/experiments/test_tracer.py
```

## Limitations

- **FLOPs are not time.** The counts ignore memory traffic, and count an `exp` the same as an add. Decode at batch 1 is usually limited by reading the weights and the KV cache (see kv_cache_size), not by FLOPs.
- **The overheads are for the simulated schemes** in `../quantization.py`. Real kernels may fuse the rounding into the matmul or expand int4 weights in hardware, which would change those rows.
- **Batch 1 only.** Larger batches multiply every row except int4's weight expansion, which is shared by the batch.
- **Masked attention is counted.** Prefill includes the masked half of the score matrix, as `operators/attention.py` computes it.

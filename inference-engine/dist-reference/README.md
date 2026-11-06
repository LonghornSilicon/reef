# dist-reference: GPT-Neo across chips

The reference for distributed inference: GPT-Neo split across N chips with
tensor parallelism (TP), computed on the host rank by rank. Multi-chip runs
on Spike or `reef_perf` are checked against it, the way `coral/` is checked
against `reference/`.

It changes nothing in `reference/`. It calls the engine's operators
(`linear`, `layernorm`, `gelu`, `attend_head`, the embeddings) and only adds
the splitting and the combining. A change to the engine's arithmetic reaches
this model too, and the tests below fail if the two drift apart.

```
include/dist_inference/
  shard.hpp      ranges, slicing, concatenation, rank-order sums,
                 column_parallel_linear and row_parallel_linear
  tp_model.hpp   TpGptNeo: the model, rank by rank, with every sync recorded
tests/           Google Test: test_shard.cpp, test_tp_model.cpp
tp_cli.cpp       gpt_neo_cli with --tp RANKS and --scheme megatron|column
tp_compare.cpp   Test B: teacher-forced logits, TpGptNeo against GPTNeo
test_b.py        runs tp_compare on real TinyStories text, prints the table
```

## What each rank does

Rank r holds a contiguous share of the heads and of the MLP width.
Per layer it:

1. runs LayerNorm on the whole residual stream, which every rank holds;
2. projects query, key and value for its heads, keeps those heads' key and
   value cache, and attends them;
3. after the attention output projection and the residual add, computes its
   columns of `c_fc` and applies GELU to them.

The host embeds the tokens: `wte` is also the lm_head, which is split by
vocabulary rows, so no rank holds the whole table. The host gathers the
logits at the end.

The two projections that read every head or the whole MLP width, the
attention output and `c_proj`, are split by one of two schemes:

| Scheme | Those two projections | Syncs per layer | Against one chip |
|---|---|---|---|
| `kColumnOnly` | by output columns; the host concatenates | 4 gathers | same bits, any N |
| `kMegatron` | by input rows; the host sums partials in rank order, then adds the bias once | 2 sums | same bits at N = 1; rounding only above |

Concatenating moves values without computing anything, so column-only keeps
every bit. Megatron changes the order of each sum, and in W8A8 each rank
quantizes its own input slice with its own scale, so its logits differ by
rounding. The host driver must sum exactly as `sum_in_rank_order()` does:
rank 0 first, bias last.

`TpGptNeo::syncs()` lists the last forward call's syncs (gather or sum, rows,
width), for checking the schedule and costing the links.

## Tests

`dist_tests` runs under CTest with the engine's tests:

- column-only gives `GPTNeo`'s logits bit for bit for N = 1, 2, 4, in fp32
  and W8A8, for a prefill and every decode step, and the same generated
  tokens;
- Megatron at N = 1 gives `GPTNeo`'s logits bit for bit; at N = 2 and 4
  they are within 1e-5 (fp32) or 5% (W8A8) of the largest logit, and not
  identical;
- each scheme syncs where it should: 4L + 1 or 2L + 1 times per forward;
- the parallel linears add the bias once and sum in rank order.

## Test B: Megatron on TinyStories-28M

`test_b.py` packs TinyStories validation stories into token sequences and
runs `tp_compare`, which feeds every model the same tokens (teacher forcing)
and compares the logits at every position with the single chip's. Rows are
relative logit error (largest difference over the largest single-chip
logit), how often the top token agrees, and perplexity on the real next
token. The first table compares each split model with one chip at the same
precision; the second compares W8A8 with fp32 (`--reference fp32`), so it
shows which W8A8 model is closer to full precision.

1,024 positions (2 sequences of 512 tokens, past the 256-token local
window), measured on 5 Nov 2026:

| Precision | Scheme | Ranks | Max rel | Mean rel | Top token agrees | PPL one chip | PPL split |
|---|---|---|---|---|---|---|---|
| fp32 | Megatron | 2 | 3.1e-6 | 1.4e-6 | 100% | 3.34951 | 3.34951 |
| fp32 | Megatron | 4 | 3.2e-6 | 1.4e-6 | 100% | 3.34951 | 3.34951 |
| fp32 | Megatron | 8 | 3.9e-6 | 1.4e-6 | 100% | 3.34951 | 3.34951 |
| W8A8 | Megatron | 2 | 9.0% | 4.3% | 97.1% | 3.35445 | 3.36482 |
| W8A8 | Megatron | 4 | 11.0% | 4.3% | 96.8% | 3.35445 | 3.34662 |
| W8A8 | Megatron | 8 | 10.4% | 4.3% | 96.6% | 3.35445 | 3.38239 |
| fp32 | column-only | 4 | 0 | 0 | 100% | 3.34951 | 3.34951 |
| W8A8 | column-only | 4 | 0 | 0 | 100% | 3.35445 | 3.35445 |

W8A8 Megatron against one chip in fp32 (ranks 1 is one chip in W8A8):

| Ranks | Max rel | Mean rel | Top token agrees | PPL fp32 | PPL W8A8 |
|---|---|---|---|---|---|
| 1 | 8.3% | 3.70% | 97.6% | 3.34951 | 3.35445 |
| 2 | 8.3% | 3.51% | 98.3% | 3.34951 | 3.36482 |
| 4 | 9.0% | 3.43% | 98.4% | 3.34951 | 3.34662 |
| 8 | 6.8% | 3.34% | 97.2% | 3.34951 | 3.38239 |

- **fp32 Megatron passes.** Its logits move by a few parts per million,
  never change the top token, and leave perplexity unchanged to six digits.
- **W8A8 Megatron moves the logits about as much as W8A8 itself does.**
  Each rank quantizes its own slice with its own scale, so the noise is new
  quantization noise, not rounding. About 3% of positions change their top
  token, so free-running W8A8 generation depends on the rank count. If
  results must not depend on N, all ranks need the whole row's activation
  scale, which costs a small sync per row-parallel projection.
- **Against fp32, split W8A8 is slightly closer than one chip.** The mean
  logit error falls with every doubling of ranks, from 3.70% to 3.34%: a
  slice's maximum is at most the row's, so its scale is finer. Perplexity
  does not show it: it moves within 1% in no consistent direction (+0.5%,
  -0.1%, +1.0% against fp32), the noise of 1,024 positions, larger than the
  0.15% W8A8 costs on one chip.
- **Column-only is bit-exact on the real model too**, in fp32 and W8A8.

Reproduce from `inference-engine/` after downloading the model (see the
inference-engine README) and the validation text:

```sh
curl -fL -o models/data/TinyStories-valid.txt \
    https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt
cmake --build build --target tp_compare
uv run python dist-reference/test_b.py --model models/tinystories-28m \
    --stories models/data/TinyStories-valid.txt
```

It takes about an hour and a half; each `tp_compare` run needs about 650 MB, and
`--jobs` (default 2) sets how many run at once.

## Running it

From `inference-engine/`:

```sh
cmake -S . -B build && cmake --build build
ctest --test-dir build -R 'Shard|TpGptNeo' --output-on-failure
build/bin/tp_cli --model DIR --tp 4 --scheme megatron 5 17 42
```

## Next

- Goldens for multi-chip firmware runs: each rank's inputs and outputs at
  every sync, written by a host tool like `coral/host/gen_golden.cpp`.
- A host driver that runs N chips on Spike, compared with these goldens bit
  for bit.

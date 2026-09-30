# kv_cache_size

How much memory does TinyStories-Instruct's KV cache take, for each model size, at every sequence length up to the 2,048-token limit, in fp32, bf16, int8 and int4? Reef has to keep that cache on or next to the chip, so its size bounds how long a story the chip can write without going to off-chip memory.

## Method

The sizes are computed from each config (`configs/gpt_neo.py`), at batch 1. Every layer caches a key and a value vector per token, each `hidden size` wide, since GPT-Neo gives every head its own keys and values:

```
bytes = sum over layers of  2 (K and V) × cached tokens × hidden size × bytes per element
```

- **Precisions:** fp32 is 4 bytes per element, bf16 is 2, int8 is 1 and int4 is half a byte. int8 and int4 also need one fp16 scale (2 bytes) per token, per head, for keys and for values, so the stored numbers can be turned back into real values.
- **Two cache sizes.** GPT-Neo alternates global layers, which attend to every earlier token, with local layers, which attend only to the last 256.
  - **Stored:** what `models/gpt_neo.py` keeps today, which is every token in every layer.
  - **Needed:** local layers keep only their last 256 tokens. This is the least a Reef implementation has to hold.

## Running

From `workloads/`:

```sh
uv run python src/workloads/experiments/kv_cache_size/main.py                          # all five sizes
uv run python src/workloads/experiments/kv_cache_size/main.py TinyStories-Instruct-8M  # a subset
```

It computes rather than runs anything, so it finishes in seconds and downloads nothing.

## Output

The run writes two files to `results/kv_cache_size/`:

- **`gpt_neo.csv`** (committed): one row per model, precision, cache and length.

  | column | meaning |
  |---|---|
  | `model` | the size, e.g. `TinyStories-Instruct-8M` |
  | `precision` | `fp32`, `bf16`, `int8` or `int4` |
  | `cache` | `stored` or `needed` |
  | `tokens` | sequence length: powers of two from 1 to 2,048 |
  | `bytes` | KV cache size |

  The size grows in straight lines between these lengths. Its only bend is at 256, which is one of them, so the CSV captures the whole curve.
- **`gpt_neo.png`** (gitignored): one panel per size, from 1 to 256 tokens of generation, with a line for each precision. Stored and needed are the same size up to 256 tokens, the local window, so the plot shows one line per precision; the CSV has the full range to 2,048.

## Observations

KV cache at the 2,048-token limit, stored / needed:

| model | fp32 | bf16 | int8 | int4 |
|---|---|---|---|---|
| TinyStories-Instruct-1M | 8.4 / 4.7 MB | 4.2 / 2.4 MB | 3.1 / 1.8 MB | 2.1 / 1.2 MB |
| TinyStories-Instruct-3M | 16.8 / 9.4 MB | 8.4 / 4.7 MB | 5.2 / 2.9 MB | 3.1 / 1.8 MB |
| TinyStories-Instruct-8M | 33.6 / 18.9 MB | 16.8 / 9.4 MB | 9.4 / 5.3 MB | 5.2 / 2.9 MB |
| TinyStories-Instruct-28M | 67.1 / 37.7 MB | 33.6 / 18.9 MB | 17.8 / 10.0 MB | 9.4 / 5.3 MB |
| TinyStories-Instruct-33M | 50.3 / 28.3 MB | 25.2 / 14.2 MB | 13.1 / 7.4 MB | 6.8 / 3.8 MB |

- **Capping the local layers saves 44% at 2,048 tokens,** at every size and precision. Half the layers stop growing at 256 tokens, so the saving is `(2048 − 256) / 2048 / 2`. Below 256 tokens the two caches are the same size.
- **The scales cost the most on the smallest models.** Every size has 16 heads, so heads get narrower as the model shrinks: 4 numbers wide at 1M and 48 at 33M. The 2-byte scale per head adds 50% to 1M's int8 elements and 100% to its int4 elements, but only 4% and 8% at 33M. So at 1M, int4 is only half of bf16's size and two thirds of int8's; at 33M it is 27% of bf16 and 52% of int8.
- **The window and int8 are worth about the same.** At 8M, bf16 with the window (needed) and int8 without it (stored) both come to exactly 9.4 MB. Doing both brings 8M from 33.6 MB to 5.3 MB in int8, or 2.9 MB in int4.
- **28M has the largest cache, not 33M.** The cache grows with layers × hidden size. 28M is 8 × 512 and 33M is 4 × 768, so 33M's cache is 25% smaller even though it has more parameters.

## Tests

`tests/experiments/test_kv_cache_size.py`:

- **`test_runs_standalone_with_help`:** runs `main.py --help` as a script.
- **`test_stored_fp32_size_matches_the_model_cache`:** builds every size on PyTorch's meta device, runs a 1-, 300- and 2,048-token prompt, and checks that the bytes in the cache the model returns equal the formula's fp32 stored size.
- **`test_needed_matches_stored_up_to_the_window_then_falls_behind`:** the two caches agree up to 256 tokens and differ after.
- **`test_committed_csv_matches_the_formula`:** every committed row equals a fresh computation.

```sh
uv run pytest tests/experiments/test_kv_cache_size.py
```

## Limitations

- **Batch 1 only.** The cache grows linearly with the batch size.
- **Size, not bandwidth.** Each generated token reads the whole cache once, so bandwidth tracks these numbers, but this experiment doesn't model it.
- **The int8 and int4 caches aren't implemented.** Their sizes assume a per-token, per-head fp16 scale; nothing here runs them or measures how they affect the output.
- **No allocation overhead.** It assumes one exact-size buffer per layer, with no paging, padding or fragmentation.

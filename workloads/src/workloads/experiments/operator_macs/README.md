# operator_macs

Which operators do TinyStories-Instruct's multiply-adds (MACs) go to? The answer tells us which operators Reef has to make fast. An operator with 1% of the MACs can run on a slow path; one with 95% sets the chip's throughput.

## Method

- **Counting:** a forward hook on every operator module records its MACs, using a hand-derived formula per operator type (`FORMULAS` in `../tracer.py`). The comment above those formulas gives the counting convention:
  - a d-wide dot product costs d MACs;
  - every other scalar arithmetic op or nonlinearity costs one;
  - lookups, reshapes and copies are free.
- **Meta device:** the model is built on PyTorch's meta device, which tracks shapes without storing weights or computing values. Nothing is downloaded, and a run takes a few seconds.
- **Input:** one 128-token prompt at batch 1, as a single prefill with no decode steps.
- **Not counted:** the residual adds and the token-plus-position embedding add, which the hooks can't see. They are elementwise and small.

## Running

From `workloads/`:

```sh
uv run python src/workloads/experiments/operator_macs/main.py
```

This covers all five TinyStories-Instruct sizes.

## Output

The run writes two files to `results/operator_macs/`:

- **`gpt_neo.csv`** (committed): one row per model and operator, sorted by MACs, then a `Total` row for each model.

  | column | meaning |
  |---|---|
  | `model` | the size, e.g. `TinyStories-Instruct-8M` |
  | `operator` | the operator class, or `Total` |
  | `calls` | how many times the operator ran in one forward pass |
  | `macs` | MACs summed over those calls |
  | `share` | fraction of the model's total MACs |

- **`gpt_neo.png`** (gitignored): a ranked bar chart for each size.

## Observations

| model | total MACs | Linear | attention | Softmax | GELU + LayerNorm |
|---|---|---|---|---|---|
| TinyStories-Instruct-1M | 0.49 G | 93.4% | 3.8% | 2.1% | 0.6% |
| TinyStories-Instruct-3M | 1.08 G | 95.1% | 3.3% | 1.0% | 0.6% |
| TinyStories-Instruct-8M | 2.55 G | 96.4% | 2.7% | 0.4% | 0.5% |
| TinyStories-Instruct-28M | 6.69 G | 97.4% | 2.0% | 0.2% | 0.4% |
| TinyStories-Instruct-33M | 8.69 G | 98.5% | 1.2% | 0.1% | 0.2% |

- **Linear layers do almost all the work:** 93–99% of the MACs at every size. Attention's matrix multiplies take most of the rest.
- **The output layer (`lm_head`) is the largest single cost.** It multiplies by the 50,257-word vocabulary, which is 83% of all MACs at 1M, 65% at 8M and 49% at 28M. This run computes it at all 128 positions, but generating the next token only needs the last one. A generation-only design would skip most of that work.
- **Softmax costs the same at every size except 33M:** 10.5 M MACs. Its cost depends on heads, layers and prompt length, not on hidden size, so its share falls from 2.1% at 1M to 0.2% at 28M. 33M has half as many layers, so half the Softmax.
- **Attention's share is small at this prompt length.** The attention score matrix grows with the square of the prompt length, so 128 tokens understates attention for longer contexts.

The `lm_head` shares come from tracing the model, not from the CSV, which sums every Linear under one row.

## Tests

`tests/experiments/test_operator_macs.py` has two tests:

- It checks that `main.py --help` runs as a standalone script.
- It re-traces TinyStories-Instruct-8M and checks that the total MACs and call counts match the committed CSV. If a change to the model or to the counting rules alters any count, rerun the experiment, check the CSV diff, and commit the new CSV with the change.

```sh
uv run pytest tests/experiments/test_operator_macs.py
```

## Limitations

- **MACs are not time.** The counts ignore memory traffic, and count an `exp` the same as a multiply-add. They show where the arithmetic is, not what limits speed on Reef.
- **Only one input shape.** Every count is for one 128-token prefill. Token-by-token decoding and longer contexts would shift the shares toward attention and away from `lm_head`'s all-positions cost.
- **No precision.** A MAC is counted the same whether it runs in fp32, bf16 or int8.

# story_quality

How well does each size of TinyStories-Instruct write a story it was asked for, and how much of that survives at the lower precisions Reef might run it at? If the club picks a model of this kind as the workload for the final Reef chip, this experiment is the evidence for which size is good enough, and its regression test is what tells us that the operator library still produces that quality.

The experiment runs our from-scratch GPT-Neo (`models/gpt_neo.py`, built only from `operators/`) with the published weights. Every size runs at fp32, bf16, int8 and int4, gets the same five prompts at each, and the experiment records what it writes.

## Models

| model | parameters | hidden size × layers |
|---|---|---|
| TinyStories-Instruct-1M | 3.7M | 64 × 8 |
| TinyStories-Instruct-3M | 8.3M | 128 × 8 |
| TinyStories-Instruct-8M | 19.7M | 256 × 8 |
| TinyStories-Instruct-28M | 52.0M | 512 × 8 |
| TinyStories-Instruct-33M | 68.5M | 768 × 4 |

The names are the published ones and roughly count only the parameters outside the embeddings. The tied 50,257-token embedding by itself is 86% of the 1M model.

## Method

- **Weights:** loaded from `roneneldan/TinyStories-Instruct-*` on Hugging Face into our model. `tests/models/test_checkpoints.py` checks that the 8M model's logits match `transformers`.
- **Decoding:** greedy (argmax, which is temperature 0), on the CPU. Each run generates 320 new tokens and cuts the text at the first `<|endoftext|>`.
- **Prompts:** these use the TinyStories-Instruct training format. Each prompt adds constraints to the one before:

  | # | name | fields |
  |---|---|---|
  | 1 | `summary_only` | Summary |
  | 2 | `summary_words` | Summary, Words |
  | 3 | `summary_dialogue` | Features: Dialogue; Summary |
  | 4 | `summary_twist_badending` | Features: Twist, BadEnding; Summary |
  | 5 | `everything_at_once` | Features: Dialogue, MoralValue; Words; Summary; Random sentence |

  The exact text is `PROMPTS` in `main.py`. A comment there explains why each prompt ends in a blank line after `Story:`.
- **Precisions:** each starts from the same fp32 weights.

  | precision | what changes | stays fp32 |
  |---|---|---|
  | `fp32` | nothing | everything |
  | `bf16` | the whole model, weights and every activation, is cast to bf16 | nothing |
  | `int8` | every Linear weight except `lm_head` is rounded to int8, one scale per output channel, and every input to those Linears is rounded to int8, one scale per token | embeddings and `lm_head`, LayerNorm, attention scores and softmax, KV cache |
  | `int4` | every Linear weight except `lm_head` is rounded to int4, one scale per group of 32 inputs; the inputs stay fp32 | the same, plus the Linear inputs |

  int8 and int4 are simulated. Values are rounded onto a symmetric integer grid (±127 or ±7 times the scale), then the arithmetic runs in fp32. That measures what the rounding does to the stories and runs the same on every machine, but says nothing about integer speed. `lm_head` stays fp32 because it shares its weight with the token embedding.

## Running

From `workloads/`:

```sh
uv run python src/workloads/experiments/story_quality/main.py        # all sizes and precisions
uv run python src/workloads/experiments/story_quality/main.py TinyStories-Instruct-8M --precision int8 --precision int4
```

The first run downloads the checkpoints, which take about 1.5 GB in the Hugging Face cache for all five sizes. After that, a full run takes about 100 s on an Apple-silicon laptop CPU.

A run rewrites every CSV with only the models and precisions it was given, so commit results from a full run.

## Output

`results/story_quality/<#>_<name>.csv` holds one file per prompt, with one row per model and precision, so each model's four versions of a story sit next to each other:

| column | meaning |
|---|---|
| `model` | the size, e.g. `TinyStories-Instruct-8M` |
| `precision` | `fp32`, `bf16`, `int8` or `int4` |
| `words` | the story's word count, split on whitespace |
| `story` | the generated text |

These files are committed, and the regression test uses them as its expected output.

## Observations

### Across sizes (fp32)

- **Repetition** drops with size but never goes away.
  - 1M loops on a single sentence in prompts 2 and 3.
  - 3M repeats one sentence until it hits the 320-token cap in prompts 3 and 4.
  - 33M still ends prompt 3 with "They loved their mom. They loved their mom."
- **Plot:** 8M is the smallest size that follows prompt 2's plot, where a wave washes the castle away and Tom rebuilds further up the beach. 1M never gets to the wave.
- **Dialogue** (prompt 3) is the hardest format. Four of the five sizes start with a `Possible story:` header, and only 28M finishes before the token cap.
- **The twist** in prompt 4 (the rock turns out to be painted mud) defeats every size. 28M mentions mud but then contradicts itself.
- **Prompt 5:** only 28M uses the random sentence word for word and covers all three plot points (the fear of the dark, the brother's lantern, sharing it with a younger child). The wider but shallower 33M is not better than 28M.

### Across precisions

- **bf16 and int8 don't visibly change the quality.** Every size keeps the prompt's main character at both, and from 8M up the stories stay coherent and follow the summary. How much a story loops depends on the size, not on these two precisions.
- **int4 breaks the 1M model.** Its int4 stories are ungrammatical ("The birds were singing and birds and birds") and drop the prompt's main character in 4 of 5 prompts, which no other size or precision does. From 3M up, int4 keeps the characters. 8M and 28M at int4 still follow prompt 2's plot, and 28M at int4 still covers all of prompt 5's.
- **The exact text changes almost immediately, and fastest at int4.** Only 1 of the 50 bf16 and int8 stories matches fp32 word for word; the rest depart within their first 168 words. Every int4 story departs within its first 22, and at 1M from the first word. Under greedy decoding, one near-tied token that rounds the other way rewrites the rest of the story.
- **One story per prompt is noisy.** 33M at int4 loops through the second half of prompt 4, and 33M at bf16 loops in prompt 3, but neither size loops elsewhere. These single samples can rule out a collapse like 1M's at int4, not a small drop in quality.

## Regression test

`tests/experiments/test_story_quality.py` has four tests:

- **`test_runs_standalone_with_help`:** runs `main.py --help` as a script.
- **`test_int8_rounds_each_linear_row_and_leaves_the_tied_head`** and **`test_int4_rounds_each_group_of_32_and_leaves_the_tied_head`:** on a tiny random model, check that every Linear weight lands on its integer grid, that `lm_head` and the embedding stay untouched and tied, and that the fp32 model being copied isn't modified.
- **`test_greedy_stories_match_the_committed_results`** (marked `slow`): regenerates every model × precision × prompt and requires the text to match the committed CSV exactly. Greedy decoding is deterministic, so a change to the operators, models, weight loading, tokenizer, or PyTorch/`transformers` versions fails this test if it changes even one token. pytest then prints both texts so you can see where they diverge.

```sh
uv run pytest tests/experiments/test_story_quality.py
```

When a change is meant to alter the output (a new prompt, a different `MAX_NEW_TOKENS`), rerun the experiment, read the diff of the CSVs, and commit them along with the change.

## Limitations and future work

- **Greedy decoding only.** `generate()` has no temperature or sampling, so sampled decoding is neither implemented nor tested.
- **Exact match depends on the platform.** The committed stories came from an Apple-silicon CPU with torch 2.14.0 and transformers 5.17.0. A different BLAS or torch version can round differently, flip a near-tied argmax, and fail the test with no real regression. If the test fails, look at where the stories first diverge before assuming a bug.
- **Simulated int8 and int4 cover only the Linears.** Attention, softmax, LayerNorm, the KV cache, the embeddings and `lm_head` stay fp32. A chip that also runs those in integers would need them added here.
- **Checking quality instead of exact tokens.** A larger judge model (around 30B parameters) could decide whether new stories look like they come from the same distribution as the committed ones. That would still pass after harmless numerical changes and would also cover sampled decoding. The judge would need tests of its own: for example, it must tell 1M/3M output or hard-coded sentences apart from 8M+ output. It would also need an opt-in pytest marker, since not every tester can download and run a 30B model.

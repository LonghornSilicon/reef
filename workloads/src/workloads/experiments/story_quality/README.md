# story_quality

How well does each size of TinyStories-Instruct write a story it was asked for? If the club picks a model of this kind as the workload for the final Reef chip, this experiment is the evidence for which size is good enough, and its regression test is what tells us that the operator library still produces that quality.

The experiment runs our from-scratch GPT-Neo (`models/gpt_neo.py`, built only from `operators/`) with the published weights. Every size gets the same five prompts, and the experiment records what each one writes.

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
- **Decoding:** greedy (argmax, which is temperature 0), in fp32 on the CPU. Each run generates 320 new tokens and cuts the text at the first `<|endoftext|>`.
- **Prompts:** these use the TinyStories-Instruct training format. Each prompt adds constraints to the one before:

  | # | name | fields |
  |---|---|---|
  | 1 | `summary_only` | Summary |
  | 2 | `summary_words` | Summary, Words |
  | 3 | `summary_dialogue` | Features: Dialogue; Summary |
  | 4 | `summary_twist_badending` | Features: Twist, BadEnding; Summary |
  | 5 | `everything_at_once` | Features: Dialogue, MoralValue; Words; Summary; Random sentence |

  The exact text is `PROMPTS` in `main.py`. A comment there explains why each prompt ends in a blank line after `Story:`.

## Running

From `workloads/`:

```sh
uv run python src/workloads/experiments/story_quality/main.py                          # all five sizes
uv run python src/workloads/experiments/story_quality/main.py TinyStories-Instruct-8M  # a subset
```

The first run downloads the checkpoints, which take about 1.5 GB in the Hugging Face cache for all five sizes. After that, a full run takes about 30 s on an Apple-silicon laptop CPU.

## Output

`results/story_quality/<#>_<name>.csv` holds one file per prompt, with one row per model:

| column | meaning |
|---|---|
| `model` | the size, e.g. `TinyStories-Instruct-8M` |
| `words` | the story's word count, split on whitespace |
| `story` | the generated text |

These files are committed, and the regression test uses them as its expected output.

## Observations

- **Repetition** drops with size but never goes away.
  - 1M loops on a single sentence in prompts 2 and 3.
  - 3M repeats one sentence until it hits the 320-token cap in prompts 3 and 4.
  - 33M still ends prompt 3 with "They loved their mom. They loved their mom."
- **Plot:** 8M is the smallest size that follows prompt 2's plot, where a wave washes the castle away and Tom rebuilds further up the beach. 1M never gets to the wave.
- **Dialogue** (prompt 3) is the hardest format. Four of the five sizes start with a `Possible story:` header, and only 28M finishes before the token cap.
- **The twist** in prompt 4 (the rock turns out to be painted mud) defeats every size. 28M mentions mud but then contradicts itself.
- **Prompt 5:** only 28M uses the random sentence word for word and covers all three plot points (the fear of the dark, the brother's lantern, sharing it with a younger child). The wider but shallower 33M is not better than 28M.

## Regression test

`tests/experiments/test_story_quality.py` has two tests:

- **`test_runs_standalone_with_help`:** runs `main.py --help` as a script.
- **`test_greedy_stories_match_the_committed_results`** (marked `slow`): regenerates every model × prompt and requires the text to match the committed CSV exactly. Greedy decoding is deterministic, so a change to the operators, models, weight loading, tokenizer, or PyTorch/`transformers` versions fails this test if it changes even one token. pytest then prints both texts so you can see where they diverge.

```sh
uv run pytest tests/experiments/test_story_quality.py
```

When a change is meant to alter the output (a new prompt, a different `MAX_NEW_TOKENS`), rerun the experiment, read the diff of the CSVs, and commit them along with the change.

## Limitations and future work

- **Greedy decoding only.** `generate()` has no temperature or sampling, so sampled decoding is neither implemented nor tested.
- **Exact match depends on the platform.** The committed stories came from an Apple-silicon CPU with torch 2.14.0 and transformers 5.17.0. A different BLAS or torch version can round differently, flip a near-tied argmax, and fail the test with no real regression. If the test fails, look at where the stories first diverge before assuming a bug.
- **Checking quality instead of exact tokens.** A larger judge model (around 30B parameters) could decide whether new stories look like they come from the same distribution as the committed ones. That would still pass after harmless numerical changes and would also cover sampled decoding. The judge would need tests of its own: for example, it must tell 1M/3M output or hard-coded sentences apart from 8M+ output. It would also need an opt-in pytest marker, since not every tester can download and run a 30B model.

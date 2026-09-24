# workloads

Workload characterization for Reef. This directory rebuilds TinyStories-Instruct (GPT-Neo), the language model Reef might run, using only a small library of operators written from scratch in PyTorch. Tests check every operator and the model against PyTorch and `transformers`. Experiments then measure the model, for example where its multiply-adds go and how good its output is, to inform the chip's design.

```
src/workloads/
  operators/    the kernels, from scratch in PyTorch (the README gives the math)
  models/       TinyStories-Instruct (GPT-Neo), built only from operators/
  configs/      hyperparameters for each published model size
  experiments/  one folder per experiment, each with a runnable main.py,
                plus helpers the experiments share (tracer.py, plots.py)
tests/          mirrors src/workloads/
results/        experiment outputs, one folder per experiment
```

## Install

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). uv installs Python 3.12 itself.

```sh
git clone https://github.com/LonghornSilicon/reef.git
cd reef/workloads
uv sync
```

On Linux, torch comes from the CUDA 12.6 wheel index; everywhere else it comes from PyPI. Nothing here needs a GPU.

To update:

```sh
git pull
uv sync
```

`uv sync` installs dependency changes and rebuilds the project whenever `pyproject.toml` has changed. `uv run` does the same sync before every command, so running everything through `uv run` also keeps you current. Editing files under `src/` needs no reinstall, because the project is installed in editable mode.

## Test

```sh
uv run pytest -m "not slow"   # ~10 s, CPU only, no network
uv run pytest                 # adds the slow tests below
```

The slow tests compare against published checkpoints:

- `tests/models/test_checkpoints.py` checks TinyStories-Instruct-8M's config and logits against `transformers`.
- `tests/experiments/test_story_quality.py` regenerates every story in `results/story_quality/` and requires an exact match.

The first slow run needs network access to Hugging Face and about 1.5 GB of disk for its cache. Once the downloads are cached, the slow tests take about 2 min on an Apple-silicon laptop CPU.

## Experiments

Run experiments from `workloads/`. Each one takes `--help`.

| experiment | question | command | output |
|---|---|---|---|
| `operator_macs` | Which operators do each model's multiply-adds (MACs) go to? | `uv run python src/workloads/experiments/operator_macs/main.py [family ...]` | `results/operator_macs/gpt_neo.csv` and `.png` |
| `kv_cache_size` | How big is the KV cache for each size, up to 2,048 tokens, in fp32, bf16, int8 and int4? | `uv run python src/workloads/experiments/kv_cache_size/main.py [model ...]` | `results/kv_cache_size/gpt_neo.csv` and `.png` |
| `story_quality` | How well does each TinyStories size write the story it is asked for, at fp32, bf16, int8 and int4? | `uv run python src/workloads/experiments/story_quality/main.py [model ...] [--precision p]` | `results/story_quality/<#>_<prompt>.csv` |

`operator_macs` and `kv_cache_size` download nothing and finish in seconds. Each experiment has its own README covering its method, findings and tests: [operator_macs](src/workloads/experiments/operator_macs/README.md), [kv_cache_size](src/workloads/experiments/kv_cache_size/README.md), [story_quality](src/workloads/experiments/story_quality/README.md).

## Contributing

- **Core code** (operators, models, configs) lives in `src/workloads/` and never imports from `experiments/`.
- **An experiment** is a folder `src/workloads/experiments/<name>/` with a `main.py` that:
  - runs on its own as `uv run python src/workloads/experiments/<name>/main.py`, and parses its arguments with `argparse` so that `--help` works;
  - writes its outputs to `results/<name>/`;
  - has a `README.md` giving the question it answers, its method, how to run it, its outputs and what they show;
  - uses absolute imports (`from workloads.models.gpt_neo import gpt_neo`).

  Code that several experiments share goes directly in `src/workloads/experiments/`.
- **Tests** mirror `src/workloads/` under `tests/`.
  - Each experiment has a `tests/experiments/test_<name>.py` that imports its `main` module and, at minimum, checks that `--help` runs.
  - Mark tests that download files or take more than a few seconds with `@pytest.mark.slow`.
- **Results:** commit small text outputs (CSV), since tests may compare against them. Plots (PNG) are gitignored; regenerate them locally.
- **Lint** before pushing. CI runs the same checks with ruff 0.16.5, which is also the version `uv sync` installs. ruff requires a docstring on every module, package and class.

  ```sh
  uv run ruff format
  uv run ruff check --fix
  ```
- **Pull requests** use the template in `.github/pull_request_template.md`. Fill in its Test section with the commands a reviewer should run.

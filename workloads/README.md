# workloads

Workload characterization for Reef. This directory rebuilds the neural networks Reef might run, using only a small library of operators written from scratch in PyTorch. Tests check every operator and model against PyTorch, `transformers` and `torchvision`. Experiments then measure the models, for example where the multiply-adds go and how good a small language model's output is, to inform the chip's design.

```
src/workloads/
  operators/    the kernels, from scratch in PyTorch (the README gives the math)
  models/       GoogLeNet, ResNet, EfficientNet, GPT-2, SmolLM2 (Llama), TinyStories (GPT-Neo),
                built only from operators/
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

On Linux, torch and torchvision come from the CUDA 12.6 wheel index; everywhere else they come from PyPI. Nothing here needs a GPU.

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

- `tests/models/test_checkpoints.py` covers SmolLM2-135M and TinyStories-Instruct-8M against `transformers`, and ResNet-18/50 against `torchvision`.
- `tests/experiments/test_story_quality.py` regenerates every story in `results/story_quality/` and requires an exact match.

The first slow run needs network access to Hugging Face and download.pytorch.org, and about 2 GB of disk for the Hugging Face and torch caches. Once the downloads are cached, the slow tests take about 30 s on an Apple-silicon laptop CPU.

## Experiments

Run experiments from `workloads/`. Each one takes `--help`.

| experiment | question | command | output |
|---|---|---|---|
| `operator_macs` | Which operators do each model's multiply-adds (MACs) go to? | `uv run python src/workloads/experiments/operator_macs/main.py [family ...]` | `results/operator_macs/<family>.csv` and `.png` |
| `story_quality` | How well does each TinyStories size write the story it is asked for? | `uv run python src/workloads/experiments/story_quality/main.py [model ...]` | `results/story_quality/<#>_<prompt>.csv` |

`operator_macs` counts on PyTorch's meta device, so it downloads nothing and finishes in seconds. `story_quality` has [its own README](src/workloads/experiments/story_quality/README.md) covering its method, findings and regression test.

## Contributing

- **Core code** (operators, models, configs) lives in `src/workloads/` and never imports from `experiments/`.
- **An experiment** is a folder `src/workloads/experiments/<name>/` with a `main.py` that:
  - runs on its own as `uv run python src/workloads/experiments/<name>/main.py`, and parses its arguments with `argparse` so that `--help` works;
  - writes its outputs to `results/<name>/`;
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

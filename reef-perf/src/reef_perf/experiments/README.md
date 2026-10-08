# Experiments

An experiment is a program that runs the simulator and produces artifacts:
results we can analyse on their own, or feed into another experiment.

Each experiment has its own directory with:

| File | Purpose |
| --- | --- |
| `README.md` | What the experiment measures and how to run it. |
| `main.py` | Collects **one** artifact. Command-line arguments describe which one. |
| `run.py` | Collects a **sweep** of artifacts described by a config file. |
| `plot.py` | Plots a sweep, given the same config file. |
| `configs/` | Config files describing sweeps. |

Artifacts are written under `results/<experiment>/` in the reef-perf directory,
so later experiments can find and reuse them. `results/` is not committed.

Experiments are the only code in `src/` that runs standalone. Shared logic
belongs in the `reef_perf` package itself.

## Experiments

| Experiment | What it does |
| --- | --- |
| [`run`](run/README.md) | Runs workloads on the simulator and records cycles, IPC and stall breakdowns. |

# Experiments

An experiment is a program that runs the simulator and produces artifacts:
results we can analyze on their own, or feed into another experiment.

Each experiment has its own directory with a `README.md` saying what it
measures and how to run it, and a `main.py` that collects **one** artifact.
Experiments that collect many add scripts as needed (for example `sweep.py`
and `plot.py`, with sweep configs in `sweeps/`).

Artifacts are written under `results/<experiment>/` in the reef-perf directory,
so later experiments can find and reuse them. `results/` is not committed.

Experiments are the only Python code that runs standalone. Shared logic
belongs in the `reef_perf` package itself.

## Experiments

| Experiment | What it does |
| --- | --- |
| [`sweep`](sweep/README.md) | Runs microbenchmarks under parameter variants and records cycles, IPC and stall breakdowns. |
| [`topdown`](topdown/README.md) | Top-down analysis: where each microbenchmark's dispatch slots went ([docs/topdown.md](../../../docs/topdown.md)). |
| [`correlation`](correlation/README.md) | Compares the model's ROI cycles with the M3 RTL; the PR bot posts its report. |

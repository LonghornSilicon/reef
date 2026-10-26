# `sweep` experiment

Runs the microbenchmarks (`reef-perf/microbenchmarks/`) on the reef_perf
simulator, under one or more parameter variants, and records for each run the
instructions retired, cycles, IPC, where dispatch stalled, and how busy each
execution resource was. Design exploration compares these across variants.

## Artifacts

One JSON file per run:

```json
{
  "microbenchmark": "gemv_int8",
  "sim_config": ".../configs/machines/m3.yaml",
  "params": {"top.frontend.params.fetch_interval": "2"},
  "result": {"instructions": ..., "cycles": ..., "ipc": ..., "dispatch": {...}, "pools": {...}}
}
```

`result` is exactly the simulator's `--json` output.

## Running it

From `reef-perf/`, inside the Docker image (`../tools/docker.sh shell`):

```sh
# One artifact
uv run python -m reef_perf.experiments.sweep.main --microbenchmark gemv_int8
uv run python -m reef_perf.experiments.sweep.main --microbenchmark alu_indep \
    --param top.frontend.params.fetch_interval=2

# A sweep (runs in parallel), then its plots
uv run python -m reef_perf.experiments.sweep.sweep \
    --config python/reef_perf/experiments/sweep/sweeps/all.yaml
uv run python -m reef_perf.experiments.sweep.plot \
    --config python/reef_perf/experiments/sweep/sweeps/all.yaml
```

| Script | Output |
| --- | --- |
| `main.py` | `results/sweep/single/<microbenchmark>.json` (or `--out`) |
| `sweep.py` | `results/sweep/<name>/<variant>/<microbenchmark>.json` and `summary.csv` |
| `plot.py` | `results/sweep/<name>/ipc.png` and `stalls.png` |

## Sweep configs

`sweeps/all.yaml` runs every microbenchmark once on the M3 config. To sweep a
parameter, add variants:

```yaml
name: fetch_sweep
sim_config: configs/machines/m3.yaml
microbenchmarks: [alu_indep, branch_tight, gemv_int8]
variants:
  baseline: {}
  fetch_every_2: {top.frontend.params.fetch_interval: 2}
```

Every (variant, microbenchmark) pair is a separate simulator process.
`sweep.py` runs them in parallel, one per CPU by default (`--jobs N` to change that).

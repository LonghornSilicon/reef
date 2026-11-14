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

### Config stacks, pre-built programs and grids

The fields below are optional; a config without them runs as before.
`sweeps/matrix_presets.yaml` and `sweeps/e2e_kernels.yaml` use them.

| Field | Meaning |
| --- | --- |
| `sim_config` | One file, or a list: the first is the base and the rest are applied in order, as repeated `-c`. |
| `microbenchmarks` | Names may include a subdirectory, e.g. `matrix/mac_indep`. Defaults to `all`, or to none when `programs` is given. |
| `programs` | Name -> pre-built ELF, e.g. the kernels from `inference-engine/coral`. `${VAR}` is expanded; relative paths are from `reef-perf/`. Missing files stop the sweep before any run. |
| `sim_args` | Further `reef_perf` options for every run, e.g. `[--ext-mem-mb, "160"]`. |
| `variants` | As before. A variant may also list `overlay:` config files, applied after `sim_config`. |
| `grid` | Parameter path (or `overlay`) -> list of values. Every combination becomes a variant, crossed with `variants`. Names join the variant name and one label per key, e.g. `coral_128pe__rob_entries-16`. |

```yaml
name: rob_by_preset
sim_config: [configs/machines/m3.yaml, configs/machines/m3_vme.yaml]
microbenchmarks: [matrix/mac_indep, matrix/gemm_tile]
grid:
  overlay: [configs/matrix/coral_vme.yaml, configs/matrix/coral_128pe.yaml]
  top.core.rob.params.rob_entries: [8, 16]
```

`--list` prints the variants and programs without running anything.
YAML `true`/`false` are passed as `true`/`false`.

`summary.csv` keeps its first five columns (`variant`, `microbenchmark`,
`instructions`, `cycles`, `ipc`) and adds `roi_cycles` and
`roi_instructions` (between the first and last `mcycle` read; use these for
kernels that time their own work), `matrix_*` (utilization, MACs, busy
cycles, memory beats, area and energy; whole run, not only the ROI; area and
energy are empty until configured), `overlay_configs`, and one column per
overridden parameter.

## Matrix-engine experiments

`sweeps/matrix_*.yaml` and `matrix/` hold the matrix-engine experiments on
the PR #57 kernels: `matrix/build_kernels.sh` builds the kernel ELFs and
`matrix/run_all.sh` runs every sweep. The plan, how to reproduce a run and
where the results are kept are in `docs/matrix-experiments.md`.

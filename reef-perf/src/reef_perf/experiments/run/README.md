# `run` experiment

Runs RISC-V workloads on the reef_perf simulator and records, for each one,
the instructions retired, cycles, IPC, where dispatch stalled, and how busy
each execution resource was.

This is the basic measurement every other study builds on. The correlation
work compares these numbers against the M3 RTL, and design exploration
compares them across parameter variants.

## Workloads

The `workloads/` directory holds small assembly programs, each stressing one
feature, plus `gemv_int8`, a kernel shaped like LLM decode. They link to the
Reef M3 default memory map (code at `0x0`, data at `0x10000`) and end with
`mpause`, so the same ELF runs on the simulator and on the M3 RTL simulator.
Each loop is bracketed by `csrr mcycle` into `s10`/`s11`, so its RTL cycle
count is `s11 - s10`.

| Workload | Stresses |
| --- | --- |
| `alu_indep`, `alu_chain` | Fetch/dispatch width; ALU latency |
| `branch_tight`, `branch_group` | Taken-branch cost; a branch ends the dispatch group |
| `mul_chain`, `div_chain` | Multiplier latency; data-dependent divider |
| `fp_chain`, `fp_mix` | FPU latency; FP dispatches alone |
| `load_stream`, `load_chain`, `addr_forward` | LSU throughput; load-to-use latency; address forwarding |
| `vec_add_m1`, `vec_add_m4`, `vec_memcpy` | Vector ALU and vector memory at different LMUL |
| `gemv_int8` | int8 matrix-vector product (widening multiply + reduction) |

To add one, drop a `<name>.S` file in `workloads/` that uses `common.inc`. It
is picked up automatically.

## Artifacts

One JSON file per run:

```json
{
  "workload": "gemv_int8",
  "sim_config": ".../configs/m3.yaml",
  "params": {"top.core.fetch.params.fetch_interval": "2"},
  "result": {"instructions": ..., "cycles": ..., "ipc": ..., "dispatch": {...}, "pools": {...}}
}
```

`result` is exactly the simulator's `--json` output.

## Running it

From `reef-perf/`, inside the Docker image (`tools/docker.sh shell`):

```sh
# One artifact
uv run python -m reef_perf.experiments.run.main --workload gemv_int8
uv run python -m reef_perf.experiments.run.main --workload alu_indep \
    --param top.core.fetch.params.fetch_interval=2

# A sweep (runs in parallel), then its plots
uv run python -m reef_perf.experiments.run.run \
    --config src/reef_perf/experiments/run/configs/all.yaml
uv run python -m reef_perf.experiments.run.plot \
    --config src/reef_perf/experiments/run/configs/all.yaml
```

| Script | Output |
| --- | --- |
| `main.py` | `results/run/single/<workload>.json` (or `--out`) |
| `run.py` | `results/run/<name>/<variant>/<workload>.json` and `summary.csv` |
| `plot.py` | `results/run/<name>/ipc.png` and `stalls.png` |

## Sweep configs

`configs/all.yaml` runs every workload once on the M3 config. To sweep a
parameter, add variants:

```yaml
name: fetch_sweep
sim_config: configs/m3.yaml
workloads: [alu_indep, branch_tight, gemv_int8]
variants:
  baseline: {}
  fetch_every_2: {top.core.fetch.params.fetch_interval: 2}
```

Every (variant, workload) pair is a separate simulator process. `run.py` runs
them in parallel, one per CPU by default (`--jobs N` to change that).

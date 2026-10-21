# `topdown` experiment

Runs microbenchmarks and breaks each one's dispatch slots into the top-down
tree described in [docs/topdown.md](../../../../docs/topdown.md): retiring,
frontend bound (branch resteer, fetch bandwidth) and backend bound (core,
memory, vector, matrix, each by stall reason).

## Running it

From `reef-perf/`, inside the Docker image (`../tools/docker.sh shell`):

```sh
# Every microbenchmark: prints the level-2 table
uv run python -m reef_perf.experiments.topdown.main

# One microbenchmark: prints its full tree
uv run python -m reef_perf.experiments.topdown.main --microbenchmark load_chain

# With a parameter changed, to see a category move
uv run python -m reef_perf.experiments.topdown.main --microbenchmark load_chain \
    --param top.memory.params.dtcm_latency=4
```

## Artifacts

Under `results/topdown/` (or `--out`):

| File | Contents |
| --- | --- |
| `<microbenchmark>.json` | The tree: `{"name", "slots", "children": [...]}` |
| `summary.md` | Level-2 shares of every microbenchmark, as a Markdown table |
| `topdown.png` | The same shares as a stacked bar chart |

# `correlation` experiment

Runs every microbenchmark on the M3 config and compares the model with the M3
RTL and with a baseline:

- **ROI cycles.** Each microbenchmark reads `mcycle` before and after its loop
  (`ROI_BEGIN`/`ROI_END` in `microbenchmarks/common.inc`). The model reports
  the cycles between the two reads as `roi_cycles`; the RTL numbers are in
  `microbenchmarks/rtl_cycles.yaml`. Error is (model − RTL) / RTL.
- **Δ cycles.** The change in total model cycles against a baseline in the
  golden-file format, `tests/golden/m3_results.json` by default.

## Running it

From `reef-perf/`, inside the Docker image (`../tools/docker.sh shell`):

```sh
uv run python -m reef_perf.experiments.correlation.main
```

It writes `results/correlation/correlation.json` (one row per
microbenchmark) and `results/correlation/report.md`, and prints the report.

## The PR bot

`.github/workflows/reef-perf-correlation.yml` (at the repository root) runs
this experiment on every push to a PR that changes `reef-perf/`, with the base
branch's golden file as the baseline. It posts the report as a PR comment and
updates that comment on later pushes. The report is also in the job summary
and in the `reef-perf-correlation` artifact (the only places it appears for
PRs from forks, whose token can't comment).

The bot only reports; the golden test (`tests/python/integration/test_golden.py`)
is what fails when timing changes without the golden file being updated.

## Adding RTL numbers

Measure a microbenchmark on the M3 RTL simulator (roadmap ticket B01) and
replace its `null` in `microbenchmarks/rtl_cycles.yaml` with `s11 - s10`,
noting the RTL commit next to the value.

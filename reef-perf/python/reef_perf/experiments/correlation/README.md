# `correlation` experiment

Runs every microbenchmark on the M3 config and compares the model with the M3
RTL and with a baseline:

- **ROI cycles.** Each microbenchmark reads `mcycle` before and after its loop
  (`ROI_BEGIN`/`ROI_END` in `microbenchmarks/common.inc`). The model reports
  the cycles between the two reads as `roi_cycles`; the RTL numbers are in
  `microbenchmarks/rtl_m3.csv`. Error is (model - RTL) / RTL.
- **ROI instructions.** Both sides also count the instructions retired
  between the reads. If they differ, the two cycle counts measure different
  work: the report flags the row and leaves it out of the mean error.
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

## RTL numbers

`microbenchmarks/rtl_m3.csv` comes from `rvv_core_mini_verification_axi_sim
--instr_trace` at CoralNPU tag `M3-2026-04-27` (roadmap ticket B01).
`rtl_trace.py` reads a trace log and returns the ROI cycles
(`s11 - s10`) and the instructions retired in between; see B01 in
`docs/roadmap.md` for the commands.

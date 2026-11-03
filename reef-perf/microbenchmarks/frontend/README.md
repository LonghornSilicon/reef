# Frontend microbenchmarks

Microbenchmarks that measure one frontend behavior on the M3 RTL, each next to
a control that differs only in that behavior. They calibrate frontend
parameters and are not part of the correlation report:
`list_microbenchmarks()` only globs `microbenchmarks/*.S`, so the sweep, the
correlation experiment and the golden file skip this directory.

Every loop runs 1000 times and starts on a 16-byte boundary (`.balign 16`), so
fetch alignment does not mix into the result. `rtl_m3.csv` holds the RTL ROI
cycles, measured as in roadmap ticket B01; `tests/python/integration/
test_frontend_microbenchmarks.py` checks the model against them.

| Microbenchmark | Branch in the loop | M3 predicts | Actual | RTL cycles/iteration |
| --- | --- | --- | --- | --- |
| `branch_bwd_not_taken` | `bne x0, x0, 1b` | taken | not taken | 6 (mispredict) |
| `branch_fwd_not_taken` | `bne x0, x0, 2f` | not taken | not taken | 2 (control) |
| `branch_fwd_taken` | `beq x0, x0, 2f` | not taken | taken | 6 (mispredict) |
| `jalr_chain` | `jalr x0, 0(t0)` | never | taken | 6 (mispredict) |
| `jal_fwd` | `jal x0, 2f` | taken | taken | 4 (control) |

Each mispredict costs 2 cycles more than `jal_fwd`, which needs the same two
fetches per iteration: `mispredict_penalty: 2`. `branch_fwd_not_taken` fits
the whole loop in one fetch.

Build one by hand (the sweep doesn't), inside the Docker image:

```sh
mkdir -p build/microbenchmarks/frontend
uv run python -c "from pathlib import Path; from reef_perf.microbenchmarks import build_microbenchmark; print(build_microbenchmark('frontend/jal_fwd', Path('build/microbenchmarks')))"
```

The frontend benchmarks also participate in the optional held-out RTL
parameter sweep (`../sweeps.json`). Their `ITERATIONS` assembler parameter
changes loop length while keeping the aligned-loop body and control-flow
behavior the same; see the parent `../README.md` for execution instructions.

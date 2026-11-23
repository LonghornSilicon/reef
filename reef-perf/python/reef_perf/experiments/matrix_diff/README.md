# matrix_diff

Measures what one matrix operation costs on the model and on the CoralNPU VME
RTL, with the one-time costs cancelled.

Each looping program in `microbenchmarks/matrix/` (`mac_chain`, `mac_indep`,
`tile_load`, `tile_store`) is built with N and 2N iterations (the `ITERS`
symbol). `(ROI(2N) - ROI(N)) / (N x ops per iteration)` is the steady-state
cost of one operation. Setup, the work still in flight at `ROI_END`, and most
of the difference between the model's M3 core and the RTL's `8a5e1b31` core
cancel out. These are the numbers the matrix parameters are calibrated
against ([docs/matrix.md](../../../../docs/matrix.md), "Measuring against the RTL").

The model steps run in the Docker image. The RTL step runs on the host, where
the VME verification simulator was built (docs/matrix-correlation-log.md, step C1):

```
uv run python -m reef_perf.experiments.matrix_diff.main build
uv run python -m reef_perf.experiments.matrix_diff.main model
PYTHONPATH=python python3 -m reef_perf.experiments.matrix_diff.main rtl \
    --sim <coralnpu>/bazel-bin/tests/verilator_sim/vme_core_mini_verification_axi_sim \
    [--record]
uv run python -m reef_perf.experiments.matrix_diff.main compare
```

Artifacts go to `results/matrix_diff/`: `elfs/<iters>/*.elf`, `model.csv`,
`rtl.csv` and `compare.csv`. `rtl --record` also updates the committed
reference `microbenchmarks/matrix/rtl_vme.csv`. Without a fresh `rtl.csv`,
`compare` uses that reference, so the RTL step only reruns when the programs
or the RTL change.

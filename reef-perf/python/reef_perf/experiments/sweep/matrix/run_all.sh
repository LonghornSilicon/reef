#!/usr/bin/env bash
# Runs every matrix-engine experiment sweep (docs/matrix-experiments.md) and
# the E6 epilogue analysis. Run from reef-perf/ inside the Docker image, with
# E2E_ROOT naming the kernel builds (build_kernels.sh's OUT):
#
#   docker run --rm -v <reef checkout>:/app/reef -w /app/reef/reef-perf \
#       -e E2E_ROOT=build/e2e akull-reef \
#       bash python/reef_perf/experiments/sweep/matrix/run_all.sh [OUT]
#
# OUT is the results root (default results/sweep). Each sweep writes
# OUT/<name>/ (summary.csv and one JSON per run) and OUT/<name>.log. A sweep
# rerun into the same OUT replaces that sweep's results: give each run of the
# experiments its own OUT, or archive the previous one first.
set -uo pipefail

out="${1:-results/sweep}"
configs=python/reef_perf/experiments/sweep/sweeps
status=0
mkdir -p "$out"
for name in matrix_baseline_tokens matrix_e1_bandwidth matrix_e2_tokens \
    matrix_e3_presets matrix_e4_latency matrix_e5_feed matrix_e6_readout \
    matrix_e6_epilogue matrix_e7_int4 matrix_e8_arrays; do
    if uv run python -m reef_perf.experiments.sweep.sweep \
        --config "$configs/$name.yaml" --out "$out" >"$out/$name.log" 2>&1; then
        echo "$name ok"
    else
        echo "$name FAILED (see $out/$name.log)"
        status=1
    fi
done
uv run python python/reef_perf/experiments/sweep/matrix/epilogue_share.py \
    "$out/matrix_e6_epilogue" || status=1
# The container runs as root; hand the outputs back to the host user.
chown -R "${HOST_UID:-1000}:${HOST_GID:-1000}" "$out" .venv 2>/dev/null || true
exit $status

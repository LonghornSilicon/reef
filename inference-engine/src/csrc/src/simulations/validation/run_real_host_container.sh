#!/usr/bin/env bash
# Host validation only. No Coral mount or simulator invocation.
set -euo pipefail
REEF_HOST_CONTEXT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REEF_HOST_REPO=$(git -C "$REEF_HOST_CONTEXT" rev-parse --show-toplevel)
REEF_HOST_WORK=$(mktemp -d /tmp/reef-real-host.XXXXXX)
trap 'printf "%s\n" "$?" > "$REEF_HOST_WORK/exit-status"' EXIT
printf 'Host validation evidence: %s\n' "$REEF_HOST_WORK"
mkdir -p "$REEF_HOST_WORK"/{build,tmp,home,uv-cache,docker-config}
export DOCKER_CONFIG="$REEF_HOST_WORK/docker-config"
mkdir -p "$REEF_HOST_REPO/inference-engine/build"
git -C "$REEF_HOST_REPO" rev-parse HEAD > "$REEF_HOST_WORK/source-revision"
git -C "$REEF_HOST_REPO" status --short > "$REEF_HOST_WORK/source-status"
docker build --progress=plain -t reef-real-host:uv0.10.10 \
    -f "$REEF_HOST_CONTEXT/Dockerfile.host" "$REEF_HOST_CONTEXT" \
    2>&1 | tee "$REEF_HOST_WORK/image-build.log"
docker run --rm --user "$(id -u):$(id -g)" \
    --mount "type=bind,source=$REEF_HOST_REPO,target=/repo,readonly" \
    --mount "type=bind,source=$REEF_HOST_WORK,target=/work" \
    --mount "type=bind,source=$REEF_HOST_WORK/build,target=/repo/inference-engine/build" \
    --env HOME=/work/home --env TMPDIR=/work/tmp \
    --env UV_CACHE_DIR=/work/uv-cache --env UV_PYTHON_DOWNLOADS=never \
    --env RUFF_CACHE_DIR=/work/ruff-cache \
    --env 'PYTEST_ADDOPTS=-o cache_dir=/work/pytest-cache' \
    --env PYTHONDONTWRITEBYTECODE=1 --env HF_HOME=/work/hf-cache \
    --env CUDA_VISIBLE_DEVICES= \
    --env PYTHONPATH=/repo/inference-engine/src:/repo/workloads/src \
    reef-real-host:uv0.10.10 bash -euxo pipefail -c '
        export UV_PROJECT_ENVIRONMENT=/work/inference-venv
        uv sync --locked
        uv run --locked ruff format --check .
        uv run --locked ruff check .
        bash tools/check_cpp.sh
        uv run --locked pytest
        uv run --locked clang-format --dry-run --Werror \
          src/csrc/src/simulations/validation/host_bridge.cc
        uv run --locked clang-tidy --quiet \
          src/csrc/src/simulations/validation/host_bridge.cc -- -std=c++17
        cd /repo/workloads
        export UV_PROJECT_ENVIRONMENT=/work/workloads-venv
        uv sync --locked --no-default-groups --group dev
        uv run --locked --no-default-groups --group dev pytest \
          /repo/inference-engine/src/csrc/src/simulations/validation/test_real_numerics.py
        uv run --locked --no-default-groups --group dev python -m \
          inference_engine.experiments.w8a8_mlp.export_real \
          --synthetic-check --output /work/synthetic
        uv run --locked --no-default-groups --group dev python \
          /repo/inference-engine/src/csrc/src/simulations/validation/compare_real_host.py \
          --fixture /work/synthetic --allow-synthetic \
          --report /work/synthetic-report.json
        uv run --locked --no-default-groups --group dev python -m \
          inference_engine.experiments.w8a8_mlp.export_real \
          --cache /work/hf-cache --output /work/real
        uv run --locked --no-default-groups --group dev python \
          /repo/inference-engine/src/csrc/src/simulations/validation/compare_real_host.py \
          --fixture /work/real --report /work/real-report.json
        printf "PASS: real-model host validation; no Coral execution\n"
    ' 2>&1 | tee "$REEF_HOST_WORK/checks.log"

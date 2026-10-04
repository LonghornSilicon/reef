#!/usr/bin/env bash
# The only tool you run on the host. Everything else runs inside the image.
# Needs Docker (no sudo, no other host packages).
#
#   tools/docker.sh build          Build the image (first time ~1 hour).
#   tools/docker.sh shell          Interactive shell, repo mounted at /app/reef/reef-perf.
#   tools/docker.sh test [ARGS]    Run `uv run pytest ARGS` in a container.
#   tools/docker.sh run CMD...     Run any command in a container.
#
# Environment:
#   REEF_PERF_IMAGE  image name (default: reef-perf)
#   JOBS             parallel compile jobs for `build` (default: 4; Sparta
#                    needs ~1 GB of RAM per job)
#   REEF_PERF_EXTRA_CA  PEM root certificate to trust inside the image, for
#                    networks that intercept TLS (see README)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${REEF_PERF_IMAGE:-reef-perf}"
WORKDIR=/app/reef/reef-perf

# Runs a command in a fresh container with the checkout mounted, so edits on
# the host are visible inside and build outputs land in the checkout's build/.
in_container() {
    local tty=()
    if [[ -t 0 && -t 1 ]]; then
        tty=(-it)
    fi
    docker run --rm "${tty[@]}" \
        -v "${ROOT}:${WORKDIR}" -w "${WORKDIR}" \
        "${IMAGE}" "$@"
}

case "${1:-}" in
build)
    if [[ ! -f "${ROOT}/ext/map/sparta/CMakeLists.txt" ||
          ! -f "${ROOT}/ext/spike/configure" ]]; then
        echo "== fetching submodules"
        git -C "${ROOT}" submodule update --init --recursive
    fi
    extra=()
    if [[ -n "${REEF_PERF_EXTRA_CA:-}" ]]; then
        extra=(--secret "id=extra_ca,src=${REEF_PERF_EXTRA_CA}")
    fi
    DOCKER_BUILDKIT=1 docker build "${extra[@]}" \
        --build-arg "JOBS=${JOBS:-4}" -t "${IMAGE}" "${ROOT}"
    ;;
shell)
    in_container bash
    ;;
test)
    shift
    in_container uv run pytest "$@"
    ;;
run)
    shift
    in_container "$@"
    ;;
*)
    sed -n '2,15p' "$0"
    exit 2
    ;;
esac

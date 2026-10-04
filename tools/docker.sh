#!/usr/bin/env bash
# The only tool you run on the host. Everything else runs inside the image.
# Needs Docker (no sudo, no other host packages).
#
#   tools/docker.sh build          Build the image (potentially hours).
#   tools/docker.sh shell          Interactive shell, repo mounted at /app/reef.
#
# Environment:
#   JOBS             parallel compile jobs for `build` (default: 4; Sparta
#                    needs ~1 GB of RAM per job)
#   REEF_PERF_EXTRA_CA  PEM root certificate to trust inside the image, for
#                    networks that intercept TLS (see README)
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)" 
NAME="$(whoami | LC_ALL=C tr -cd 'a-zA-Z0-9')" 
IMAGE="${REEF_PERF_IMAGE:-$NAME-reef}"
CONTAINER="${NAME}-container" 

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
    state="$(docker container inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null || true)"
    case "${state}" in
    true)  ;;                                    # running: just attach
    false) docker start "${CONTAINER}" >/dev/null ;;
    *)     docker run -dit \
               --name "${CONTAINER}" \
               -v "${ROOT}:/app/reef" \
               -w /app/reef \
               "${IMAGE}" bash >/dev/null ;;
    esac
    exec docker exec -it -w /app/reef "${CONTAINER}" bash
    ;;
*)
    sed -n '2,15p' "$0"
    exit 2
    ;;
esac

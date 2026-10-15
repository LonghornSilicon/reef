#!/usr/bin/env bash
# The only thing you run on the host for pd/. Everything else runs in the image.
#
#   pd/docker/run.sh build          Build the image.
#   pd/docker/run.sh install-pdk    Install SKY130 into $PDK_ROOT (outside the repo), once.
#   pd/docker/run.sh shell          Shell with the repo at /app/reef and $PDK_ROOT at /pdk, read only.
#   pd/docker/run.sh run <cmd...>   Run one command in pd/ and exit.
#
# PDK_ROOT_PRIVATE, if set, is mounted read only at /pdk_private.
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
NAME="$(whoami | LC_ALL=C tr -cd 'a-zA-Z0-9')"
IMAGE="${REEF_PD_IMAGE:-${NAME}-reef-pd}"

case "${1:-}" in
build)
    exec docker build -t "${IMAGE}" "${ROOT}/pd/docker"
    ;;
esac

: "${PDK_ROOT:?set PDK_ROOT to a directory outside the repo}"
mounts=(-v "${ROOT}:/app/reef" -v "${PDK_ROOT}:/pdk:ro")
if [[ -n "${PDK_ROOT_PRIVATE:-}" ]]; then
    mounts+=(-v "${PDK_ROOT_PRIVATE}:/pdk_private:ro" -e PDK_ROOT_PRIVATE=/pdk_private)
fi

case "${1:-}" in
install-pdk)
    mkdir -p "${PDK_ROOT}"
    exec docker run --rm -v "${ROOT}:/app/reef" -v "${PDK_ROOT}:/pdk" -w /app/reef/pd \
        "${IMAGE}" bash docker/install_pdk.sh
    ;;
shell)
    exec docker run --rm -it "${mounts[@]}" -w /app/reef/pd "${IMAGE}" bash
    ;;
run)
    shift
    exec docker run --rm "${mounts[@]}" -w /app/reef/pd "${IMAGE}" bash -lc "$*"
    ;;
*)
    sed -n '2,9p' "$0"
    exit 2
    ;;
esac

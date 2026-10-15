#!/usr/bin/env bash
# The only thing you run on the host for rtl/. Everything else runs in the image.
#
#   rtl/docker/run.sh build          Build the image.
#   rtl/docker/run.sh shell          Interactive shell, repo mounted at /app/reef, cwd rtl/.
#   rtl/docker/run.sh run <cmd...>   Run one command in rtl/ and exit.
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
NAME="$(whoami | LC_ALL=C tr -cd 'a-zA-Z0-9')"
IMAGE="${REEF_RTL_IMAGE:-${NAME}-reef-rtl}"

case "${1:-}" in
build)
    docker build -t "${IMAGE}" "${ROOT}/rtl/docker"
    ;;
shell)
    exec docker run --rm -it -v "${ROOT}:/app/reef" -w /app/reef/rtl "${IMAGE}" bash
    ;;
run)
    shift
    exec docker run --rm -v "${ROOT}:/app/reef" -w /app/reef/rtl "${IMAGE}" bash -lc "$*"
    ;;
*)
    sed -n '2,6p' "$0"
    exit 2
    ;;
esac

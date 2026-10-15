#!/usr/bin/env bash
# Install SKY130 into $PDK_ROOT at the pinned open_pdks commit, with ciel.
# Run through `docker/run.sh install-pdk`. Does nothing if already there.
set -euo pipefail

: "${PDK_ROOT:?}"
PDK_VERSION=1e931c9417df0478df9ee6b7289202f3e87440ab   # the commit ChipFoundry pins for the OpenFrame
CIEL_VERSION=3.0.1

if [[ "$(cat "${PDK_ROOT}/.reef_pdk_version" 2>/dev/null)" == "${PDK_VERSION}" ]]; then
    echo "sky130 ${PDK_VERSION:0:8} already in ${PDK_ROOT}"
    exit 0
fi
uvx --from "ciel==${CIEL_VERSION}" ciel enable --pdk-root "${PDK_ROOT}" --pdk-family sky130 "${PDK_VERSION}"
echo "${PDK_VERSION}" > "${PDK_ROOT}/.reef_pdk_version"

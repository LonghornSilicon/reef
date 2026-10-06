#!/usr/bin/env bash
# Builds pinned third-party dependencies from the ext/ submodules and
# installs them into PREFIX. The Dockerfile runs this, one component per
# image layer; there is normally no reason to run it by hand.
#
#   tools/build_deps.sh PREFIX JOBS COMPONENT...
#
#   yaml-cpp  ext/yaml-cpp  (Sparta 3.x needs yaml-cpp >= 0.8)
#   sparta    ext/map       (Sparta library and headers; needs yaml-cpp)
#   spike     ext/spike     (Spike static libraries and headers)
set -euo pipefail

PREFIX="${1:?usage: tools/build_deps.sh PREFIX JOBS COMPONENT...}"
JOBS="${2:?usage: tools/build_deps.sh PREFIX JOBS COMPONENT...}"
shift 2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
mkdir -p "${PREFIX}"

# Runs a command quietly, printing its output only if it fails.
quiet() {
    local log="${WORK}/last.log"
    if ! "$@" >"${log}" 2>&1; then
        echo "FAILED: $*" >&2
        tail -n 60 "${log}" >&2
        exit 1
    fi
}

# Git checkouts without symlink support (e.g. Windows) store a symlink as a
# small text file holding the target path. Replace such a stub with a copy of
# its target.
materialize_link_stub() {
    local file="$1" target
    [[ -L "${file}" ]] && return 0
    target="$(head -c 256 "${file}")"
    if [[ "${target}" != *$'\n'* && -f "$(dirname "${file}")/${target}" ]]; then
        cp "$(dirname "${file}")/${target}" "${file}"
    fi
}

build_yaml_cpp() {
    echo "== yaml-cpp"
    quiet cmake -S "${ROOT}/ext/yaml-cpp" -B "${WORK}/yaml-cpp" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF
    quiet cmake --build "${WORK}/yaml-cpp" -j "${JOBS}"
    quiet cmake --install "${WORK}/yaml-cpp"
}

build_sparta() {
    echo "== Sparta"
    local src="${ROOT}/ext/map/sparta"
    # Sparta's CMake runs `git describe` on its own sources. Inside the Docker
    # build context the submodule has no git metadata, so give this copy a
    # throwaway repository tagged with the pinned release.
    if ! git -C "${src}" describe --tags --always >/dev/null 2>&1; then
        git -C "${src}" init -q
        git -C "${src}" -c user.name=reef-perf -c user.email=none \
            commit -q --allow-empty -m map_v3.0.2
        git -C "${src}" tag map_v3.0.2
    fi
    quiet cmake -S "${src}" -B "${WORK}/sparta" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
        -DCMAKE_PREFIX_PATH="${PREFIX}"
    cmake --build "${WORK}/sparta" -j "${JOBS}" --target sparta
    quiet cmake --install "${WORK}/sparta"
}

build_spike() {
    echo "== Spike"
    local src="${ROOT}/ext/spike"
    materialize_link_stub "${src}/spike_dasm/spike_dasm_option_parser.cc"
    mkdir -p "${WORK}/spike"
    (
        cd "${WORK}/spike"
        quiet "${src}/configure" --prefix="${PREFIX}"
        quiet make -j "${JOBS}"
        quiet make install
        # The model links Spike statically; install the archives alongside
        # the shared libraries that `make install` provides.
        cp libriscv.a libdisasm.a libsoftfloat.a libfesvr.a libfdt.a \
            "${PREFIX}/lib/"
        # Spike's headers include the generated config.h.
        cp config.h "${PREFIX}/include/riscv/"
    )
}

for component in "$@"; do
    case "${component}" in
    yaml-cpp) build_yaml_cpp ;;
    sparta) build_sparta ;;
    spike) build_spike ;;
    *)
        echo "unknown component: ${component}" >&2
        exit 2
        ;;
    esac
done

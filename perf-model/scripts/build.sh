#!/usr/bin/env bash
# Builds everything from the pinned submodules in ext/:
#   1. yaml-cpp 0.8.0          (ext/yaml-cpp)        -> ${DEPS}
#   2. Sparta map_v3.0.2       (ext/map/sparta)      -> ${DEPS}
#   3. MPACT driver            (ext/coralnpu-mpact)  -> ${BUILD}/driver
#   4. the perf model                                  -> ${BUILD}/model
#   5. the workloads                                   -> workloads/build
#
# Steps 1-2 are skipped once installed; step 3 is incremental (Bazel).
# First run: ~1 hour (Sparta and MPACT's protobuf/abseil). Later runs: seconds.
#
#   scripts/build.sh
#   JOBS=8 scripts/build.sh          # more parallelism if you have >16 GB RAM
set -euo pipefail
source "$(dirname "$0")/env.sh"

cd "${PM}"
if [ ! -f ext/map/sparta/CMakeLists.txt ] || [ ! -f ext/coralnpu-mpact/WORKSPACE ] \
   || [ ! -f ext/map/sparta/simdb/CMakeLists.txt ]; then
  echo "== fetching submodules"
  git submodule update --init --recursive
fi
mkdir -p "${BUILD}" "${DEPS}"

# 1. yaml-cpp. Ubuntu 22.04's yaml-cpp 0.7 exports a CMake target name that
#    Sparta 3.x does not accept, so we build the pinned 0.8.0.
if [ ! -f "${DEPS}/lib/cmake/yaml-cpp/yaml-cpp-config.cmake" ]; then
  echo "== yaml-cpp"
  cmake -S ext/yaml-cpp -B "${BUILD}/yaml-cpp" -GNinja -DCMAKE_BUILD_TYPE=Release \
    -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX="${DEPS}" >/dev/null
  ninja -C "${BUILD}/yaml-cpp" -j"${JOBS}" >/dev/null
  cmake --install "${BUILD}/yaml-cpp" >/dev/null
fi

# 2. Sparta (only the library target, not its tests/examples).
if [ ! -f "${DEPS}/lib/libsparta.a" ]; then
  echo "== Sparta (slow the first time)"
  cmake -S ext/map/sparta -B "${BUILD}/sparta" -GNinja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${DEPS}" -DCMAKE_INSTALL_PREFIX="${DEPS}" >/dev/null
  ninja -C "${BUILD}/sparta" -j"${JOBS}" sparta
  cmake --install "${BUILD}/sparta" >/dev/null
fi

# 3. MPACT driver. The driver sources are copied into the MPACT workspace as
#    //perf_driver so they can use MPACT's Bazel targets; .gitmodules marks
#    untracked files in that submodule as ignored, so this does not dirty it.
echo "== MPACT driver"
rm -rf ext/coralnpu-mpact/perf_driver
mkdir -p ext/coralnpu-mpact/perf_driver
cp driver/BUILD driver/coralnpu_driver.h driver/coralnpu_driver.cc driver/cn_trace.cc \
   ext/coralnpu-mpact/perf_driver/
(
  cd ext/coralnpu-mpact
  # --symlink_prefix=/ : do not create bazel-* symlinks in the submodule.
  bazelisk build -c opt --jobs="${JOBS}" --symlink_prefix=/ \
    //perf_driver:libcoralnpu_driver.so //perf_driver:cn_trace
  BIN="$(bazelisk info -c opt bazel-bin 2>/dev/null)"
  mkdir -p "${DRIVER_DIR}"
  cp -f "${BIN}/perf_driver/libcoralnpu_driver.so" "${BIN}/perf_driver/cn_trace" "${DRIVER_DIR}/"
  chmod u+w "${DRIVER_DIR}"/*
)

# 4. The model.
echo "== model"
cmake -S "${PM}" -B "${BUILD}/model" -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DDEPS_PREFIX="${DEPS}" -DCORALNPU_DRIVER_SO="${DRIVER_DIR}/libcoralnpu_driver.so" >/dev/null
ninja -C "${BUILD}/model" -j"${JOBS}"

# 5. Workloads.
echo "== workloads"
bash "${PM}/workloads/build.sh" >/dev/null

echo
echo "Built:"
echo "  model  ${MODEL}"
echo "  trace  ${DRIVER_DIR}/cn_trace"
echo "Next: scripts/run_all.sh"

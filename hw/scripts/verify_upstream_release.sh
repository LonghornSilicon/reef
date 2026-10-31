#!/usr/bin/env bash
# Verify a CoralNPU upstream tag as a Reef baseline candidate.
# Builds the tag's own dev container, emits CoreMiniAxi / RvvCoreMiniAxi Verilog,
# runs both cocotb regressions, and copies the generated SV out for diffing.
#
# Usage: scripts/verify_upstream_release.sh [TAG]   (default: M3-2026-04-27)
set -euo pipefail

TAG="${1:-M3-2026-04-27}"
SRC="${SRC:-$HOME/coralnpu-m3}"                  # checkout of $TAG
CACHE="${CACHE:-$HOME/coralnpu-bazel-cache-m3}"  # persistent bazel cache
OUT="${OUT:-$HOME/coralnpu-m3-artifacts}"        # logs + generated SV
IMAGE="${IMAGE:-coralnpu:m3}"
# WSL here has ~7.6 GB RAM and no swap; an unthrottled Bazel build (20 cores)
# exhausted it and crashed the WSL VM. Cap the container and Bazel's parallelism.
MEM="${MEM:-5g}"
BAZEL_FLAGS="${BAZEL_FLAGS:---jobs=4 --local_resources=memory=HOST_RAM*.5}"

mkdir -p "$CACHE" "$OUT"
if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch "$TAG" https://github.com/google-coral/coralnpu.git "$SRC"
fi
# M3's own dockerfile fetches Bazel's apt key from a URL that no longer works
# (key rotated); configs/ has the same file with only that URL updated.
DOCKERFILE="${DOCKERFILE:-$(dirname "$0")/../configs/coralnpu-m3-fixedkey.dockerfile}"
docker image inspect "$IMAGE" >/dev/null 2>&1 || \
  docker build -t "$IMAGE" -f "$DOCKERFILE" "$SRC"

run() {
  docker run --rm --memory="$MEM" --memory-swap="$MEM" \
    -v "$SRC":/home/builder/coralnpu \
    -v "$CACHE":/home/builder/.cache/bazel \
    -v "$OUT":/home/builder/artifacts \
    -w /home/builder/coralnpu "$IMAGE" bash -c "$1"
}

# 1. Emit Verilog for the scalar and RVV variants.
run "
  set -e
  bazel build $BAZEL_FLAGS //hdl/chisel/src/coralnpu:core_mini_axi_cc_library_emit_verilog \
              //hdl/chisel/src/coralnpu:rvv_core_mini_axi_cc_library_emit_verilog
  cp -L bazel-bin/hdl/chisel/src/coralnpu/CoreMiniAxi.sv    /home/builder/artifacts/CoreMiniAxi.gen.sv
  cp -L bazel-bin/hdl/chisel/src/coralnpu/RvvCoreMiniAxi.sv /home/builder/artifacts/RvvCoreMiniAxi.gen.sv
" 2>&1 | tee "$OUT/emit.log"

# 2. cocotb regressions (Verilator, via Bazel's hermetic toolchain).
for suite in core_mini_axi_sim_cocotb rvv_core_mini_axi_sim_cocotb; do
  run "bazel test $BAZEL_FLAGS //tests/cocotb:$suite --test_output=errors --keep_going" \
    2>&1 | tee "$OUT/$suite.log" || true
done

echo "Done. Logs and generated SV in $OUT"

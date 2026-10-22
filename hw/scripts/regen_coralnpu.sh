#!/usr/bin/env bash
# Vendor the pinned CoralNPU RTL into rtl/vendor/coralnpu/.
#
# Reef baseline: release M3-2026-04-27 (commit 72d700e), top RvvCoreMiniAxi.
# SV-first: by default we take the release's pre-generated RvvCoreMiniAxi.sv
# (byte-identical to what Bazel emits, see results/m3_verification.md) -- no
# Chisel, no Docker. --build re-emits it from Chisel instead; only needed if a
# generator flag ever changes, and that is a leads decision.
#
# Either way the bundle is sha256-checked, then split into real files plus
# synth.f / sim.f by scripts/split_generated_sv.py.
#
# Usage: scripts/regen_coralnpu.sh [--build]
# Env:   SRC, CACHE, IMAGE, MEM, BAZEL_FLAGS  (--build only, same as verify_upstream_release.sh)
set -euo pipefail

TAG=M3-2026-04-27
COMMIT=72d700ef724669163e1b6f88e8622676ef6b0a74
TOP=RvvCoreMiniAxi
SHA256=74d3bc7e3295353fd879f70a513f7c3fde79912010a1f62f08d815fbcd6f4ff4

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/rtl/vendor/coralnpu"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

if [ "${1:-}" = "--build" ]; then
  SRC="${SRC:-$HOME/coralnpu-m3}"
  CACHE="${CACHE:-$HOME/coralnpu-bazel-cache-m3}"
  IMAGE="${IMAGE:-coralnpu:m3}"
  # WSL here has ~7.6 GB RAM, no swap: an uncapped 20-core Bazel build crashed the VM.
  MEM="${MEM:-5g}"
  BAZEL_FLAGS="${BAZEL_FLAGS:---jobs=4 --local_resources=memory=HOST_RAM*.5}"
  [ -d "$SRC/.git" ] || git clone --depth 1 --branch "$TAG" https://github.com/google-coral/coralnpu.git "$SRC"
  [ "$(git -C "$SRC" rev-parse HEAD)" = "$COMMIT" ] || { echo "$SRC is not at $TAG ($COMMIT)" >&2; exit 1; }
  docker image inspect "$IMAGE" >/dev/null 2>&1 || \
    docker build -t "$IMAGE" -f "$REPO_ROOT/configs/coralnpu-m3-fixedkey.dockerfile" "$SRC"
  docker run --rm --memory="$MEM" --memory-swap="$MEM" \
    -v "$SRC":/home/builder/coralnpu -v "$CACHE":/home/builder/.cache/bazel -v "$TMP":/out \
    -w /home/builder/coralnpu "$IMAGE" bash -c "
      set -e
      bazel build $BAZEL_FLAGS //hdl/chisel/src/coralnpu:rvv_core_mini_axi_cc_library_emit_verilog
      cp -L bazel-bin/hdl/chisel/src/coralnpu/$TOP.sv /out/$TOP.sv"
else
  gh release download "$TAG" --repo google-coral/coralnpu --pattern "$TOP.sv" --dir "$TMP"
fi

echo "$SHA256  $TMP/$TOP.sv" | sha256sum -c --quiet || {
  echo "sha256 mismatch: $TOP.sv is not the pinned $TAG output" >&2; exit 1; }

rm -rf "$OUT_DIR"
python3 "$REPO_ROOT/scripts/split_generated_sv.py" "$TMP/$TOP.sv" "$OUT_DIR"
# Read-only as a "do not hand-edit" guard.
find "$OUT_DIR" -type f -exec chmod 444 {} +
echo "Vendored $TAG ($COMMIT) $TOP into ${OUT_DIR#$REPO_ROOT/}. Now run scripts/lint_vendor.sh."

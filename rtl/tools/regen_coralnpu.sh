#!/usr/bin/env bash
# Regenerate src/reef/gen/ from the Coral Chisel in src/reef/. Runs on the host
# (it starts Docker itself), not inside the rtl image.
#
#   tools/regen_coralnpu.sh                 Chisel -> RvvCoreMiniAxi.sv bundle -> split into
#                                           src/reef/gen/ and filelists/coral_m3.f
#   tools/regen_coralnpu.sh --check         Regenerate into a temp dir and fail on any diff
#                                           against the committed gen/ and filelist (what CI runs)
#   tools/regen_coralnpu.sh --bundle FILE   Only build the bundle, copy it to FILE
#   tools/regen_coralnpu.sh --src DIR ...   Build from DIR instead of src/reef/
#
# Bazel runs in the coralnpu:m3-bazel image (docker/coralnpu-m3-bazel.dockerfile on top of
# docker/coralnpu-m3-fixedkey.dockerfile), built here if missing. The build happens in a temp
# copy of the source, so src/reef/ never gets bazel-* symlinks.
#
# Env: IMAGE (coralnpu:m3-bazel), CACHE (bazel cache dir, ~/.cache/reef-coralnpu-bazel),
#      MEM (container memory cap, 6g), BAZEL_FLAGS (--jobs=4 ...).
set -euo pipefail

RTL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOP=RvvCoreMiniAxi
TARGET=//hdl/chisel/src/coralnpu:rvv_core_mini_axi_cc_library_emit_verilog
# M3-2026-04-27 release asset RvvCoreMiniAxi.sv. Unmodified M3 Chisel reproduces it exactly.
M3_SHA256=74d3bc7e3295353fd879f70a513f7c3fde79912010a1f62f08d815fbcd6f4ff4
GEN=src/reef/gen
FILELIST=filelists/coral_m3.f

IMAGE="${IMAGE:-coralnpu:m3-bazel}"
CACHE="${CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/reef-coralnpu-bazel}"
# WSL with ~8 GB and no swap crashed on an uncapped 20-core Bazel build.
MEM="${MEM:-6g}"
BAZEL_FLAGS="${BAZEL_FLAGS:---jobs=4 --local_resources=memory=HOST_RAM*.5}"

mode=regen
src="$RTL/src/reef"
bundle_out=""
while [[ $# -gt 0 ]]; do
    case "$1" in
    --check) mode=check ;;
    --bundle) mode=bundle; bundle_out="$(realpath -m "$2")"; shift ;;
    --src) src="$(realpath "$2")"; shift ;;
    *) sed -n '2,16p' "$0"; exit 2 ;;
    esac
    shift
done

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    # The fixedkey dockerfile copies nothing in, so any context works.
    docker build -t coralnpu:m3 -f "$RTL/docker/coralnpu-m3-fixedkey.dockerfile" "$RTL/docker"
    docker build -t "$IMAGE" -f "$RTL/docker/coralnpu-m3-bazel.dockerfile" "$RTL/docker"
fi

# The image runs as builder (uid 1000); CI runners are uid 1001. Let it write.
mkdir -p "$CACHE" "$tmp/ws" "$tmp/out"
cp -a "$src/." "$tmp/ws/"
rm -rf "$tmp/ws/gen" "$tmp"/ws/bazel-*
chmod -R a+rwX "$tmp"
chmod a+rwx "$CACHE"

# --workspace_status_command: M3's .bazelrc runs utils/get_workspace_status.sh, which needs
# a git checkout of coralnpu. The emitted SV does not contain the git revision.
docker run --rm --memory="$MEM" --memory-swap="$MEM" \
    -v "$tmp/ws":/home/builder/coralnpu \
    -v "$CACHE":/home/builder/.cache/bazel \
    -v "$tmp/out":/out \
    -w /home/builder/coralnpu "$IMAGE" bash -c "
        set -e
        bazel build $BAZEL_FLAGS --workspace_status_command=/bin/true $TARGET
        cp -L bazel-bin/hdl/chisel/src/coralnpu/$TOP.sv /out/$TOP.sv"

sv="$tmp/out/$TOP.sv"
sha="$(sha256sum "$sv" | cut -d' ' -f1)"
if [[ "$sha" == "$M3_SHA256" ]]; then
    echo "$TOP.sv sha256 $sha: identical to the M3-2026-04-27 release asset"
else
    echo "$TOP.sv sha256 $sha: differs from the M3-2026-04-27 release asset ($M3_SHA256)"
fi

case "$mode" in
bundle)
    cp "$sv" "$bundle_out"
    ;;
regen)
    cd "$RTL"
    rm -rf "$GEN"
    python3 tools/split_generated_sv.py "$sv" "$GEN" "$FILELIST"
    echo "$sha  $TOP.sv" >"$GEN/BUNDLE.sha256"
    ;;
check)
    mkdir -p "$tmp/rtl/filelists"
    (cd "$tmp/rtl" && python3 "$RTL/tools/split_generated_sv.py" "$sv" "$GEN" "$FILELIST" >/dev/null)
    echo "$sha  $TOP.sv" >"$tmp/rtl/$GEN/BUNDLE.sha256"
    if diff -r "$RTL/$GEN" "$tmp/rtl/$GEN" && diff "$RTL/$FILELIST" "$tmp/rtl/$FILELIST"; then
        echo "OK: $GEN and $FILELIST match the Chisel in src/reef"
    else
        echo "FAIL: $GEN or $FILELIST differ from what the Chisel in src/reef generates." >&2
        echo "Never edit gen/. Change the Chisel or hdl/verilog, then run tools/regen_coralnpu.sh." >&2
        exit 1
    fi
    ;;
esac

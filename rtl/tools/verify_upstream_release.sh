#!/usr/bin/env bash
# Check src/reef/ against the CoralNPU release it was copied from. Runs on the host.
#
#   tools/verify_upstream_release.sh           1. the release asset RvvCoreMiniAxi.sv has the
#                                                 pinned sha256
#                                              2. every file in src/reef/ (except gen/) is
#                                                 byte-identical to the tag, or is listed as a
#                                                 Reef change
#   tools/verify_upstream_release.sh --build   3. also: the tag's own copy of those files,
#                                                 built with tools/regen_coralnpu.sh, gives
#                                                 exactly the release asset
#
# Needs git and curl; --build needs Docker (see regen_coralnpu.sh).
set -euo pipefail

TAG=M3-2026-04-27
COMMIT=72d700ef724669163e1b6f88e8622676ef6b0a74
M3_SHA256=74d3bc7e3295353fd879f70a513f7c3fde79912010a1f62f08d815fbcd6f4ff4
URL=https://github.com/google-coral/coralnpu

RTL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$RTL/src/reef"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "== 1. release asset"
curl -fsSL -o "$tmp/release.sv" "$URL/releases/download/$TAG/RvvCoreMiniAxi.sv"
echo "$M3_SHA256  $tmp/release.sv" | sha256sum -c

echo "== 2. src/reef vs $TAG"
git -c advice.detachedHead=false clone -q --depth 1 --branch "$TAG" "$URL.git" "$tmp/tag"
[[ "$(git -C "$tmp/tag" rev-parse HEAD)" == "$COMMIT" ]] || {
    echo "$TAG no longer points at $COMMIT" >&2
    exit 1
}
# Every file and symlink in src/reef/ except gen/.
files() { (cd "$SRC" && find . -path ./gen -prune -o \( -type f -o -type l \) -print0 | sort -z); }
changed=()
added=()
while IFS= read -r -d '' f; do
    f="${f#./}"
    if [[ ! -e "$tmp/tag/$f" && ! -L "$tmp/tag/$f" ]]; then
        added+=("$f")
    elif [[ -L "$SRC/$f" || -L "$tmp/tag/$f" ]]; then
        # Symlinks (toolchain wrappers): compare where they point.
        [[ "$(readlink "$SRC/$f")" == "$(readlink "$tmp/tag/$f")" ]] || changed+=("$f")
    elif ! cmp -s "$SRC/$f" "$tmp/tag/$f"; then
        changed+=("$f")
    fi
done < <(files)
echo "$(files | tr -cd '\0' | wc -c) files," \
    "${#changed[@]} changed and ${#added[@]} added by Reef:"
for f in "${changed[@]+"${changed[@]}"}"; do echo "  M $f"; done
for f in "${added[@]+"${added[@]}"}"; do echo "  A $f"; done

if [[ "${1:-}" == "--build" ]]; then
    echo "== 3. build the $TAG copy of src/reef"
    mkdir "$tmp/pristine"
    files | while IFS= read -r -d '' f; do
        f="${f#./}"
        [[ -e "$tmp/tag/$f" || -L "$tmp/tag/$f" ]] || continue
        mkdir -p "$tmp/pristine/$(dirname "$f")"
        cp -P "$tmp/tag/$f" "$tmp/pristine/$f"
    done
    "$RTL/tools/regen_coralnpu.sh" --src "$tmp/pristine" --bundle "$tmp/built.sv"
    if ! cmp "$tmp/built.sv" "$tmp/release.sv"; then
        mkdir -p "$RTL/build"
        cp "$tmp/built.sv" "$RTL/build/verify_built.sv"
        diff "$tmp/built.sv" "$tmp/release.sv" | head -20
        echo "FAIL: kept the build in build/verify_built.sv" >&2
        exit 1
    fi
    echo "OK: the $TAG files in src/reef build to the release asset, byte for byte"
fi

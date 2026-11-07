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
git clone -q --depth 1 --branch "$TAG" "$URL.git" "$tmp/tag"
[[ "$(git -C "$tmp/tag" rev-parse HEAD)" == "$COMMIT" ]] || {
    echo "$TAG no longer points at $COMMIT" >&2
    exit 1
}
changed=()
added=()
while IFS= read -r -d '' f; do
    f="${f#./}"
    if [[ ! -f "$tmp/tag/$f" ]]; then
        added+=("$f")
    elif ! cmp -s "$SRC/$f" "$tmp/tag/$f"; then
        changed+=("$f")
    fi
done < <(cd "$SRC" && find . -path ./gen -prune -o -type f -print0 | sort -z)
echo "$(cd "$SRC" && find . -path ./gen -prune -o -type f -print | wc -l) files," \
    "${#changed[@]} changed and ${#added[@]} added by Reef:"
for f in "${changed[@]+"${changed[@]}"}"; do echo "  M $f"; done
for f in "${added[@]+"${added[@]}"}"; do echo "  A $f"; done

if [[ "${1:-}" == "--build" ]]; then
    echo "== 3. build the $TAG copy of src/reef"
    mkdir "$tmp/pristine"
    (cd "$SRC" && find . -path ./gen -prune -o -type f -print0) | while IFS= read -r -d '' f; do
        f="${f#./}"
        [[ -f "$tmp/tag/$f" ]] || continue
        mkdir -p "$tmp/pristine/$(dirname "$f")"
        cp "$tmp/tag/$f" "$tmp/pristine/$f"
    done
    "$RTL/tools/regen_coralnpu.sh" --src "$tmp/pristine" --bundle "$tmp/built.sv"
    cmp "$tmp/built.sv" "$tmp/release.sv"
    echo "OK: the $TAG files in src/reef build to the release asset, byte for byte"
fi

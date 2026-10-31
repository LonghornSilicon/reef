#!/usr/bin/env bash
# Run before every push to LonghornSilicon/reef (a PUBLIC repo).
# Scans the files a push would add or change and fails on anything that looks
# like foundry/PDK/NDA material, secrets, or school-server paths.
#
#   scripts/check_no_confidential.sh                 # files changed vs. upstream branch
#   scripts/check_no_confidential.sh origin/rtl-dev  # files changed vs. a given base
#   scripts/check_no_confidential.sh --all           # every tracked file
#
# Exit 0 = clean, 1 = something to look at. A hit is not always a leak, but
# never push until a human has looked at every hit.
#
# Binary files cannot be scanned. One a human has opened and cleared goes in
# configs/reviewed_binaries.sha256 (`sha256sum <file>` output, path relative to this
# repo root); a binary is skipped only while its content still matches that hash.
set -uo pipefail

if [ "${1:-}" = "--all" ]; then
  files=$(git ls-files)
else
  base="${1:-$(git rev-parse --abbrev-ref --symbolic-full-name @{upstream} 2>/dev/null || echo origin/main)}"
  files=$(git diff --name-only --diff-filter=AMR "$base"...HEAD)
fi
[ -z "$files" ] && { echo "Nothing to check."; exit 0; }

fail=0
hit() { echo "  [$1] $2"; fail=1; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REVIEWED="$ROOT/configs/reviewed_binaries.sha256"
reviewed() {  # file: listed with its current hash
  [ -f "$REVIEWED" ] || return 1
  local rel; rel="$(realpath --relative-to="$ROOT" "$f")"
  grep -qxF "$(sha256sum "$f" | cut -d' ' -f1)  $rel" "$REVIEWED"
}

echo "Checking $(echo "$files" | wc -l) file(s)..."

# 1. File types that are foundry/PDK/EDA-database or key material.
bad_ext='\.(lib|lef|techlef|tlef|gds|gds2|gdsii|oas|oasis|db|ndm|milkyway|cdl|spi|spice|sp|scs|tf|itf|tluplus|nxtgrd|qrcTechFile|v\.enc|vp|svp|encrypted|pem|key|p12|pfx|env)$'
bad_name='(^|/)(\.ssh|\.config/gh|\.bash_history|id_rsa|id_ed25519|hosts\.yml)(/|$)|tsmc|tcbn|imec|memory_compiler|memcomp'
while IFS= read -r f; do
  echo "$f" | grep -qiE "$bad_ext"  && hit "file type" "$f"
  echo "$f" | grep -qiE "$bad_name" && hit "file name" "$f"
  [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt 52428800 ] && hit "over 50 MB" "$f"
done <<< "$files"

# 2. Content that marks foundry/vendor confidential material or secrets.
patterns=(
  'TSMC (Confidential|Proprietary|Property)|Taiwan Semiconductor|Copyright.{0,40}TSMC'
  'Company Confidential|Proprietary and Confidential|CONFIDENTIAL AND PROPRIETARY|Under NDA|Subject to NDA'
  '(Synopsys|Cadence|Mentor|Siemens|Arm|Artisan|imec).{0,40}(Confidential|Proprietary)'
  '^\s*library\s*\(.*\)\s*\{|^\s*MACRO\s+\S+\s*$|^\s*SITE\s+core'
  'tcbn[0-9]|tsmcN[0-9]|/pdk/tsmc|/tsmc[0-9]'
  'ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{20,}|gho_[A-Za-z0-9]{36}|-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|sk-ant-[A-Za-z0-9_-]{20,}'
  'LM_LICENSE_FILE|SNPSLMD_LICENSE_FILE|CDS_LIC_FILE|MGLS_LICENSE_FILE|[0-9]{4,5}@[A-Za-z0-9.-]+\.(edu|com)'
  '/home/ecelrc/|/misc/scratch/|/misc/linux/|/usr/local/packages/'
)
while IFS= read -r f; do
  [ -s "$f" ] || continue                                    # missing or empty
  [ "$(basename "$f")" = check_no_confidential.sh ] && continue  # its own patterns
  if ! grep -Iq . "$f" 2>/dev/null; then
    reviewed "$f" || hit "binary file (not in configs/reviewed_binaries.sha256)" "$f"
    continue
  fi
  for p in "${patterns[@]}"; do
    m=$(grep -nE "$p" "$f" 2>/dev/null | head -2 | cut -c1-120)
    [ -n "$m" ] && hit "content" "$f: $m"
  done
done <<< "$files"

if [ $fail -eq 0 ]; then
  echo "OK: no confidential-looking files or content found."
else
  echo
  echo "STOP: look at every hit above before pushing. reef is public; anything"
  echo "pushed stays in git history even after it is deleted."
fi
exit $fail

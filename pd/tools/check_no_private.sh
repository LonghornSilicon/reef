#!/usr/bin/env bash
# Fails on private-looking tracked files under pd/: anything in tech/private
# but README.md and .gitignore, PDK or EDA file types, denylisted whole words.
set -euo pipefail
cd "$(dirname "$0")/.."

fail=0
hit() { echo "  [$1] $2"; fail=1; }

for f in $(git ls-files tech/private); do
    case "$f" in tech/private/README.md|tech/private/.gitignore) ;; *) hit "tech/private" "$f" ;; esac
done

bad_ext='\.(lib|db|lef|tlef|tf|itf|gds|gds2|oas|spef|drc|lvs|cdl|sp|spice)$'
# Allowlist: none yet. A fixture needing one of these types gets listed here with a reason.
while IFS= read -r f; do
    echo "$f" | grep -qiE "$bad_ext" && hit "file type" "$f"
done < <(git ls-files .)

words="$(grep -vE '^\s*(#|$)' tools/private_denylist.txt | paste -sd'|' -)"
while IFS= read -r f; do
    [[ "$f" == tools/private_denylist.txt || "$f" == tools/check_no_private.sh ]] && continue
    grep -Iq . "$f" || continue
    m="$(grep -niwE "$words" "$f" | head -2 | cut -c1-100 || true)"
    [[ -n "$m" ]] && hit "denylist" "$f: $m"
done < <(git ls-files .)

if [[ $fail -eq 0 ]]; then
    echo "no private-looking files under pd/"
else
    echo "STOP: private-looking content under pd/. If it is already pushed, tell a lead now so history can be rewritten."
fi
exit $fail

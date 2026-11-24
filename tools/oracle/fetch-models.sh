#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
wanted=${1:-all}
matched=0
while read -r kind name file revision size sha url; do
    [[ $kind == model && ( $wanted == all || $wanted == "$name" ) ]] || continue
    matched=1
    target=$models_dir/$name/$file
    mkdir -p -- "$(dirname -- "$target")"
    candidate=$target
    if [[ ! -f $target ]]; then
        candidate=$target.part
        if [[ ! -f $candidate || $(stat -c %s -- "$candidate") != "$size" ]]; then
            curl --fail --location --retry 4 --continue-at - --output "$candidate" "$url?download=true"
        fi
    fi
    [[ $(stat -c %s -- "$candidate") == "$size" ]]
    printf '%s  %s\n' "$sha" "$candidate" | sha256sum --check
    if [[ $candidate != "$target" ]]; then mv -- "$candidate" "$target"; fi
done < "$oracle_dir/models.txt"
(( matched )) || { echo "Unknown oracle model: $wanted" >&2; exit 2; }

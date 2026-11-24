#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if (( $# < 3 || $# > 5 )); then
    echo "usage: $0 MODEL TEXT_FILE OUT.ids [ADD_SPECIAL=1] [PARSE_SPECIAL=0]" >&2
    exit 2
fi
temp=$(mktemp -d)
trap 'rm -rf -- "$temp"' EXIT
{
    printf 'prompt\t%s\t%s\t' "${4:-1}" "${5:-0}"
    od -An -v -tx1 -- "$2" | tr -d ' \n'
    printf '\n'
} > "$temp/corpus.tsv"
oracle_run cpu tokenize "$1" "$temp/corpus.tsv" "$temp/tokens.tsv"
awk -F '\t' 'NR == 2 && $5 != "" { n = split($5, ids, ","); for (i = 1; i <= n; i++) print ids[i] }' \
    "$temp/tokens.tsv" > "$3"

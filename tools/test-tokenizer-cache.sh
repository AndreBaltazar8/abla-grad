#!/usr/bin/env bash
# Verify tokenizer cache hits and recovery using the existing Inkling fixtures. No GPU is used.
set -euo pipefail
bin=${ABLA_GRAD_BIN:-build/abla-grad}
json=${1:-$HOME/models/inkling-2-mini/tokenizer.json}
model=${2:-$HOME/models/inkling-2-mini/Inkling-2-Mini-UD-Q4_K_M.gguf}
dir=$(mktemp -d)
trap 'rm -rf -- "$dir"' EXIT
export XDG_CACHE_HOME="$dir/cache"
cache="$XDG_CACHE_HOME/abla-grad/tokenizers-v1"
tokenize() {
    "$bin" tokenize "$1" tests/prompts/03.txt --add-special >"$2"
}

tokenize "$json" "$dir/json.ids"
digest=$(sha256sum "$json")
entry="$cache/json-${digest%% *}.bin"
[[ -s $entry ]]
cp "$entry" "$dir/expected.bin"
before=$(stat -c %y "$entry")
tokenize "$json" "$dir/json-warm.ids"
[[ $(stat -c %y "$entry") == "$before" ]]
cmp "$dir/json.ids" "$dir/json-warm.ids"

printf 'damaged' >>"$entry"
tokenize "$json" "$dir/json-repaired.ids"
cmp "$dir/expected.bin" "$entry"
cmp "$dir/json.ids" "$dir/json-repaired.ids"

# A correct checksum still cannot make a truncated tables payload valid.
printf '\377\377\377\377' >"$dir/invalid"
bad_digest=$(sha256sum "$dir/invalid")
key=${entry##*/}
printf '%s\n%s\n' "${key%.bin}" "${bad_digest%% *}" >"$entry"
cat "$dir/invalid" >>"$entry"
tokenize "$json" "$dir/json-structure-repaired.ids"
cmp "$dir/expected.bin" "$entry"
cmp "$dir/json.ids" "$dir/json-structure-repaired.ids"

cp "$json" "$dir/changed.json"
printf '\n' >>"$dir/changed.json"
tokenize "$dir/changed.json" "$dir/changed.ids"
changed=$(sha256sum "$dir/changed.json")
[[ "${changed%% *}" != "${digest%% *}" && -s "$cache/json-${changed%% *}.bin" ]]
cmp "$dir/json.ids" "$dir/changed.ids"

tokenize "$model" "$dir/gguf.ids"
gguf_entry=$(find "$cache" -type f -name 'gguf-*.bin' -print -quit)
[[ -s $gguf_entry ]]
before=$(stat -c %y "$gguf_entry")
tokenize "$model" "$dir/gguf-warm.ids"
[[ $(stat -c %y "$gguf_entry") == "$before" ]]
cmp "$dir/gguf.ids" "$dir/gguf-warm.ids"
echo 'test-tokenizer-cache: JSON and GGUF hits, source changes, and corruption recovery passed'

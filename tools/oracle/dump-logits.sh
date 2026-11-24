#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if (( $# < 3 || $# > 4 )); then
    echo "usage: $0 MODEL IDS_FILE OUT.grdl [--cpu|--cuda]" >&2
    exit 2
fi
case ${4:---cuda} in
    --cpu) backend=cpu; layers=0 ;;
    --cuda) backend=cuda; layers=999 ;;
    *) echo "Expected --cpu or --cuda" >&2; exit 2 ;;
esac
temp=$(mktemp -d -- "$(dirname -- "$3")/.grdl.XXXXXX")
trap 'rm -rf -- "$temp"' EXIT
oracle_run "$backend" logits "$1" "$2" "$temp/data.grdl" "$temp/prompt.txt" "$layers"
read -r digest _ < <(sha256sum < "$1")
reads=0
if [[ -n ${ORACLE_PROMPT_FILE:-} ]]; then
    cp -- "$ORACLE_PROMPT_FILE" "$temp/prompt.txt"
    reads=1
fi
jq -n --rawfile prompt "$temp/prompt.txt" --arg model "${1##*/}" --arg sha "$digest" \
    --arg commit "$llama_commit" --arg backend "$backend" --argjson layers "$layers" \
    --argjson original "$reads" \
    '{prompt:$prompt,prompt_source:(if $original == 1 then "ORACLE_PROMPT_FILE" else "decoded_ids" end),
      model:$model,model_sha256:$sha,llama_cpp_commit:$commit,backend:$backend,
      flags:{context:512,threads:4,threads_batch:4,batch:512,ubatch:512,tokens_per_decode:1,
             cache_k:"f16",cache_v:"f16",flash_attention:"auto",gpu_layers_requested:$layers}}' > "$temp/data.json"
mv -- "$temp/data.json" "$3.json"
mv -- "$temp/data.grdl" "$3"

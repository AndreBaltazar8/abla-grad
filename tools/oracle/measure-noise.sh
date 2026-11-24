#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
out=${1:?usage: measure-noise.sh OUTPUT_DIRECTORY [cpu|cuda|compare|all]}
mode=${2:-all}
case $mode in
    cpu|cuda|compare|all) ;;
    *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
repo=$(cd -- "$oracle_dir/../.." && pwd)
mkdir -p -- "$out"
if [[ $mode == all || $mode == compare ]]; then
    printf 'model\tprompt\tpositions\tmax_abs_logit\tmax_kl_cpu_cuda\tmean_kl_cpu_cuda\tmatching_argmax\n' > "$out/noise.tsv"
fi
while read -r kind name file revision bytes digest url; do
    [[ $kind == model ]] || continue
    model=$models_dir/$name/$file
    for prompt in "$repo"/tests/prompts/*.txt; do
        case_name=${prompt##*/}
        case_name=${case_name%.txt}
        ids=${prompt%.txt}.ids
        if [[ $name == inkling-2-mini ]]; then ids=${prompt%.txt}.inkling.ids; fi
        for backend in cpu cuda; do
            [[ $mode == "$backend" || $mode == all ]] || continue
            directory=$out/$backend/$name
            mkdir -p -- "$directory"
            ORACLE_PROMPT_FILE=$prompt "$oracle_dir/dump-logits.sh" "$model" "$ids" \
                "$directory/$case_name.grdl" "--$backend" > "$directory/$case_name.log" 2>&1 ||
                { tail -n 20 "$directory/$case_name.log" >&2; exit 1; }
        done
        if [[ $mode == all || $mode == compare ]]; then
            metrics=$(oracle_run cpu noise "$out/cpu/$name/$case_name.grdl" "$out/cuda/$name/$case_name.grdl")
            printf '%s\t%s\t%s\n' "$name" "$case_name" "$metrics" >> "$out/noise.tsv"
        fi
    done
done < "$oracle_dir/models.txt"

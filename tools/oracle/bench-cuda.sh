#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
out=${1:?usage: bench-cuda.sh OUTPUT_DIRECTORY}
mkdir -p -- "$out"
export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LC_ALL=C
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv > "$out/gpu.csv"
while read -r kind name file revision bytes digest url; do
    [[ $kind == model ]] || continue
    "$llama_root/cuda/bin/llama-bench" -m "$models_dir/$name/$file" \
        -p 512 -n 128 -r 3 -t 4 -b 512 -ub 512 -ngl 999 -fa auto -o json \
        > "$out/$name.json" 2> "$out/$name.log"
    jq -e 'length == 2 and all(.[]; (.samples_ts | length == 3) and .avg_ts > 0)' "$out/$name.json" > /dev/null
done < "$oracle_dir/models.txt"

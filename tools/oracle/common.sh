#!/usr/bin/env bash
oracle_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
read -r _ llama_commit llama_url < <(awk '$1 == "llama.cpp" { print; exit }' "$oracle_dir/models.txt")
models_dir=${ORACLE_MODELS_DIR:-$HOME/models}
llama_root=$models_dir/llama.cpp-$llama_commit
llama_source=${LLAMA_SOURCE_DIR:-$HOME/src/llama.cpp}

oracle_run() {
    local backend=$1
    shift
    local binary=$llama_root/$backend/abla-grad-oracle
    [[ -x $binary ]] || { echo "Build oracle backend $backend with $oracle_dir/build-llama.sh" >&2; return 1; }
    if [[ $backend == cuda ]]; then
        export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    LC_ALL=C "$binary" "$@"
}

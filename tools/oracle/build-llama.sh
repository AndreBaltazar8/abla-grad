#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
mode=${1:-all}
case $mode in
    cpu|cuda|all) ;;
    *) echo "usage: $0 [cpu|cuda|all]" >&2; exit 2 ;;
esac

if [[ ! -d $llama_source/.git ]]; then
    mkdir -p -- "$(dirname -- "$llama_source")"
    git init -q "$llama_source"
    git -C "$llama_source" remote add origin "$llama_url"
    git -C "$llama_source" fetch --depth=1 origin "$llama_commit"
    git -C "$llama_source" checkout --detach "$llama_commit"
fi
[[ $(git -C "$llama_source" rev-parse HEAD) == "$llama_commit" ]] ||
    { echo "Expected $llama_commit at $llama_source" >&2; exit 1; }
[[ -z $(git -C "$llama_source" status --porcelain --untracked-files=no) ]] ||
    { echo "Pinned llama.cpp source has tracked edits" >&2; exit 1; }
mkdir -p -- "$llama_root"

export oracle_dir llama_source llama_root
if [[ $mode != cuda ]]; then
    if command -v cmake >/dev/null && command -v ninja >/dev/null; then
        "$oracle_dir/build-one.sh" cpu
    else
        nix-shell -p cmake ninja --run 'bash "$oracle_dir/build-one.sh" cpu'
    fi
fi
if [[ $mode != cpu ]]; then
    nix-shell "$oracle_dir/cuda-shell.nix" --run 'bash "$oracle_dir/build-one.sh" cuda'
fi

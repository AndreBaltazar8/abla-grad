#!/usr/bin/env bash
# Speculation gate: greedy speculative decoding (DSpark drafter, adaptive) against plain greedy decoding on the prompts 01-05: the streams are identical or
# differ first at a near-tie (target logit margin <= 1.0), see `abla-grad spec-check`. Run in nix-shell shell.nix; each prompt takes the GPU lock itself.
# Usage: tools/gate-spec.sh [abla-grad binary] [tokens, default 256]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
binary=${1:-build/abla-grad}
tokens=${2:-256}
dir=${INKLING_DIR:-$HOME/models/inkling-2-mini}
for prompt in 01 02 03 04 05; do
    echo "== prompt $prompt"
    tools/gpu-lock.sh -t 900 -- "$binary" spec-check "$dir/Inkling-2-Mini-UD-Q4_K_M.gguf" "$dir/dspark-Inkling-2-Mini-Q8_0.gguf" "tests/prompts/$prompt.inkling.ids" --n "$tokens" | tail -3
done
echo "gate-spec: all prompts identical or near-tie"

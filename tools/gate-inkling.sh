#!/usr/bin/env bash
# The proof-model gate (docs/plan.md, S3): Inkling-2-Mini logits from the GPU against llama.cpp's CUDA build on the four fixed prompts.
# Usage: tools/gate-inkling.sh [abla-grad binary]   (run in nix-shell shell.nix; needs ~/models and the oracle's GRDL files)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
binary=${1:-build/abla-grad}
model=$(ls ~/models/inkling-2-mini/*.gguf)
oracle=${ORACLE_DIR:-$HOME/src/abla-grad-oracle/artifacts/cuda/inkling-2-mini}
out=build/gate-inkling
mkdir -p "$out"
positions=0
argmax=0
for prompt in 01 02 03 04; do
    "$binary" logits "$model" "tests/prompts/$prompt.inkling.ids" ${GATE_LOGITS_FLAGS:-} -o "$out/$prompt.grdl" > /dev/null
    status=0
    "$binary" compare "$oracle/$prompt.grdl" "$out/$prompt.grdl" --tolerances tests/tolerances-inkling.txt > "$out/$prompt.txt" || status=$?
    summary=$(tail -1 "$out/$prompt.txt")
    echo "prompt $prompt: $summary"
    [[ $status == 0 ]] || { echo "gate-inkling: prompt $prompt is outside the tolerances" >&2; exit 1; }
    count=$(sed -E 's/^compare: ([0-9]+) positions.*/\1/' <<<"$summary")
    misses=$(sed -E 's/.* ([0-9]+) argmax differences.*/\1/' <<<"$summary")
    positions=$((positions + count))
    argmax=$((argmax + misses))
done
((argmax * 100 <= positions)) || { echo "gate-inkling: $argmax argmax differences in $positions positions (more than 1%)" >&2; exit 1; }
echo "gate-inkling: $positions positions within the tolerances, $argmax argmax differences"

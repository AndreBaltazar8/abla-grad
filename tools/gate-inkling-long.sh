#!/usr/bin/env bash
# The long-prompt gate (slow, a few minutes): the 1,624-token natural-text prompt tests/prompts/05.inkling.ids (Alice in Wonderland excerpt) through the
# step-mode decoder against llama.cpp 33aeb17's CUDA logits. Natural text is noisier than the four short prompts, so the rule counts instead of
# bounding every row: see tests/tolerances-inkling-long.txt for the limits and the measured noise floor they come from.
# Usage: tools/gate-inkling-long.sh [abla-grad binary]   (nix-shell shell.nix; run it WITHOUT gpu-lock.sh: it takes the lock for the logits run only; needs the GGUF and the oracle 05.grdl, LONG_ORACLE_DIR)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
binary=${1:-build/abla-grad}
model=${INKLING_MODEL:-$HOME/models/inkling-2-mini/Inkling-2-Mini-UD-Q4_K_M.gguf}
oracle=${LONG_ORACLE_DIR:-$HOME/src/abla-grad-oracle/long-depth}/05.grdl
out=build/gate-inkling-long
mkdir -p "$out"
# only the logits run holds the GPU; the several-minute compare below is CPU-only
tools/gpu-lock.sh -t 900 -- "$binary" logits "$model" tests/prompts/05.inkling.ids --step --ctx 1700 -o "$out/05.grdl" > /dev/null
"$binary" compare "$oracle" "$out/05.grdl" --tolerances tests/tolerances-inkling.txt > "$out/05.txt" || true
summary=$(tail -1 "$out/05.txt")
echo "long prompt 05: $summary"
positions=$(sed -E 's/^compare: ([0-9]+) positions.*/\1/' <<<"$summary")
outside=$(sed -E 's/^compare: [0-9]+ positions, ([0-9]+) outside.*/\1/' <<<"$summary")
misses=$(sed -E 's/.* ([0-9]+) argmax differences.*/\1/' <<<"$summary")
kl=$(sed -E 's/.*worst KL ([0-9.e+-]+).*/\1/' <<<"$summary")
((positions == 1624)) || { echo "gate-inkling-long: expected 1624 positions, got $positions" >&2; exit 1; }
((outside <= 50)) || { echo "gate-inkling-long: $outside rows outside the .04 KL limit (limit 50)" >&2; exit 1; }
((misses * 100 <= positions * 3)) || { echo "gate-inkling-long: $misses argmax differences (limit 3%)" >&2; exit 1; }
awk -v kl="$kl" 'BEGIN { exit !(kl + 0 <= 1.1) }' || { echo "gate-inkling-long: worst KL $kl above 1.1" >&2; exit 1; }
echo "gate-inkling-long: $outside rows over the limit, $misses argmax differences, worst KL $kl: within the long-prompt rule"

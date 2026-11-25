#!/usr/bin/env bash
# The long-prompt gate (slow: minutes for prompt 05, about half an hour for 06): natural-text prompts through the step-mode decoder against llama.cpp 33aeb17's
# CUDA logits. 05 = tests/prompts/05.inkling.ids (1,624 tokens), 06 = tests/prompts/06.inkling.ids (4,300 tokens, crosses the 1,024 window and 4,096).
# Natural text is noisier than the four short prompts, so the rule counts instead of bounding every row: tests/tolerances-inkling-long.txt has the limits and the
# measured noise floor they come from.
# Usage: [LONG_PROMPTS="05 06"] tools/gate-inkling-long.sh [abla-grad binary]   (nix-shell shell.nix; run it WITHOUT gpu-lock.sh: it takes the lock for the
# logits runs only; needs the GGUF and the oracle GRDLs in LONG_ORACLE_DIR, default ~/src/abla-grad-oracle/long-depth).
# Override the step mode with GATE_LOGITS_FLAGS="--prefill --chunk 512" to apply the same limits to batched prefill.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
binary=${1:-build/abla-grad}
model=${INKLING_MODEL:-$HOME/models/inkling-2-mini/Inkling-2-Mini-UD-Q4_K_M.gguf}
oracle_dir=${LONG_ORACLE_DIR:-$HOME/src/abla-grad-oracle/long-depth}
out=build/gate-inkling-long
mkdir -p "$out"
read -r -a logits_flags <<<"${GATE_LOGITS_FLAGS:---step}"
# per prompt: positions, max rows over the .04 KL limit, max argmax differences in percent, max KL
declare -A rule=([05]="1624 50 3 1.1" [06]="4300 40 3 0.3")
for prompt in ${LONG_PROMPTS:-05 06}; do
    read -r positions max_rows max_percent max_kl <<<"${rule[$prompt]}"
    # only the logits run holds the GPU; the compare below is CPU-only
    tools/gpu-lock.sh -t 3000 -- "$binary" logits "$model" "tests/prompts/$prompt.inkling.ids" "${logits_flags[@]}" --ctx $((positions + 100)) -o "$out/$prompt.grdl" > /dev/null
    "$binary" compare "$oracle_dir/$prompt.grdl" "$out/$prompt.grdl" --tolerances tests/tolerances-inkling.txt > "$out/$prompt.txt" || true
    summary=$(tail -1 "$out/$prompt.txt")
    echo "long prompt $prompt: $summary"
    got=$(sed -E 's/^compare: ([0-9]+) positions.*/\1/' <<<"$summary")
    outside=$(sed -E 's/^compare: [0-9]+ positions, ([0-9]+) outside.*/\1/' <<<"$summary")
    misses=$(sed -E 's/.* ([0-9]+) argmax differences.*/\1/' <<<"$summary")
    kl=$(sed -E 's/.*worst KL ([0-9.e+-]+).*/\1/' <<<"$summary")
    ((got == positions)) || { echo "gate-inkling-long: $prompt: expected $positions positions, got $got" >&2; exit 1; }
    ((outside <= max_rows)) || { echo "gate-inkling-long: $prompt: $outside rows outside the .04 KL limit (limit $max_rows)" >&2; exit 1; }
    ((misses * 100 <= positions * max_percent)) || { echo "gate-inkling-long: $prompt: $misses argmax differences (limit $max_percent%)" >&2; exit 1; }
    awk -v kl="$kl" -v max="$max_kl" 'BEGIN { exit !(kl + 0 <= max + 0) }' || { echo "gate-inkling-long: $prompt: worst KL $kl above $max_kl" >&2; exit 1; }
    echo "gate-inkling-long: $prompt: $outside rows over the limit, $misses argmax differences, worst KL $kl: within the long-prompt rule"
done

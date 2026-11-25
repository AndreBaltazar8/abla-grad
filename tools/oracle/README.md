# llama.cpp oracle

[models.txt](models.txt) pins GGUF revisions, sizes and SHA-256s, and the
llama.cpp source commit. The shell tools use a C helper linked to that
revision. Defaults are `~/models` and `~/src/llama.cpp`. Override them
with `ORACLE_MODELS_DIR` and `LLAMA_SOURCE_DIR`. CUDA builds use the
[Nix shell](../../shell.nix) and default to `sm_89` (`ORACLE_CUDA_ARCH`).

```sh
tools/oracle/fetch-models.sh
tools/oracle/build-llama.sh
model="$HOME/models/llama-3.2-1b-instruct/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
tools/oracle/dump-tokens.sh "$model" tests/prompts/01.txt /tmp/01.ids
tools/gpu-lock.sh -- env ORACLE_PROMPT_FILE=tests/prompts/01.txt \
    tools/oracle/dump-logits.sh "$model" /tmp/01.ids /tmp/01.grdl --cuda
tools/gpu-lock.sh -- tools/oracle/dump-tensors.sh "$model" /tmp/01.ids /tmp/01-tensors --cuda
tools/gpu-lock.sh -- tools/oracle/dump-greedy.sh "$model" tests/prompts/01.txt /tmp/01-greedy 8 10 --cuda
```

- Token files contain one decimal ID per line. The optional fourth and fifth arguments of `dump-tokens.sh` set `add_special` and `parse_special`; [tests/prompts](../../tests/prompts) has fixed IDs and tokenizer TSVs.
- GRDL v1 stores `GRDL`, little-endian u32 version, vocabulary size, token count, u32 IDs and row-major F32 logits. Row i follows input ID i. The JSON sidecar records model hash, commit, prompt, backend and flags.
- Tensor dumps contain contiguous little-endian F32 files and an index of shapes, ops and original types. The output directory must be new.
- Greedy dumps save IDs, raw F32 logits, top logits and row metadata. Ties go to the lowest ID. The final output ID is listed without feeding it back.

## Correctness gates

```sh
tools/gpu-lock.sh -- tools/gate-llama.sh
GATE_LOGITS_FLAGS=--step tools/gpu-lock.sh -- tools/gate-inkling.sh
GATE_LOGITS_FLAGS="--prefill --chunk 256" tools/gpu-lock.sh -- tools/gate-inkling.sh
tools/gate-spec.sh
```

`abla-grad compare` reports argmax, top five, logit error and KL. The
[short](../../tests/tolerances-inkling.txt) and [long](../../tests/tolerances-inkling-long.txt)
Inkling gates use calibrated limits. `measure-noise.sh OUTPUT_DIR` measures
llama.cpp CPU/CUDA differences. `bench-cuda.sh OUTPUT_DIR` saves individual
benchmark samples. See [the current performance tables](../../docs/speed.md).

## Long context

Build [context-logits.c](context-logits.c) against the matching llama.cpp
headers and libraries for each reference build. It requests F16 KV with
one sequence and sliding windows. Compare the same IDs and chunk size in
Abla, mainline and Unsloth. Record the ID file hash. Modes `global32`,
`global16` and `global8` in Abla change global precision only.

```sh
ablac build tools/context-logits.ab -o build/context-logits --fast
cc -O3 -fopenmp -fno-math-errno tools/oracle/compare-context.c -lm -o build/compare-context
set -o pipefail
tools/gpu-lock.sh -t 3000 -- build/context-logits "$model" "$ids" q8 "$chunk" |
    build/compare-context "$mainline_raw" - "$positions" 200058
```

[compare-context.c](compare-context.c) defines the TSV columns.
[score-context.c](score-context.c) scores next-ID perplexity and saves
greedy IDs. [context-quality.mjs](../context-quality.mjs) generates fixed
needle/retrieval prompts and scores final JSON. Its pinned-engine runner
[context-greedy.c](context-greedy.c) also decodes answer IDs in `text` mode.

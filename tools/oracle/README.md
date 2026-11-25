# llama.cpp reference oracle

The oracle provides reference data for abla-grad. The application itself has
no dependency on llama.cpp. The shell entry points use a C helper linked to the
pinned libllama. The model files and compiled binaries live outside the repo.

```sh
tools/oracle/fetch-models.sh
tools/oracle/build-llama.sh
```

`models.txt` pins both Hugging Face revisions, exact sizes, SHA-256s and the
llama.cpp commit. The small model is Unsloth Llama 3.2 1B Instruct Q4_K_M, an
ungated 807,694,368-byte GGUF with the standard Llama architecture. The
16,610,544,608-byte Inkling 2 Mini UD-Q4_K_M uses Unsloth's mainline layout
(`inkling2_loop`). The name is the quant preset. Its 744 tensors use these
stored formats.

| Type | Tensors | Stored bytes |
| --- | ---: | ---: |
| F32 | 518 | 147,544,064 |
| Q8_0 | 150 | 3,112,445,952 |
| Q5_K | 4 | 1,868,562,432 |
| BF16 | 32 | 3,833,856 |
| NVFP4 | 40 | 11,466,178,560 |

The source checkout defaults to `~/src/llama.cpp`, the downloads to
`~/models/<name>/`, and the builds to
`~/models/llama.cpp-<commit>/{cpu,cuda}/`. Set `LLAMA_SOURCE_DIR` or
`ORACLE_MODELS_DIR` to override the paths. The build script also accepts
`cpu` or `cuda`. The CUDA shell uses Nix CUDA 12.9, with `sm_89` the default
compile target for the RTX 4090. Set `ORACLE_CUDA_ARCH` for another device.
Scripts use Bash, coreutils, Git, curl and jq.

The CUDA scripts add `/run/opengl-driver/lib` to the library search path on
NixOS. The driver there provides `libcuda.so.1`. To resolve NVRTC separately
for the abla-grad CUDA backend, use the `lib` output of
`cudaPackages_12_9.cuda_nvrtc` and load `libnvrtc.so.12` from its `lib/`
directory.

## Fixed prompts and GRDL v1

```sh
model="$HOME/models/llama-3.2-1b-instruct/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
tools/oracle/dump-tokens.sh "$model" tests/prompts/01.txt /tmp/01.ids
ORACLE_PROMPT_FILE=tests/prompts/01.txt \
  tools/oracle/dump-logits.sh "$model" /tmp/01.ids /tmp/01.grdl --cuda
tools/oracle/dump-tensors.sh "$model" /tmp/01.ids /tmp/01-tensors --cuda
tools/oracle/dump-greedy.sh "$model" tests/prompts/01.txt /tmp/01-greedy 8 10 --cuda
```

`dump-tokens.sh` writes one decimal ID per line. By default it allows
model-configured BOS/EOS insertion and treats literal control-token names as
text. The optional fourth and fifth arguments set `add_special` and
`parse_special` independently. `tests/prompts/NN.ids` are the small model's
IDs and `NN.inkling.ids` the Inkling IDs. The tokenizer TSVs cover the four flag
combinations on 15 byte-exact strings. `tokenizer-cases.tsv` has no header.
Its fields are case name, add flag, parse flag and input bytes in hex. The
expected TSVs add comma-separated IDs and comma-separated hex token pieces.
To regenerate them, source `tools/oracle/common.sh` and call
`oracle_run cpu tokenize MODEL tests/prompts/tokenizer-cases.tsv OUTPUT.tsv`.

`dump-logits.sh` consumes supplied IDs one at a time in a 512-token context
and emits raw logits after each position. The binary contains `GRDL`, little-
endian u32 version 1, vocabulary size, and token count, then the u32 IDs and
a row-major matrix of float32 logits. Row `i` is the prediction *after*
consuming ID `i`. The `.grdl.json` sidecar records hash, commit, backend, flags
and prompt. It reconstructs the prompt from the IDs unless `ORACLE_PROMPT_FILE`
points to the original bytes.

`dump-tensors.sh` evaluates the first supplied token. The output directory
must not exist. `index.txt` indexes named F32, F16 and BF16 intermediate nodes
observed by the llama.cpp eval callback. Files hold contiguous little-endian
F32 in GGML dimension order. The index lists the name, four extents, op,
original type and filename. Repeated names get distinct occurrence suffixes.

`dump-greedy.sh` writes `tokens.tsv`, `rows.tsv`, `topk.tsv`,
`logits.f32le`, `prompt.txt` and `meta.tsv`. It first evaluates the
tokenized prompt, then selects the largest raw logit at each step, with ties
going to the lowest ID. It stops on the requested number of tokens or an EOG
token. Every row comes after consuming its `input_id`. `rows.tsv` records
whether the next ID was supplied by the prompt or chosen greedily.
`logits.f32le` has no header. Use `meta.tsv` for the vocabulary size and row
count, and `topk.tsv` for the raw logits ranked within each row. The final
generated token is listed in `tokens.tsv` and is not fed back for evaluation.

## Calibration and speed

`tools/oracle/measure-noise.sh OUTPUT_DIR` generates CPU and CUDA GRDL files
for both models and all four prompts. It reports max absolute logit error,
max and mean KL of CPU against CUDA, and matching argmax positions in
`noise.tsv`. Optional `cpu`, `cuda` and `compare` modes split the work.

`tools/oracle/bench-cuda.sh OUTPUT_DIR` measures CUDA with llama-bench at
pp512 and tg128, three repetitions each. The JSON files preserve the three
individual throughput samples.

The CUDA build of `33aeb17` measured the following on the RTX 4090 with
driver 580.95.05. Figures are mean ± sample standard deviation in tokens/s
over three repetitions. The host used E-cores 24–27, with batch and ubatch
512, F16 K/V and flash attention set to auto.

| Model | pp512 | tg128 |
| --- | ---: | ---: |
| Llama 3.2 1B Instruct | 55,059.74 ± 9,262.76 | 758.24 ± 8.79 |
| Inkling 2 Mini | 3,506.31 ± 69.86 | 77.96 ± 0.21 |

The individual Llama pp512 samples were 44,364.5, 60,497.8 and 60,316.9
tokens/s. At context 512, llama.cpp reported 49 of 49 Inkling layers
offloaded and 15,360.37 MiB in the CUDA model buffer. F16 KV buffers used
96.00 MiB and the CUDA compute buffer used 787.86 MiB, before driver overhead.

### All-row long-context comparison

`context-logits.c` writes one raw float32 vocabulary row per input ID in absolute position order. Build it separately against the headers and libraries for each pinned llama.cpp build. Pass the model path, an IDs file, the number of positions and the chunk size. It requests F16 K/V, uses one sequence, and disables full-size sliding caches. The portable Unsloth build needs its backend plugin directory. For each run save the IDs file’s SHA-256 alongside the output and check the GPU layer and cache allocation logs.

The Abla producer takes the model path, the same IDs file, the cache mode (`auto`, `f16` or `q8`) and chunk size. The Q8 mode stores global K/V in 32-channel blocks and sliding K/V in F16. Use the same chunk size in both llama.cpp builds and Abla. The 32,768-position calibration used 256. The 102,400-position reference used 64 to fit the full-attention compute buffer.

```sh
ablac build tools/context-logits.ab -o build/context-logits --fast
cc -O3 -fopenmp -fno-math-errno tools/oracle/compare-context.c -lm -o build/compare-context
tools/gpu-lock.sh -t 3000 -- build/context-logits "$model" "$ids" q8 "$chunk" |
    build/compare-context "$mainline_raw" - "$positions" 200058
```

Run the pipeline with `set -o pipefail`. The comparator streams batches of 16 rows and checks both input lengths. Its TSV columns are position count, rows over KL .04, argmax differences, worst KL, zero-based worst position and maximum absolute logit difference. Run the same comparison with Unsloth logits and the identical mainline reference to obtain the worst-KL limit in `tests/tolerances-inkling-long.txt`. At 102,400 positions one uncompressed reference is 81,943,756,800 bytes. Both candidate streams can go directly into the comparator.

## Long-context retrieval

`node tools/context-quality.mjs make ABLA_GRAD MODEL CORPUS OUT_DIR`
inserts four records into a natural-text corpus with at least 131,000
tokens. It writes exact 32,256-, 65,024- and 130,560-token prompts, with 512
tokens reserved for a greedy answer. The same ID file goes to both engines.
`node tools/context-quality.mjs score RESPONSE` prints badge retrieved
(0/1), facts retrieved (0–3), all facts retrieved (0/1), and all four JSON keys present (0/1). The four
values are scored against the final JSON object in the answer.

Compile `context-greedy.c` against the pinned CUDA llama.cpp from the
project root. Use the source tree at the same revision as the library.

```sh
source tools/oracle/common.sh
libs="$llama_root/cuda/bin"
cc -O2 -I"$llama_source/include" -I"$llama_source/ggml/include" \
    tools/oracle/context-greedy.c -L"$libs" -Wl,-rpath,"$libs" \
    -lllama -lggml -lggml-base -o build/context-greedy
export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
tools/gpu-lock.sh -t 1200 -- build/context-greedy MODEL 32768.ids f16 32768 32256 512 64 > answer.ids
build/context-greedy MODEL answer.ids text > answer.txt
node tools/context-quality.mjs score answer.txt
```

The `text` mode loads the vocabulary only. It also decodes Abla's
output IDs, using the same token pieces and end-of-generation rule.

To compare global cache precision on the same teacher-forced prefix, run
`context-logits` with `global32`, `global16` or `global8`.
These modes preserve the sliding-cache precision selected for the context.
`score-context` writes the greedy ID at each input position and reports
perplexity on the following input IDs. For a prefix of N tokens it scores
N−1 predictions, including the first prediction after BOS.

```sh
cc -O3 -fopenmp -fno-math-errno tools/oracle/score-context.c -lm -o build/score-context
set -o pipefail
tools/gpu-lock.sh -- build/context-logits "$model" "$ids" global16 256 |
    OMP_NUM_THREADS=6 build/score-context "$ids" 32768 200058 global16.choices
```

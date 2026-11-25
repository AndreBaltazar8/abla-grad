# Inkling-2-Mini benchmarks

RTX 4090 (24 GiB), Intel Core i9-13900K, NVIDIA 580.95.05, CUDA 12.9.
The runners pin CPUs 24–27 and take the GPU lock per invocation.
The original target GGUF and llama.cpp mainline pin (`33aeb17`) are in
[models.txt](../tools/oracle/models.txt). The other reference is the Unsloth
`app-b12649-mix-c7059d6-linux-x64-cuda12-portable.tar.gz` build. See
[Serving](serving.md) for the DSpark drafter hash. Abla builds use ablac
0.2.73 with `--fast`. Rows identify the measured build. [Unsloth release](https://github.com/unslothai/llama.cpp/releases/tag/b12649-mix-c7059d6)

## Plain decode by cached depth

128 timed greedy tokens, excluding prefix ingestion. Abla uses Dickens
prefixes and one run per depth; llama-bench uses synthetic prefixes, F16
K/V, CUDA graphs, full offload and the mean of three runs. A dash means
unmeasured.

| Cached tokens | abla-grad `2a12ee8` tok/s | llama.cpp mainline tok/s | Unsloth tok/s |
|---:|---:|---:|---:|
| 0 | 138.32 | 77.90 | 102.85 |
| 8,192 | 128.98 | 74.03 | 95.03 |
| 32,768 | 119.88 | 64.63 | 89.92 |
| 65,536 | 113.73 | — | — |
| 131,072 | 100.50 | 48.88 | 64.95 |

Run from the Nix shell with the [README model path](../README.md#build-and-run).
Use the archived `depth-release` binary or build `2a12ee8`. Step, chunk-256,
fixed 05/06, both speculation caps and all-row 32k/102,400 gates passed.

```sh
h="$HOME/archive/abla-grad-depth-1254"
for n in 0 8192 32768 65536 131072; do
    ctx=$(((n + 169 + 255) / 256 * 256))
    tools/gpu-lock.sh -t 240 -- taskset -c 24-27 "$h/depth-release" decode-bench \
        "$model" "$h/d$n.ids" --ctx "$ctx" --mode graph --n 128 --batch-seed
done
```

The archive contains `run-monitored`, the input IDs and `time-depth-release-d*.log`. The reference commands and JSON live
in `~/src/abla-grad-oracle/tg128-current-main` (0 and 8,192) and
`~/src/abla-grad-oracle/tg128-128k` (32,768 and 131,072).
Set `bench` to either command below. Run the Unsloth command from
`$mix_dir/prebuilt`. Set `batch=512` except mainline at 131,072 uses 64.

```sh
lock="$PWD/tools/gpu-lock.sh"
bench=("$HOME/models/llama.cpp-33aeb17f85869279e3821f746bc645357139bc2c/cuda/bin/llama-bench")
# For Unsloth instead:
# source "$HOME/src/abla-grad-oracle/unsloth-b12649-mix-c7059d6/runtime.sh"
# bench=("$glibc/ld-linux-x86-64.so.2" --library-path "$mix_libpath" "$mix_dir/prebuilt/llama-bench")
"$lock" -- taskset -c 24-27 "${bench[@]}" -m "$model" \
    -p 0 -n 128 -d "$n" -r 3 -t 4 -b "$batch" -ub "$batch" \
    -ngl 999 -ctk f16 -ctv f16 -fa auto -o json
```

## Prompt processing

Synthetic prompts use each engine's own IDs. Abla reports its second timed
sample with chunk 512 and default int8 experts. References use three-run means
with batch and microbatch 512 and F16 KV.

| Prompt tokens | abla-grad `515ce8f` tok/s | llama.cpp mainline tok/s | Unsloth tok/s |
|---:|---:|---:|---:|
| 512 | 5547.60 | 3506.31 | 5474.87 |
| 2,048 | 5377.98 | 2625.59 | 4545.45 |
| 4,096 | 5259.57 | 2349.93 | 4068.31 |

```sh
tools/gpu-lock.sh -t 180 -- taskset -c 24-27 "$grad" prefill-bench "$model" --tokens "$n" --chunk 512 --reps 2
"$lock" -t 180 -- taskset -c 24-27 "${bench[@]}" -m "$model" \
    -p "$n" -n 0 -d 0 -r 3 -t 4 -b 512 -ub 512 -ngl 999 -ctk f16 -ctv f16 -fa auto -o json
```

The synthetic samples are in `~/src/abla-grad-oracle/prefill-length/summary.tsv`.
The long runs below use identical fixed IDs across engines. Results are the
median of three runs through 32k and the midpoint of two at 128k. Chunk size
is 512 except mainline at 128k uses 32. The Abla column uses the int8 projection
benchmark snapshot (production implementation `4bae40a`, timed on a patched
`e2a0fdf` snapshot). Reference rows use F16 KV.

| Prompt tokens | abla-grad int8 projection tok/s | llama.cpp mainline tok/s | Unsloth tok/s |
|---:|---:|---:|---:|
| 8,192 | 4,776.15 | 2017.35 | 3673.16 |
| 32,768 | 4,152.54 | 1213.45 | 3192.81 |
| 131,072 | 2,365.08 | 340.86 | 2602.87 |

Use the archived fixed-ID drivers. Each locks the GPU around the timed engine.

```sh
p="$HOME/src/abla-grad-oracle/prefill-1256"
g="$HOME/archive/abla-grad-prefill-gemm-cx5.PYyfewbp"
# Set n to 8192, 32768 or 131072; use reps=2 at 131072, otherwise 3.
# Set batch=32 for mainline at 131072, otherwise 512.
"$p/run-pp.sh" mainline "$n" "$batch" "$reps"
"$p/run-pp.sh" unsloth "$n" 512 "$reps"
"$g/measure.sh" x5 "repeat-x5-64" "$n"
```

The expert GEMM pass (`18752f2`) adds Q5_K int8 MMA, a three-block target
for NVFP4, and fused NVFP4 gate/up plus SiTU. Both arms were built from
`cf6578c` with the same fixed IDs, chunk 512 and 64-token expert tiles.
The sample counts are the same as in the preceding table.

| Prompt tokens | Before tok/s | Q5_K only tok/s | Full pass tok/s | Change |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 4,771.38 | 4,858.75 | 5,178.37 | +8.53% |
| 32,768 | 4,083.60 | 4,101.45 | 4,375.06 | +7.14% |
| 131,072 | 2,636.43 | 2,635.70 | 2,755.95 | +4.53% |

The full pass matched the grouped CUDA CPU-reference checks. Inkling
prefill passed 324 positions with zero argmax differences. Llama passed
266 positions with one argmax difference. All positions met the gates'
tolerances. The logs are under `$g/prefill-r2-main-*.log`. Repeat with
`"$g/measure.sh" prefill-r2-main-fused-tuned repeat-fused-tuned "$n" --moe-tile 64`.

## Serve TTFT

First streamed output on `9aaccb1`, with Inkling and DSpark already loaded,
no prefix reuse, 512-token prefill chunks, context capacity prompt + 256.
The timed endpoint is Abla's server; the two reference binaries above were
benchmarked for decode and prefill.

| Rendered prompt tokens | First streamed output (s) |
|---:|---:|
| 8,192 | 3.55 |
| 32,768 | 10.30 |
| 131,072 | 53.95 |

```sh
ttft="$HOME/src/abla-grad-oracle/serve-ttft-1261"
# grad is the absolute path of the abla-grad binary built at 9aaccb1.
tools/gpu-lock.sh -- taskset -c 24-27 "$ttft/run-ttft.sh" "$n" 18082 "$grad" repeat-batch
```

The harness warms up once, sends the matching `chat$n.json` and records SSE
and the server log. Source results are `raw/ttft-batch-d*.server.log`.

## Serve replay

Same private eight-request capture, original GGUF and DSpark. Session HTTP
is the sum of the requests and excludes model load. Other columns describe
the final request. Request 5 stopped after 261 tokens on `b22b960`
and hit the 512-token limit from `cf6578c` through `45da9c5`. The title request produced 167
tokens on `2ea30b6`, down from 198 on `86df7a9`.

| Main landing | Prompt tokens | Reused tokens | Generated tokens | First raw token (s) | Decode (tok/s) | Final HTTP (s) | Session HTTP (s) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| `ab55314` | 74917 | 56660 | 202 | 13.820 | 62.11 | 17.073 | 65.086 |
| `8af8468` | 74917 | 56660 | 202 | 12.943 | 65.74 | 16.016 | 65.097 |
| `4bae40a` | 74917 | 56663 | 202 | 12.657 | 65.84 | 15.725 | 60.803 |
| `b22b960` | 74917 | 56657 | 202 | 11.945 | 67.34 | 14.945 | 59.208 |
| `cf6578c` | 74917 | 56656 | 202 | 10.838 | 101.22 | 12.834 | 55.565 |
| `9aaccb1` | 74917 | 56656 | 202 | 9.602 | 101.44 | 11.593 | 47.226 |
| `6ef2608` | 74917 | 56656 | 202 | 9.354 | 101.30 | 11.349 | 45.760 |
| `9785f44` | 74917 | 56662 | 202 | 9.040 | 105.41 | 10.956 | 45.178 |
| `86df7a9` | 74917 | 56662 | 202 | 9.026 | 105.48 | 10.941 | 45.138 |
| `2ea30b6` | 74917 | 56662 | 202 | 9.005 | 108.13 | 10.874 | 45.100 |
| `45da9c5` | 74917 | 56662 | 202 | 9.353 | 107.43 | 11.233 | 47.126 |

```sh
capture="$HOME/archive/abla-grad-serve-replay/capture/requests.jsonl"
tools/serve-replay-landing 9aaccb1 "$capture" "$model" "$new_results_dir" 18081
```

Capture SHA-256 is `db68928941920ea8fbdf486c13c930b51cbd0cc1d3b0df73edf2edfc9515e478`.
See `~/archive/abla-grad-serve-landings/<sha>/` for per-request results and
`~/archive/abla-grad-serve-replay/results.md` for the initial run.

## Recording new results

Update the matching table with verified results on a named commit. Append
the landing tool's row for a fixed replay. Record the date, pin, workload,
flags, sample count/statistic, GPU, gate outcome and raw artifact path in
one short note if they differ from the existing setup. Use a dash for a
missing cell. Keep exploratory runs, profiles and private captures in the
local artifact archive. Older results remain in Git history.

## Long-context retrieval by KV precision

One fixed Inkling UD-Q4_K_M fixture per context on the RTX 4090. Both
engines received identical prompt token IDs. Four records at roughly
6%, 28%, 53% and 79% of each prompt supply a badge lookup and a
three-fact question that joins a dispatch, ticket and site directory.
A deterministic scorer checks the final JSON, with 512 tokens reserved
for greedy generation. Table cells show badge retrieval and facts
retrieved.

| Input tokens | Abla global F32 | Abla global F16 | Abla global Q8 | llama.cpp F16 | llama.cpp Q8_0 |
|---:|:---:|:---:|:---:|:---:|:---:|
| 32,256 | 1/1, 3/3 | 1/1, 3/3 | 1/1, 3/3 | 1/1, 3/3 | 1/1, 3/3 |
| 65,024 | 1/1, 3/3 | 1/1, 3/3 | 1/1, 3/3 | 1/1, 3/3 | 1/1, 3/3 |
| 130,560 | Unscored (out of memory) | 1/1, 3/3 | 1/1, 3/3 | 1/1, 3/3 | 1/1, 3/3 |

Abla used `cf6578c`, target-only graph decode and 512-token prefill
chunks. Sliding KV stayed F16 while global KV varied. Abla's Q8 stores
a block of 32 signed bytes with an FP32 scale. llama.cpp `33aeb17`
used its selected F16 or Q8_0 format for both global and sliding KV,
with batch/microbatch 64. At 128k its F16 run required batch/microbatch
32 after CUDA graph instantiation exhausted memory at 64. Abla's 128k
global F32 attempt failed in KV allocation with `cuMemAlloc` status 2.

The ID hashes, raw answers and launch commands are in
`~/archive/abla-grad-context-quality/`. Generate the fixture and score
answers with `tools/context-quality.mjs`. Run Abla with
`decode-bench MODEL IDS --mode graph --ctx CONTEXT --batch-seed --f16-kv --global-kv 8 --n 512 --log answer.ids`
under the GPU lock, selecting global KV 8, 16 or 32. Decode answer IDs
with the vocabulary-only mode of `tools/oracle/context-greedy.c`.
These results cover the fixed lookup and fact-join fixture at each depth.

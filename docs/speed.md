# Inkling-2-Mini decode and prefill speed (RTX 4090)

Goal: beat mainline llama.cpp and Unsloth's build at every depth and in prefill, with `tools/gate-inkling.sh` passing; stretch line 200 tok/s decode.

## Current plain decode through 128k

RTX 4090, Inkling UD-Q4_K_M, 128 timed greedy tokens. The Abla rows use
the same Dickens prefixes and exclude loading, graph capture and prefix
ingestion. All use CUDA C RMSNorm and depth mix. Global attention uses
Q8 and 64 partitions from an 8,192-position allocation.

| Plain tg128, tok/s | d0 | d8,192 | d32,768 | d65,536 | d131,072 |
|---|---:|---:|---:|---:|---:|
| Initial grouped-Q8 run | 137.69 | 123.76 | 115.25 | 108.60 | 92.50 |
| Confirmation before prefill GEMM change | 138.69 | 126.78 | 119.09 | 110.29 | 95.90 |
| Release `cf6578c` | **138.57** | **126.72** | **119.01** | **111.35** | **95.58** |
| Pinned Unsloth CUDA mix | 102.85 | 95.03 | 89.92 | — | 64.95 |

Each Abla cell is one unprofiled run. The release retained 85.9% of d0 throughput at 32k and 85.8% of 64k
throughput at 128k. It was 0.99 tok/s short of 120 at 32k. The confirmation
before the prefill GEMM change reproduced all 128 generated IDs of the
previous plain implementation at 128k. With the new prefill arithmetic,
the generated sequence first differed from that confirmation on token eight.
The Unsloth results are three-run means with llama-bench synthetic
inputs. The 64k reference was not measured in this series. Per-family
timings, bandwidth and precision comparisons are in the
[global-attention measurements](#what-grows-during-plain-decode).

## Earlier short-context measurements

Method: ours = `abla-grad decode-bench <gguf> <ids> --n 128 --mode graph` (greedy, N cached tokens then 128 generated; the cached tokens are prompt 04's ids cycled, so the text is junk but the routing realistic); gate = `GATE_LOGITS_FLAGS=--step tools/gate-inkling.sh`. llama.cpp mainline 33aeb17 and Unsloth b12649-mix-c7059d6: llama-bench tg128 -d N, same UD-Q4_K_M, CUDA graphs, F16 KV (ablac-df, raw data in ~/src/abla-grad-oracle/).

| tg128, tok/s | d0 | d15 | d300 | d512 |
|---|---:|---:|---:|---:|
| llama.cpp mainline 33aeb17 | 74.43 | 78.10 | 77.50 | 77.29 |
| Unsloth b12649-mix-c7059d6 | 104.06 | 106.58 | 106.15 | 105.50 |
| abla-grad a02be45 | 96.5 | 95.7 | 81.5 | 73.8 |
| abla-grad + split attention, 8 blocks/head (4c1ee1f) | | 98.9 | 96.1 | 93.7 |
| abla-grad, 16 blocks/head (default now) | | 98.7 | | 95.7 |
| abla-grad parallel controllers (85b7bee, fixture 06) | 105.39 | 105.63 | 103.77 | 102.27 |

| Depth 0, tok/s | pp512 | tg1024 |
|---|---:|---:|
| Unsloth b12649-mix-c7059d6 | 5,474.87 | 103.96 |
| llama.cpp mainline 33aeb17 | 3,506.31 (prior run) | |
| abla-grad c01d3f3 | ~3,550 ([prefill benchmark](prefill.md)) | |
| abla-grad prefill + int8 experts (34dfad6, ablac-90) | 5,545 (gate 324 positions, 1 argmax difference, worst KL 0.0144) | |
| abla-grad prefill 22b35db | 4,685 (`prefill-bench --tokens 512`, gate with `--prefill --chunk 256`: 0 argmax differences in 324, worst KL 0.0094) | |
| abla-grad prefill ec2366d | 4,150 (`prefill-bench --tokens 512 --moe-tile 64`, gate `GATE_LOGITS_FLAGS="--prefill --chunk 256"`: 0 to 1 argmax differences in 324, worst KL 0.0094) | |

The llama-bench results are means of three repetitions on E-cores 24–27.
`-d N` fills N tokens of KV cache before timing; `tg1024 -d 0` averages
a 1,024-token generation starting with an empty cache. Unsloth used its
verified CUDA 12 portable prebuilt (compiled commit `189f431bf`). Both
llama-bench builds had access to the full 4090 and used no draft model.

The Unsloth logits check reused the pinned CUDA oracle’s four prompts
and matched all 324 argmax IDs. Maximum absolute logit difference was
10.9349, the largest row mean absolute difference was 2.9223, and
maximum KL was 0.013805. All three passed the Inkling tolerances.
Raw benchmark samples are in
`~/src/abla-grad-oracle/artifacts/bench-depth-33aeb17/` and
`~/src/abla-grad-oracle/unsloth-b12649-mix-c7059d6/`.

## Prefill by prompt length

Prompt processing in tokens/s on the RTX 4090 with Inkling UD-Q4_K_M.

| Engine or expert path | pp512 | pp2048 | pp4096 |
|---|---:|---:|---:|
| llama.cpp mainline `33aeb17` | 3506.31 | 2625.59 | 2349.93 |
| Unsloth `b12649-mix-c7059d6` | 5474.87 | 4545.45 | 4068.31 |
| abla-grad `515ce8f`, f16 experts | 4941.15 | 4838.67 | 4766.66 |
| abla-grad `515ce8f`, default int8 experts | 5547.60 | 5377.98 | 5259.57 |

The llama.cpp rows are three-run means from `llama-bench -p N -n 0`
with batch and microbatch 512 and F16 KV. Their pp512 values come from
the earlier reference runs. Both abla-grad rows use F32 KV and report
the second timed sample from `prefill-bench --tokens N --chunk 512 --reps 2`.
The f16 expert row adds `--no-int8-experts`. Each benchmark generates
its own synthetic input IDs. New runs held the GPU lock separately
with CPU affinity 24–27.

Raw samples and commands are in `~/src/abla-grad-oracle/prefill-length/`.

## Plain graph decode through depth 16,384

tg128 in tokens/s on the RTX 4090 with the same Inkling UD-Q4_K_M GGUF.

| Cached depth | abla-grad `2029657` | llama.cpp `33aeb17` | Unsloth `b12649-mix-c7059d6` |
|---:|---:|---:|---:|
| 0 | 133.80 | 77.90 | 102.85 |
| 512 | 129.29 | 75.78 | 103.91 |
| 1,024 | 125.33 | 74.00 | 100.76 |
| 2,048 | 119.77 | 75.77 | 99.22 |
| 4,096 | 119.47 | 75.29 | 97.56 |
| 8,192 | 117.31 | 74.03 | 95.03 |
| 16,384 | n/a (device memory for F32 context) | 71.34 | 94.81 |

abla-grad ran `decode-bench --mode graph --n 128` once per depth with
F32 KV. Input was the first depth-plus-one IDs of the pinned natural
text fixture, including one seed. Context capacity was
`depth + 129` rounded up to 256. The two reference builds averaged
three `llama-bench -d N -n 128` repetitions with synthetic tokens,
F16 KV, batch and microbatch 512, CUDA graphs, and full GPU offload.
The timers exclude prefix ingestion. Each invocation held the GPU
lock separately with CPU affinity 24–27.

The stock build exited before timing at d16384. At a
context of 16,513, its model-load probe reported insufficient
device memory with 22,076 MiB free.

Exact input IDs, commands, raw samples and any failures are in
`~/src/abla-grad-oracle/tg128-current-main/`.

## Decode and first-token latency through depth 131,072

The historical capacity measurements below precede the [five-depth grouped-attention results](#grouped-attention-through-131072-cached-tokens).

The RTX 4090 used the original Inkling UD-Q4_K_M GGUF for all four
runs at each depth. Decode rates are in tokens/s for 128 timed tokens.

| Cached depth | abla-grad plain graph | abla-grad adaptive DSpark | llama.cpp mainline | Unsloth CUDA mix |
|---:|---:|---:|---:|---:|
| 8,192 | 117.21 | 116.80 | 74.03 | 95.03 |
| 32,768 | 92.71 | 85.78 | 64.63 | 89.92 |
| 131,072 | 35.44 | 35.44 | 48.88 | 64.95 |

abla-grad `56d0e48` used Dickens prefixes of depth-plus-one IDs, with
one seed ID. Its plain benchmark batched the untimed prefix in a
scratch build with the same model and timed graph. The adaptive run
used stock `spec-bench --adaptive --nmax 2 --n 129`. The first token
comes from prompt ingestion and the timed loop generates the next 128.
The capacity build uses F32 KV for the 8,192 and 32,768 rows and F16
sliding rings with block-Q8 global KV for the 131,072 row. The two
abla-grad modes use different prompt ingestion paths. Their generated
streams shared 30 initial IDs at depth 8,192, shared 48 initial IDs at depth 32,768, shared 7 initial IDs at depth 131,072. Each adaptive run starts a fresh controller and includes
its exploration. At 32,768, the controller switched four times and
produced 100 plain tokens and 28 tokens from 18 speculative cycles.

The reference builds, mainline `33aeb17` and Unsloth
`b12649-mix-c7059d6`, used their synthetic prefixes, F16 KV, CUDA
graphs, and full GPU offload. Each result is the mean of three
`llama-bench -d N -n 128` repetitions, with batch and microbatch 512
for the original configurations. mainline at depth 131,072 used batch 64 and microbatch 64. All runs used CPU affinity 24–27
and one GPU lock per invocation.

| Rendered chat prompt tokens | abla-grad server TTFT (s) |
|---:|---:|
| 32,768 | 28.68 |
| 131,072 | 66.45 |

TTFT uses stock `serve` at the same pin, context capacity 256 tokens
above each prompt length, and the first emitted server token event.
The server was model-warm from a short request and reused zero prompt
tokens for each timed request. It chose prefill chunks of 128 at 32,768
and 512 at 131,072. The timer includes request parsing,
rendering, tokenization, and prefill. Model loading and network
transport are outside the timer.

Raw commands, exact IDs, traces and the untimed seed patch are in
`~/src/abla-grad-oracle/tg128-128k/`.

## Long-context prefill on identical prompts

Prompt tokens/s on RTX 4090 using the original Inkling UD-Q4_K_M and
the same fixed token IDs for all three engines. The 8,192-token prompt
is a Dickens prefix. The 32,768- and 131,072-token prompts are the
rendered chat requests used for the TTFT measurements above.

| Prompt tokens | abla-grad `56d0e48` | llama.cpp `33aeb17` | Unsloth `b12649-mix-c7059d6` |
|---:|---:|---:|---:|
| 8,192 | 4620.34 | 2017.35 | 3673.16 |
| 32,768 | 4036.87 | 1213.45 | 3192.81 |
| 131,072 | 2252.38 | 340.86 | 2602.87 |

The timers include prefill and the last token's logits. Each engine
warms up on one chunk and clears the sequence before the measured
runs. The table reports the median of three runs through 32k and the
midpoint of two at 128k. Chunk size is 512 except mainline at 128k,
which uses batch and microbatch 32 to fit its prefill workspace. Both
llama.cpp builds use F16 KV. abla-grad uses F32 at 8k and 32k and
F16 sliding KV with Q8 global KV at 128k.

Nsight Systems traced one abla-grad 128k prefill with 512-token chunks.
The profiled prefill took 58.90 s wall time.

| GPU work | Launches | Kernel seconds | Share of kernel time |
|---|---:|---:|---:|
| Global attention (Q8) | 2,048 | 30.12 | 51.62% |
| Matrix products | 151,553 | 20.67 | 35.43% |
| Other kernels | 456,707 | 5.36 | 9.19% |
| Sliding attention (F16 ring) | 10,240 | 2.15 | 3.69% |
| KV writes | 12,288 | 0.04 | 0.07% |

From the first measured embedding launch through the final kernel,
58.35 of 58.90 s were GPU-active. The GPU timeline had
0.55 s of gaps, including 0.01 s immediately before
embedding launches at chunk boundaries after staging copies. CUDA
copies accounted for 0.20 ms.

Raw samples, token IDs, reference driver and Nsight artifacts are in
`~/src/abla-grad-oracle/prefill-1256/`.

### Shared K/V staging for 128k global prefill

On a full 512-token prefill chunk, one CTA stages each global Q8 K/V
tile for all three query heads in its group. Each head still uses
two warps for its QK, softmax and value product. Chunks below 256 tokens and
other head ratios use the existing per-head launch.

Matched screen on `56d0e48` with the same exact prompts and harness above.

| Prompt tokens | Baseline pp (tok/s) | Shared-K/V pp (tok/s) |
|---:|---:|---:|
| 8,192 | 4620.34 | 4632.13 |
| 32,768 | 4036.87 | 4037.89 |
| 131,072 | 2252.38 | 2573.48 |

| Prompt tokens | Baseline TTFT (s) | Shared-K/V TTFT (s) | Prefill chunks, baseline / candidate |
|---:|---:|---:|---:|
| 32,768 | 28.60 | 28.81 | 128 / 128 |
| 131,072 | 66.93 | 61.00 | 512 / 512 |

At 128k, prompt throughput rose 14.26% and
TTFT fell 8.86%.
Both lengths used a warmed server, zero prompt reuse and
identical streamed outputs across the two builds. All 1,624
forced-Q8 prompt 05 logit rows matched byte for byte. A paired
640-token forced-Q8 check on the later int8 GEMM base also matched.

## Cold server first-token latency

The older 32k server used a 128-token prefill chunk under device
memory pressure. On the exact prompt, standalone prefill on
`b22b960` took 20.56 s with chunk 128 and 7.98 s with chunk 512.
The later compact global KV layout let the server use chunk 512
at all three lengths.

On `cf6578c`, a 32k server phase probe took 15.91 s to feed the
prompt. Batched target prefill took 8.29 s, DSpark preparation
took 3.31 s and the 193-token exact tail took 4.31 s.
Workspace selection took 1.1 ms and the
checkpoint copies took 2.7 ms. Standalone
prefill with the same IDs, build and chunk took 8.02 s.

The exact tail now replays the existing target step graph with each
known token. Its final argmax supplies the first completion token.
DSpark also rotates each group of contiguous prompt K rows with one
launch and stores the group’s K/V with the existing batched writer.

| Prompt tokens | Baseline (s) | Graph exact tail (s) | Graph + batched DSpark (s) |
|---:|---:|---:|---:|
| 8,192 | 7.32 | 4.24 | 3.55 |
| 32,768 | 16.44 | 13.45 | 10.30 |
| 131,072 | 58.11 | 55.93 | 53.95 |

These are first streamed output times on the RTX 4090. Each run
loaded the same Inkling model and Q8_0 DSpark, warmed up on a short
request, then served the exact prompt with chunk 512 and zero reuse.
Context capacity was 256 tokens above each prompt length. The
16-token streamed completions matched at all three lengths.
Raw first-token times, prompt IDs and phase logs are in
`~/src/abla-grad-oracle/serve-ttft-1261/`.

## Decode and speculation across cache depths

Throughput on the RTX 4090 in tokens/s on the pinned Inkling UD-Q4_K_M.
abla-grad `5104a3f` used the same fixture 06 prefix in every mode, with
cached depth equal to the input ID count minus one. All three used
F32 KV, step seeding, and context capacity equal to input IDs + requested
tokens + 28. The speculative rows used the Q8_0 DSpark drafter with two
proposals and a replay batch of four cycles. Adaptive switches between
that cycle and plain decode.
Each cell is one GPU-locked run on E-cores 24–27. Timing starts after
seeding. Throughput divides the requested count by elapsed time to exclude
extra tokens from the final replay batch. The adaptive timer includes
recording the plain graph.

| tg128 | d0 | d512 | d1024 | d2048 | d4096 |
|---|---:|---:|---:|---:|---:|
| abla-grad plain graph | 130.06 | 126.44 | 122.06 | 120.62 | 117.74 |
| abla-grad DSpark cycle graph | 115.20 | 110.62 | 111.37 | 103.62 | 90.72 |
| abla-grad DSpark adaptive | 113.43 | 108.43 | 105.24 | 104.18 | 101.61 |

| tg1024 | d0 | d512 | d1024 | d2048 | d4096 |
|---|---:|---:|---:|---:|---:|
| abla-grad plain graph | 126.98 | 123.11 | 121.30 | 120.33 | 117.22 |
| abla-grad DSpark cycle graph | 118.01 | 157.89 | 157.21 | 159.62 | 136.03 |
| abla-grad DSpark adaptive | 123.18 | 162.34 | 153.73 | 146.94 | 140.99 |

Plain step-mode validation on fixture 05 found 28 of 1,624 rows above KL
0.04. Mainline CPU versus CUDA found 42 and Unsloth CUDA versus mainline
CUDA found 36. The 4,300-position fixture 06 gate reported 25 rows over
0.04, 97 argmax differences and worst KL 0.209, within the measured
CPU/CUDA noise floor of 35, 111 and 0.230.

### llama-bench reference

The following means are from three CUDA graph runs with synthetic tokens
and F16 KV. Both llama-bench builds ran without a drafter.

| Generated tokens | Engine | d0 | d512 | d1024 | d2048 | d4096 |
|---:|---|---:|---:|---:|---:|---:|
| 128 | llama.cpp 33aeb17 | 77.39 | 77.97 | 76.90 | 75.58 | 74.92 |
| 128 | Unsloth b12649-mix-c7059d6 | 105.27 | 105.53 | 104.13 | 102.62 | 100.74 |
| 1024 | llama.cpp 33aeb17 | 77.80 | 76.58 | 75.92 | 75.58 | 74.85 |
| 1024 | Unsloth b12649-mix-c7059d6 | 104.62 | 103.74 | 102.94 | 102.56 | 100.78 |

### DSpark on fixed prompts

Tokens per cycle include the target token and accepted proposals. For adaptive
runs, the acceptance column only counts tokens produced by speculative
cycles. The speculative CLI seeds one anchor in advance. ID checks
align the logged sequences after that anchor.

| Prompt (cached depth) | Requested | Plain tok/s | Cycle tok/s | Cycle tokens/cycle | Adaptive tok/s | Adaptive tokens/cycle | Unsloth DSpark n2 tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 03 (14) | 128 | 131.30 | 154.85 | 2.36 | 145.97 | 2.58 | 126.31 |
| 03 (14) | 1024 | 126.64 | 168.51 | 2.62 | 167.16 | 2.66 | — |
| 05 (1623) | 128 | 121.15 | 104.86 | 1.75 | 110.46 | 1.75 | 82.51 |
| 05 (1623) | 1024 | 120.73 | 154.49 | 2.56 | 152.23 | 2.72 | — |

The aligned greedy logs agree through every compared ID in 25 of
34 mode comparisons. On the 1,024-token prompt 05 run, cycle-only
matched all 1023 aligned IDs and adaptive shared a 504-ID prefix.

### DSpark on matched cached prefixes

Both engines used identical depth-plus-one prefixes of fixture 05.
Unsloth values are means of three server completions. Unsloth used F16 KV
and abla-grad used F32 KV. Each engine used its native generation timer.

| Cached depth | abla-grad plain | abla-grad cycle | abla-grad adaptive | Unsloth plain | Unsloth DSpark n2 |
|---:|---:|---:|---:|---:|---:|
| 0 | 131.16 | 121.25 | 127.15 | 100.07 | 94.20 |
| 15 | 130.98 | 109.85 | 113.36 | 100.64 | 87.83 |
| 300 | 127.92 | 150.95 | 154.33 | 99.44 | 125.16 |

Raw timings, emitted IDs, and reproduction commands are in
`~/src/abla-grad-oracle/spec-5104a3f/`. The Unsloth server samples are in
`~/src/abla-grad-oracle/dspark-b12649/`.

## Decode profiles at d15 and d512

On the same 4090, CUPTI recorded 128 decode steps per depth with zero
dropped records. abla-grad `9a46740` used direct mode and context capacity
700. Its forward pass uses the same step-buffer kernels as graph replay.
Graph replay adds the two device-side token feedback kernels. CUPTI and
Nsight node profiling both crashed at abla-grad graph launch, so its
direct kernel durations are a screen for hot kernels. The untraced d15
graph run measured 98.78 tok/s. Direct launch cadence and profiling can
change clock and cache behavior. Unsloth used its CUDA graphs and synthetic
llama-bench token stream.

| Engine | Depth | GPU kernel ms/token | Kernels/token | GPU busy ms/token | Gap ms/token |
|---|---:|---:|---:|---:|---:|
| abla-grad direct | 15 | 10.719 | 2,613 | 10.720 | |
| abla-grad direct | 512 | 10.980 | 2,613 | 10.981 | |
| Unsloth graph | 15 | 9.335 | 3,281 | 9.393 | 1.137 |
| Unsloth graph | 512 | 9.409 | 3,281 | 9.467 | 1.228 |

GPU busy is the union of kernel and copy intervals. Gap is the rest of
the timed decode window. abla-grad direct mode synchronizes for host
feedback per token, so its gaps are omitted.

| Kernel or group | abla-grad d15 → d512, GPU ms/token (launches/token) | Unsloth d15 → d512, GPU ms/token (launches/token) |
|---|---:|---:|
| Q8_0 matvec | 2.626 → 2.598 (289) | 2.549 → 2.552 (293) |
| Indexed NVFP4 matvec / fused NVFP4 matvec | 1.935 → 1.920 (88) | 1.806 → 1.807 (81) |
| RMSNorm | 1.240 → 1.190 (434) | 0.547 → 0.543 (342, two largest variants) |
| F32 matvec | 1.173 → 1.141 (336) | 0.799 → 0.793 (343, two largest variants) |
| Depth mix | 1.126 → 1.082 (120) | 0.448 → 0.428 (86) |
| Convolution | 0.452 → 0.435 (192) | 0.224 → 0.224 (176, conv chain) |
| Relative split attention + merge / flash attention | 0.302 → 0.823 (96) | 0.161 → 0.254 (48) |

The controller accounts for 576 kernel launches per token. At d15 its
two norms take 0.257 GPU ms/token, the down projection takes 0.615, and
gain, forget and mix together take 0.468. The whole chain takes 1.339 ms
at d15 and 1.296 ms at d512.

Exact symbols, launch counts and raw traces are in
`~/src/abla-grad-oracle/profile-9a46740/`.

## Log
- a02be45: graph decode, e7's router/RMSNorm/matvec work. Per-token cost grew with context (one attention block per head walked the context serially).
- split attention: positions of a head over 8 blocks plus a merge (sink once); d512 73.9 -> 93.7, ids identical, gate passes (324 positions, 0 argmax differences).
- Next: F16 K/V and an SWA ring, MTP speculative decoding (acceptance rate first), fusion chains, matvecs to the bandwidth roof.

## Kernel tuning at a02be45

Matched the [300-position Inkling CUPTI profile](inkling-decode-profile.md) on
the RTX 4090 with a context capacity of 512. Each arm used the pinned d926ff4
source with the specified kernel changes. RMSNorm and packed matvec timing
came from separate arms.

| Kernel | Baseline GPU ms/token | Tuned GPU ms/token | Baseline → tuned effective GB/s |
|---|---:|---:|---:|
| MoE routing | 2.429 | 0.241 | |
| RMSNorm | 2.110 | 1.187 | |
| Q8_0 matvec | 3.103 | 2.604 | 601.5 → 716.7 |
| Indexed NVFP4 matvec | 2.453 | 1.914 | 646.3 → 828.2 |

Bandwidth counts useful logical payload. In separate resident-payload
microbenchmarks, Q8_0 at 136 MiB reached 918 GB/s and indexed NVFP4 at
182.25 MiB reached 894 GB/s, against the 4090's rated 1,008 GB/s. Routing
and RMSNorm kept the 300 logit rows byte-identical. The packed matvec arm
passed all 300 positions against llama.cpp with zero argmax differences.

| Matched graph decode, 300-token prefix and 128 generated | tok/s |
|---|---:|
| b5d99be | 70.54 |
| a02be45 | 81.78 |

This single A/B held the context capacity at 512. The cached RMSNorm and
packed loads preserved all 128 generated IDs.

## First generated fusion

Three alternating pairs on the RTX 4090 with the four-token prompt 01,
context capacity 512 and 128 generated tokens. Values are medians in tok/s.
All 324 gate positions produced byte-identical logits across the two paths.

| Generated block | Direct, unfused → fused | Graph, unfused → fused |
|---|---:|---:|
| Indexed NVFP4 gate/up → SiTU | 84.48 → 84.73 | 98.76 → 98.51 |

## DSpark small-row projections

On the RTX 4090, the eight-row draft block and three-row tap injection use
small-row matvecs. The recorded head reads the shared output matrix once for
both proposals. Q, K and V prepare in parallel, and the first Markov correction
runs alongside the shared output projection.

Recorded proposal time at depth 15 fell from 3.309 to 1.365 ms over 40
CUPTI-traced replays. The following `spec-bench --prefill --n 128` runs compare
main `0e9214d` with the draft changes. The complete generated ID logs matched
byte-for-byte, including the final batch.

| Prompt and depth | Baseline ms/cycle | Tuned ms/cycle | Baseline tok/s | Tuned tok/s |
|---|---:|---:|---:|---:|
| 03, depth 15 | 18.12 | 15.73 | 130.08 | 149.89 |
| 05, depth 1624 | 19.30 | 16.96 | 83.55 | 95.10 |

## GPU etiquette
Every model run, gate and benchmark goes through `tools/gpu-lock.sh [-t hold-seconds] -- command...` (flock on /tmp/abla-grad-gpu.lock, waits for 20 GB free, hard hold timeout, one run per hold). Timings taken next to another run are not trustworthy.
- attention splits 16 (--splits N): d512 93.7 -> 95.7, d15 unchanged; 32 and 64 no better. The context cost is now ~0.3 ms/token between d15 and d512, near Unsloth's 0.1; what remains is kernel time elsewhere (RMSNorm, F32 matvec, depth_mix, copy_f32).

### Depth mix and attention input normalization

On the 4090, prompt 04 with 300 cached positions and 128 graph-decoded
tokens went from 119.3 to 128.2 tok/s with identical generated IDs.
The layer saves 96 launches per token by grouping the three attention-input
norms in one launch, or using one shared norm for sliding-window
visits without the loop controller. Total recorded kernel launches went from 2,235 to 2,139 per token.

The decode and batched depth-mix entries share a warp-per-state scoring
implementation. The decode launch uses 1,024 threads and the batch launch
uses 256, which also performed well at 128 rows. CUDA event measurements on
resident synthetic inputs (width 2,304, three queries) give the median
microseconds per launch below. The count includes only bank states, with
one extra current state in each case.

| Bank states | Decode before → after | 3-row batch before → after | 128-row batch before → after |
| ---: | ---: | ---: | ---: |
| 1 | 5.95 → 3.38 | 5.05 → 3.87 | 7.87 → 6.39 |
| 6 | 16.95 → 3.85 | 6.17 → 4.40 | 13.65 → 12.82 |
| 12 | 29.30 → 4.40 | 12.53 → 7.83 | 21.91 → 20.99 |
| 24 | 54.14 → 7.19 | 22.46 → 13.28 | 39.36 → 36.70 |

On the 512-token model prefill benchmark, the five-run median was 4,682
tok/s before and 4,669 tok/s after (−0.28%).

## Host replay of a suffix-derived branch

At `3fe9947`, `spec-accept --nmax 4 --trace FILE` logged DSpark's
four proposals at each anchor of a plain greedy continuation. The replay
compared a four-proposal chain with a five-row tree. Both include the
anchor. The tree keeps the first three DSpark proposals and can spend the
fourth edge on an alternative retrieved from the committed prefix.
Lookup uses the longest matching suffix of 2–32 tokens, breaks ties by
recency, and takes the first differing edge after the shared prefix.
If there is no eligible branch it keeps the full chain. Each policy
advances by its own committed length after each verification.

| Prompt | Anchors | Chain, tokens/cycle | Suffix branch, tokens/cycle | Optimistic one-sibling bound |
|---|---:|---:|---:|---:|
| 03, multilingual greeting | 1,020 | 3.148 | 3.938 | 4.378 |
| 05, Alice | 1,020 | 3.481 | 3.764 | 4.232 |
| Fog explanation, chat with thinking | 508 | 1.946 | 1.961 | 3.006 |
| Museum scene, chat with thinking | 508 | 2.330 | 2.341 | 3.320 |
| Ferry clock, original story continuation | 508 | 2.363 | 2.442 | 3.342 |

The last column grants perfect foresight. At every position it can repair
the first draft miss with the correct sibling, end the path there, or take
any shorter prefix. Dynamic programming minimizes total cycles under the
five-row budget. A tree with descendants on multiple branches would need
a separate analysis. These figures count accepted tokens on a fixed trace.
They exclude tree execution costs and verifier-induced near-tie changes.

Suffix branching saves two of 261 cycles on fog, one of 218 on the museum
scene, and seven of 215 on the ferry continuation. Fixture 03 repeats its
greeting, which makes its larger gain a poor estimate for original prose.
Exact prompt text and IDs, traces, replay scripts and baseline R5
`spec-bench` logs are in `~/archive/abla-grad-retrieval-1244/`.

## Native BF16 reads for exact router and controller weights

On the RTX 4090, the matrix-only packed GGUF saved 0.097 and 0.084 ms/token
at d15 and 0.092 and 0.068 ms/token at d512 in two counterbalanced file
pairs. Each graph decode-bench run generated 128 tokens from the same fixed
prefix with context capacity 700. Timed generation excludes loading. All
128 generated IDs matched the original at each depth. The native reader
preserves 70,778,880 bytes of weight storage savings on the device. The
converter, model hashes, per-arm rates and full-logit comparisons are in
[the GGUF packing measurements](gguf-packing.md).

## Small-row verification on the RTX 4090

The table follows the five changes in commit order, from the same Q4_K_M
target and Q8_0 DSpark drafter. Each row is a full fixed-four
`spec-bench --n 256 --nmax 4` run. Cycle time includes drafting,
verification and commit. Fixture 03 produced the same 256 IDs in 87
cycles on every build.

| Cumulative change | 03 tok/s | 03 ms/cycle | 05 tok/s | 05 ms/cycle | 05 cycles |
|---|---:|---:|---:|---:|---:|
| Starting point, `235db08` | 157.6 | 18.60 | 77.9 | 20.72 | 158 |
| Small-row Q8 LM head, `b256489` | 160.1 | 18.31 | 84.1 | 19.20 | 158 |
| Stable parallel expert map, `30cb1e7` | 170.2 | 17.22 | 87.3 | 18.48 | 158 |
| Four-tap convolution with history snapshots, `ca9c8a5` | 173.6 | 16.88 | 88.9 | 18.16 | 158 |
| One block per BF16 projection output, `ffeda30` | 177.2 | 16.54 | 147.0 | 17.34 | 100 |
| Fixed-shape scaled NVFP4 union, `8db7496` | 184.2 | 15.91 | 150.4 | 16.95 | 100 |

The first three changes also preserved the 05 ID stream. The BF16
projection multiplies BF16 weights by FP32 activations with FP32 FMA
and reduces each dot product in one block. The previous split-K tile
rounded inputs to FP16 and combined partial sums. On 05 the two builds
first differ at generated token 20 (one-based). The jump in tok/s
includes the resulting change in draft acceptance. The specialized
NVFP4 union preserved the BF16 build’s fixed-four output IDs on 03,
05, museum and fog.

On 512 generated tokens, museum went from 120.4 to 153.0 tok/s
(228 to 212 cycles, 18.61 to 15.76 ms/cycle). Fog went from 106.9
to 125.4 tok/s (258 cycles on both builds, 18.53 to 15.79 ms/cycle).
At the observed yield these two runs would need about 12.1 and
9.9 ms/cycle respectively to sustain 200 tok/s.

The five-row graph on the museum prompt at position 46 took 17.11 ms
before the changes. With all five, three separate 150-replay runs
took 13.82, 13.85 and 14.03 ms (median 13.85). In a separate diagnostic recording we froze the
expert map at each visit, retained the graph's forks, and omitted one
family at a time. The mean of two frozen-map controls was 15.94 ms.
Omitting the 88 NVFP4 expert kernels removed 6.39 ms. The small-row
F32 matvecs cost 1.69 ms, the original LM head 1.66 ms, the dense Q8
row matvecs 1.56 ms, and the Q5_K/Q8 expert tail 0.78 ms. These
differences measure time removed from the graph and overlap across
branches. The 13.58 ms verifier microbenchmark includes additional
experimental controller changes. It excludes drafting and commit.

### Adaptive generation

Each cell is tok/s for one 512-token run on the same GPU and prompt
IDs, pinned to cores 24–27. The plain rows use graph `decode-bench`.
Speculation uses `spec-bench --adaptive --n 512 --nmax 2|4` with the
product default of two proposals in the first adaptive column.
The controller times its arms and can make different choices across runs.

| Prompt | Plain, before → after | Adaptive 2, before → after | Adaptive 4, before → after |
|---|---:|---:|---:|
| 03, multilingual greeting | 131.4 → 131.5 | 167.8 → 200.3 | 162.8 → 192.0 |
| 05, Alice | 122.6 → 123.0 | 121.0 → 148.8 | 138.4 → 147.8 |
| Museum, chat with thinking | 131.3 → 131.4 | 138.4 → 158.5 | 130.5 → 159.2 |
| Fog, chat with thinking | 130.4 → 130.3 | 128.7 → 142.3 | 128.5 → 134.1 |

All four plain runs preserved every generated ID. At nmax 2 the optimized
verification graph took 244 speculative cycles and 16 plain steps on
museum, and 275 speculative cycles and 16 plain steps on fog. On fog,
nmax 4 switched to plain for 224 steps. Prompt 03 is the only case in
this matrix to reach 200 tok/s.

All GPU runs used `tools/gpu-lock.sh`. Bench logs, token IDs and the
frozen diagnostic profiles are in `~/archive/abla-grad-r5-ed/`.
The museum and fog IDs are `story.bos.ids` and `explain.bos.ids` in
`~/archive/abla-grad-retrieval-1244/`.

The five-change build passed the step gate (324 positions, zero argmax
differences), chunk-256 prefill (324 positions, one argmax difference),
and the speculation gate on 01–05 at maxima two and four. The long
step gate counted 28 rows outside the short-prompt limits on 05, with
34 argmax differences, and 25 rows on 06, with 97 differences.
`verify-check` covered each accepted length from zero through four.
At prompt 05, forty windows starting at position 200 produced zero
argmax differences from plain decode at both three and five rows.

### Which verification changes transfer to plain decode

These graph-mode screens use the 16- and 513-token fixed prefixes in
`~/src/abla-grad-oracle/profile-9a46740/`, 128 generated tokens,
context capacity 700, and `8db7496` as the implementation baseline.
All arms generated the same 128 IDs at each depth. The figures are
single runs unless the row specifies a median. They price separate
experiments and do not add to the released throughput.

| Plain graph | d15 tok/s | d512 tok/s |
|---|---:|---:|
| Five-change baseline | 132.81 | 127.96 |
| One block per small BF16 projection output | 132.73 | 127.91 |
| Fixed eight-slot NVFP4 down projection | 124.20 | 126.78 |
| Four-tap specialization of the existing fused convolutions | 135.13 | 129.78 |
| All three transfers together | 134.70 | 130.30 |
| Paired norms plus fused gain/forget/message mix | 135.66 | 129.68 |
| Generated norm/down controller plus fused gain/forget/message mix | 137.83 | 131.72 |

The one-row route already writes the selected IDs straight to the expert
matvec. There is no union map or extra history snapshot to remove.
The eight-slot gate/up already includes SiTU. The four-tap screen
specializes the loops inside the existing plain convolution and residual
kernels. The small-row LM head handles multiple activations sharing a
weight read, an opportunity absent at one row.

The strongest measured plain change is fusing the loop-controller work
on the three existing attention lanes and the FFN lane. The generated
norm/down and the gain/forget/mix tail together remove 384 launches
per token, with 1,571 remaining. The two-depth result is well short of
the proposed 150–165 tok/s. At d15, 150 tok/s still requires removing
0.59 ms/token from the fastest observed arm. The standalone cooperative
controller prototype saved only 0.94 µs per invocation at one row and
ran slower at five rows. Its best plain saving over all 96 controllers
would be about 0.09 ms/token before accounting for branch overlap.

Replaying the generated norm/down kernel from `b7d402c` by itself, on
the current three-lane schedule, gave medians of 135.95 and 130.17 tok/s
across three runs per depth. The corresponding baseline medians were
132.37 and 127.78. The kernel matched the separate operations bit for bit
on its 64-output fixture. Both the norm/down-only build and the combined
controller build reproduced every original FP32 logit on the 324
short-gate rows in step mode.
The combined controller also reproduced the original step logits byte for byte on both long prompts (5,924 rows).

### Plain controller fusion on the original and packed models

The default plain graph uses the generated norm/down and the fused gain/forget/mix for matching F32 device matrices. Each stage checks its own matrix types. Native packed BF16 matrices use the standalone FP32-accumulating kernels. On the same release executable, the original model went from 1,955 graph launches per token to 1,571.

| Cached depth | Original, fusion disabled | Original, default | Packed BF16, default |
|---|---:|---:|---:|
| d15 | 133.15 | 137.87 | 132.94 |
| d512 | 126.11 | 130.40 | 127.06 |

Rates are tok/s for 128 generated tokens at context capacity 700, using the prefixes from the plain graph screen above. Each arm matched all 128 output IDs at each depth. The two release files reproduced every reference logit byte across all 6,248 stepped positions (01–06) and all 324 short chunk-256 positions. The short and long gates and the two- and four-proposal speculation gates passed. The F32 control is available with `decode-bench --no-fuse-controller`.

### Fixed-four proposal shortlist and controller experiment

Two more fixed-four runs used the same 512-token museum and fog prompts.
Within groups of 32 vocabulary IDs, DSpark evaluated the Markov
projection for the four highest base-plus-bias scores. The target still
verified the proposed tokens. The verify graph also used a fused
controller tail, a 256-thread controller-down reduction and 20 expert
grid entries per pass with a strided loop over the remaining entries.

| Prompt | Timed tokens | Cycles | Milliseconds per full cycle, two runs | Tok/s, two runs |
|---|---:|---:|---:|---:|
| Museum | 511 | 212 | 14.69, 14.62 | 164.1, 164.8 |
| Fog | 511 | 260 | 14.54, 14.76 | 135.2, 133.1 |

Both repeats matched the 512 output IDs from the shortlist-only build on
each prompt. Fog took 260 cycles with the shortlist, versus 258 with the
released full-proposal build. Restricting candidates can change draft
acceptance. The corrected compact expert grid alone removed about
0.11 ms from the isolated verifier. An earlier grid probe accumulated
the wrong expert address on its second iteration; its timings have been
excluded. The corrected kernel passed the independent multi-expert
comparison. The frozen executable and patch are `combo-correct` and
`combo-fixed.patch` in the experiment archive.

The combined fixed-four diagnostic passed `make test-host`, `verify-check`,
the two- and four-proposal spec gates on 01–05, the short step and
chunk-256 gates, and the long step gates on 05 and 06. These figures
describe the archived experimental build. The released five-change
throughput appears in the adaptive table above.

### Default adaptive two-proposal comparison

Each pair used the original GGUF and the same 512-token prompt run on the RTX 4090. The first pair ran the baseline first and the second ran it last. Both used `spec-bench --n 512 --nmax 2 --adaptive` with cores 24–27 and a separate GPU lock per invocation. The first 511 generated-token intervals were timed.

The first comparison used main at `2a5e004` and the combined prototype.

| Prompt | Pair | Baseline tok/s | Candidate tok/s | Accepted / offered, baseline → candidate | Plain steps, baseline → candidate | Matching output IDs |
|---|---:|---:|---:|---|---|---:|
| 03 | 1 | 199.01 | 207.79 | 308/376 → 308/376 | 16 → 16 | 512/512 |
| 03 | 2 | 199.61 | 207.44 | 308/376 → 308/376 | 16 → 16 | 512/512 |
| 05, Alice | 1 | 148.28 | 151.40 | 177/222 → 177/222 | 224 → 224 | 512/512 |
| 05, Alice | 2 | 147.07 | 151.03 | 177/222 → 177/222 | 224 → 224 | 512/512 |
| Museum | 1 | 157.96 | 164.92 | 251/488 → 251/488 | 16 → 16 | 512/512 |
| Museum | 2 | 155.89 | 161.48 | 251/488 → 251/488 | 16 → 16 | 512/512 |
| Fog | 1 | 141.66 | 147.85 | 220/550 → 219/552 | 16 → 16 | 512/512 |
| Fog | 2 | 139.98 | 145.82 | 220/550 → 219/552 | 16 → 16 | 512/512 |

The release comparison included native BF16 support and the shipping plain-controller fusion in both executables. The candidate additionally used the recorded DSpark shortlist, the small-row controller kernels and the compact five-row expert grid. At nmax 2 the verifier has three rows, so the five-row grid affects only the fixed-four comparison below.

| Prompt | Pair | Baseline tok/s | Candidate tok/s | Accepted / offered, baseline → candidate | Plain steps, baseline → candidate | Matching output IDs |
|---|---:|---:|---:|---|---|---:|
| 03 | 1 | 200.00 | 208.37 | 308/376 → 308/376 | 16 → 16 | 512/512 |
| 03 | 2 | 196.17 | 206.34 | 308/376 → 308/376 | 16 → 16 | 512/512 |
| 05, Alice | 1 | 150.82 | 152.15 | 177/222 → 177/222 | 224 → 224 | 512/512 |
| 05, Alice | 2 | 151.24 | 151.92 | 177/222 → 177/222 | 224 → 224 | 512/512 |
| Museum | 1 | 156.27 | 163.27 | 251/488 → 251/488 | 16 → 16 | 512/512 |
| Museum | 2 | 158.62 | 165.26 | 251/488 → 251/488 | 16 → 16 | 512/512 |
| Fog | 1 | 140.27 | 146.24 | 220/550 → 219/552 | 16 → 16 | 512/512 |
| Fog | 2 | 142.09 | 148.14 | 220/550 → 219/552 | 16 → 16 | 512/512 |

Each speculative cycle commits an anchor plus its accepted proposals. The acceptance numerator subtracts one anchor per cycle from the device token count. The denominator is two offers per cycle on these prompts. The last device batch can commit an extra token beyond the 512 logged IDs.

| Fixed four, 512 tokens | Baseline ms/cycle, two runs | Candidate ms/cycle, two runs | Baseline cycles | Candidate cycles |
|---|---:|---:|---:|---:|
| Museum | 16.07, 16.06 | 14.85, 14.80 | 212, 212 | 212, 212 |
| Fog | 16.03, 16.02 | 14.71, 14.74 | 258, 258 | 260, 260 |

The fixed-four figures divide elapsed generation time by the recorded cycle count and include drafting, verification and commit. Raw logs and the 512 output IDs per arm are under `~/archive/abla-grad-r5-ed/` (`ab-*`, `postctrl-*` and `endcycle-*`).

The combination ships as the recorded default for the supported BF16 drafter. It won every release pair. The short step and chunk-256 gates and both long gates passed. Target step logits matched the original reference at all 6,248 positions. The two- and four-proposal speculation gates passed on 01–05. Both target files passed `verify-check` at accepted lengths zero through four and produced zero R3/R5 argmax differences on forty windows of 05. The production shortlist test covers a partial warp, an excluded correction and a rounded softcap tie across blocks.

## Sliding-window KV at 32k context on the RTX 4090

The Q4_K_M target has eight full-attention visits and forty sliding-window visits. At a context limit of 32,768, their FP32 K and V allocations total 2,650,800,128 bytes (2.47 GiB). The full-length baseline allocated 12,884,901,888 bytes (12 GiB). The 1,536-slot sliding caches accommodate a 1,024-token live window plus up to 512 positions written before attention.

With the Q8_0 DSpark drafter loaded, `serve --ctx 32768` processed a 16,787-token chat prompt and returned eight tokens. A second identical request reused 16,786 prompt tokens. Time to first token was 16.22 s cold and 0.16 s warm. The largest process allocation observed in half-second samples was 21,694 MiB. These figures come from two requests to one server. A separate target-only `prefill-bench` processed 32,760 positions in 7.40 s (4,430 tok/s, one timed run at chunk 512).

Both long step-mode GRDLs (05 and 06, 5,924 positions) matched the full-cache files byte for byte. At the physical wrap, 20 windows each at verification widths three and five matched plain decode on all 160 row argmaxes. The scalar decode test also compared every output bit of two KV heads for zero through four accepted proposals. Locked runs, prompts and output files are under `~/archive/abla-grad-ring-1250/`.

Two matched plain graph tg128 runs per depth used the original target, the same prompt IDs and opposite run order. At d15 the linear baseline measured 138.52/138.71 tok/s and the ring build 138.24/136.91. At d512 they measured 132.85/129.38 and 132.48/132.96. On the 1,624-token Alice prompt they measured 126.34/121.62 and 126.19/123.28. All six pairs produced identical generated IDs. These pairs establish the memory saving without a repeatable decode gain at these depths.

## Prose proposal failures and verification depth

The RTX 4090 runs used the original Q4_K_M target, Q8_0 drafter and the recorded four-candidates-per-warp shortlist at `3dc62e0`. Each completion has 512 greedy IDs, including the seed token. The ranked traces record the two best scored draft IDs and the two best target IDs after each row, with ties broken by token ID. Only proposals reached before or at the first rejection count. Diagnostic tracing and throughput measurements used separate executables.

| Fixed nmax 4 | Zero / one / two / three / four proposals accepted per cycle | First proposal rejected | Target winner was draft runner-up at first rejection | Draft runner-up in target top two at first rejection |
|---|---|---:|---:|---:|
| Fog | 124 / 68 / 36 / 17 / 15 | 124/260 | 57/245 (23.3%) | 81/245 (33.1%) |
| Museum | 63 / 64 / 44 / 16 / 25 | 63/212 | 48/187 (25.7%) | 63/187 (33.7%) |
| Ferry | 82 / 49 / 40 / 19 / 27 | 82/217 | 50/190 (26.3%) | 76/190 (40.0%) |

Fog's first-proposal rejection accounts for almost half its cycles. Museum loses nearly as many cycles at its second proposal as at its first. On the first proposal alone the runner-up recovers 33 of Fog's 124 misses and 21 of Museum's 63. Scaling every score by a positive inverse temperature preserves the greedy ranking. We varied the relative weight on the trained Markov correction and the shortlist size instead.

| Markov projection or shortlist change | Fog cycles | Museum cycles | Ferry cycles |
|---|---:|---:|---:|
| Released | 260 | 212 | 217 |
| Half Markov correction | 269 | 228 | 224 |
| Double Markov correction | 279 | 232 | 230 |
| Omit Markov correction | 293 | 277 | 270 |
| Eight shortlist candidates per warp | 258 | 212 | 216 |

The all-position trace commits only the verified anchor, so a replay can start a cycle at every position in the same greedy chain. It compares proposals with the true subsequent token IDs and accounts for changes to the next cycle start. At five verifier rows, replacing the fourth linear proposal with a runner-up at the smallest top-two gap among proposals one to three, when that gap is below one logit, took 251 cycles on Fog and 207 on Museum. Choosing this single sibling with knowledge of future tokens would take at least 234 and 198 cycles respectively. These are replayed cycle counts. Branch-dependent KV, convolution histories and any tree-verifier overhead are excluded.

On the first proposal, the median draft gap between first and second was 0.82 logits on Fog misses and 3.66 on hits. Museum's corresponding medians were 0.49 and 2.21. The two ranked target IDs came from a separate reduction over all target logits. Exact runner-up statistics include ties after the rounded softcap.

A second diagnostic saved the 256-element Markov state at each primary proposal, restored it for each runner-up, and ran DSpark's actual correction on the resulting descendants. The anchor-only trace still generated the same 512 target IDs. At every fixed-four cycle start on all three prompts, all four primary proposal IDs agreed with the original trace. This allows comparison of branching policies with the same target chain and R5 row budget. Under four proposals there are nine binary tree shapes when every chosen second child also has its first sibling. On Fog, the best static shape used the first three primary proposals and a sibling at the first proposal (249 cycles). On Museum the best static shape kept the four primary proposals (212). An oracle choosing among all nine shapes at each boundary took 227 cycles on Fog and 187 on Museum. Even with zero extra work for tree verification, these counts would require at most 11.26 and 13.66 ms per cycle to reach 200 tok/s. The measured linear R5 cycles were 14.54 and 14.59 ms respectively.

Fog and Museum spent the entire measured 512-token chat continuations in the thinking channel. Ferry used a raw prose continuation. Applying 0.25 of the previous rejected verifier row’s logits to the next draft reduced fixed-four cycles on Fog from 260 to 250, but increased Museum from 212 to 223. Applying a 0.10 prior only to the next first proposal took 256 and 209 cycles. Each comparison used the same target chain.

Verifying through the first of the initial three proposals whose margin fell below a cutoff, or all four if none did, gave 153.94 tok/s on Fog with a two-logit cutoff, and 174.77 on Museum with a one-logit cutoff. The corresponding fixed-two baselines on the same IDs were 149.88 and 170.89 tok/s. These are prompt-specific diagnostic settings. The production controller still uses the released adaptive policy. The per-run logs, exact IDs, trace replays and fork-policy calculations are in `~/archive/abla-grad-prose-1249/`.

## 64k–128k serving and KV precision

With a 131,072-token limit the Q4_K_M target and Q8_0 DSpark drafter processed a 105,878-token chat request and returned eight greedy tokens. A repeat served the identical completion and reused 105,877 prompt tokens. On the release candidate the first token took 50.13 s cold and 0.765 s on the repeat. The server selected a 512-token prefill chunk and reported 1,211 MiB to spare after loading with 21,698 MiB free. At the 65,536-token limit, a 54,571-token prompt returned eight tokens, reproduced the same choices on repeat, and reused 54,570 tokens. First-token times were 22.57 s and 0.396 s, with 1,732 MiB to spare. Eight generated tokens establish that the request completes, but are too few for a sustained decode measurement.

At 131,072 positions the target's forty sliding F16 visits and eight global Q8 visits need 2.484375 GiB of K and V. Each global block holds 32 signed bytes and one FP32 scale. Full F16 on the target required 4.234375 GiB and ran out of memory while instantiating the plain graph in the same serving experiment. DSpark retains one full F16 cache and two 2,048-slot F16 sliding caches, including space for its noise block. The four FP32 target taps now occupy 72 MiB. The server injects them into the drafter as each prompt chunk finishes.

The numerical gate compared every vocabulary logit at every input position, using contiguous token prefixes of the Standard Ebooks edition of Dickens’s *A Tale of Two Cities* (source commit `4411856fa18102aafeebb573357dae474045efd5`). Both llama.cpp engines used the original GGUF, F16 K/V, one sequence, GPU offload and compact sliding caches. The 32,768-row run used batch and microbatch 256. The 102,400-row run used 64 in all engines after the pinned mainline CUDA build requested a 5.29 GB prefill compute buffer at 256.

| Positions | Engine against mainline CUDA | Rows KL > .04 | Argmax differences | Worst KL | Max absolute logit difference |
|---:|---|---:|---:|---:|---:|
| 32,768 | Unsloth b12649 | 136 | 806 | .676532 | 12.669900 |
| 32,768 | Abla F16 | 149 | 799 | .581824 | 13.264651 |
| 32,768 | Abla global Q8, sliding F16 | 142 | 786 | .612149 | 13.850366 |
| 102,400 | Unsloth b12649 | 356 | 2,331 | 1.418658 | 16.690143 |
| 102,400 | Abla F16 | 358 | 2,331 | .662612 | 15.573454 |
| 102,400 | Abla global Q8, sliding F16 | 350 | 2,330 | .824671 | 18.409976 |

Both Abla cache modes passed at 32,768 and 102,400 positions under the calibrated long-prompt rule. On fixed step prompt 06, F16 had 24 rows over the KL threshold, 110 argmax differences and worst KL .175718. Mixed Q8/F16 had 29, 94 and .287230. Both also passed prompt 05. At the 131,072-token allocation, adaptive nmax 2 matched 256 plain greedy IDs on each of prompts 01–04. Prompt 05 first differed at token 98 with a target margin of .139292, inside the speculation gate. The input hashes, fixed 05/06 rules and calibration limits are in `tests/tolerances-inkling-long.txt`. Commands and raw logs are in `~/archive/abla-grad-128k-1251/`. The reusable producers and streaming comparator are described in `tools/oracle/README.md`.

### Serving a rewritten prompt after long generation

At a 131,072-token capacity, the HTTP checkpoint regression started with 21,765 MiB free and selected a 512-token prefill workspace with 865 MiB to spare. Each of the two checkpoints occupies 207.66 MiB. This includes 160 MiB for the target sliding K/V, 8 MiB for the two sliding drafter visits, 36 MiB for rolling target taps and 3.66 MiB for convolution history.

A 6,040-token prompt took 4.019 s to first token cold. After generating 2,304 tokens, its repeat reused 6,039 prompt tokens and took 0.060 s to first token. A 6,026-token edit within the preceding prompt reused the saved 4,096-token boundary and took 3.025 s, compared with 3.748 s for a cold request with the identical edit. Both restored requests returned the same eight-token choices as their cold references, including reasoning. These are single-request first-token times. The test is reproducible with `tools/test-serve.sh <model.gguf> --long-checkpoint` under the GPU lock.

On main `aa4b5f8`, a 15-request opencode session completed in 329 s at the same capacity. Two edited requests reused 19,269 of 26,052 and 26,051 of 27,808 input tokens. Their first output events arrived after 4.95 and 3.57 s. Two tool calls reused 35,244 of 35,279 and 37,128 of 37,159 input tokens. Their first output events arrived after 19.34 and 19.18 s. This build timestamped the first parsed event, and the parser buffers tool arguments until their closing marker. Generating those two replies took 18.70 and 18.53 s at the logged decode rates. Separate raw-token timestamps were unavailable. Decode throughput ranged from 147.76 tok/s on the 9,795-token request to 80.18 on the 43,503-token request. The requests generated different content, so these endpoints include both depth and speculation-acceptance effects.

## OpenAI session replay

One RTX 4090 run with Inkling UD-Q4_K_M, the Q8_0 DSpark drafter, and a 131,072-token context. The original opencode logs did not include HTTP bodies, so this
capture uses a new opencode session with two read-tool calls and four follow-up turns containing inventory fixtures. Request 1 is opencode's title request.
The capture and replay match on prompt, reused and generated token counts and finish reason for every request. Replay sends the recorded tool results as part
of the chat history. It measures HTTP wall time after model startup, with first raw token and decode rate from the server.

| Request | Prompt tokens | Reused tokens | Generated tokens | First raw token (s) | Decode (tok/s) | Total HTTP (s) | Finish |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 513 | 0 | 167 | 3.333 | 182.70 | 4.247 | stop |
| 2 | 861 | 0 | 67 | 3.511 | 188.51 | 3.867 | tool_calls |
| 3 | 992 | 926 | 78 | 0.939 | 158.88 | 1.431 | tool_calls |
| 4 | 1150 | 1068 | 136 | 1.220 | 176.82 | 1.990 | stop |
| 5 | 19539 | 1279 | 215 | 8.448 | 114.39 | 10.327 | stop |
| 6 | 38007 | 19748 | 202 | 9.727 | 89.35 | 11.988 | stop |
| 7 | 56462 | 38206 | 202 | 11.594 | 78.61 | 14.163 | stop |
| 8 | 74917 | 56660 | 202 | 13.820 | 62.11 | 17.073 | stop |

Reproduce locally with `tools/serve-replay <capture.jsonl> <model.gguf> --ctx 131072 --max-tokens 512`. This run used an ablac 0.2.73 `--fast` build and cached
CUDA compilation artifacts. The private capture and the capture driver are in `~/archive/abla-grad-serve-replay/`.

### Per-landing replay

Each row uses the same eight chat POST bodies on a fresh model server.
The prompt, reused and generated counts, first raw token, decode rate
and HTTP time are for request 8. Session HTTP is the sum of the eight
request times and excludes model loading. GPU clocks and other system load can affect these single-run comparisons.

| Main landing | Prompt tokens | Reused tokens | Generated tokens | First raw token (s) | Decode (tok/s) | Final HTTP (s) | Session HTTP (s) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| `ab55314` | 74917 | 56660 | 202 | 13.820 | 62.11 | 17.073 | 65.086 |
| `8af8468` | 74917 | 56660 | 202 | 12.943 | 65.74 | 16.016 | 65.097 |
| `4bae40a` | 74917 | 56663 | 202 | 12.657 | 65.84 | 15.725 | 60.803 |
| `b22b960` | 74917 | 56657 | 202 | 11.945 | 67.34 | 14.945 | 59.208 |
| `cf6578c` | 74917 | 56656 | 202 | 10.838 | 101.22 | 12.834 | 55.565 |
| `9aaccb1` | 74917 | 56656 | 202 | 9.602 | 101.44 | 11.593 | 47.226 |

On `cf6578c` and `9aaccb1`, request 5 hit the 512-token limit. On `b22b960` it stopped after 261 tokens. The session timings include this difference.

### Int8 prefill projection GEMMs

The original 128k trace attributed 20.67 GPU seconds to GEMMs, including
12.48 s to grouped NVFP4 int8 and 4.24 s to the two Q8_0 dense paths.
The Q8_0 int8 path reuses three quantized activation packs across the five
Q/K/V/relative/gate projections and keeps the single combined launch.
Grouped Q8_0 also uses int8 MMA with 64-token expert tiles.

Matched RTX 4090 prefill throughput on the original Inkling UD-Q4_K_M,
fixed prompt IDs from the 128k trace, chunk 512, and the v0.2.73 compiler.
Before and after use the same benchmark source snapshot based on e2a0fdf
apart from the GEMM changes. Values are medians of three runs at 8k and
32k and two runs at 128k.

| Prompt tokens | Before tok/s | After tok/s | Change |
| ---: | ---: | ---: | ---: |
| 8,192 | 4,499.68 | 4,776.15 | +6.1% |
| 32,768 | 4,034.44 | 4,152.54 | +2.9% |
| 131,072 | 2,322.73 | 2,365.08 | +1.8% |

The Inkling prefill gate passed 324 positions and the Llama gate passed
266, with one argmax difference each and zero outside tolerance. A
32-token NVFP4 int8 tile reduced 8k throughput to 4,174.27 tok/s, so the
default remains 64.

## What grows during plain decode

Nsight Systems traced one recorded target step after untimed prefill of the Dickens
prefix. The baseline used the original cache policy and 16 key partitions per head.
The profiler started immediately before the graph launch. Each trace contains
1,571 GPU kernels, including eight global and forty sliding split/merge pairs.
The columns partition elapsed GPU time. Kernel overlaps are counted once and K/V
stores are included in the other-kernels column.

| Cached tokens | Global attention (ms) | Sliding attention (ms) | Other kernels (ms) | Idle (ms) | GPU span (ms) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.042 | 0.213 | 7.737 | 0.052 | 8.045 |
| 32,768 | 3.176 | 0.704 | 7.589 | 0.049 | 11.518 |
| 131,072 | 19.775 | 0.578 | 7.574 | 0.046 | 27.973 |

At 32k the eight global F32 caches contain 2.147 GB of live K/V. Dividing
unique bytes by the 3.154 ms spent in the split kernels gives 681 GB/s.
At 128k the global block-Q8 caches contain 2.416 GB. Their 19.752 ms of
split work gives 122 GB/s. These rates use the logical bytes once per KV
head and exclude the merge. They do not count repeated L2 accesses by
multiple query heads. Sliding attention reads at most 1,024 keys per
query from each 1,536-slot ring.

A separate teacher-forced comparison used the first 32,768 Dickens IDs
in chunk-256 prefill, with F32 sliding K/V in all three arms. Greedy
agreement was checked at all positions. Perplexity used the 32,767
observed next tokens.

| Global K/V | Greedy differences from F32 | Mean negative log likelihood | Perplexity |
|---|---:|---:|---:|
| F32 | 0 | 2.320695 | 10.182751 |
| F16 | 608 | 2.320309 | 10.178819 |
| Block Q8 | 643 | 2.320597 | 10.181753 |

Traces, raw timings, token logs and benchmark binaries are in
`~/archive/abla-grad-depth-1254/`. The scoring commands are in
`tools/oracle/README.md`.

### Grouped attention through 131,072 cached tokens

Plain greedy graph decode on one RTX 4090, 128 tokens per timing. Each
Abla run used the first depth-plus-one Dickens IDs, CPU cores 24–27 and
a separate GPU hold. The graph was recorded before seeding. Chunk-512
prefill, graph capture and loading the model were outside the timer.
The timer includes graph replay and stream completion. The benchmark
sets capacity to depth plus 169 rounded up to a multiple of 256.

| Global attention policy | d0 | d8,192 | d32,768 | d65,536 | d131,072 |
|---|---:|---:|---:|---:|---:|
| F16 global, one query per block, 16 splits | 134.89 | 109.42 | 89.85 | 74.80 | 54.92 |
| Q8 global, one query per block, 16 splits | 135.50 | 108.39 | 79.15 | 57.33 | 36.63 |
| F16 grouped from 8k, Q8 above 98,304 | 127.05 | 111.37 | 101.01 | 86.84 | 84.87 |
| **Q8 global, grouped/64 from 8k, CUDA C RMSNorm and depth mix** | **137.69** | **123.76** | **115.25** | **108.60** | **92.50** |

Both precision-only rows used the CUDA C norms and depth mix on
`175171f`. The F16 grouped row includes the Abla kernel ports. That diagnostic
also switched sliding K/V to F16 at 8k and used F32 globals below 8k.
At 131,072 cached tokens, the F16 grouped row reproduced all 128 token IDs of
the earlier 35.44 tok/s run.

The bold row uses F16 globals below 8k, block Q8 from 8k and F16 sliding
K/V from 65,536. At short capacity it keeps the original 16 partitions.
At 32k it improves on the 92.71 tok/s capacity baseline by 24.3%. The
32k goal of 120 tok/s still needs 0.343 ms/token. At 128k it improves
on 35.44 tok/s by 2.61×. A registry-only repeat on the integrated
build at 32k measured 115.99 tok/s with CUDA C RMSNorm and depth mix.
Selecting the Abla ports gave 107.48 and 111.26 on the identical 128 IDs.
On unmodified main at d0, three alternating measurements averaged 123.47
tok/s with the ports and 135.04 with CUDA C. Within each pair all IDs
matched. Main restored the CUDA C dispatch in `8af8468`.

With the Abla ports selected, the otherwise identical Q8 policy measured
123.44 / 114.96 / 107.48 / 104.54 / 90.68 tok/s at the five depths.
At 32k, grouping the forty sliding visits as well measured 107.17.
Making sliding K/V F16 measured 109.73 with per-query sliding attention
and 109.27 with grouped sliding attention. This screen kept the existing
sliding dispatch for the release.

An isolated replay over eight disjoint global caches includes the gated
merge for each visit. At 32k the 64-partition grouped kernel took
1.231 ms with F16 and 0.827 ms with Q8 (872 and 731 GB/s of unique
live K/V respectively). With Q8 at 128k it took 3.047 ms (793 GB/s).
Staging explicitly aligned 128-bit loads in shared memory took 1.018 ms
at 32k with 256 partitions and 3.616 ms at 128k with 512. The grouped
kernel loads each block's values into registers and uses them for all
three query heads. These rates count the stored K/V data once, including
Q8 scales. They are distinct from hardware DRAM-counter measurements.

The 32,768-row external gate used the production combination of Q8 global
and F32 sliding K/V, chunk 256 and the pinned mainline CUDA logits. It had
118 rows over KL .04, 796 argmax differences, worst KL .708463 at position
3,061, and max absolute logit difference 17.27565. It passed the calibrated
limits, including the .726532 worst-KL ceiling from the matched Unsloth
comparison on these same IDs.

The fixed step gates also passed with a short allocation and with a
33,024-position allocation that selects Q8 global K/V. The counts
below use the same pinned llama.cpp CUDA references.

| Prompt | Allocation | Rows above KL .04 | Argmax differences | Worst KL |
|---|---:|---:|---:|---:|
| 05 (1,624 rows) | 1,724 | 30 | 24 | .800399 |
| 06 (4,300 rows) | 4,400 | 27 | 96 | .208716 |
| 05 | 33,024 | 30 | 27 | .811584 |
| 06 | 33,024 | 31 | 98 | .168726 |

At the 33,024-position allocation, the five-row verification check
matched the reference argmax after each possible commit length, zero
through four. The grouped-kernel test matched per-query FP32 softmax
partials bit for bit at equal partition counts for F32, F16 and Q8 K/V.

After integrating the int8 prefill projection GEMMs from `4bae40a`, the
same 32,768-row external gate had 137 rows over KL .04, 778 argmax
differences, worst KL .595350 at position 3,061 and max absolute logit
difference 12.83186. This passes the same .726532 ceiling.

### Global attention after the Q8 release

A one-step node trace of the grouped-Q8 graph measured the following GPU
times. K/V stores are counted in other kernels. The profiled binary used
the grouped release kernels and the pre-`4bae40a` prefix arithmetic.

| Cached tokens | Global split (ms) | Global merge (ms) | Global split, unique GB/s | Sliding split + merge (ms) | Other kernels (ms) | Idle (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | .197 | .057 | 765 | .692 | 6.996 | .047 |
| 65,536 | 1.422 | .060 | 850 | .577 | 6.992 | .047 |
| 131,072 | 3.347 | .059 | 722 | .575 | 7.224 | .047 |

Doubling the depth from 64k to 128k added 1.93 ms in the eight global
visits. The forty sliding visits stayed within .003 ms. Bandwidth counts
each global K/V cache element once, including block scales.

Keeping global keys Q8, two experimental builds packed the values in
blocks of 32 with one FP32 scale per block. The four-bit format used
20 bytes per block and the six-bit format used 28. Both used the prefill
arithmetic of release `cf6578c`. The following checks compared every
logit row on the same 32,768 Dickens IDs with pinned llama.cpp CUDA.
The argmax limit is 983 rows, the KL row limit is 327, and the calibrated
worst-KL limit is .726532.

| Global values | Rows above KL .04 | Argmax differences | Worst KL | Max absolute logit difference |
|---|---:|---:|---:|---:|
| Q8 | 137 | 778 | .595350 | 12.832 |
| Four bits | 217 | 957 | .499146 | 17.419 |
| Six bits | 143 | 838 | .380045 | 16.919 |

At 8k, using 16, 32 and 64 global partitions measured 124.04, 125.00
and 126.72 tok/s, respectively. All three generated the same 128 token
IDs. At 128k, an experimental 128-partition graph measured 96.18 tok/s
against 95.58 with 64 partitions. This single pair differed by .065 ms
per token.

| Global value cache | d65,536 tg128 | d131,072 tg128 |
|---|---:|---:|
| Q8 release | 111.35 | 95.58 |
| Four bits | 105.72 | 85.88 |
| Six bits | 101.34 | 79.41 |

The packed values reduced bytes read and passed the 32k numerical screen,
but both implementations slowed end-to-end decode. The release uses Q8 values.

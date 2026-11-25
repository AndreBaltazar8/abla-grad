# Inkling-2-Mini decode and prefill speed (RTX 4090)

One table, updated with every speed change. Goal: beat mainline llama.cpp and Unsloth's build at every depth and in prefill, with `tools/gate-inkling.sh` passing; stretch line 200 tok/s decode.

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

## Decode across cache depths

Decode throughput in tokens/second on the RTX 4090, with the same
Inkling UD-Q4_K_M GGUF and no drafter.

| Cached depth | Generated tokens | abla-grad graph | llama.cpp 33aeb17 | Unsloth b12649-mix-c7059d6 |
|---:|---:|---:|---:|---:|
| 0 | 128 | 105.39 | 77.39 | 105.27 |
| 0 | 1024 | 102.60 | 77.80 | 104.62 |
| 512 | 128 | 102.27 | 77.97 | 105.53 |
| 512 | 1024 | 100.29 | 76.58 | 103.74 |
| 1024 | 128 | 99.63 | 76.90 | 104.13 |
| 1024 | 1024 | 99.21 | 75.92 | 102.94 |
| 2048 | 128 | n/a (accuracy pending 06) | 75.58 | 102.62 |
| 2048 | 1024 | n/a (accuracy pending 06) | 75.58 | 102.56 |
| 4096 | 128 | n/a (accuracy pending 06) | 74.92 | 100.74 |
| 4096 | 1024 | n/a (accuracy pending 06) | 74.85 | 100.78 |

Each llama.cpp result is the mean of three `llama-bench` repetitions
with F16 KV, CUDA graphs and synthetic tokens. abla-grad `85b7bee`
used one graph run per cell, F32 KV, and greedy continuation of a
natural-text prefix (fixture 06). Its context was
`depth + generation + 1` rounded up to 256. Depth counts cached
tokens before the seed. All timings exclude prefill and use E-cores 24–27.
Supplemental abla-grad tg1024 runs reached 102.68 at d15 and 101.19 at d300.

The 1,624-position natural-text gate qualified the sliding-window path
on fixture 05. Among its 1,624 rows, abla-grad exceeded KL 0.04 on 28,
mainline CPU vs CUDA on 42, and mainline CUDA vs Unsloth CUDA on 36.
The 4,300-position fixture 06 is the remaining accuracy check for
the d2048 and d4096 rows. Raw samples and reproduction commands are in
`~/src/abla-grad-oracle/long-depth/`.

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

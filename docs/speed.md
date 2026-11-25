# Inkling-2-Mini decode and prefill speed (RTX 4090)

Goal: beat mainline llama.cpp and Unsloth's build at every depth and in prefill, with `tools/gate-inkling.sh` passing; stretch line 200 tok/s decode.

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

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

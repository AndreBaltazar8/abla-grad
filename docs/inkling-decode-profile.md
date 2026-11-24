# Inkling decode profile on the RTX 4090

The forward and logit-readback window averaged **45.88 ms per token** in
abla-grad and **15.24 ms per token** in llama.cpp. The device spent 15.33
and 12.52 ms per token in kernels, respectively. GPU idle time explains
28.14 ms of the 30.64 ms gap. Within abla-grad's GPU work, the top
three costs were Q8_0 matvec, indexed NVFP4 matvec, and the one-thread
MoE router.

## Setup and timing

One trace per implementation, each using the 300 fixed IDs in
`tests/prompts/04.inkling.ids` and the same verified Unsloth Inkling 2 Mini
GGUF. Both consumed one ID per step and copied the full 200,058-float logit
row back. That holds the prefix fixed across implementations, including
positions where greedy choices might differ. The RTX 4090 ran driver
580.95.05. Both processes ran on E-cores 24–27 at context 512. abla-grad
was `d926ff4` and llama.cpp was `33aeb17` with CUDA graphs enabled and F16
KV. Each trace matched its own untraced GRDL byte for byte.

The measurement starts at the first device kernel and ends after the
300th 800,232-byte device-to-host logit copy. CUPTI activity records give
kernel and copy intervals. Busy time is their union, and idle time is the
rest of the window. For between-row idle, use gaps beginning at a completed
full-logit copy. All other gaps count within rows, including initial kernel
setup. These are idle intervals for the traced application's device work.

| Mean per position | abla-grad | llama.cpp CUDA |
| --- | ---: | ---: |
| Forward and readback window | 45.876 ms | 15.240 ms |
| Kernel time, sum | 15.332 ms | 12.517 ms |
| Memory copies, sum | 0.032 ms | 0.346 ms |
| GPU busy, union | 15.364 ms | 12.863 ms |
| GPU idle | 30.513 ms | 2.377 ms |
| Idle within rows | 16.715 ms | 1.875 ms |
| Idle between rows | 13.798 ms | 0.502 ms |
| Device kernels per position | 2,564 | 6,332.16 |
| Kernel records from graph nodes | 0% | 99.00% |
| Direct `cuLaunchKernel` calls per position | 2,564 | 0.40 |
| Runtime `cudaGraphLaunch` calls per position | 0 | 22.77 |

Times are trace measurements and include profiling overhead. The
untraced and traced **whole-command** times were 36.177 s and 38.362 s in
abla-grad and 10.861 s and 9.158 s in llama.cpp, in that order on shared
hardware. These single runs give a wall-time check, with cache state and
trace overhead combined. The earlier 78 ms/token estimate came from
subtracting a reported model load from a 27.2 s CLI run. It included
CPU-side handling of every full logit row and final GRDL output. In this
trace abla-grad spent 4.01 s between the first CUDA API and first kernel,
13.76 s in the forward/readback window, and 20.6 s of whole-command time
outside the CUDA timeline after accounting for those phases. The latter
is an estimate of the post-CUDA tail plus time before the first CUDA API.
The CLI decodes each logit into an Abla array, appends it to another
array, and ultimately encodes 60,017,400 floats one byte at a time. The
trace attributes the 13.80 ms of between-row idle to the interval around
those per-row host steps, and it leaves the final encoding and file
write outside the forward window. This test measures the diagnostics CLI.
The generated-text path can keep the logits on the GPU and read back a
single selected ID.
The matched llama.cpp trace processed 65.6 positions per second, while
its earlier `llama-bench tg128` measured 77.96 tokens per second.

### CPU submission

The 769,200 `cuLaunchKernel` calls totaled 9.36 ms per token on the host,
or 3.65 µs per call. Context push and pop API durations added up to
0.52 ms/token. Between direct launch calls within the same row there was
22.13 ms/token of host elapsed time outside the launch API, including
context calls, graph orchestration, registry lookup, launch validation,
argument packing, allocation and other host work. This trace does not split
that interval further by host function. The separate no-op forward
measurement in `tests/bench/host_overhead.ab` is 3.9 ms/token, and
`tests/bench/launch_cost.ab` measured 4.9 µs/launch for the tiny
`add` kernel. Those measurements have a different workload and build
mode. Host API durations and inter-call intervals can overlap GPU work,
so the CPU totals are diagnostic rather than additive terms in the
GPU busy/idle table.

abla-grad issued 300 full D-to-H copies averaging 31.8 µs of device copy
time per row (25.2 GB/s of transferred payload). The synchronous driver copy call averaged
0.741 ms per position including synchronization. There were 155,282 idle
gaps of at least 10 µs in the abla-grad window and 1,312 in llama.cpp.

## Ranked abla-grad kernels

Logical bandwidth is estimated from launch arguments and packed tensor
formats. The estimate counts selected expert weight payloads, distinct
inputs and outputs once, with separate payloads for different launches.
It can include service from cache. Repeated loads are outside this estimate, along with
local-memory spill traffic and memory-bus transactions. The driver reserves
performance counters for administrators on this host
(`RmProfilingAdminOnly=1`), so these are effective logical GB/s. NVIDIA
specifies a theoretical 1,008 GB/s for the card.
[RTX 4090 specifications](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)

| Rank | Kernel | Calls/token | GPU ms/token | Mean µs/call | CTAs/call | Logical GB/s (% of peak) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `matvec_q8_0` | 289 | 3.103 | 10.74 | 96–50,256 | 601.48 (59.67%) |
| 2 | `matvec_nvfp4_indexed` | 88 | 2.453 | 27.87 | 4,608 | 646.29 (64.12%) |
| 3 | `moe_route` | 48 | 2.429 | 50.60 | 1 | 0.02 (<0.01%) |
| 4 | `rmsnorm` | 434 | 2.110 | 4.86 | 1–8 | 4.41 (0.44%) |
| 5 | `matvec_f32` | 336 | 1.146 | 3.41 | 16–576 | 250.58 (24.86%) |
| 6 | `depth_mix` | 120 | 1.084 | 9.03 | 1–4 | 6.77 (0.67%) |
| 7 | `attention_decode_rel` | 48 | 1.033 | 21.52 | 24 | 59.11 (5.86%) |
| 8 | `conv_step` | 192 | 0.438 | 2.28 | 4–9 | 34.98 (3.47%) |
| 9 | `copy_f32` | 351 | 0.323 | 0.92 | 9 | 20.06 (1.99%) |
| 10 | `matvec_bf16` | 64 | 0.224 | 3.51 | 6–8 | 36.82 (3.65%) |

Three kernels explain the one-block bottleneck. `moe_route` uses one
thread for a serial top-k scan. One-block `rmsnorm` accounts for 2.049 of
its 2.110 ms/token. One-block `depth_mix` accounts for 0.829 of its 1.084
ms/token, with repeated shared-memory reductions across states. The
small convolution and copy grids are low-occupancy candidates too.
The two top quantized matvecs together read an estimated 3.44 GB of
weights per token and account for 5.56 ms/token of device time.

## What llama.cpp does differently

The pinned llama.cpp CUDA build ran 1,899,648 GPU kernels, of which
1,880,604 had graph node IDs. The excess kernel count over abla-grad
comes with much less host dispatch work. Its graph runtime submitted
6,831 `cudaGraphLaunch` calls over the 300 positions. Direct-runtime
`cudaLaunchKernel` also recorded 31,588 calls, including capture and
direct execution. Counting only host launches therefore understates the
device work, and counting only kernels overstates the host submissions.

| Rank | llama.cpp kernel | GPU ms/token |
| ---: | --- | ---: |
| 1 | Q8_0 `mul_mat_vec_q` | 2.573 |
| 2 | Indexed NVFP4 `mul_mat_vec_q` | 1.909 |
| 3 | `scale_f32` | 0.566 |
| 4 | Broadcast add | 0.549 |
| 5 | F32 `mul_mat_vec_f`, 32-thread variant | 0.543 |
| 6 | F32 `mul_mat_vec_f`, 256-thread variant | 0.541 |
| 7 | Broadcast multiply | 0.487 |
| 8 | `rms_norm_f32`, 1,024-thread variant | 0.443 |
| 9 | F32 scalar copy | 0.427 |
| 10 | `quantize_q8_1` | 0.399 |

llama.cpp quantizes activations to Q8_1 for the traced quantized matvecs,
and uses vectorized dot products. Include the activation quantization
cost when comparing those kernels. Its MoE top-k uses one 256-thread CTA,
averaging 2.59 µs per route or 0.124 ms/token for the same 48 calls. The
first two 1,024-thread RMSNorm variants total 0.720 ms/token. It uses
F16 KV with `flash_attn_ext_vec`; the main flash kernel averages
0.188 ms/token, with separate cache conversion and combine work.
[llama.cpp CUDA kernel source at the measured commit](https://github.com/ggml-org/llama.cpp/tree/33aeb17f85869279e3821f746bc645357139bc2c/ggml/src/ggml-cuda)

## Order of work

1. **60-2, measure generation separately and remove logit marshaling from
   its inner loop.** Use on-device greedy selection and a small token copy.
   In the GRDL diagnostic path, stream the raw F32 payload out of the host
   buffer in blocks. This targets the 13.80 ms/token between rows and the
   CPU tail while preserving per-position GRDL tests.
2. **60-2, capture and replay the full decode step.** All 300 positions had exactly
   the same ordered kernel names and grid dimensions. Weights, scratch
   and KV allocations can remain stable. The record/replay backend
   landed in `5e896fb` after these traces.
   Move token ID, position and
   window bounds into a step-data buffer so those values can change
   without rebuilding the graph. Warm kernel modules before capture.
   Try a single graph for a fixed context and a single stream, then
   measure replay and verify all 300 logit rows against the current
   path. Keep graph-update and capture costs visible in the test. The
   16.71 ms/token of within-row idle is the addressable timeline gap.
3. **e7, parallelize `moe_route` and `rmsnorm`.** Use warp/block
   reductions and preserve the router's lowest-ID tie rule. llama.cpp's
   top-k costs about 2.30 ms/token less. Extend the block-reduction work
   to `depth_mix`. Use the fixed cases for numerical checks before using
   runtime speed to choose a kernel.
4. **e7 and 60-2, remove short launches before tuning weight traffic.**
   Fuse or avoid copies, residual adds and small broadcast operations
   where the data lifetime allows it. 351 `copy_f32` calls account for
   only 0.32 ms of GPU work but still cross the submission path. For the
   uncaptured fallback, cache function/spec lookup and reuse the argument
   slab. After graph and one-block work, compare Q8_0/NVFP4 activation
   quantization and packed dot products against the current F32 input
   kernels, including the cost and any logit effects of the conversion.

## Reproduction

On André's workstation, the exact input, pinned abla-grad executable,
CUPTI preload source and binaries, trace driver, analyzer, logs and raw
tables live under `~/src/abla-grad-oracle/profile-d926ff4/`.
`trace.sh grad` and `trace.sh llama` use the pinned executables and run
`cmp` against the saved untraced GRDL files. Run them sequentially on the
4090 with the same CPU affinity. For a new abla-grad candidate, copy the
profile directory or use a fresh output prefix. Save an untraced GRDL
control for that candidate before tracing it. The preload is selected
with `LD_PRELOAD` and `GRAD_TRACE`; argument-shape recording is enabled
only by `GRAD_TRACE_SHAPES=1`. Normal invocations use the original
backend without any capture overhead. Both profiles recorded zero
dropped CUPTI records. The portable analysis inputs are the
`*.activity.bin`, `*.names.tsv` and abla-grad `*.shapes.bin` files.
Use `node analyze.mjs grad <CUPTI-include-dir>` or substitute `llama`.

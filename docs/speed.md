# Inkling-2-Mini decode and prefill speed (RTX 4090)

One table, updated with every speed change. Goal: beat mainline llama.cpp and Unsloth's build at every depth and in prefill, with `tools/gate-inkling.sh` passing; stretch line 200 tok/s decode.

Method: ours = `abla-grad decode-bench <gguf> <ids> --n 128 --mode graph` (greedy, N cached tokens then 128 generated; the cached tokens are prompt 04's ids cycled, so the text is junk but the routing realistic); gate = `GATE_LOGITS_FLAGS=--step tools/gate-inkling.sh`. llama.cpp mainline 33aeb17 and Unsloth b12649-mix-c7059d6: llama-bench tg128 -d N, same UD-Q4_K_M, CUDA graphs, F16 KV (ablac-df, raw data in ~/src/abla-grad-oracle/).

| tg128, tok/s | d0 | d15 | d300 | d512 |
|---|---:|---:|---:|---:|
| llama.cpp mainline 33aeb17 | 74.4 | 78.1 | 77.5 | 77.3 |
| Unsloth b12649-mix | 104.1 | 106.6 | 106.2 | 105.5 |
| abla-grad a02be45 | 96.5 | 95.7 | 81.5 | 73.8 |
| abla-grad + split attention (4c1ee1f) | | 98.9 | 96.1 | 93.7 |

| pp512, tok/s | |
|---|---:|
| Unsloth b12649-mix | 5,475 |
| llama.cpp mainline | 3,506 (74's figure) |
| abla-grad | prefill in progress (ablac-4e) |

## Log
- a02be45: graph decode, e7's router/RMSNorm/matvec work. Per-token cost grew with context (one attention block per head walked the context serially).
- split attention: positions of a head over 8 blocks plus a merge (sink once); d512 73.9 -> 93.7, ids identical, gate passes (324 positions, 0 argmax differences). Fused gate/up -> SiTU (75): flat, ~0%.
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

## GPU etiquette
Every model run, gate and benchmark goes through `tools/gpu-lock.sh [-t hold-seconds] -- command...` (flock on /tmp/abla-grad-gpu.lock, waits for 20 GB free, hard hold timeout, one run per hold). Timings taken next to another run are not trustworthy.

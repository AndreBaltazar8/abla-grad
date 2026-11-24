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

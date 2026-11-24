# Prefill (batched prompt processing) — owner ablac-4e

`model.prefill(backend, ws, ids, startPos)` (src/model/inkling_prefill.ab) runs a chunk of prompt tokens through the Inkling graph at once and
leaves the state ready for `forward`: the KV caches of the chunk's positions and the four short-convolution histories per layer visit. The block bank
(depth mixing) lives inside one token's forward, so it is per token and not state. `prefillWorkspace(backend, config, chunk, allLogits)` allocates the
activations of a chunk (about 0.5 GB at 512 tokens); `allLogits` also keeps the logits of every token of the chunk (for the gate).

## Numbers
Metric: prompt tokens per second, 512 tokens, logits of the last token (llama.cpp pp512 on this 4090: 3,506 tok/s). `abla-grad prefill-bench <model> --tokens 512`.
See the table at the end for each step.

## Matmul on tensor cores (src/kernels/prefill_matmul.ab)
`Y[token][row] = sum_k X[token][k] * W[row][k]`, X f32 [tokens][cols], W the GGUF rows in their stored type, Y f32 [tokens][rows]. A CTA: 64 tokens x 128
weight rows, eight warps of 32 x 32 (`mma.sync.m16n8k16` f16 inputs, f32 accumulators), k in steps of 32 with double-buffered shared memory. X is converted to
f16 on the way into shared memory; weights are dequantized (scale folded in) to f16 on the way into shared memory. NVFP4 is exact in f16; Q8_0 rounds
`scale * q` to f16 (relative 2^-11); the f16 rounding of X is the other error source. Measured against the f32 decode path: prompt 04 (300 positions) worst
KL 0.0038, 0 argmax differences; against llama.cpp's CUDA build the gate numbers in tests/tolerances-inkling.txt hold (worst KL 0.0116 of 0.04).

| kernel | parameters | problem |
|---|---|---|
| `matmul_<t>` (q8_0, nvfp4, bf16, f32) | `w, x, y, rows, cols, tokens` | `[rows, ceil(tokens/64)]` |
| `matmul_f32_exact` | same; f32 arithmetic, 16 x 16 tile (the router: expert selection must not move with rounding) | `[rows, ceil(tokens/16)]` |
| `matmul_<t>_grouped` (nvfp4, q8_0, q5_k) | `w, x, xmap, y, scales, plan, rows, cols, nExpert, expertStrideBytes` | `[rows, maxTiles]`, `maxTiles = pairs/64 + nExpert` |

Experts: `moe_route_batch` picks the experts of every token, `moe_sort` groups the (token, expert) pairs by expert and writes the plan (group offsets, tile
count, (expert, first token) tiles of 64 tokens); the grouped matmul runs one tile per block, rows of x gathered through `xmap`, y rows in sorted order,
times the expert's scale; SiTU runs over the sorted rows; the down projection reads them in place (`xmap` = identity); `moe_combine_batch` sums the
8 weighted rows of each token.

## Other batched kernels (src/kernels/prefill_small.ab)
`gather_rows_q8_0`, `copy_rows` (strided row copy: the block bank, slicing mixed queries), `depth_mix_batch` (bank `[token][block][n]`), `conv_seq` +
`conv_history` (the causal convolution over the sequence with the history in and out), `l2clip_rows` (the per-head scale repeats per token), `scale_tau`,
`kv_store_batch`, `attention_prefill_rel` (the decode kernel with the token as a grid dimension).
Existing elementwise kernels (`add`, `situ`, `sigmoid_mul`, `softcap`, `control_mix`, `rmsnorm`, `rel_table`, `copy_f32`) take their size as `n`/`rows` and run
unchanged over `tokens * size`.

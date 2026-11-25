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
`gather_rows_q8_0`, `copy_rows` (strided row copy: the block bank, slicing mixed queries), `depth_mix_batch` (shared with decode in `src/kernels/depth_mix.ab`; bank `[token][block][n]`), `conv_seq` +
`conv_history` (the causal convolution over the sequence with the history in and out), `l2clip_rows` (the per-head scale repeats per token), `scale_tau`,
`kv_store_batch`, `attention_prefill_rel` (the decode kernel with the token as a grid dimension).
Existing elementwise kernels (`add`, `situ`, `sigmoid_mul`, `softcap`, `control_mix`, `rmsnorm`, `rel_table`, `copy_f32`) take their size as `n`/`rows` and run
unchanged over `tokens * size`.

## Verification for speculative decoding (`verify`, `commitVerified`)
`prefillWorkspace(backend, config, chunk, allLogits = true, verifyRows = 3)` adds room for three snapshots of the convolution history (3.8 MB each).
`model.verify(backend, ws, ids, startPos)` runs the anchor token and up to two proposals as one causal batch: the KV rows of those positions are written,
the logits of row k are in the workspace (`readPrefillLogits(ws, k)`, `verifyArgmax(ws, rows)` gives the argmax of every row), and the live convolution histories
are **not** changed: `conv_history_rows` writes the history after row 1, 2, 3 into the snapshots instead. `model.commitVerified(ws, accepted)` copies snapshot
`accepted` over the live history (one launch; `accepted = 0` keeps the old state). KV rows beyond the accepted prefix are overwritten by later steps; the depth-mixing
bank and the loop's control state live inside one token's forward, so nothing else is stateful. `abla-grad verify-check <gguf> <ids>` checks the whole chain: prefill 100
tokens, verify 3, commit 2 or 0, one decode step, against a plain prefill (argmax identical, the decode step after the commit equals the verified row to 1e-3).
The rows use the f16 tensor-core matmuls, so a logit differs from the decode kernels' by their rounding (KL below 0.005 on the gate prompts); an argmax at a near tie can differ.

## Graph-static verification (`verifyStep`, `commitStep`; docs/spec-cycle.md)
`model.verifyStep(backend, ws, spec, step)` is the recordable form: it only launches kernels and reads everything that changes from cycle to cycle from device memory:
the three row token ids from `spec[0..2]`, the position P of row 0 from `step[1]`; it writes the argmax of the rows to `spec[4..6]`, the KV rows and drafter taps of
positions P..P+2 (taps via `tap_rows`, position from `step[1]`), and the per-row conv-history snapshots. `model.commitStep(backend, ws, step)` sets history := snapshot
`min(step[3], 2)` (`commit_select`; step[3] = accepted proposals a, so a + 1 rows are committed). The positional kernels (`kv_store_batch`, `scale_tau_rows`,
`attention_prefill_tc`, `tap_rows`) take a device pointer to P; the host prefill uploads a small control word per chunk. `abla-grad verify-check` runs both forms.

## Drafter taps
With `model.enableTaps` on, prefill and verify rows write the same position-major records as the step-mode decode: slots 0..2 = the layer inputs of visits 5, 23, 41,
slot 3 = the final-norm output (copied from the same vector the lm head reads). Measured against step-mode taps over 103 positions of prompt 04 (relative L2 error): slot 0
3e-4, slot 1 up to 0.07, slot 2 up to 0.37, slot 3 up to 0.38 — while the logits of the same positions agree to KL 1e-5. A final-norm tap that disagrees by 38% cannot
produce the same logits, so the deeper step-mode taps need a look (the prefill taps are copies of the vectors the prefill's own lm head used).

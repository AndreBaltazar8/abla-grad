# abla-grad kernel signatures (the contract between ops.ab and every backend)

Owner of this file: ablac-60-2 (ops/model). Implementer: ablac-e7 (CUDA C, `src/kernels/`), later e8's Abla kernels and other backends.
A kernel's **name** and **parameter list** are the contract. Parameters are packed by `KernelArgs` in this order: pointers (8 bytes)
and `u64`, then as written; `u32`/`f32` are 4 bytes. All activations are `f32`. `problem` is what ops.ab passes to
`Backend.launch`; the backend turns it into its launch geometry (CUDA: block size and items per block per kernel).
`<t>` is the weight storage type: `f32`, `bf16`, `q8_0`, `q4_k`, `q5_k`, `q6_k`, `nvfp4` (a type a model does not use needs no kernel).
In the CUDA C below `uchar` = `unsigned char`, `u32` = `unsigned`, `u64` = `unsigned long long`.

| kernel | parameters | problem | meaning |
|---|---|---|---|
| `matvec_<t>` | `const uchar* w, const float* x, float* y, u32 rows, u32 cols` | `[rows]` | `y[r] = dot(row r of w, x)`; row r starts at `r * rowBytes(t, cols)`; `cols` is a multiple of the type's block |
| `matvec_<t>_indexed` | `const uchar* w, const float* x, float* y, u32 rows, u32 cols, const u32* ids, u32 nUsed, u64 expertStrideBytes, u32 xStrideElems, const float* expertScales` | `[rows, nUsed]` | slot i: expert `ids[i]` starts at `ids[i]*expertStrideBytes`; `x` for slot i at `i*xStrideElems` (0 = broadcast); `y[i*rows + r]`, multiplied by `expertScales[ids[i]]` when `expertScales` is non-null (Inkling stores a per-expert f32 `ffn_*_exps.scale` tensor next to its NVFP4 weights; llama.cpp applies it to the mul_mat_id result) |
| `rmsnorm` | `const float* x, const float* w, float* y, u32 n, f32 eps` | `[nVectors]` | per vector (stride n): `y = x * rsqrt(mean(x^2) + eps) * w`; `w` has n values; in place allowed (`x == y`) |
| `add` | `const float* a, const float* b, float* out, u32 n` | `[n]` | `out = a + b` (`out == a` allowed) |
| `silu_mul` | `const float* gate, const float* up, float* out, u32 n` | `[n]` | `out = silu(gate) * up` |
| `softcap` | `float* x, f32 cap, u32 n` | `[n]` | in place: `x = cap * tanh(x / cap)` |
| `rope` | `float* x, const float* freqFactors, u32 nHeads, u32 headDim, u32 ropeDim, u32 pos, f32 freqBase, u32 mode` | `[nHeads * ropeDim / 2]` | rotate the first `ropeDim` values of each head in place; `mode` 0 = adjacent pairs (llama), 1 = split halves (neox); `freqFactors` may be null (llama 3 supplies `rope_freqs`) |
| `kv_store` | `const float* k, const float* v, float* kCache, float* vCache, u32 pos, u32 nKvHead, u32 headDim, u32 maxCtx` | `[nKvHead * headDim]` | cache layout `[kvHead][maxCtx][headDim]`; writes position `pos` |
| `attention_decode` | `const float* q, const float* kCache, const float* vCache, float* out, u32 nHead, u32 nKvHead, u32 headDim, u32 first, u32 last, u32 maxCtx, f32 scale` | `[nHead]` | per head h (kv head `h / (nHead/nKvHead)`): scores over positions `first..last` inclusive of `q . K * scale`, softmax, weighted sum of V into `out[h*headDim ..]` (`first` = window start; 0 for global layers) |
| `gather_row_<t>` | `const uchar* table, u32 token, float* out, u32 cols` | `[cols]` | `out = dequantized row token` of the embedding table |
| `topk_softmax` | `const float* logits, u32 nExpert, u32 k, u32* ids, float* weights, u32 mode` | `[1]` | the router: `mode` selects the normalization (fixed once Inkling's gating is read from llama.cpp); `ids`/`weights` hold k entries, highest first |
| `moe_combine` | `const float* expertOut, const float* weights, float* out, u32 rows, u32 nUsed` | `[rows]` | `out[r] += sum_i weights[i] * expertOut[i*rows + r]` |
| `argmax` | `const float* x, u32 n, u32* out` | `[1]` | index of the largest value (lowest index on ties) |
| `copy_f32` | `const float* src, float* dst, u32 n` | `[n]` | `dst = src` |

Order of work for e7: `matvec_f32` (the spike) -> `matvec_nvfp4_indexed` -> `matvec_q8_0` -> `matvec_bf16` -> `matvec_q5_k` ->
`rmsnorm`, `rope`, `kv_store`, `attention_decode`, `add`, `silu_mul`, `gather_row_<t>`, `argmax` (enough for the Llama-3.2-1B
bring-up, with `matvec_q4_k`, `matvec_q6_k`) -> `softcap`, `topk_softmax`, `moe_combine`, the indexed variants of the rest.
Tolerances: each kernel's test compares with an f64 CPU reference; matvec and attention at a relative 1e-4 (f32 accumulation),
elementwise ops at 1 ulp-ish (1e-6 relative); write the number in the test.
Where Inkling needs something this table does not cover (relative-position attention terms, gate/up softcap placement), I add a row
after reading llama.cpp's implementation at the oracle's pinned commit, and tell e7.

## Rope convention (Llama bring-up)
`rope` with `mode` 0 (adjacent pairs, llama.cpp's LLAMA_ROPE_TYPE_NORM): for head h and pair i in [0, ropeDim/2): the pair is `(x[2i], x[2i+1])`,
`theta_i = pos / (freqFactors[i] * freqBase^(2i/ropeDim))` (`freqFactors` all 1 when null), rotated by `theta_i`
(`x0' = x0 cos - x1 sin`, `x1' = x0 sin + x1 cos`). Llama-3.2 supplies `rope_freqs.weight` (f32, ropeDim/2 values) as `freqFactors`. Mode 1 (neox)
pairs `x[i]` with `x[i + ropeDim/2]`. The rest of each head (`ropeDim..headDim`) is unchanged.

## Inkling additions (semantics: docs/inkling-graph.md)
Weights that are small tensors (norm weights, conv weights, queries, clip scales, relative bias, router, loop controller tensors) are converted to f32
on load unless noted; kernels read them as `const float*`. Matvec kernels with `<t>` handle the big weights (Q8_0, Q5_K, BF16, NVFP4, F32).
| kernel | parameters | problem | meaning |
|---|---|---|---|
| `rmsnorm` (extended) | as above, with `w` allowed to be a null pointer (`u64(0)`) | `[nVectors]` | weightless form: `y = x * rsqrt(mean(x^2)+eps)` |
| `l2clip` | `const float* x, const float* scale, float* y, u32 rows, u32 dim, f32 t` | `[rows]` | per row: `v = scale ? x*scale : x` (scale has `rows*dim` values, one per element), `y = v * t / max(|v|_2, t)`; in place allowed |
| `scale_f32` | `float* x, f32 s, u32 n` | `[n]` | in place `x *= s` (tau, vocabulary softcap factors) |
| `sigmoid_mul` | `const float* a, const float* g, float* out, u32 n` | `[n]` | `out = a * sigmoid(g)` |
| `depth_mix` | `const float* bank, u32 nBank, const float* cur, const float* queries, u32 nQuery, float* out, u32 n, f32 eps` | `[nQuery]` | states = `bank[0..nBank)` (contiguous n floats each) + `cur`; each state weightless-rmsnormed with `eps`; per query q: `s_i = state_i . queries[q*n..]`, `/ sqrt(n)`, softmax over the nBank+1 states, `out[q*n..] = sum_i p_i state_i` (normalized states) |
| `conv_step` | `const float* x, float* history, const float* w, const float* dyn, float* out, u32 channels, u32 nConv, u32 heads, f32 outScale` | `[channels]` | the short causal conv: window = `history[0..nConv-1][c]` then `x[c]` (oldest first); `out[c] = (x[c] + sum_k w[k*channels + c]*window[k] + (dyn ? sum_lag dyn[head(c)*nConv + lag] * window[nConv-1-lag] : 0)) * outScale`; then history shifts by one and appends `x[c]`; `history` layout `[nConv-1][channels]`; `head(c) = c / (channels/heads)`; `outScale = 1/sqrt((dyn?2:1)*nConv+1)` (ops computes it); in-place on `x == out` allowed if the kernel reads before writing |
| `rel_table` | `const float* relBias, const float* r, const float* sinks, float* table, u32 nRel, u32 extent, u32 nHead` | `[nHead * (extent+1)]` | `table[h*(extent+1)+d] = sum_k relBias[k*extent+d] * r[h*nRel+k]` for d < extent; `table[h*(extent+1)+extent] = 0`; then every entry of head h `-= sinks[h]` |
| `attention_decode_rel` | `const float* q, const float* kCache, const float* vCache, const float* table, float* out, u32 nHead, u32 nKvHead, u32 headDim, u32 first, u32 last, u32 pos, u32 extent, u32 maxCtx, f32 scale` | `[nHead]` | per head: positions j in `first..last`: `logit = q.K_j*scale + table[h*(extent+1) + (pos-j < extent ? pos-j : extent)]`; softmax over those logits and one extra zero logit (the sink); `out = sum p_j V_j` |
| `moe_route` | `const float* logits, const float* bias, u32 nExpert, u32 k, u32* ids, float* weights, f32 normEps, f32 weightScale` | `[1]` | `p = sigmoid(logits)`; top `k` by `p + bias` (bias may be null; ties: lower index); `weights[i] = p[ids[i]]`; `weights /= (sum + normEps)`; `weights *= weightScale` |
| `situ` | `const float* gateUp, float* out, u32 nUsed, u32 nFf, f32 beta, f32 lb` | `[nUsed * nFf]` | per slot i, element j: `g = gateUp[i*2*nFf + j]`, `u = gateUp[i*2*nFf + nFf + j]`; `out[i*nFf+j] = beta*tanh(g/beta)*sigmoid(g) * (lb>0 ? lb*tanh(u/lb) : u)` |
| `control_mix` | `const float* cur, const float* message, const float* gain, const float* forget, float* out, u32 n` | `[n]` | `out = cur - tanh(forget)*cur + tanh(gain)*message` (gain/forget are the pre-tanh vectors) |
| `copy_at` | `const float* src, float* dst, u32 n` | `[n]` | like `copy_f32`; ops uses pointer offsets to concatenate (`dst` offset into a larger buffer) |
| `tanh_inplace`, `rmsnorm`-on-a-slice | covered by `softcap` with `cap`=1 and `rmsnorm` with pointer offsets | | |
`matvec_<t>_indexed` with `rows = 2*nFf` is the fused gate/up expert projection (`ffn_gate_up_exps`); `rows = n_embd, cols = nFf` the down projection.
Per-token scalars (`tau`, positions, windows) are computed on the host and passed as kernel parameters.
Order for Inkling after the Llama set: `rmsnorm` weightless, `l2clip`, `conv_step`, `depth_mix`, `rel_table`, `attention_decode_rel`, `sigmoid_mul`, `scale_f32`,
`moe_route`, `situ`, `matvec_nvfp4_indexed` (already first in line), `control_mix`.

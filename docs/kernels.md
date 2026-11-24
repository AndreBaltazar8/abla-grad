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
| `matvec_<t>_indexed` | `const uchar* w, const float* x, float* y, u32 rows, u32 cols, const u32* ids, u32 nUsed, u64 expertStrideBytes, u32 xStrideElems` | `[rows, nUsed]` | slot i: expert `ids[i]` starts at `ids[i]*expertStrideBytes`; `x` for slot i at `i*xStrideElems` (0 = broadcast); `y[i*rows + r]` |
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

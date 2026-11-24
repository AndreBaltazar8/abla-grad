# Inkling-2-Mini graph (decode, one token at a time), as llama.cpp implements it

Source of truth: llama.cpp at the oracle's pinned commit 33aeb17 (`/home/andre/src/llama.cpp`): `src/models/inkling2.cpp` (the graph),
`src/models/inkling2-loop.cpp` (the looped variant), `src/llama-graph.cpp` (`build_moe_ffn`, `LLM_FFN_SITU`),
`src/llama-kv-cache.cpp` (`set_input_relative_pos`). Where this note and that source disagree, the source wins; per-layer tensor dumps
(tools/oracle/dump-tensors.sh) are how we find out. Every ggml op name below is the llama.cpp one. This is **much more** than
"GQA + MoE": relative-position bias with sinks, short causal convolutions with state, block attention residuals ("depth mixing"),
clipped q/k, gated attention, a custom activation (SiTU), sigmoid routing with a selection bias, and (for `inkling2_loop`) a looped core
with controller gates. Our kernel list (docs/kernels.md) grows accordingly.

## Hyper-parameters (GGUF metadata, arch prefix `inkling2` or `inkling2_loop`)
`attention.layer_norm_rms_epsilon`, `attention.qk_clip`, `attn_res.block_size`, `shortconv.l_cache` (n_conv, 2..64),
`attention.relative_length` (n_rel), `attention.relative_extent`, `attention.relative_clip`, `attention.log_scale_floor`,
`attention.log_scale_alpha`, `vocab_size` (padded), `vocab_size_unpadded`, `expert_feed_forward_length` (scalar or per layer),
`expert_count`, `expert_used_count` (per layer), `activation.situ_beta`, `activation.situ_linear_beta`, `expert_weights_scale`,
`final_logit_softcapping`, `attention.sliding_window` (n_swa), `attention.sliding_window_pattern` (bool per layer: SWA or full),
`attention.head_count` / `head_count_kv` (per layer), `embedding_length`, `block_count`; loop variant: `loop.start`, `loop.end`,
`loop.count` (passes), `loop.gate_rank`, `loop.varying`. Fixed choices: no RoPE (the relative bias carries positions), router in f32,
expert weight normalization eps 1e-20.

## Tensors (names via llama.cpp's `tn(...)`; shapes {ne0, ne1} = {fast, slow})
Global: `token_embd` {n_embd, n_vocab_padded}, `token_embd_norm` {n_embd}, `output` {n_embd, n_vocab_padded}, `output_norm` {n_embd},
`output_res_query` {n_embd}. Per layer i: `attn_q` {n_embd, hd*n_head}, `attn_k`/`attn_v` {n_embd, hd*n_head_kv}, `attn_output`
{hd*n_head, n_embd}, `attn_gate` {n_embd, hd*n_head}, `attn_q_clip` {hd, n_head}, `attn_sink_proj` {n_embd, n_head}, `attn_rel`
{n_embd, n_rel*n_head}, `attn_rel_bias` {n_rel, n_extent} (n_extent = n_swa for SWA layers else relative_extent), `attn_rel_clip`
{n_rel, n_head}, `attn_k_conv`/`attn_v_conv` {n_conv, hd*n_head_kv}, `attn_post_conv`/`ffn_post_conv` {n_conv, n_embd}; SWA layers
`attn_norm` {n_embd}; full layers `attn_q_inp_norm`/`attn_k_inp_norm`/`attn_v_inp_norm` {n_embd} and `attn_k_conv_proj`/`attn_v_conv_proj`
{n_embd, n_conv*n_head_kv}; for i > 0 `attn_res_query` {n_embd, 1 (SWA) or 3 (full)}; `ffn_res_query` {n_embd}; `ffn_norm` {n_embd};
`ffn_gate_inp` {n_embd, n_expert}; optional `ffn_exp_probs_b` {n_expert}; `ffn_gate_up_exps` {n_embd, 2*n_ff_exp, n_expert} (fused:
rows 0..n_ff-1 gate, n_ff..2n_ff-1 up); `ffn_down_exps` {n_ff_exp, n_embd, n_expert}. Loop variant per core layer: `msg_query`
{n_embd, 4}, and for ctrl in (Q,K,V,FFN): `ctrl_down` {2*n_embd, rank}, `ctrl_gain` {rank, n_embd}, `ctrl_forget` {rank, n_embd}.
In the loop variant the core layers' weights are shared by every pass; each pass (cache slot) has its own KV cache and conv state.

## Primitives
- `rmsnorm(x, w, eps)`: x / sqrt(mean(x^2)+eps) * w; the weightless form (w absent) is the same without `* w`.
- `l2clip(x, scale?, t)` per row of length `hd` (or n_rel): `x' = x * scale` if scale (per channel), then `x' * t / max(|x'|_2, t)`.
- `mix(blocks, cur, query[n_q])` (depth attention): states = [blocks..., cur], each an n_embd vector; each state rms-normalized (weightless, eps 1e-6);
  for each query q: scores_s = (state_s . query_q) / sqrt(n_embd), softmax over s, output_q = sum_s scores_s * state_s (the **normalized** states).
- `convolve(cur, w, dynamic?, history)` (short causal depthwise conv with residual): window of the last n_conv inputs of each channel (history
  + the current input, current last); `out[c] = sum_{k<n_conv} w[k][c] * window[k][c]` (w[n_conv-1] multiplies the current input; ggml_ssm_conv);
  if `dynamic` ([n_heads][n_conv] taps from `attn_k_conv_proj`/`attn_v_conv_proj`·input): add `sum_{lag<n_conv} dynamic[head(c)][lag] *
  window_lag[c]` where window_lag is the input `lag` steps back (lag 0 = current), head(c) = c / (channels/n_heads); result
  `(cur + out) / sqrt((dynamic ? 2 : 1) * n_conv + 1)`. The history keeps the last n_conv-1 raw inputs of each convolution.
- `situ(gate, up)` = `beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)` (lb > 0; if lb == 0 the up factor is just `up`).
- `tau(pos)` = `1 + alpha * log(max(1, (pos + 1) / floor))`, a scalar per token (computed on the host).

## One layer (visit `ip`, cache slot `il`), single token, input `cur`, bank `blocks`
```
inpSA = cur
inputs = [cur, cur, cur]
if i > 0 (attn_res_query exists):  mixed = mix(blocks, cur, attn_res_query);  SWA: inputs = [mixed[0]]*3;  full: inputs[j] = mixed[j] (j = Q,K,V)
(loop pass > 0: inputs[j] = control(inputs[j], message(Q+j)) for j in 0..2)
SWA:   inputs[j] = rmsnorm(inputs[j], attn_norm, eps)        (all three the same unless looped)
full:  inputs[0] = rmsnorm(inputs[0], attn_q_inp_norm), inputs[1] = rmsnorm(inputs[1], attn_k_inp_norm), inputs[2] = rmsnorm(inputs[2], attn_v_inp_norm)
Q = attn_q . inputs[0]                                         (no conv)
K = convolve(attn_k . inputs[1], attn_k_conv, dynamic = full ? attn_k_conv_proj . inputs[1] : none, history offset 0)
V = convolve(attn_v . inputs[2], attn_v_conv, dynamic = full ? attn_v_conv_proj . inputs[2] : none, history offset n_embd_k_gqa)
R = attn_rel . inputs[0]                                       ([n_rel, n_head])
Q = l2clip(Q per head, attn_q_clip, qk_clip);  K = l2clip(K per kv-head, none, qk_clip);  V = rmsnorm_noweight(V per kv-head, eps);  R = l2clip(R per head, attn_rel_clip, relative_clip)
full layers: Q *= tau, R *= tau
sinks[h] = (attn_sink_proj . inputs[0])[h]
table[h][d] = sum_r attn_rel_bias[r][d] * R[h][r]  for d in [0, n_extent); table[h][n_extent] = 0   (the pad row);  then table[h][*] -= sinks[h]
KV store: K, V at the slot's cache position (cache holds the processed K and V)
attention over cached positions j in the layer's window (SWA: last n_swa positions; full: all, with distance extent clamp):
    logit_j = (Q_h . K_j) / sqrt(hd) + table[h][ d_j ],   d_j = pos - pos_j  if 0 <= d_j < n_extent else n_extent
    softmax over j with one extra zero logit (the sink); out_h = sum_j p_j V_j      (sink contributes no value)
cur = out * sigmoid(attn_gate . inputs[0])
cur = convolve(attn_output . cur, attn_post_conv, none, history offset 2*n_embd_k_gqa)
if ip % attn_res_block_size == 0:  blocks.append(inpSA)   (cur is NOT added: the block output replaces the stream)
else:                              cur = inpSA + cur
ffn_inp = cur
cur = mix(blocks, cur, ffn_res_query)    (loop pass > 0: cur = control(cur, message(FFN)))
cur = rmsnorm(cur, ffn_norm)
MoE: logits = ffn_gate_inp . cur (f32); p = sigmoid(logits); select top n_used by (p + ffn_exp_probs_b) (bias only for selection);
     w = p[selected]; w = w / (sum(w) + 1e-20); w *= expert_weights_scale (if set and != 1)
     gu = ffn_gate_up_exps[e] . cur for each selected e  (rows 2*n_ff); a = situ(gu[:n_ff], gu[n_ff:]);  d = ffn_down_exps[e] . a
     moe = sum_i w_i * d_i
cur = convolve(moe, ffn_post_conv, none, history offset 2*n_embd_k_gqa + n_embd)
cur = ffn_inp + cur
```
Embedding: `cur = rmsnorm(gather(token_embd, token), token_embd_norm)`. Output: `cur = rmsnorm(mix(blocks, cur, output_res_query), output_norm)`;
`logits = output . cur` (f32 accumulate), `if final_logit_softcapping: logits = cap * tanh(logits / cap)`, then the first `vocab_size_unpadded` entries.
`control(cur, message)` (loop variant): `g = ctrl_down . concat(rmsnorm0(cur), rmsnorm0(message))` (rank), `gain = tanh(ctrl_gain . g)`,
`forget = tanh(ctrl_forget . g)`, result `cur - forget*cur + gain*message` (rmsnorm0 = weightless, eps 1e-6). The loop's messages come from
`mix(bank, bank_out, msg_query)` ([4][n_embd] = Q, K, V, FFN messages) where `bank` = the previous pass's blocks after the first core block.
Visit order (loop): layers [0, core_start), then `loop.count` passes over [core_start, core_end) (each pass starts from the prelude's output and the
prelude's blocks), then [core_end, n_layer). Block size divides core_start and core_end.

## Consequences for the plan
- New kernels beyond the Llama set: see docs/kernels.md "Inkling additions": `rmsnorm` with an optional weight, `l2clip`, `depth_mix`,
  `conv_step`, `rel_table`, `attention_decode_rel` (bias table + sink + window), `sigmoid_mul`, `scale_f32`, `moe_route`, `situ`, `control_mix`
  (+ `concat`/`copy` at offsets).
- State per cache slot: KV cache, conv history ([n_conv-1][channels] with channels = 2*n_embd_kv + 2*n_embd), the block bank (one n_embd vector per block).
- Prefill = repeated decode (conv history and KV are sequential anyway).
- The small proof model (Llama-3.2-1B) exercises only the Llama subset; the Inkling-specific kernels are proven layer by layer against
  `dump-tensors.sh` on Inkling itself.

## What the actual file has (Inkling-2-Mini-UD-Q4_K_M.gguf, 744 tensors, read with our reader)
Metadata: arch `inkling2_loop`, 24 layers (block size 6), n_embd 2304, 24 heads x 128 head dim, 8 kv heads, 128 experts (8 used, n_ff_exp 1152), vocab 201024 padded /
200058 real, loop start 6 / end 18 / count 3 / gate rank 64, n_conv 4, n_swa 1024, relative length 16 / extent 1024 / clip 8, qk_clip 22.627, log scale floor 8192 /
alpha 0.1, situ beta 4 / linear beta 25, expert weights scale 8, final logit softcap 30, rms eps 1e-6. Visits: layers 0-5, three passes over 6-17, then 18-23 = 48 visits.
Full-attention layers are 5, 11, 17, 23 (they carry attn_q/k/v_inp_norm and attn_k/v_conv_proj, BF16 {2304, 32}).
Types: token_embd and output Q8_0; attn q/k/v/o/gate and attn_rel Q8_0; attn_sink_proj and the conv_proj tensors BF16; routers, norms, conv weights, clips, rel bias, loop controllers,
expert bias `exp_probs_b.bias` and the per-expert `ffn_*_exps.scale` ({128}) F32; experts: gate_up NVFP4 (layers 0-19) / Q5_K (20-23), down NVFP4 (0-19) / Q8_0 (20-23).
The `.scale` tensors multiply the mul_mat_id result per selected expert. So the indexed kernels needed: nvfp4, q5_k, q8_0 (all with `expertScales`).
Sizes: weights 16.6 GB; the embedding and output are 492 MB each.

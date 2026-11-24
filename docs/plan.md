# abla-grad plan

## Targets and artifacts
- Hardware: RTX 4090 (24 GB, sm_89), driver 580.95, libcuda from the driver, NVRTC 12.9 (nix `cudaPackages.cuda_nvrtc`).
- Small standard model (proof): an ungated llama-family q4_K_M GGUF of at most 1 GB (df picks and pins it; preference:
  Qwen2.5-0.5B-Instruct Q4_K_M, byte-level BPE like Inkling's tokenizer). Tensor types: Q4_K, Q6_K, F32 (+ Q8_0 if present).
- Inkling-2-Mini: df's chosen Unsloth "Mainline_Support" Q4_K_M. 90's census of its first 16 MiB: 744 tensors, F32 x518,
  Q8_0 x150, BF16 x32, NVFP4 x40 (the fused experts, the bulk of the weights), Q5_K x4; **no Q4_K/Q6_K**. NVFP4 is the
  original checkpoint's own format, so this is the faithful artifact. Kernel scope follows the artifact: NVFP4, Q8_0,
  Q5_K, BF16, F32 for Inkling; Q4_K and Q6_K only for the small test model.
- Inkling text-only architecture (from #1232, to verify against the GGUF metadata and llama.cpp's inkling2): GQA 24/8
  heads, head 128; sliding window 1024 with 4 global layers; relative positions (d_rel / rel_extent); MoE with 128
  experts, top-8, gate/up softcaps; a looped core (layers 6-17 run 3 times); final logit softcap; byte-level BPE
  (199,998 base tokens, 60 specials, ignore_merges, GPT-4o pre-tokenizer, no BOS). Vision and MTP later.

## Stages (each ends with a measurable gate; the first three run in parallel)
S0  repo, contracts, backend interface (this commit).                                       owner: 60-2
S1a GGUF reader + quant layouts + `abla-grad info` (type census of any GGUF).               owner: 90
S1b tokenizer (tokenizer.json, and from GGUF metadata for the small model), `tokenize` cmd.  owner: 90
S1c oracle: llama.cpp built, models in ~/models, `dump-logits.sh`, `dump-tokens.sh`,
    `dump-tensors.sh` (per-layer tensors through llama.cpp's eval callback).                 owner: df
S1d CUDA spike -> backend: libcuda + NVRTC from Abla, compile + launch `matvec_f32`,
    then `CudaBackend` behind the `Backend` interface (docs/interfaces.md).            owner: e7 (+ e8 for #1233 later)
S2  small model forward on the GPU (decode, one token at a time; prefill = repeated decode);
    gate: on the pinned prompts the top-1 matches the oracle at every position and the
    logits are within the calibrated tolerance below.                                         owner: 60-2 (model), e7 (kernels)
S3  Inkling text-only: NVFP4/Q8_0/Q5_K/BF16 matvec, MoE (router, top-8, indexed experts),
    sliding+global attention, looped core, softcaps; same gate on Inkling fixed prompts.     owner: 60-2 + e7
S4  speed: prefill as matmul, fused norms, coop/tensor-core paths, a chat `generate` command.
S5  swap CUDA C for Abla kernels (#1233); a second backend (Vulkan or CPU) on the same kernel names.

## How we compare with llama.cpp
- Oracle files: GRDL v1 (docs/interfaces.md): the logits after every prompt position, f32, plus a JSON sidecar with the
  prompt token ids, model SHA-256, llama.cpp commit and flags. Prompts are fixed token-id files in tests/prompts/
  (the text and llama.cpp's own tokenization are dumped next to them, so the tokenizer is tested separately).
- Metrics per position: argmax equal, top-5 set equal, max |dlogit|, mean |dlogit|, KL(oracle || ours) over softmax.
- Tolerance is calibrated, not guessed: llama.cpp quantizes activations (q8_K / q8_1) in its CPU and GPU dot products,
  and we use f32 activations, so exact equality is not expected. The noise floor is measured as llama.cpp CPU vs
  llama.cpp CUDA (same model, same prompts); our tolerance is a fixed multiple of it, written into tests/tolerances.txt.
- When a gate fails, `dump-tensors.sh` gives llama.cpp's named intermediate tensors for the first token; the compare
  tool finds the first tensor that leaves tolerance (embedding, layer N attn_norm, ... ), so the bug is localized by layer.

## Kernel list (CUDA C, one `.cu` source string per kernel, names are the contract between ops.ab and every backend)
Matvec y = W x with W stored as GGUF rows (one contiguous row per output; x, y f32):
  matvec_f32, matvec_bf16, matvec_q8_0, matvec_q4_k, matvec_q5_k, matvec_q6_k, matvec_nvfp4
Expert-indexed matvec (W is a fused [n_expert][rows][cols] tensor; ids[n_used] selects):
  matvec_nvfp4_indexed, matvec_q8_0_indexed (the types the Inkling experts use; others on demand)
Elementwise / small: rmsnorm, add, scale, silu_mul (swiglu), softcap (tanh), gather_row_<type> (embedding row dequant),
  rope (variants fixed when Inkling's relative positions are verified), softmax, topk_softmax (router), argmax.
Attention (decode): scores = q . K^T with GQA, mask (causal, sliding window), softmax, . V; KV cache f16 or f32 (flag).
Prefill matmul (S4): matmul_<type> tiled; until then prefill = repeated decode.

## Risks
- NVRTC needs CUDA headers for cuda_fp16.h / cuda_bf16.h (nix `cuda_cudart` dev output): e7's spike settles it.
- Inkling's exact attention (relative positions) and the NVFP4 block layout must be read from llama.cpp's source at the
  oracle's pinned commit (the GGUF type id and layout are defined there), not guessed.
- VRAM: file size of the Inkling GGUF is unknown to me; df reports it. 24 GB may need expert offload or a smaller quant.

## Status (updated as stages finish)
- S0 contracts, S1a GGUF reader + `info`, S1b tokenizer, S1c oracle (llama.cpp 33aeb17, models pinned, GRDL dumps, tensor dumps, CPU/CUDA noise), S1d CUDA backend + kernel registry: done.
- S2 small model: done. `tools/gate-llama.sh`: 266 positions over 4 prompts vs llama.cpp CUDA, all inside tests/tolerances.txt (worst KL 0.0047), 1 argmax flip at a near tie.
- S3 Inkling: model code (src/model/inkling.ab, inkling_config.ab) written to docs/inkling-graph.md; debug dumps with the oracle's names (`abla-grad logits --dump DIR`,
  `abla-grad compare-tensors ORACLE_DIR DIR`) find the first diverging tensor. Waiting on the quantized kernels (Q8_0, BF16, NVFP4/Q5_K/Q8_0 indexed, gather Q8_0).

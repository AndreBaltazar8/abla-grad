# Inference engine

- [GGUF reader](../src/gguf/reader.ab) maps the file read-only. Tensor addresses stay valid until close. [Quantization layouts](../src/gguf/quant.ab) define block sizes and row strides.
- [Tokenizer](../src/tokenizer/tokenizer.ab) loads byte-level BPE from GGUF or Hugging Face JSON. Fixed IDs and special-token cases live in [tests/prompts](../tests/prompts).
- Model code calls [typed ops](../src/ops.ab) over [Backend](../src/backend/backend.ab). A device buffer belongs to its allocating backend; drop releases it. Kernel parameters are ordered by the op and launch geometry comes from the backend registry.
- [CUDA](../src/backend/cuda.ab) uses the driver API, CUDA graphs and NVRTC. Kernel sources and ABIs live in [src/kernels](../src/kernels); CPU reference implementations live in the tests. The backend also loads compiler-emitted PTX for Abla `@kernel` definitions.
- [The fusion generator](../src/fuse) emits CUDA kernels from supported op chains. `tools/test-fuse.sh` covers accepted and rejected chains.

## Inkling

[Configuration](../src/model/inkling_config.ab) reads the model's metadata.
[Decode](../src/model/inkling.ab) implements relative-position attention with
sinks and gates, short causal convolution, depth mixing and SiTU experts.
Loop passes share weights and keep independent KV and convolution state.
The pinned model visits 48 layers, with eight global and forty sliding visits.

Global KV switches from F16 to block Q8 at 8,192 allocated positions. On
Ada, E4M3 replaces block Q8 from 65,536 positions. The
grouped decode kernel shares KV across three query heads. Sliding attention
sees 1,024 keys and uses a 1,536-slot ring for long contexts, switching from
F32 to F16 at 8,192 allocated positions. Physical ring offsets wrap;
logical positions still drive masking and relative bias.

[Prefill and verification](../src/model/inkling_prefill.ab) operate on rows in
batches and keep sequential convolution histories. Large prefill batches
segment KV stores and attention so sliding queries retain their full window.
Packed expert weights stay on device. Eligible prefill projections and
experts use int8 tensor cores; verifier kernels reuse weight loads across
rows and group the union of routed experts.

The 64-token expert tile uses int8 activations for NVFP4, Q8_0 and Q5_K.
The Q5_K kernel uses the activation sum to apply the weight block's
affine minimum. NVFP4 targets three CTAs per Ada SM and the gate/up
kernel applies SiTU before writing the down projection's input.

## DSpark

[Speculation](../src/model/speculate.ab) drafts from target hidden-state taps.
The target checks the anchor plus proposals in a CUDA graph and accepts a
proposal when it matches the target argmax after the previous row. It commits
the anchor and accepted prefix, restores the corresponding convolution
snapshot, and overwrites unused KV rows later. The next anchor is the argmax
after the last committed row. [Step kernels](../src/kernels/step.ab) define the
device buffer layout.

The controller compares plain decode with cycle graphs up to the configured
proposal cap using measured costs and recent acceptance. The default cap is
two. `spec-bench` measures throughput, `spec-accept` traces proposal
acceptance, and `spec-check` reports logit margins for greedy differences.
Numerical near-ties can change the token selected by multi-row verification.
See [the oracle guide](../tools/oracle/README.md) for the correctness gates.

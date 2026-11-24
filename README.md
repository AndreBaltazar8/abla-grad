# abla-grad

Run large language models entirely in Abla, on the GPU. First target: `thinkingmachines/Inkling-2-Mini` (Unsloth
GGUF), text only. Proof model first: a small standard GGUF. llama.cpp is the **oracle** (logits to compare against),
never a dependency of the program.

Status (2026-11-24): the small model (Llama-3.2-1B) runs on the GPU and matches llama.cpp (tools/gate-llama.sh); Inkling is written and waits for its quantized kernels. See `docs/plan.md` (stages, kernel list, how we compare) and `docs/interfaces.md` (the
contracts between the parts and who owns what). Local repo only until André OKs a GitHub repo. Model files live in
`~/models`, never in the repo.

## Layers (nothing above a layer touches the one below it)

```
src/main.ab, src/cli/        commands: info, tokenize, logits, compare, generate
src/model/                   config from GGUF metadata, weights, KV cache, the graph (llama-family, then Inkling)
src/ops.ab                   typed ops over Backend: matvec(format, ...), rmsnorm, rope, attention, moe, ...
src/backend/backend.ab       the backend interface: buffers, kernel launch, synchronisation
src/backend/cuda.ab          first implementation: the CUDA driver API (libcuda) + NVRTC, `trusted extern` in this module only
src/kernels/*.cu             CUDA C kernels, embedded in Abla (src/kernels/*.ab holds the source) and compiled by NVRTC at startup
src/gguf/, src/tokenizer/    host side: GGUF reader (mmap), quant layouts, tokenizer (byte-level BPE)
tools/oracle/                llama.cpp build, model fetch, per-token logit dumps (shell, no Python)
tests/                       one test per module; CPU references live in tests only
```

Later backends (Vulkan through abla-graphics, CPU, Metal) and Abla-written kernels (#1233, PTX through LLVM's NVPTX)
implement the same `Backend` interface and the same kernel names; the model code does not change.

## Running (once the pieces exist)

```
nix-shell shell.nix --run 'make test'
abla-grad info    ~/models/<model>.gguf              # header, metadata, tensor type census
abla-grad logits  ~/models/<model>.gguf prompt.ids -o out.grdl
abla-grad compare out.grdl oracle.grdl               # per-token argmax / top-5 / max abs diff / KL
```

## Rules

- GPU is the primary target; a CPU path comes later from the same kernels. No separate CPU engine first.
- Weights stay in their stored quantization on the GPU (dequantized inside the matvec kernels); activations are f32.
- Every kernel has a test with a CPU reference **in the test only**, and a tolerance written down.
- A change to a contract in `docs/interfaces.md` is announced to the owners of both sides first.

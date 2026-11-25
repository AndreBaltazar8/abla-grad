# abla-grad

Run large language models entirely in Abla, on the GPU. First target: `thinkingmachines/Inkling-2-Mini` (Unsloth
GGUF), text only. Proof model first: a small standard GGUF. llama.cpp is the **oracle** (logits to compare against),
never a dependency of the program.

Status (2026-11-24): the small model (Llama-3.2-1B) runs on the GPU and matches llama.cpp (tools/gate-llama.sh); Inkling is written and waits for its quantized kernels. See `docs/plan.md` (stages, kernel list, how we compare) and `docs/interfaces.md` (the
contracts between the parts and who owns what). Local repo only until André OKs a GitHub repo. Model files live in
`~/models`, never in the repo.

## Layers (nothing above a layer touches the one below it)

```
src/main.ab, src/cli/        commands: info, tokenize, logits, compare, generate, chat (+ benchmarks and checks, see docs/speed.md)
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
abla-grad generate ~/models/<model>.gguf --chat "What is 2 + 2?" --max-tokens 32
abla-grad compare out.grdl oracle.grdl               # per-token argmax / top-5 / max abs diff / KL
```

`generate` accepts one of `--prompt TEXT`, `--prompt-file PATH`, or `--chat TEXT`. The chat option renders a single
user message with the model’s text chat template. The embedded GGUF tokenizer is used by default. For Inkling,
`--tokenizer PATH` selects its separate `tokenizer.json`. Generation is greedy by default. Use `--temp T` with
`--top-p P` and `--seed S` for reproducible sampling. `--ctx N` sets the total prompt and continuation budget.
Output streams to stdout as complete UTF-8 and ends at the model’s end-of-turn token or the token limit.

## Chatting with Inkling

```
nix-shell shell.nix --run 'abla-grad chat ~/models/inkling-2-mini/Inkling-2-Mini-UD-Q4_K_M.gguf --tokenizer ~/models/inkling-2-mini/tokenizer.json'
```

A multi-turn chat on the 4090. What you type is read line by line; the reply streams as it is decoded. The model, the
drafter and the whole conversation stay on the GPU between turns: the KV caches, the short-convolution histories and the
drafter's taps are kept, so a turn processes only its own new tokens (a few dozen for a typical message, a chat-length paste
through the batched prefill) and never the history again. Replies are greedy and use speculative decoding with the DSpark
drafter (`docs/speculation.md`): a cycle drafts a few tokens and the target verifies them in one batch, and a controller
falls back to plain decoding whenever the text does not repay speculation. Typical speeds on a 4090: about 125-130 tok/s
on prose, 150-250 tok/s on predictable text such as lists and code. The first turn also loads the model (about ten seconds).

- The drafter file `dspark-Inkling-2-Mini-Q8_0.gguf` is looked up next to the model (or `--drafter PATH`; Unsloth's
  `Mainline_Support/dspark-Inkling-2-Mini-Q8_0.gguf`, Apache-2.0, sha256 `3da00cf7c31bbe5662b3c0d373ecf782d74d32ef43990acc158ce60c2add72e0`).
- Thinking is hidden (shown as a dim `...` while it runs); `--show-thinking` prints it in grey.
- `--system TEXT` sets a system message, `--ctx N` the conversation budget in tokens (default 8192), `--max-tokens N` the longest reply (default 1024),
  `--speculate off` plain decoding only.
- At the prompt: `/reset` starts a new conversation, `/quit` leaves (so does end of input), `/help` lists them. A conversation that no longer fits
  `--ctx` is refused until `/reset`.
- After every turn a line on stderr reports the tokens processed, the tokens generated and the decode speed.
- Piped input works (`printf 'Hi\n/quit\n' | abla-grad chat ...`): one line per turn.

## Lossless packing for the Inkling target

The packer checks every FP32 word in each router and loop-controller matrix. It requires every selected word to have zero low 16 bits. An inexact word produces an error identifying its tensor and element before the converter publishes the output. For the pinned UD-Q4_K_M model this saves 70,778,880 bytes across 168 matrices. Abla reads the packed router and controller weights natively. Decode and verification expand each BF16 value exactly inside their FP32 reductions. Prefill keeps its original router and controller reduction order. Loading the original GGUF uses its existing F32 path.

From the project’s Nix shell, choose a new output path.

```sh
ablac build tools/pack-inkling.ab --fast -o build/pack-inkling
dir="$HOME/models/inkling-2-mini"
build/pack-inkling "$dir/Inkling-2-Mini-UD-Q4_K_M.gguf" "$dir/Inkling-2-Mini-UD-Q4_K_M-lossless-BF16.gguf"
```

The packed file for the pinned original has SHA-256 `d081fc3aafe0e2868a016ef3c9d0a891a01ba7b5f7ad36751357a0a6b4625267`. llama.cpp can load this file and selects its own kernels based on the stored types.

## Rules

- GPU is the primary target; a CPU path comes later from the same kernels. No separate CPU engine first.
- Quantized weights stay packed on the GPU and matvec kernels dequantize them. Activations are f32. The losslessly packed router and controller matrices also retain BF16 storage on device.
- Every kernel has a test with a CPU reference **in the test only**, and a tolerance written down.
- A change to a contract in `docs/interfaces.md` is announced to the owners of both sides first.

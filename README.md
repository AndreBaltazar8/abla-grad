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

## Serving Inkling over HTTP

`abla-grad serve <model.gguf> [--port N] [--ctx N] [--tokenizer PATH] [--drafter PATH] [--model-name NAME] [--max-tokens N] [--capture-requests FILE]` starts an OpenAI-compatible
server on 127.0.0.1 (default port 8080; without `--ctx` the context is chosen from the free device memory at startup and printed; the drafter `dspark-Inkling-2-Mini-Q8_0.gguf` is looked for next to the model). Point a coding
tool at `http://127.0.0.1:8080/v1`; any API key works.

- `GET /v1/models`, `GET /health`, `POST /v1/chat/completions` (JSON, or `"stream": true` as server-sent events with `stream_options.include_usage`).
- Tools: `tools` / `tool_choice` go in, `tool_calls` come out (`finish_reason: "tool_calls"`), `role: "tool"` messages go back. The prompt is rendered the way
  Inkling's own chat template does it. The model's thinking arrives as `reasoning_content`, the answer as `content`.
- `max_tokens` (default: all the context the prompt leaves), `stop` and `reasoning_effort` are honoured. Decoding is greedy: `temperature` and `top_p` are accepted and ignored.
- One request at a time (the rest wait in the listen queue). The KV cache and speculation state stay between requests: a request whose prompt starts with
  the previous conversation only processes the tokens after it (`usage.prompt_tokens_details.cached_tokens` tells how many were reused). Each request is
  logged on stderr with prompt, reused and generated tokens, time to first token and tok/s.
- `--ctx 32768` fits the Q4_K_M target and Q8_0 drafter on the RTX 4090. At this capacity the target uses 2.47 GiB for K and V. Prefill uses a smaller chunk if its initial 512-token workspace cannot fit.
- For an explicit long serving session, pass `--ctx 65536` or `--ctx 131072` to `serve`. Starting at 65,536 allocated positions, target K/V stores round to F16 and DSpark projects committed taps into its own cache as the prompt arrives. Above 98,304 allocated positions the eight global target visits use 32-channel blocks with signed 8-bit values and an FP32 scale. The forty sliding target visits keep F16 rings. DSpark keeps one full F16 visit, two sliding F16 rings and a 2,048-position buffer of FP32 taps. The target’s three passes through a repeated layer retain independent K/V.
- The server saves a checkpoint immediately before the final prompt token, including convolution histories and the live sliding-cache windows. A repeat can reuse this prefix after long generation. With space for a second checkpoint, a prompt edit can also rewind to the most recent shared 4,096-token stride, aligned to a prefill chunk boundary. The server reports the number of tokens actually reused.
- `tools/test-serve.sh` (under `tools/gpu-lock.sh`) checks the shapes, a tool-call round trip, prefix reuse and that greedy output equals `generate`.

### Record and replay a session

To record a session, start the server under `tools/gpu-lock.sh` with `--capture-requests /path/to/new.jsonl` and point the client at it. The server creates a new
mode-0600 file and writes each chat POST body as a JSONL string in `body`. These bodies can contain private code, tool results and credentials pasted into messages.
Keep the capture local. The capture contains neither HTTP authorization headers nor model responses. Stop the capture server by its listening PID before replaying.

From the Nix shell, replay the capture with the same model and serving options.

```sh
tools/serve-replay /path/to/session.jsonl /path/to/model.gguf --ctx 131072 --port 18081
```

The tool takes the GPU lock, starts a fresh loopback server, sends the original bodies in order and stops its own listener on exit. It supports streaming and plain
responses and never executes tool calls. Recorded tool results and assistant turns travel in subsequent request bodies. The Markdown table pairs server prompt
and reused-token counts, first raw token and decode rate with client HTTP wall time. It excludes model startup. Set `ABLA_GRAD_BIN` to select a binary. The tool
uses Bash, curl, jq and ss. Use an outer `tools/gpu-lock.sh -t SECONDS --` when the session needs longer than the default hold.

For a main landing that changes decode, prefill or serving, run `tools/serve-replay-landing SHA CAPTURE MODEL NEW_RESULTS_DIR [PORT]` from the Nix shell.
It builds the exact landing in a detached worktree, replays the fixed eight-request 75k capture and saves the full table and one comparison row. Record that row
in `docs/speed.md`. The fixed capture remains local to the benchmark machine.

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

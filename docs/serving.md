# Serving Inkling

Put `dspark-Inkling-2-Mini-Q8_0.gguf` next to the target or pass `--drafter`.
The drafter is Unsloth's Apache-2.0 `Mainline_Support/dspark-Inkling-2-Mini-Q8_0.gguf`,
SHA-256 `3da00cf7c31bbe5662b3c0d373ecf782d74d32ef43990acc158ce60c2add72e0`.
[Pinned download](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/8078d149286eec7193795413b9e0643da6d11fa1/Mainline_Support/dspark-Inkling-2-Mini-Q8_0.gguf_file)

`--tokenizer PATH` overrides the tokenizer embedded in the GGUF.

- `serve` listens on 127.0.0.1:8080 by default. Omit `--ctx` to size the context from free device memory.
- `GET /health`, `GET /v1/models`, and `POST /v1/chat/completions` support OpenAI clients. No API key is required.
- Chat accepts JSON or SSE, including streaming usage, tool calls and tool results. Reasoning text is returned in `reasoning_content`.
- The server honors `max_tokens`, `stop` and `reasoning_effort`. Generation is greedy.
- Tool calls enforce `required` fields in the argument object and in nested objects declared with `properties`. If a call closes early, the server resumes generation at the missing field within the request budget.
- The server warms the short and batched prefill kernels before it starts listening. A fresh CUDA cache can add compilation time to startup.
- CUDA images are cached in `$XDG_CACHE_HOME/abla-grad/cuda-images-v1` (default `~/.cache/abla-grad/cuda-images-v1`). A kernel source, GPU architecture, or driver change selects a new entry. Each entry carries a checksum. Invalid entries are rebuilt. The cache prefers cubins and falls back to PTX if CUDA cannot link a native image. `tools/test-cuda-cache.sh` exercises cold and warm loads, invalidation and corruption recovery.
- Tokenizer tables are cached in `abla-grad/tokenizers-v1` under the same cache root. The source key is the exact JSON SHA-256 for `--tokenizer`, or the GGUF header and metadata SHA-256 for an embedded tokenizer. The loader checks the checksum and table bounds before use.
- Weight loading uses two CUDA streams and a pair of bounded, pinned staging slots per stream. If pinning is unavailable, it retries with synchronous uploads.
- One request generates at a time. Complete requests queue in FIFO order within each session, and another ready session gets a turn before the previous one can run again. Partial HTTP requests stay out of the ready queue and have a 30-second deadline.
- Send `X-Session-ID` to identify a conversation. Serve also recognizes the OpenCode session headers. KV and convolution state, including both prompt checkpoints, stay resident only while that session owns the single GPU slot. Switching sessions invalidates them, so a returning session prefills again. Requests without a session header keep the original unlabelled prefix-reuse behavior. Usage reports the actual reuse in `prompt_tokens_details.cached_tokens`.
- `tools/gpu-lock.sh -- tools/test-serve.sh` checks response shapes, tool calls, prefix reuse and greedy parity.
- `tools/test-allocation-failure.sh` injects CUDA allocation failures in `decode-bench` and `serve` and checks the requested byte counts and exit codes. It takes the GPU lock for each run.
- For local interactive use, `chat MODEL` supports `/reset`, `/quit`, `/help`, `--show-thinking` and `--speculate off`. `generate MODEL --chat TEXT` streams one response; use `--temp`, `--top-p` and `--seed` for sampling.

## Capture and replay

Start the server with `--capture-requests FILE` to record each chat POST body as
JSONL. The new file has mode 0600. Keep it private: bodies can contain source,
tool results and secrets. It omits HTTP headers and server responses.
Stop the capture server before replay.

```sh
tools/serve-replay "$capture" "$model" --ctx 131072 --max-tokens 512 --port 18081
tools/serve-replay-landing "$sha" "$capture" "$model" "$new_results_dir" 18081
```

The helpers take the GPU lock, send the original bodies to a fresh loopback
server and stop their own listener. They replay recorded tool results without
executing tools. Timings exclude model load. Set `ABLA_GRAD_BIN` for a custom
binary with `serve-replay`. The landing helper builds the requested commit and
writes the full results and a row for [the benchmark table](speed.md#serve-replay).

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
- Requests are serialized. A shared prompt prefix can reuse KV and convolution state. The server checkpoints before the final prompt token and can keep a second checkpoint for edits at 4,096-token boundaries. Usage reports the actual reuse in `prompt_tokens_details.cached_tokens`.
- `tools/gpu-lock.sh -- tools/test-serve.sh` checks response shapes, tool calls, prefix reuse and greedy parity.
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

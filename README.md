# abla-grad

Abla inference for GGUF language models on CUDA. Inkling-2-Mini supports
greedy generation, DSpark speculation, and an OpenAI-compatible server.
Llama 3.2 1B is the small correctness fixture. llama.cpp supplies reference
logits for tests.

## Build and run

Use an NVIDIA driver and installed ablac 0.2.73. The Nix shell supplies CUDA
12.9 NVRTC. Keep model files outside the repository.

```sh
nix-shell shell.nix
mkdir -p build
ablac build src/main.ab --fast -o build/abla-grad
tools/oracle/fetch-models.sh
model="$HOME/models/inkling-2-mini/Inkling-2-Mini-UD-Q4_K_M.gguf"
tools/gpu-lock.sh -- build/abla-grad decode-bench "$model" tests/prompts/03.inkling.ids --mode graph --n 128
tools/gpu-lock.sh -- build/abla-grad serve "$model" --ctx 32768 --port 8080
```

Serve at `http://127.0.0.1:8080/v1`. Place the pinned DSpark drafter next
to the model. See [Serving](docs/serving.md) for setup, endpoints and replay.

## RTX 4090 results

| Workload | Result | Measured build |
|---|---:|---|
| Plain decode, 0 / 32k / 128k cached | 138.69 / 125.54 / 101.39 tok/s | `9b3bbf3` |
| Batched decode, 2 / 3 / 4 sessions (aggregate) | 207.6 / 241.5 / 283.7 tok/s | `ce57198` |
| Prompt processing, 512 tokens | 5547.60 tok/s | `515ce8f` |
| Server TTFT, 32k / 128k prompt | 10.30 / 53.95 s | `9aaccb1` |
| 75k serve replay, final request / whole session | 7.843 / 33.721 s | `34e6098` |

[Benchmarks and reproduction](docs/speed.md) · [Engine](docs/engine.md) ·
[GGUF packing](docs/gguf-packing.md) · [Reference oracle](tools/oracle/README.md)

Run `make test` for host tests and `tools/gpu-lock.sh -- make test-gpu`
for the CUDA kernel suite.

Licensed under the [Mozilla Public License 2.0](LICENSE), like ablac. Model weights are not part of this
repository; they keep their own licenses.

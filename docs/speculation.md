# Speculative decoding (DSpark drafter)

`abla-grad generate` decodes greedily (temperature 0) with a drafter when `dspark-Inkling-2-Mini-Q8_0.gguf` sits next to the model (or `--drafter PATH`);
`--speculate off` turns it off, sampling (temperature > 0) always decodes plainly. The output is the plain greedy output up to numerical near-ties (the
verification rows go through different kernels than one-token decode; `abla-grad spec-check` measures the logit margin at the first difference).

## One cycle (src/model/speculate.ab, docs/spec-cycle.md for the buffers)
1. the drafter (90, src/model/dspark.ab) proposes n tokens from the target's hidden-state taps (inputs of visits 5, 23, 41 and the final-norm output) of the committed prefix;
2. the target verifies the anchor and the n proposals as one (n + 1)-row batch (4e/e7: row kernels that read each weight once for all rows, an expert-union map for the MoE);
3. greedy acceptance: a proposal is accepted when it is the target's argmax after the previous row; a + 1 rows commit, the position advances by a + 1, the argmax after the last committed row is the next anchor;
4. the conv histories of the committed rows are restored from per-row snapshots (the KV rows past the accepted prefix are overwritten later).
Everything runs on the device: one recorded CUDA graph per cycle (and per n = 2, 3, 4), replayed back to back; the host reads the token log every few cycles.

With a BF16 Markov W2 matrix, the recorded drafter ranks each group of 32 vocabulary IDs by base logit plus bias and evaluates the rank-256 correction for the best four. It compares their rounded, softcapped logits and breaks ties by the lowest token ID. A strong correction on an excluded ID can change the proposal. The target verifies every proposal. The host reference used by `spec-accept` and other supported W2 storage types evaluate the full vocabulary. The paired throughput and acceptance results appear in `docs/speed.md`.

## The controller
Arms: the plain decode graph and the cycle graphs with 2..4 proposals (default: up to 2). It starts on plain, sweeps each arm (a warm-up batch and a measured one), then prices
every arm in tokens/s from stable costs (seconds per plain token, seconds per cycle per arm) and the device's histogram of accepted counts (one histogram of the top
arm prices every smaller arm), with hysteresis, and refreshes the top arm's histogram at a backing-off interval. Batches never run past the tokens still wanted.

## Measured (RTX 4090, 512 generated tokens, tok/s; plain / n-max 2 / 3 / 4 / adaptive n-max 2)
| prompt | plain | 2 | 3 | 4 | adaptive 2 | tokens per cycle (2 / 3 / 4) |
|---|---:|---:|---:|---:|---:|---|
| 01 | 129 | 157 | 143 | 153 | 137 | 2.35 / 2.58 / 2.91 |
| 02 | 130 | 150 | 144 | 148 | 151 | 2.25 / 2.59 / 2.82 |
| 03 | 129 | 168 | 166 | 159 | 169 | 2.59 / 3.04 / 3.07 |
| 04 | 126 | 190 | 212 | 247 | 187 | 3 / 4 / 5 |
| 05 (Alice, 1,624 prompt tokens) | 121 | 143 | 139 | 151 | 141 | 2.36 / 2.71 / 3.10 |
At 128 tokens the adaptive controller is at plain speed (prose) or above (prompt 03: 161 vs 131). Acceptance depends on the text and on where cycles start, so tokens per cycle
vary by +-15% between runs that differ by a few near-tie token flips. Tools: `spec-bench`, `spec-check`, `spec-accept` (host-side acceptance chain, `--nmax`, `--cost`, `--union`),
`tools/gate-spec.sh`.

# One speculation cycle as a recorded graph (ablac-60-2, design)

Goal: a cycle (draft, verify, accept, commit) is one graph replay with no host work between cycles, like the decode graph. Per-cycle values live in the
device `step` buffer (u32 words; the decode graph already uses [0] = token, [1] = position and the token log from word 4 on):

| word | meaning |
|---|---|
| step[0] | anchor token (the target's greedy token after the last committed position; not yet evaluated) |
| step[1] | P = the anchor's position = number of committed positions |
| step[2] | proposals p1, p2 count in this cycle (0..2), written by the drafter |
| step[3] | accepted proposals of the last cycle (0..2), written by `spec_accept` |
| `spec[0..3]` | (a separate small buffer) rows of the verification batch: anchor, p1, p2, padding; and the argmax of rows 0..2 |

Cycle (every kernel reads P, tokens, counts from the buffers, never from kernel parameters):
1. drafter: reads taps [0, P) of the committed prefix and the anchor, writes p1, p2 (and the count) to `spec`. No host download.
2. verify: rows = (anchor, p1, p2), R = 3 fixed (rows beyond the count run on padding and are ignored); positions P, P+1, P+2 from step[1]; writes KV rows
   P..P+2, taps for the rows, conv-history snapshot after each row, logits argmax per row to `spec`.
3. `spec_accept` (one thread): a = 0; if (n >= 1 and argmax0 == p1) a = 1 and if (n >= 2 and argmax1 == p2) a = 2. Committed rows = a + 1.
   Writes the token log (anchor, p1.., at positions P..P+a), the new anchor = argmax[a], step[1] += a + 1, step[3] = a.
4. commit: history := snapshot[a] (copy selected by the device value), then the next cycle's drafter first pushes the taps of rows 0..a.
5. host reads the token log and step[3] every K cycles (or when the log reaches a stop condition) and stops the loop.

Acceptance rule is greedy equality (llama.cpp's at temperature 0): a proposal is accepted only if the target's argmax after the previous row equals it.
Near-tie differences from plain decode (f16 verify rows) are accepted as in llama.cpp; the margin of any argmax that differs from plain decode is recorded in
tests (not on the hot path). `--no-speculate` keeps plain decode.

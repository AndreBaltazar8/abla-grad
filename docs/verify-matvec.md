# Speculative verification matvecs

The CUDA kernels accept one to three activation vectors per launch and reuse
each packed weight across the vectors that need it. The number of vectors and
launch dimensions stay fixed when the kernels are recorded in a graph.

| Kernel | Parameters in launch order | Problem |
| --- | --- | --- |
| `matvec_f32_rows`, `matvec_q8_0_rows` | `w, x, y, u32 rows, u32 cols, u32 vectors` | `[rows]` |
| `matvec_nvfp4_indexed_rows` | `w, x, y, u32 rows, u32 cols, u32 vectors, map, u32 capacity, u64 expertStrideBytes, u32 slotsPerVector, u32 xVectorStride, u32 xSlotStride, expertScales` | `[rows, capacity]` |

Dense inputs and outputs are contiguous FP32 arrays of shape `[vectors][cols]`
and `[vectors][rows]`, respectively. Weight packing is the same as for
the single-vector kernels.

For indexed NVFP4, each map entry is four u32s `{expertId, slot0, slot1, slot2}`.
Use one entry per distinct expert and write `0xffffffff` for absent slots.
Entries with no valid slots are skipped, including padding up to the graph's
fixed capacity. The producer must supply in-range expert IDs for active
entries, and each active pair `(vector, slot)` must appear exactly once.
The input offset in floats is `vector * xVectorStride + slot * xSlotStride`.
Gate/up broadcasts across slots with strides `cols, 0`. Down projections
use `slotsPerVector * cols, cols`. Outputs are contiguous
`[vectors][slotsPerVector][rows]`. Pass a null `expertScales` pointer with
`u64(0)` to omit expert scaling.

The GPU test compares each output bit against the corresponding single-vector
kernel. It covers shared, partially shared and disjoint expert sets, reordered
slots, graph padding, per-slot inputs, expert scales and unaligned input loads.

For a focused run, build `tests/kernels/verify_matvec_test.ab` with `ablac` in
the Nix shell and run the executable through `tools/gpu-lock.sh`. Pass
`--bench` for the larger parity cases and five batches of 64 timed launches
per shape. Those batch timings include host submission and one stream sync.

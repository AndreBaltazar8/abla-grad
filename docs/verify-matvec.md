# Speculative verification matvecs

The dense CUDA kernels accept one to five or eight activation vectors per
launch. Indexed kernels accept one to five. Each weight is reused across the
vectors that need it. The vector count and launch dimensions stay fixed during
graph capture.

| Kernel | Parameters in launch order | Problem |
| --- | --- | --- |
| `matvec_f32_rows`, `matvec_bf16_rows`, `matvec_q8_0_rows` | `w, x, y, u32 rows, u32 cols, u32 vectors` | `[rows]` |
| `matvec_nvfp4_indexed_rows`, `matvec_q5_k_indexed_rows`, `matvec_q8_0_indexed_rows` | `w, x, y, u32 rows, u32 cols, u32 vectors, map, u32 capacity, u64 expertStrideBytes, u32 slotsPerVector, u32 xVectorStride, u32 xSlotStride, expertScales` | `[rows, capacity]` |

Dense inputs and outputs are contiguous FP32 arrays of shape `[vectors][cols]`
and `[vectors][rows]`, respectively. Weight packing is the same as for
the single-vector kernels.

For indexed matvecs with one to three vectors, each entry is four u32s
`{expertId, slot0, slot1, slot2}`. With four or five vectors, each entry is
six u32s `{expertId, slot0, slot1, slot2, slot3, slot4}`. With eight slots
per vector, capacity 40 covers the largest five-vector union.
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

## RTX 4090 microbenchmarks

CUPTI kernel durations in microseconds, median of five batches of 64 launches.
Weights and activations were already on the device. The inputs were synthetic
and used decode-sized dimensions plus two dense matrices larger than L2.
All timed shapes also passed the bit-for-bit comparison. A second run
reproduced each R=3 ratio within 0.02×.

| Kernel and shape | Existing single | R=1 | R=2 | R=3 | R=3 / R=1 | R=3 / existing single |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q8_0, 3072 × 2304 | 4.49 | 4.56 | 5.56 | 6.15 | 1.35× | 1.37× |
| Q8_0, 51200 × 2304 | 135.32 | 135.40 | 137.12 | 140.24 | 1.04× | 1.04× |
| F32, 2304 × 2304 | 6.71 | 6.76 | 7.24 | 7.96 | 1.18× | 1.19× |
| F32, 16384 × 2048 | 142.65 | 142.72 | 143.31 | 143.92 | 1.01× | 1.01× |
| NVFP4, 2304 × 1152, shared experts | 11.96 | 15.95 | 17.44 | 20.66 | 1.30× | 1.73× |
| NVFP4, 2304 × 1152, partial overlap | 12.00 | 15.99 | 22.07 | 27.23 | 1.70× | 2.27× |
| NVFP4, 2304 × 1152, disjoint experts | 12.00 | 16.00 | 24.89 | 34.61 | 2.16× | 2.89× |

NVFP4 activates eight experts per vector. The shared, partial and disjoint
R=3 cases read 8, 17 and 24 distinct experts, respectively. The new kernel's
graph-compatible grid has capacity 24 at every R, so it issues some empty
blocks when the union is smaller. The existing single kernel launches a
grid of eight experts. A 64/128/256/512-thread sweep kept 128 threads as the
best overall geometry across the overlap patterns.

The Q5_K (2304 × 2304) and Q8_0 (2304 × 1152) expert projections use
the same union layout in layers 20–23. With eight slots per vector and
17 distinct experts across three vectors, their R=3 kernel times were
72.54 µs and 22.17 µs, respectively. Three existing single-vector calls
would take 93.40 µs and 26.38 µs at their measured per-call times.

Replacing the grouped tile kernels in the recorded cycle reduced cycle
time from 18.1 to 17.4 ms on prompt 03 and from 19.4 to 17.7 ms on
prompt 05. The verification argmax check passed.

### Four- and five-vector expert unions

RTX 4090 kernel times in µs, using the same CUPTI method. With eight slots
per vector, the mixed pattern selects 17, 22, and 26 distinct experts at
R=3, R=4, and R=5. R=4 and R=5 both use capacity 40 in this benchmark.

| Kernel and expert pattern | R=3 before | R=3 after | R=4 | R=5 |
| --- | ---: | ---: | ---: | ---: |
| NVFP4, shared | 20.69 | 20.46 | 30.02 | 34.52 |
| NVFP4, mixed | 27.19 | 28.59 | 39.78 | 45.30 |
| NVFP4, disjoint | 34.54 | 36.33 | 50.22 | 60.73 |
| Q5_K, mixed | 72.25 | 73.34 | 100.68 | 123.46 |
| Q8_0, mixed | 22.07 | 23.99 | 33.63 | 39.20 |

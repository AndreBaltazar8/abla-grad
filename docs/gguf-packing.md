# Lossless packing of Inkling-2-Mini UD-Q4_K_M

The pinned original is 16,610,544,608 bytes. Packing 168 router and
loop-controller matrices produces a 16,539,765,728-byte GGUF, a saving
of 70,778,880 bytes (67.5 MiB, 0.426%). Each packed FP32 word has zero
in its low 16 bits. Independent comparison recovered all original words
and matched all 576 untouched tensor payloads byte for byte.

| Original storage | Tensors | Payload bytes | Exact whole-tensor saving found | Saving in the distributed file |
|---|---:|---:|---:|---:|
| F32 | 518 | 147,544,064 | 73,276,416 | 70,778,880 |
| BF16 | 32 | 3,833,856 | 0 | 0 |
| Q8_0 | 150 | 3,112,445,952 | 0 | 0 |
| Q5_K | 4 | 1,868,562,432 | 0 | 0 |
| NVFP4 | 40 | 11,466,178,560 | 0 | 0 |

Of the F32 tensors, 394 fit BF16 exactly. The 199 that fit IEEE FP16
are a subset of those 394. The other 124 F32 tensors each contain a
word that fails both round trips. No entire BF16 tensor fits either
E4M3FN or E5M2 FP8. In Q8_0, none of the 91,542,528 blocks can be
re-expressed on a signed or affine 4-, 5- or 7-bit integer grid with
exactly representable FP16 parameters. Each Q8_0 tensor also contains
a 16-value group whose span, in units of the greatest common divisor
of its code differences, exceeds the 63 intervals available in Q6_K.
Each Q5_K tensor contains a nonzero-scale group with more than
16 distinct values, excluding Q4_K. Each NVFP4 tensor has both a
16-value group with more than eight distinct values and a 32-value
block that cannot use a shared MXFP4 power-of-two scale. The figures
cover tensor encodings directly usable in a GGUF. General-purpose
archive compression has a different access pattern.

A trial file converting all 394 eligible tensors failed during
llama.cpp CPU evaluation. Its elementwise kernel rejected an F32
activation paired with a BF16 operand. The distributed file therefore
packs the 24 router matrices and 144 controller matrices. Abla expands
them at upload, retaining FP32 router decisions and the existing
prefill, verify and decode kernels. llama.cpp 33aeb17 loaded the
matrix-only file and generated tokens on CPU and on the RTX 4090.

The original SHA-256 is
`78fcedcf93c5d1bfe83e225513b71392bb52259be0eaa5f8899c856a2e5834ae`.
The packed SHA-256 is
`d081fc3aafe0e2868a016ef3c9d0a891a01ba7b5f7ad36751357a0a6b4625267`.
The standalone Abla conversion command is in the [README](../README.md#lossless-packing-for-the-inkling-target).

## Original and packed files in llama.cpp

Each cell reports **argmax differences / worst per-position KL / maximum absolute logit difference**. All comparisons use fixed input IDs and all 200,058 vocabulary logits at each position. For file comparisons, KL uses the original distribution as its reference. For the noise floor, both engines read the original file and the CPU distribution is the reference.

| Gate prompt | Positions | CPU, original vs packed | CUDA, original vs packed | Original, CPU vs CUDA |
|---|---:|---:|---:|---:|
| 01 | 4 | 0 / 0.00039581 / 0.53662 | 0 / 0 / 0 | 0 / 0.00031625 / 0.4686 |
| 02 | 5 | 0 / 0.00058562 / 0.47708 | 0 / 0 / 0 | 0 / 0.00035782 / 0.42979 |
| 03 | 15 | 1 / 0.0089027 / 1.2978 | 0 / 0 / 0 | 1 / 0.017128 / 1.5119 |
| 04 | 300 | 0 / 0.0069705 / 9.7211 | 0 / 0 / 0 | 0 / 0.0042416 / 9.8654 |
| 05 | 1,624 | 23 / 0.26165 / 5.5825 | 0 / 0 / 0 | 27 / 0.32447 / 9.4334 |
| 06 | 4,300 | 89 / 0.23346 / 10.057 | 0 / 0 / 0 | 111 / 0.2296 / 10.261 |
| All | 6,248 | 113 / 0.26165 / 10.057 | 0 / 0 / 0 | 139 / 0.32447 / 10.261 |

The CUDA logit files matched byte for byte on all six prompts. The CPU packed-file comparison had fewer argmax changes than the original CPU/CUDA comparison on both long prompts. On 06, its worst KL was slightly higher (0.23346 versus 0.22960).

On short prompts 01–04, every row in each comparison passed the limits in `tests/tolerances-inkling.txt` (maximum absolute difference 25, mean absolute difference 8 and KL 0.04). On long prompts 05 and 06, respectively, 30 and 29 CPU file-comparison rows exceeded a short-prompt limit. The original-file noise floor had 42 and 35 such rows. See `tests/tolerances-inkling-long.txt` for the separate per-prompt limits.

We used llama.cpp `33aeb17f85869279e3821f746bc645357139bc2c`, F16 K/V, four threads, and one token per decode. The CPU used zero GPU layers. CUDA used full offload on the RTX 4090. Context capacities were 512 for 01–04, 1,792 for 05 and 4,352 for 06. Fresh original-file logits for 01–04 matched the pinned originals on each backend. For 05 and 06 we compared against the archived originals from the same commit, model hash, prompt IDs and settings. Both packed-file backends evaluated all six prompts. The raw scores, script (`llama-pair`) and GRDL comparator are in `~/archive/abla-grad-r5-ed/`.

## Abla exactness

The upload-widening build produced byte-identical original and packed
GRDLs for every position on 01–06 in step mode (6,248 rows), and on
01–04 in chunk-256 prefill (324 rows). Each row contains all 200,058
FP32 logits. The original tensor payload audit, the gate comparisons
and the final file's SHA-256 were checked independently. The distributed
file and its checksum are under `~/models/inkling-2-mini/`.

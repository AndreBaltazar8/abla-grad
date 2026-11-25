# Lossless Inkling GGUF packing

The packer stores the 168 router and loop-controller matrices as BF16. It
rejects any selected F32 word with nonzero low bits before publishing the
output. Other tensors keep their original payloads. The pinned original
is 16,610,544,608 bytes. The packed file is 16,539,765,728 bytes, saving
70,778,880 bytes on disk and in Abla's device arena.

```sh
mkdir -p build
ablac build tools/pack-inkling.ab --fast -o build/pack-inkling
dir="$HOME/models/inkling-2-mini"
build/pack-inkling "$dir/Inkling-2-Mini-UD-Q4_K_M.gguf" \
    "$dir/Inkling-2-Mini-UD-Q4_K_M-lossless-BF16.gguf"
```

Packed SHA-256 is `d081fc3aafe0e2868a016ef3c9d0a891a01ba7b5f7ad36751357a0a6b4625267`.
Original provenance is in [models.txt](../tools/oracle/models.txt).

Abla widens packed weights during FP32 accumulation and preserves prefill
reduction order. Original and packed GRDLs matched byte for byte for 6,248
step rows and 324 prefill rows at chunk 256. llama.cpp `33aeb17` also
loaded the packed file. Its CUDA logits matched on all six prompts. The
CPU comparisons differed and passed the calibrated prompt tolerances. Raw scores
and the packing audit are in `~/archive/abla-grad-r5-ed/`.

# abla-grad interfaces (contracts between the parts)

Signatures are Abla and are the agreed shape; the owner may refine details but tells the other side first. `int` is 64-bit,
float values are `f32` unless a doc says `f64`. A function that can fail returns a nullable or `bool` and the module keeps a
`...LastError(): string` (no panics for bad input files).

Owners: GGUF/quant/tokenizer = ablac-90. Oracle (`tools/oracle/`) = ablac-df. CUDA backend implementation and kernels
(`src/backend/cuda.ab`, `src/kernels/`) = ablac-e7. Backend interface, ops, model, CLI, compare = ablac-60-2 (this file's owner).

## 1. GGUF reader (ablac-90): `src/gguf/reader.ab`, `src/gguf/quant.ab`
```
class GgufTensorInfo(val name: string, val type: int, val dims: array<int>, val offset: int, val bytes: int)
//   type = the ggml type id (see quant.ab); dims in GGUF order (dims[0] is the fastest-varying, the row length);
//   offset = bytes from the start of the tensor data section; bytes = the tensor's exact stored size.
class GgufFile                       // opaque; the file is mmap'd read-only until close()
fun ggufOpen(path: string): GgufFile?    // null on any error (bad magic/version, truncated, an unknown tensor type)
fun ggufLastError(): string
fun GgufFile.version(): int
fun GgufFile.has(key: string): bool
fun GgufFile.getString(key: string): string?     fun GgufFile.getInt(key: string): int?
fun GgufFile.getFloat(key: string): f64?         fun GgufFile.getBool(key: string): bool?
fun GgufFile.getStringArray(key: string): array<string>?   fun GgufFile.getIntArray(key: string): array<int>?
fun GgufFile.getFloatArray(key: string): array<f64>?
fun GgufFile.tensorCount(): int
fun GgufFile.tensorAt(index: int): GgufTensorInfo
fun GgufFile.tensor(name: string): GgufTensorInfo?
fun GgufFile.tensorAddress(tensor: GgufTensorInfo): int   // address of the mmap'd bytes, valid until close(); for upload
fun GgufFile.close(): void
```
`quant.ab`: ggml type id constants taken from the oracle's pinned llama.cpp (`GGML_TYPE_F32 = 0`, `F16 = 1`, `Q8_0 = 8`,
`Q4_K = 12`, `Q5_K = 13`, `Q6_K = 14`, `BF16 = 30`, NVFP4 = whatever that commit says), and
```
fun ggmlTypeName(type: int): string
fun ggmlBlockElements(type: int): int          // elements per block (1 for F32/F16/BF16, 32 for Q8_0, 256 for K-quants, ...)
fun ggmlBlockBytes(type: int): int
fun ggmlRowBytes(type: int, elements: int): int  // elements must be a multiple of the block; checked
```
CPU reference dequantization is **not** part of the program: it lives in `tests/` (90 writes it for the layout tests;
e7 reuses it for the kernel tests).

## 2. Tokenizer (ablac-90): `src/tokenizer/tokenizer.ab`
```
class Tokenizer                       // byte-level BPE, pre-tokenizer pattern, specials, ignore_merges, no/optional BOS (config)
fun tokenizerFromJson(path: string): Tokenizer?          // HF tokenizer.json (Inkling)
fun tokenizerFromGguf(file: GgufFile): Tokenizer?        // tokenizer.ggml.* metadata (the small model)
fun Tokenizer.encode(text: string, addSpecial: bool): array<int>
fun Tokenizer.decode(ids: array<int>): string
fun Tokenizer.vocabSize(): int      fun Tokenizer.bosId(): int?     fun Tokenizer.eosId(): int?
```
Gate: `abla-grad tokenize` output equals the oracle's `dump-tokens.sh` ids on the fixed prompts in `tests/prompts/*.txt`.

## 3. Oracle (ablac-df): `tools/oracle/`
Shell only (no Python). Models live in `~/models/<name>/<file>.gguf`; `tools/oracle/models.txt` pins name, URL, size,
SHA-256 (the small model and the Inkling GGUF). llama.cpp is built from a pinned commit (written in the same file).
```
tools/oracle/build-llama.sh                       # builds llama.cpp (CPU and CUDA builds) into ~/models/llama.cpp-<commit>/
tools/oracle/fetch-models.sh                      # downloads + verifies the pinned models
tools/oracle/dump-tokens.sh  <gguf> <text-file> <out.ids>        # llama.cpp's own tokenization, one id per line
tools/oracle/dump-logits.sh  <gguf> <ids-file> <out.grdl> [--cpu|--cuda]   # logits after every position
tools/oracle/dump-tensors.sh <gguf> <ids-file> <outdir>           # named intermediate tensors of the first token(s)
```
**GRDL v1** (little-endian): `"GRDL"`, u32 version=1, u32 n_vocab, u32 n_tokens, `u32 tokens[n_tokens]`,
`f32 logits[n_tokens][n_vocab]` (row i = the logits after consuming tokens[0..i]); a `<out>.json` sidecar with the
prompt, model SHA-256, llama.cpp commit and flags. `ids-file` = one decimal id per line.
Tensor dump: `<outdir>/<tensor-name>.f32` raw little-endian f32 plus `<outdir>/index.txt` (name, dims, op).
Report: the Inkling GGUF's file size, tensor census and whether llama.cpp's `inkling2` implementation is in the pinned commit.

## 4. Backend interface (60-2): `src/backend/backend.ab`
The only thing the model, ops and CLI know about devices. **`src/backend/backend.ab` is the contract** (it compiles; tests/backend_interface_test.ab
checks it with a recording test double). Implementations: `CudaBackend` (first, ablac-e7), later Vulkan, CPU, Metal.
- `DeviceBuffer(owner: Backend, handle: int, bytes: int)` is a resource class created by `Backend.allocate`; dropping it calls
  `owner.release(handle, bytes)`. `owner` carries the context identity; a buffer is only valid with its owner.
- `KernelArgs` packs the parameters in the kernel's order: `.pointer(buffer, offset = 0)` (bounds-checked: offset within the buffer),
  `.u32`, `.i32`, `.u64`, `.f32`. Pointers and u64 take 8 bytes, the rest 4 (aligned to their size).
- `Backend`: `name`, `allocate(bytes): DeviceBuffer?`, `upload(dst, offset, hostAddress, bytes)`, `download(src, offset, hostAddress, bytes)`
  (waits for queued work), `launch(kernel, args, problem)`, `synchronize()`, `freeMemory()`, `release(handle, bytes)` (drop only).

`launch` takes the kernel by **name** and a `problem` (op-specific sizes, e.g. `[rows]` for a matvec); the backend maps
`problem` to its launch geometry from its own registry (for CUDA: block dims and items-per-block per kernel, `grid =
ceilDiv(problem[i], perBlock[i])`). Launches on one backend run in order on one stream, so no explicit barrier is needed
between dependent launches; `synchronize` is only for reads and timing. Kernel names are the contract (section 5).

## 5. Kernels (ablac-e7): `src/kernels/<name>.ab` + `src/backend/cuda.ab`
CUDA C source for each kernel as an Abla string (`val kernelMatvecQ8_0 = "extern \"C\" __global__ void matvec_q8_0(...) {...}"`
or a `$cuda` block later), compiled by NVRTC for the device's compute capability at first use and cached on disk
(`~/.cache/abla-grad/<hash>.cubin`), loaded with `cuModuleLoadData`. Host calls (`cuInit`, `cuCtxCreate`, `cuMemAlloc`,
`cuMemcpyHtoDAsync`, `cuLaunchKernel`, streams, NVRTC) are `trusted extern:"c"` declarations in `src/backend/cuda.ab` only,
wrapped in checked types; no other module imports an extern.
Kernel parameter conventions (so ops.ab can call every kernel the same way):
- weights: `const unsigned char* w` (raw GGUF bytes of the tensor; row r starts at `r * rowBytes`), then `const float* x`,
  `float* y`, then scalars `u32 rows, u32 cols` (cols = elements per row, a multiple of the type's block).
- indexed (MoE): `... const unsigned* ids, u32 nUsed, u64 expertStrideBytes, u32 xStrideElems` after rows/cols; problem = `[rows, nUsed]`; expert `ids[i]` starts at `ids[i] * expertStrideBytes`; `x` for slot i is at `i * xStrideElems` (0 = one x broadcast to every expert, as gate/up use; `cols` = a separate x per expert, as down uses); `y` is `nUsed * rows` contiguous.
- activations f32; norm/scale weights f32 (converted at load if stored f16/bf16).
- Each kernel has `tests/kernels/<name>_test.ab` with a CPU reference **in the test file** and its tolerance.
The kernel list and order is in `docs/plan.md`.

## 6. Ops (60-2): `src/ops.ab`
`fun matvec(backend: Backend, type: int, w: borrow DeviceBuffer, wOffset: int, x: ..., xOffset: int, y: ..., yOffset: int, rows: int, cols: int): bool`
and one function per op in the plan; they build `KernelArgs` and call `launch`. Model code calls only these.

## 7. Compare (60-2): `abla-grad compare a.grdl b.grdl [--tolerances file]`
Prints per position and in summary: argmax equal, top-5 equal, max |dlogit|, mean |dlogit|, KL(a||b); exit 0 when all
positions are within the tolerances file (`tests/tolerances.txt`, calibrated as in docs/plan.md).

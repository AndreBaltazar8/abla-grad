#!/usr/bin/env bash
set -euo pipefail
backend=${1:?specify cpu or cuda}
gpu=()
if [[ $backend == cuda ]]; then
    gpu=(-DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=${ORACLE_CUDA_ARCH:-89}" -DGGML_CUDA_NCCL=OFF)
fi
build=$llama_root/$backend
cmake -S "$llama_source" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_TOOLS=ON -DLLAMA_OPENSSL=OFF "${gpu[@]}"
cmake --build "$build" --target llama llama-bench -j "${ORACLE_JOBS:-6}"
"${CC:-cc}" -std=c11 -O2 -Wall -Wextra -Werror \
    -I"$llama_source/include" -I"$llama_source/ggml/include" \
    "$oracle_dir/oracle.c" -L"$build/bin" -Wl,-rpath,"$build/bin" \
    -lllama -lggml -lggml-base -lm -o "$build/abla-grad-oracle"

#!/usr/bin/env bash
# Exercise real CUDA image loading with an isolated cache. Each process takes a GPU lock.
set -euo pipefail
compiler=${1:-ablac}
dir=$(mktemp -d)
trap 'rm -rf -- "$dir"' EXIT

cat >"$dir/trace.c" <<'C'
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

int cuDriverGetVersion(int *version) {
    int (*real)(int *) = dlsym(RTLD_NEXT, "cuDriverGetVersion");
    if (!real) abort();
    int status = real(version);
    if (status == 0) *version += atoi(getenv("TEST_DRIVER_OFFSET"));
    return status;
}

int nvrtcCompileProgram(void *program, int count, const char **options) {
    int (*real)(void *, int, const char **) = dlsym(RTLD_NEXT, "nvrtcCompileProgram");
    if (!real) abort();
    fputs("cache-test: compile\n", stderr);
    return real(program, count, options);
}

int cuLinkComplete(void *state, void **image, size_t *size) {
    int (*real)(void *, void **, size_t *) = dlsym(RTLD_NEXT, "cuLinkComplete");
    if (!real) abort();
    fputs("cache-test: link\n", stderr);
    return real(state, image, size);
}

int cuModuleLoadData(void **module, const unsigned char *image) {
    int (*real)(void **, const unsigned char *) = dlsym(RTLD_NEXT, "cuModuleLoadData");
    if (!real) abort();
    if (image[0] == 127 && image[1] == 'E' && image[2] == 'L' && image[3] == 'F')
        fputs("cache-test: load cubin\n", stderr);
    return real(module, image);
}
C
"${CC:-cc}" -Wall -Wextra -Werror -shared -fPIC -o "$dir/trace.so" "$dir/trace.c" -ldl

# This fixture launches one kernel. Rebuild it with the GPU source edited in
# a scratch tree, leaving the actual repo and the expected GPU result intact.
fixture=tests/kernels/gather_row_f32_test.ab
mkdir -p "$dir/tree/tests/kernels" "$dir/cache"
cp -R src "$dir/tree/"
cp abla.toml "$dir/tree/"
cp "$fixture" tests/kernels/device_test_support.ab "$dir/tree/tests/kernels/"
"$compiler" build "$fixture" -o "$dir/original" --fast --no-cache
sed -i '/^extern "C" __global__ void gather_row_f32/i // cache test source change' "$dir/tree/src/kernels/matvec_f32.ab"
(
    cd "$dir/tree"
    "$compiler" build "$fixture" -o "$dir/modified" --fast --no-cache
)

run() {
    local label=$1 binary=$2 driver=$3 expected=$4
    local log="$dir/$label.log"
    if ! tools/gpu-lock.sh -t 90 -- env "XDG_CACHE_HOME=$dir/cache" CUDA_CACHE_DISABLE=1 \
        "LD_PRELOAD=$dir/trace.so" "TEST_DRIVER_OFFSET=$driver" "$binary" >"$log" 2>&1; then
        cat "$log" >&2
        return 1
    fi
    local counts
    counts=$(awk '/^cache-test: compile$/{c++} /^cache-test: link$/{l++} /^cache-test: load cubin$/{b++}
        END {print c+0,l+0,b+0}' "$log")
    if [[ $counts != "$expected $expected 1" ]]; then
        printf '%s: expected compile/link/load counts %s %s 1, got %s\n' "$label" "$expected" "$expected" "$counts" >&2
        cat "$log" >&2
        return 1
    fi
}

run cold "$dir/original" 0 1
run warm "$dir/original" 0 0
entry=$(find "$dir/cache" -name '*.image' -type f -print -quit)
[[ -n $entry ]]
# Keep the key and format intact while damaging the payload's checksum.
printf 'invalid cubin bytes' >> "$entry"
run corrupt "$dir/original" 0 1
run repaired "$dir/original" 0 0
# A self-consistent entry can still be rejected by CUDA. Recovery must be the same.
key=${entry##*/}
key=${key%.image}
invalid_ptx="invalid PTX for the driver"
digest=$(printf "%s" "$invalid_ptx" | sha256sum)
printf "%sP%s\n%s" "$key" "${digest%% *}" "$invalid_ptx" >"$entry"
run driver-rejected "$dir/original" 0 1
run driver-repaired "$dir/original" 0 0
run source-change "$dir/modified" 0 1
run source-warm "$dir/modified" 0 0
run driver-change "$dir/original" 1 1
run driver-warm "$dir/original" 1 0
run original-still-warm "$dir/original" 0 0
echo 'test-cuda-cache: cold, warm, corruption, kernel and driver invalidation passed'

#!/usr/bin/env bash
# Force CUDA OOM in each CLI, including after serve has accepted a request.
# Run from the repo root after building build/abla-grad. Each process gets its own GPU hold.
set -euo pipefail
model=${1:-$HOME/models/inkling-2-mini/Inkling-2-Mini-UD-Q4_K_M.gguf}
bin=${ABLA_GRAD_BIN:-build/abla-grad}
port=${PORT:-18377}
dir=$(mktemp -d)
trap 'rm -rf -- "$dir"' EXIT

# The shim records the exact byte count CUDA received. No production fault-injection flag is needed.
cat >"$dir/oom.c" <<'C'
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int cuMemAlloc_v2(uint64_t *address, size_t bytes) {
    static unsigned calls;
    const char *at = getenv("FAIL_ALLOC_AT");
    const char *marker = getenv("FAIL_ALLOC_MARKER");
    const char *once = getenv("FAIL_ALLOC_ONCE");
    ++calls;
    if ((at && strtoul(at, NULL, 10) == calls) ||
        (marker && access(marker, F_OK) == 0) || (once && unlink(once) == 0)) {
        fprintf(stderr, "forced CUDA OOM: %zu bytes\n", bytes);
        return 2;  // CUDA_ERROR_OUT_OF_MEMORY
    }
    int (*allocate)(uint64_t *, size_t) = dlsym(RTLD_NEXT, "cuMemAlloc_v2");
    if (!allocate) abort();
    return allocate(address, bytes);
}
C
"${CC:-cc}" -Wall -Wextra -Werror -shared -fPIC -o "$dir/oom.so" "$dir/oom.c" -ldl

check_failure() {
    local status=$1 log=$2 prefix=$3 purpose=$4 bytes
    bytes=$(sed -n 's/^forced CUDA OOM: \([0-9][0-9]*\) bytes$/\1/p' "$log" | tail -n 1)
    if [[ $status != 1 || -z $bytes ]] ||
        ! grep -E "^$prefix: .*could not allocate $purpose.*\\($bytes bytes\\)" "$log" >/dev/null; then
        printf 'Expected %s to exit 1 and identify %s with the requested byte count (exit %s).\n' "$prefix" "$purpose" "$status" >&2
        cat "$log" >&2
        exit 1
    fi
}

# The third model allocation is the 128k key cache. This exercises the formerly silent CLI failure.
status=0
tools/gpu-lock.sh -t 90 -- env "LD_PRELOAD=$dir/oom.so" FAIL_ALLOC_AT=3 \
    "$bin" decode-bench "$model" tests/prompts/03.inkling.ids --ctx 131072 --n 1 --mode graph >"$dir/decode.out" 2>"$dir/decode.err" ||
    status=$?
check_failure "$status" "$dir/decode.err" decode-bench 'Inkling key cache'

status=0
tools/gpu-lock.sh -t 90 -- env "LD_PRELOAD=$dir/oom.so" FAIL_ALLOC_AT=1 \
    "$bin" serve "$model" --ctx 131072 --port "$port" >"$dir/start.out" 2>"$dir/start.err" ||
    status=$?
check_failure "$status" "$dir/start.err" serve 'weight arena'

status=0
tools/gpu-lock.sh -t 150 -- bash -s -- "$bin" "$model" "$port" "$dir" <<'SH' || status=$?
set -euo pipefail
bin=$1 model=$2 port=$3 dir=$4
LD_PRELOAD="$dir/oom.so" FAIL_ALLOC_MARKER="$dir/fail" FAIL_ALLOC_ONCE="$dir/once" \
    "$bin" serve "$model" --ctx 4096 --port "$port" >"$dir/request.out" 2>"$dir/request.err" &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
url="http://127.0.0.1:$port"
ready=false
for _ in $(seq 1 600); do
    if curl --max-time 1 -fs "$url/health" >/dev/null 2>&1; then ready=true; break; fi
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    sleep 0.1
done
if [[ $ready != true ]]; then
    echo 'serve failed to reach health before fault injection' >&2
    exit 2
fi
prompt=$(awk 'BEGIN { for (i=0; i<100; ++i) printf "The index lists each number. " }')
touch "$dir/once"
curl --max-time 30 -fs -o "$dir/retry" "$url/v1/chat/completions" \
    -d '{"messages":[{"role":"user","content":"'"$prompt"'"}],"max_tokens":1}'
[[ ! -e $dir/once ]]
touch "$dir/fail"
http=$(curl --max-time 15 -sS -o "$dir/reply" -w '%{http_code}' "$url/v1/chat/completions" \
    -d '{"messages":[{"role":"user","content":"'"Different index. $prompt"'"}],"max_tokens":1}')
if [[ $http != 500 ]]; then
    printf 'Expected HTTP 500, got %s.\n' "$http" >&2
    cat "$dir/reply" >&2
    exit 2
fi
wait "$pid"
SH
check_failure "$status" "$dir/request.err" serve 'prefill workspace \(chunk 1\)'
echo 'test-allocation-failure: decode allocation, serve startup and serve request passed'

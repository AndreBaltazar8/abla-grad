#!/usr/bin/env bash
# Run one command with the 4090 to yourself: tools/gpu-lock.sh [-t SECONDS] [-w WAIT_SECONDS] -- command args...
# Holds flock on /tmp/abla-grad-gpu.lock (override with ABLA_GRAD_GPU_LOCK) for the whole command, waits until the card has at least 20000 MiB free
# (ABLA_GRAD_GPU_FREE_MIB) before starting, and kills the command after the hold limit (default 900 s). One run per hold: never loop inside a hold.
# Exit codes: the command's, 124 on hold timeout, 75 when the lock or the free memory did not come within the wait (default 3600 s).
set -uo pipefail
hold=900
wait=3600
while [[ ${1:-} == -* ]]; do
    case $1 in
        -t) hold=$2; shift 2 ;;
        -w) wait=$2; shift 2 ;;
        --) shift; break ;;
        *) echo "gpu-lock: unknown option $1" >&2; exit 2 ;;
    esac
done
[[ $# -gt 0 ]] || { echo "usage: tools/gpu-lock.sh [-t hold-seconds] [-w wait-seconds] -- command..." >&2; exit 2; }
# Already inside a hold (a script run under this wrapper calls it again): just run, never wait for our own lock.
if [[ -n ${ABLA_GRAD_GPU_LOCK_HELD:-} ]]; then exec timeout -k 10 "$hold" "$@"; fi
lock=${ABLA_GRAD_GPU_LOCK:-/tmp/abla-grad-gpu.lock}
need=${ABLA_GRAD_GPU_FREE_MIB:-20000}
exec 9>"$lock"
flock -w "$wait" 9 || { echo "gpu-lock: no lock after ${wait}s" >&2; exit 75; }
deadline=$((SECONDS + wait))
while true; do
    used=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{print $2 - $1}')
    [[ ${used:-0} -ge $need ]] && break
    ((SECONDS < deadline)) || { echo "gpu-lock: only ${used} MiB free after ${wait}s (a process outside the lock holds the card)" >&2; exit 75; }
    sleep 5
done
ABLA_GRAD_GPU_LOCK_HELD=1 timeout -k 10 "$hold" "$@"

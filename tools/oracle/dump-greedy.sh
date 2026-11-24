#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if (( $# < 3 || $# > 6 )); then
    echo "usage: $0 MODEL PROMPT OUT_DIR [STEPS=8] [TOP_K=10] [--cpu|--cuda]" >&2
    exit 2
fi
case ${6:---cuda} in
    --cpu) backend=cpu; layers=0 ;;
    --cuda) backend=cuda; layers=999 ;;
    *) echo "Expected --cpu or --cuda" >&2; exit 2 ;;
esac
oracle_run "$backend" decode "$1" "$2" "$3" "${4:-8}" "${5:-10}" "$layers"

#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if (( $# < 3 || $# > 4 )); then
    echo "usage: $0 MODEL IDS_FILE OUT_DIR [--cpu|--cuda]" >&2
    exit 2
fi
case ${4:---cuda} in
    --cpu) backend=cpu; layers=0 ;;
    --cuda) backend=cuda; layers=999 ;;
    *) echo "Expected --cpu or --cuda" >&2; exit 2 ;;
esac
oracle_run "$backend" tensors "$1" "$2" "$3" "$layers"

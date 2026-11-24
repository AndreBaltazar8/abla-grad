#!/usr/bin/env bash
# `abla-grad compare`: identical files exit 0, a large change exits 1 under a tolerances file, mismatched token ids exit 2.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
compiler=${1:-ablac}
out=build/compare-cli
mkdir -p "$out"
"$compiler" build src/main.ab -o "$out/abla-grad" --fast --no-cache > "$out/build.log" 2>&1 || { cat "$out/build.log" >&2; exit 1; }
"$compiler" build tests/grdl_make.ab -o "$out/make" --fast --no-cache > "$out/make.log" 2>&1 || { cat "$out/make.log" >&2; exit 1; }
"$out/make" "$out/a.grdl" 0
"$out/make" "$out/b.grdl" 0
"$out/make" "$out/c.grdl" 4
printf 'maxAbs 0.5\nrequireArgmax 1\n' > "$out/tol.txt"
"$out/abla-grad" compare "$out/a.grdl" "$out/b.grdl" --tolerances "$out/tol.txt" > "$out/same.txt" || { echo "identical files failed" >&2; cat "$out/same.txt" >&2; exit 1; }
grep -q "0 outside the tolerances" "$out/same.txt"
status=0
"$out/abla-grad" compare "$out/a.grdl" "$out/c.grdl" --tolerances "$out/tol.txt" > "$out/moved.txt" || status=$?
[[ $status == 1 ]] || { echo "a moved logit exited $status, not 1" >&2; cat "$out/moved.txt" >&2; exit 1; }
grep -q "OUT:" "$out/moved.txt"
status=0
"$out/abla-grad" compare "$out/a.grdl" "$out/missing.grdl" > "$out/missing.txt" || status=$?
[[ $status == 2 ]] || { echo "a missing file exited $status, not 2" >&2; exit 1; }
echo "compare cli: identical 0, moved 1, missing 2"

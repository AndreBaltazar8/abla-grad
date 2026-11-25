#!/usr/bin/env bash
set -euo pipefail
compiler=${1:-ablac}
if [[ $compiler == */* && $compiler != /* ]]; then compiler="$PWD/$compiler"; fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
out=build/fuse-tests
mkdir -p "$out"
"$compiler" build tests/fuse/emit.ab --strict-imports --fast --no-cache -o "$out/emit"
for name in tiny reordered clipped controller; do
    "$out/emit" "$name" > "$out/$name.cu"
    sed -n '/^extern "C" __global__ void abla_fuse_/,$p' "$out/$name.cu" > "$out/$name.wrapper.cu"
    cmp "tests/fuse/golden/$name.cu" "$out/$name.wrapper.cu"
done

reject() {
    local name=$1 substitution=$2 message=$3 fixture=${4:-fixture}
    sed "$substitution" "tests/fuse/$fixture.ab" > "$out/$name.ab"
    if "$compiler" build "$out/$name.ab" --strict-imports --fast --no-cache \
        -o "$out/$name" > "$out/$name.log" 2>&1; then
        echo "fuse: expected $name to fail" >&2
        exit 1
    fi
    grep -F 'E_SUBPARSER_FAILURE' "$out/$name.log" >/dev/null
    grep -F "$message" "$out/$name.log" >/dev/null
}
reject unknown-op 's/matvec_indexed(/rotate(/' "unknown operation 'rotate'"
reject width 's/x: f32\[n\]/x: f32[32]/' 'indexed gate/up shape mismatch'
reject type 's/ids: u32/ids: f32/' "'ids' needs type 'u32' and rank 1"
reject forward 's/w, x, ids/w, g, ids/' "'g' used before definition"
reject output 's/out a:/out missing:/' 'every output must be assigned'
reject scalar 's/beta = 4.0/beta = 0.0/' "invalid 'beta' constant"
reject ssa 's/g = matvec/w = matvec/' 'intermediate name must be fresh'
reject rank 's/nvfp4\[experts, 2\*ff, n\]/nvfp4[experts, 2*ff, n, 1]/' 'tensor rank or element count exceeds the limit'
reject extent 's/n = 64/n = 2147483647*2/' 'dimension product exceeds 2147483647'
reject launch-boundary '/a = situ/a\    d = matvec_indexed(w, a, ids, scales)' 'launch boundary after gate/up and SiTU'
reject controller-concat 's/concat(a, b)/concat(b, a)/' 'concat must consume the two normalized vectors in order' controller-fixture
reject controller-width 's/rank, 2\*n/rank, n/' 'controller matvec needs a [rows, 4608] F32 weight tensor' controller-fixture
echo 'fuse: 4 generator goldens and 12 rejected blocks passed'

#!/usr/bin/env bash
# End-to-end test of `abla-grad serve` on the 4090: tools/gpu-lock.sh -t 800 -- tools/test-serve.sh [model.gguf]
# Append --long-checkpoint after the model to exercise both checkpoint slots at context 131072 (requires jq).
# Checks the OpenAI shapes (models, completion, stream, usage, errors), a tool-call round trip, prefix reuse (second turn reports cached_tokens > 0 and a
# rewritten history reports fewer), and that greedy output equals `abla-grad generate` on the same rendered prompt.
set -euo pipefail
model=${1:-$HOME/models/inkling-2-mini/Inkling-2-Mini-UD-Q4_K_M.gguf}
bin=${ABLA_GRAD_BIN:-build/abla-grad}
port=${PORT:-18080}
context=4096
if [[ ${2:-} == --long-checkpoint ]]; then context=131072; fi
log=$(mktemp)
"$bin" serve "$model" --port "$port" --ctx "$context" >"$log" 2>&1 &
pid=$!
cleanup() {
    if [[ -n $pid ]]; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    rm -f "$log"
}
trap cleanup EXIT
for _ in $(seq 1 240); do curl -fs "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break; sleep 2; done
u="http://127.0.0.1:$port"
fail() { echo "FAIL: $*"; cat "$log"; exit 1; }
# The long mode checks target, drafter and tap rings, and an edit inside the previous prompt.
if [[ ${2:-} == --long-checkpoint ]]; then
    prefix=$(awk 'BEGIN { for (i=0;i<900;i++) printf "The index lists each number. " }')
    suffix=$(awk 'BEGIN { for (i=0;i<100;i++) printf "There is a new entry. " }')
    request() {
        curl -fs "$u/v1/chat/completions" --data-binary "$(jq -nc --arg text "$1" --argjson n "$2" \
            '{messages:[{role:"user",content:$text}],max_tokens:$n,temperature:0}')"
    }
    a="$prefix First edition. $suffix List all the integers from 1 through 5000 one by one, without skipping any."
    b="$prefix Second edition. $suffix What changed in this edition?"
    cold=$(request "$a" 8)
    count=$(jq -er '.usage.prompt_tokens' <<<"$cold")
    ((count > 4096 && count < 8192)) || fail "unexpected stride fixture length: $count"
    generated=$(request "$a" 2304)
    [[ $(jq -er '.usage.completion_tokens' <<<"$generated") == 2304 ]] || fail "generation did not wrap both rings"
    again=$(request "$a" 8)
    [[ $(jq -er '.usage.prompt_tokens_details.cached_tokens' <<<"$again") == "$((count-1))" ]] || fail "end checkpoint lost"
    [[ $(jq -c '.choices' <<<"$again") == "$(jq -c '.choices' <<<"$cold")" ]] || fail "end checkpoint changed choices"
    edited=$(request "$b" 8)
    stride=$(jq -er '.usage.prompt_tokens_details.cached_tokens' <<<"$edited")
    [[ $stride == 4096 ]] || fail "prompt edit reused $stride tokens instead of 4096"
    request "Name a colour." 1 >/dev/null
    fresh=$(request "$b" 8)
    [[ $(jq -er '.usage.prompt_tokens_details.cached_tokens' <<<"$fresh") == 0 ]] || fail "expected cold reference"
    [[ $(jq -c '.choices' <<<"$edited") == "$(jq -c '.choices' <<<"$fresh")" ]] || fail "stride checkpoint changed choices"
    cat "$log"
    echo "test-serve: long prompt and stride checkpoints passed"
    exit 0
fi
curl -fs "$u/v1/models" | grep -q '"object":"list"' || fail models
q='{"messages":[{"role":"user","content":"What is 2+2? Answer with one word."}],"max_tokens":200,"temperature":0}'
r1=$(curl -fs "$u/v1/chat/completions" -d "$q")
echo "$r1" | grep -q '"object":"chat.completion"' || fail "completion shape: $r1"
echo "$r1" | grep -q '"usage"' || fail "usage: $r1"
echo "$r1" | grep -q '"cached_tokens":0' || fail "first cached_tokens should be 0: $r1"
r2=$(curl -fs "$u/v1/chat/completions" -d "$q")
echo "$r2" | grep -Eq '"cached_tokens":[1-9]' || fail "repeat prompt must hit the cache: $r2"
c1=$(echo "$r1" | sed 's/.*"content":"\([^"]*\)".*/\1/'); c2=$(echo "$r2" | sed 's/.*"content":"\([^"]*\)".*/\1/')
[ "$c1" = "$c2" ] || fail "greedy output differs between a cold and a warm run: [$c1] [$c2]"
r3=$(curl -fs "$u/v1/chat/completions" -d '{"messages":[{"role":"user","content":"Name a colour."}],"max_tokens":50}')
echo "$r3" | grep -Eq '"cached_tokens":[0-9]+' || fail "miss shape: $r3"
s=$(curl -fsN "$u/v1/chat/completions" -d '{"stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"Say hi."}],"max_tokens":60}')
echo "$s" | grep -q 'chat.completion.chunk' || fail "stream chunks"
echo "$s" | tail -2 | grep -q '\[DONE\]' || fail "stream DONE"
t='{"messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],"max_tokens":300,"tools":[{"type":"function","function":{"name":"get_weather","description":"Weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}'
tc=$(curl -fs "$u/v1/chat/completions" -d "$t")
echo "$tc" | grep -q '"tool_calls"' || fail "tool call missing: $tc"
echo "$tc" | grep -q '"name":"get_weather"' || fail "tool name: $tc"
echo "$tc" | grep -q '"finish_reason":"tool_calls"' || fail "tool finish: $tc"
id=$(echo "$tc" | sed 's/.*"id":"\(call_[^"]*\)".*/\1/')
back='{"messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."},{"role":"assistant","content":null,"tool_calls":[{"id":"'$id'","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},{"role":"tool","tool_call_id":"'$id'","content":"18 degrees and sunny"}],"max_tokens":200,"tools":[{"type":"function","function":{"name":"get_weather","description":"Weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}'
rb=$(curl -fs "$u/v1/chat/completions" -d "$back")
echo "$rb" | grep -Eq '"cached_tokens":[1-9]' || fail "tool result turn must reuse the prefix: $rb"
echo "$rb" | grep -q '18' || fail "answer should mention the result: $rb"
# A repeated prompt must recover the saved window after generation evicts it.
prefix=$(awk 'BEGIN { for (i=0;i<350;i++) printf "The index lists each number. " }')
long='{"messages":[{"role":"user","content":"'"$prefix"'List the integers from 1 through 1000 one by one, without skipping any."}],"max_tokens":576,"temperature":0}'
short=${long/576/8}
cold=$(curl -fs "$u/v1/chat/completions" -d "$short")
echo "$cold" | grep -q '"cached_tokens":0' || fail "expected a cold long prompt"
first=$(curl -fs "$u/v1/chat/completions" -d "$long")
echo "$first" | grep -q '"completion_tokens":576' || fail "long response did not cross the rewind boundary"
echo "$first" | grep -Eq '"cached_tokens":[1-9]' || fail "live prompt snapshot was discarded"
again=$(curl -fs "$u/v1/chat/completions" -d "$short")
prompt_count=$(sed -n 's/.*"prompt_tokens":\([0-9]*\).*/\1/p' <<<"$cold")
[[ -n $prompt_count && $(sed -n 's/.*"cached_tokens":\([0-9]*\).*/\1/p' <<<"$again") == "$((prompt_count-1))" ]] || fail "lost prompt checkpoint: $again"
echo "$again" | grep -q '"completion_tokens":8' || fail "restored request failed: $again"
# Compare the complete choices, including reasoning, excluding IDs and usage.
choices() { sed -n 's/^.*"choices":\(.*\),"usage":.*$/\1/p'; }
cold_choices=$(choices <<< "$cold")
[[ -n $cold_choices && $cold_choices == "$(choices <<< "$again")" ]] || fail "restoring the checkpoint changed the greedy response"
code=$(curl -s -o /dev/null -w '%{http_code}' "$u/v1/chat/completions" -d '{"messages":')
[ "$code" = 400 ] || fail "bad json should be 400, got $code"
# greedy output equals `generate` (the server is stopped first: the card holds one model)
kill "$pid"; wait "$pid" 2>/dev/null || true
pid=
gen=$("$bin" generate "$model" --chat "What is 2+2? Answer with one word." --max-tokens 200 --temp 0 --ctx 2048 2>/dev/null | tail -1)
reasoning=$(echo "$r1" | sed 's/.*"reasoning_content":"\(.*\)"},"finish_reason.*/\1/; s/\\"/"/g')
expect="<|message_model|><|content_thinking|>${reasoning}<|end_message|><|message_model|><|content_text|>${c1}<|end_message|>"
[ "$gen" = "$expect" ] || fail "serve and generate differ:\n$gen\n$expect"
echo "test-serve: ok"

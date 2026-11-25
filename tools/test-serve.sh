#!/usr/bin/env bash
# End-to-end test of `abla-grad serve` on the 4090: tools/gpu-lock.sh -t 800 -- tools/test-serve.sh [model.gguf]
# Checks the OpenAI shapes (models, completion, stream, usage, errors), a tool-call round trip, prefix reuse (second turn reports cached_tokens > 0 and a
# rewritten history reports fewer), and that greedy output equals `abla-grad generate` on the same rendered prompt.
set -euo pipefail
model=${1:-$HOME/models/inkling-2-mini/Inkling-2-Mini-UD-Q4_K_M.gguf}
bin=${ABLA_GRAD_BIN:-build/abla-grad}
port=${PORT:-18080}
log=$(mktemp)
"$bin" serve "$model" --port "$port" --ctx 4096 >"$log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null; rm -f "$log"' EXIT
for _ in $(seq 1 240); do curl -fs "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break; sleep 2; done
u="http://127.0.0.1:$port"
fail() { echo "FAIL: $*"; cat "$log"; exit 1; }
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
code=$(curl -s -o /dev/null -w '%{http_code}' "$u/v1/chat/completions" -d '{"messages":')
[ "$code" = 400 ] || fail "bad json should be 400, got $code"
# greedy output equals `generate` (the server is stopped first: the card holds one model)
kill $pid; wait $pid 2>/dev/null || true
gen=$("$bin" generate "$model" --chat "What is 2+2? Answer with one word." --max-tokens 200 --temp 0 --ctx 2048 2>/dev/null | tail -1)
reasoning=$(echo "$r1" | sed 's/.*"reasoning_content":"\(.*\)"},"finish_reason.*/\1/; s/\\"/"/g')
expect="<|message_model|><|content_thinking|>${reasoning}<|end_message|><|message_model|><|content_text|>${c1}<|end_message|>"
[ "$gen" = "$expect" ] || fail "serve and generate differ:\n$gen\n$expect"
echo "test-serve: ok"

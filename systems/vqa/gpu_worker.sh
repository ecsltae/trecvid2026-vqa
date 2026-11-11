#!/usr/bin/env bash
# Work-stealing GPU worker.  $1 = wait-for pattern, $2 = ollama port, $3 = tag
#
# Each worker claims the next unclaimed chunk with `mkdir` — atomic on POSIX, so
# two workers can never take the same chunk — processes it, and repeats until
# the pool is empty.
#
# WHY WORK-STEALING RATHER THAN A FIXED SPLIT. The three GPUs differ in speed
# (the 80GB card is ~1.6x faster than the 20GB slices) AND become free at
# different times, so any static division leaves a card idle at the end. With a
# 25-topic chunk (~4.7 min on a slow card) the worst-case idle tail is half a
# chunk, and a faster card simply completes more chunks. No rebalancing needed.
#
# A chunk is only marked .done when its CSV actually holds every topic; an
# interrupted chunk releases its claim so another worker retries it.
set -u
cd .
source venv/bin/activate
D=runs/vqa/ag/2026/everything/chunks

# NB: match on "run_vqa_v4.py.*<pattern>", never the bare pattern. This script
# receives the pattern as argv, so a bare `pgrep -f "$1"` matches THIS PROCESS
# and waits for itself forever. (Same self-match trap as `pkill -f` — it cost
# two GPUs eight idle minutes before it was spotted.)
if [ "$1" != "none" ]; then
    while pgrep -f "run_vqa_v4.py.*$1" >/dev/null 2>&1; do sleep 15; done
fi
echo "[$3] GPU free, starting work-stealing $(date +%H:%M:%S)"

while true; do
    got=""
    for c in "$D"/chunk_*.json; do
        base=$(basename "$c" .json)
        [ -f "$D/$base.done" ] && continue
        if mkdir "$D/claims/$base" 2>/dev/null; then got="$base"; break; fi
    done
    if [ -z "$got" ]; then echo "[$3] pool empty, exiting $(date +%H:%M:%S)"; break; fi
    echo "[$3] claim $got $(date +%H:%M:%S)"
    OLLAMA_HOST=127.0.0.1:"$2" python systems/vqa/run_vqa_v4.py \
        --topics "$D/$got.json" --out "$D/$got.csv" \
        --frames data/vqa/2026/frames --model qwen3-vl:8b-instruct --samples 8 \
        --slate consensus --fewshot 4 \
        --transcripts data/vqa/transcripts_2026.jsonl --transcript-mode routed \
        --resume >> /tmp/worker_"$3".log 2>&1
    n=$(( $(wc -l < "$D/$got.csv" 2>/dev/null || echo 0) / 10 ))
    want=$(python -c "import json,sys;print(len(json.load(open('$D/$got.json'))))")
    if [ "$n" -ge "$want" ]; then
        touch "$D/$got.done"
        echo "[$3] done  $got ($n topics) $(date +%H:%M:%S)"
    else
        rmdir "$D/claims/$got" 2>/dev/null
        echo "[$3] INCOMPLETE $got ($n/$want) — claim released"
    fi
done

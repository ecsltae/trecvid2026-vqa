#!/usr/bin/env bash
# Work-stealing MC worker.  $1 = chunk dir, $2 = ollama port, $3 = tag, $4 = model
#
# Same design as gpu_worker.sh (see its header): atomic `mkdir` claims, chunk
# marked .done only when its CSV holds every query, released on failure so
# another worker retries. Work-stealing rather than a static split because the
# 80GB card is ~1.6x the 20GB slices and a fixed division leaves cards idle.
set -u
cd .
source venv/bin/activate
D="$1"; PORT="$2"; TAG="$3"; MODEL="$4"
while true; do
    got=""
    for c in "$D"/chunk_*.json; do
        base=$(basename "$c" .json)
        [ -f "$D/$base.done" ] && continue
        if mkdir "$D/claims/$base" 2>/dev/null; then got="$base"; break; fi
    done
    [ -z "$got" ] && { echo "[$TAG] pool empty, exiting $(date +%H:%M:%S)"; break; }
    echo "[$TAG] claim $got $(date +%H:%M:%S)"
    OLLAMA_HOST=127.0.0.1:"$PORT" python systems/vqa/run_vqa_mc_v5.py \
        --recipe v3 --model "$MODEL" --num-ctx 16384 \
        --topics "$D/$got.json" --frames data/vqa/2026/frames \
        --out "$D/$got.csv" --resume >> "logs/mc2026_$TAG.log" 2>&1
    want=$(python -c "import json;print(len(json.load(open('$D/$got.json'))))")
    got_n=$(cut -d, -f1 "$D/$got.csv" 2>/dev/null | sort -u | wc -l)
    if [ "$want" -eq "$got_n" ]; then
        touch "$D/$base.done" 2>/dev/null; touch "$D/$got.done"
        echo "[$TAG] done $got ($got_n/$want) $(date +%H:%M:%S)"
    else
        rmdir "$D/claims/$got" 2>/dev/null
        echo "[$TAG] INCOMPLETE $got ($got_n/$want) — released $(date +%H:%M:%S)"
    fi
done

#!/usr/bin/env bash
# Production 2026 MC worker: 32b, the harness E2 selected, translated transcripts.
# Work-stealing with atomic mkdir claims; a chunk is .done only when its CSV
# holds every query, and an incomplete chunk releases its claim for a retry.
set -u
cd .
source "${VENV:-.venv}/bin/activate"
D="$1"; PORT="$2"; TAG="$3"
while true; do
    got=""
    for c in "$D"/chunk_*.json; do
        base=$(basename "$c" .json)
        [ -f "$D/$base.done" ] && continue
        if mkdir "$D/claims/$base" 2>/dev/null; then got="$base"; break; fi
    done
    [ -z "$got" ] && { echo "[$TAG] pool empty $(date +%H:%M:%S)"; break; }
    echo "[$TAG] claim $got $(date +%H:%M:%S)"
    OLLAMA_HOST=127.0.0.1:"$PORT" python systems/vqa/run_vqa_mc_v5.py \
        --recipe v3 --no-draft --parse-repair --stage2-predict 1024 \
        --model qwen3-vl:32b-instruct --num-ctx 16384 --adaptive --frames-per-sample 16 \
        --transcripts /tmp/transcripts_2026_translate.jsonl \
        --topics "$D/$got.json" --frames data/vqa/2026/frames_dense \
        --out "$D/$got.csv" --resume >> "logs/prod32bdense_$TAG.log" 2>&1
    want=$(python -c "import json;print(len(json.load(open('$D/$got.json'))))")
    have=$(cut -d, -f1 "$D/$got.csv" 2>/dev/null | sort -u | wc -l)
    if [ "$want" -eq "$have" ]; then touch "$D/$got.done"; echo "[$TAG] done $got ($have/$want)"
    else rmdir "$D/claims/$got" 2>/dev/null; echo "[$TAG] INCOMPLETE $got ($have/$want) released"; fi
done

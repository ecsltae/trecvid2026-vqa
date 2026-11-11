#!/usr/bin/env bash
# Noise-floor replicates: identical flags, identical model, different seeds.
set -u
cd .
source "${VENV:-.venv}/bin/activate"
BASE="--recipe v3 --no-draft --parse-repair --stage2-predict 1024 --num-ctx 16384 --transcript-mode off"
for r in 1 2; do
  port=$([ $r -eq 1 ] && echo 11435 || echo 11436)
  OLLAMA_HOST=127.0.0.1:$port python systems/vqa/run_vqa_mc_v5.py $BASE \
    --model qwen3-vl:8b-instruct --seed $((20260904 + r)) \
    --topics /tmp/e2_topics.json --frames data/vqa/frames_pool16_512 \
    --out /tmp/e5_floor_r$r.csv > logs/e5_floor_r$r.log 2>&1 &
done
wait
echo "[e5] floor replicates complete $(date +%H:%M:%S)"

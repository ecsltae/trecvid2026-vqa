#!/usr/bin/env bash
# E3: model choice, at 8 options, under the harness E2 selected (--no-draft).
#
# Why this matters more than anything else queued: on the full unbiased 2025 set
# 8b is LAST of three (70.12 / 73.31 / 74.82, p=9.4e-04 and 9.8e-07), and it was
# chosen on the contiguous Q<=300 window — the one slice where it wins (-1.35pp
# there, +3.98pp everywhere else). Every model number we own comes from that
# window, and none of it was measured at 8 options.
set -u
cd .
source "${VENV:-.venv}/bin/activate"
LOG=logs
T=/tmp/e2_topics.json
F=data/vqa/frames_pool16_512
FLAGS="--recipe v3 --no-draft --parse-repair --stage2-predict 1024 --num-ctx 16384"

run () { # $1 label  $2 port  $3 model
  OLLAMA_HOST=127.0.0.1:"$2" python systems/vqa/run_vqa_mc_v5.py $FLAGS \
    --model "$3" --topics "$T" --frames "$F" --out "/tmp/e3_$1.csv" \
    > "$LOG/e3_$1.log" 2>&1
  echo "[e3] done $1 $(date +%H:%M:%S)"
}

# 8b on a 20GB slice; the two large models share the 80GB card SEQUENTIALLY —
# 19-20GB each, they must never be co-resident with anything else on it.
run 8b 11435 qwen3-vl:8b-instruct &
( run 30b 11437 qwen3-vl:30b-a3b-instruct
  OLLAMA_HOST=127.0.0.1:11437 ollama stop qwen3-vl:30b-a3b-instruct 2>/dev/null
  run 32b 11437 qwen3-vl:32b-instruct ) &
wait
echo "[e3] ALL DONE $(date +%H:%M:%S)"

#!/usr/bin/env bash
# Tier-1 ROUTING SIGNAL over the 2026 test set: an 8b pass with the abstention
# option injected and the Borda tally dumped. Its ANSWERS are discarded -- the
# abstention option costs -6.64pp when it sits in the answering pass, so tier 1's
# answers come from the clean mc2026_8b_v3 run. This pass exists only to produce
# the router's features.
set -u
cd .
source "${VENV:-.venv}/bin/activate"
for s in 0 1 2; do
  port=$((11439 + s)); rm -f /tmp/sig26_$s.jsonl
  OLLAMA_HOST=127.0.0.1:$port python systems/vqa/run_vqa_mc_v5.py \
    --recipe v3 --no-draft --parse-repair --stage2-predict 1024 --num-ctx 16384 \
    --model qwen3-vl:8b-instruct --abstain --dump-votes /tmp/sig26_$s.jsonl \
    --topics /tmp/sig26_s$s.json --frames data/vqa/2026/frames \
    --out /tmp/sig26_$s.csv > logs/sig26_$s.log 2>&1 &
done
wait
echo "[sig26] DONE $(date '+%F %H:%M:%S')  $(cat /tmp/sig26_*.jsonl | wc -l)/2443"

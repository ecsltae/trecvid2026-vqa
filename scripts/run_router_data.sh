#!/usr/bin/env bash
# Collect router training data at n=1822: abstention flag + full Borda tally +
# rank1-rank2 margin, for both the 8b (tier 1) and the 32b (tier 2 target).
#
# The 16%-escalation policy was chosen by hand on 497 items. That is a strong
# overfitting risk: the thresholds were picked after seeing the same data they
# are scored on. This gets a 3.7x larger set so a router can be TRAINED with
# cross-validation and then tested where it was not fitted.
set -u
cd .
source "${VENV:-.venv}/bin/activate"
L=logs
B="--recipe v3 --no-draft --parse-repair --stage2-predict 1024 --num-ctx 16384 \
   --frames data/vqa/frames_pool16_512"
for s in 0 1 2; do
  port=$((11439 + s))
  rm -f /tmp/rt8_$s.jsonl
  OLLAMA_HOST=127.0.0.1:$port python systems/vqa/run_vqa_mc_v5.py $B \
    --model qwen3-vl:8b-instruct --abstain --dump-votes /tmp/rt8_$s.jsonl \
    --topics /tmp/rt_s$s.json --out /tmp/rt8_$s.csv > "$L/rt8_$s.log" 2>&1 &
done
wait
echo "[router] 8b abstain+tally done $(date '+%F %H:%M:%S')"

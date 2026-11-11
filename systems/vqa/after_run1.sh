#!/usr/bin/env bash
# Post-processing + next-experiment chain, started while run 1 is still going.
# Arg $1 = PID of the running consensus_8b inference.
#
#   1. repair refusal answers ("the video is unavailable...") — run 1 was
#      launched before that prompt bug was fixed, and a refusal is a guaranteed
#      0.000 on every official metric
#   2. regenerate just those topics with the fixed prompt (--resume)
#   3. build + officially validate the submission
#   4. dev-validate the replacements for the dropped 30b runs, since 30b alone
#      measured no better than 8b under consensus (-0.006, -0.3 SE)
set -u
cd .
source venv/bin/activate
SP=${SCRATCH:-/tmp}
RUN=runs/vqa/ag/2026/consensus_8b.csv
TOPICS=data/vqa/2026/vqa.2026.testingDataset.Task1.json

while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "=== run1 inference finished $(date) ==="

echo "=== repairing refusals ==="
python systems/vqa/repair_refusals.py --run "$RUN" 2>&1 | tail -14

echo "=== regenerating repaired topics with fixed prompt ==="
python systems/vqa/run_vqa_v4.py \
    --topics "$TOPICS" --frames data/vqa/2026/frames --out "$RUN" \
    --model qwen3-vl:8b-instruct --slate consensus --samples 8 --resume 2>&1 | tail -3

echo "=== building submission ==="
python eval/vqa/make_submission.py --run "$RUN" --topics "$TOPICS" --task ag \
    --out submissions/vqa2026/BiTeM_ag_consensus_8b.json 2>&1 | tail -8

echo "=== dev: 8b samples=16 $(date) ==="
python systems/vqa/run_vqa_v4.py \
    --topics data/vqa/topics/vqa_2025_ag_topics.csv --frames data/vqa/frames_pool16_512 \
    --out "$SP/v4_dev80_s16.csv" --model qwen3-vl:8b-instruct \
    --samples 16 --slate consensus --limit 80 2>&1 | tail -2

echo "=== dev: hybrid 8b+30b $(date) ==="
python systems/vqa/run_vqa_v4.py \
    --topics data/vqa/topics/vqa_2025_ag_topics.csv --frames data/vqa/frames_pool16_512 \
    --out "$SP/v4_dev80_hybrid.csv" \
    --model "qwen3-vl:8b-instruct,qwen3-vl:30b-a3b-instruct" \
    --samples 8 --slate consensus --limit 80 2>&1 | tail -2

echo "=== dev: diverse control (8b) $(date) ==="
python systems/vqa/run_vqa_v4.py \
    --topics data/vqa/topics/vqa_2025_ag_topics.csv --frames data/vqa/frames_pool16_512 \
    --out "$SP/v4_dev80_diverse.csv" --model qwen3-vl:8b-instruct \
    --samples 8 --slate diverse --limit 80 2>&1 | tail -2

echo "=== scoring dev candidates on the official metric $(date) ==="
python eval/vqa/compare_official.py --exclude-dead --runs \
    "$SP/v4_dev80_v2style.csv=consensus-8b-s8" \
    "$SP/v4_dev80_s16.csv=consensus-8b-s16" \
    "$SP/v4_dev80_hybrid.csv=consensus-hybrid" \
    "$SP/v4_dev80_diverse.csv=diverse-8b-control" 2>&1 | tail -18

echo "=== after_run1 COMPLETE $(date) ==="

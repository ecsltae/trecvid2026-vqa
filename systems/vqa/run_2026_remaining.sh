#!/usr/bin/env bash
# Remaining TRECVID 2026 AG runs. Run 1 (consensus_8b) is done, validated and
# built. Deadline 2026-08-17, max 5 runs.
#
# What the dev evidence actually supports (n=76, official UMBC metric):
#   - slate policy (consensus vs diverse) is a NO-OP for mean@10: -0.0003 +/-
#     0.0013 on identical candidate pools. Do not spend a run on it.
#   - 30b alone is not better than 8b under consensus (-0.006, -0.3 SE), and
#     the 8b+30b hybrid pool adds nothing (+0.004, +0.3 SE).
#   - 16 samples is the best point estimate (+0.014 over 8, 1.2 SE — inside the
#     ~0.015 run-to-run noise, but it also raises best@10 0.408 -> 0.435, i.e.
#     it genuinely enriches the candidate pool).
#
#   run 2  consensus_8b_s16   ~11h  best point estimate, richer pool
#   run 3  v3_style_control    ~6h  THE scientific control: ranks 2-10 as
#                                   deliberately speculative alternatives,
#                                   which is what v3 did and what the mean
#                                   punishes. Submitting it buys an official
#                                   NIST measurement of the aggregation effect
#                                   rather than only our local one.
set -u
cd .
source venv/bin/activate

TOPICS=data/vqa/2026/vqa.2026.testingDataset.Task1.json
FRAMES=data/vqa/2026/frames
OUT=runs/vqa/ag/2026

echo "=== run2 consensus_8b_s16 — $(date) ==="
python systems/vqa/run_vqa_v4.py \
    --topics "$TOPICS" --frames "$FRAMES" --out "$OUT/consensus_8b_s16.csv" \
    --model qwen3-vl:8b-instruct --slate consensus --samples 16 --resume
python systems/vqa/repair_refusals.py --run "$OUT/consensus_8b_s16.csv" 2>&1 | tail -6
python systems/vqa/run_vqa_v4.py \
    --topics "$TOPICS" --frames "$FRAMES" --out "$OUT/consensus_8b_s16.csv" \
    --model qwen3-vl:8b-instruct --slate consensus --samples 16 --resume
python eval/vqa/make_submission.py --run "$OUT/consensus_8b_s16.csv" \
    --topics "$TOPICS" --task ag \
    --out submissions/vqa2026/BiTeM_ag_consensus_8b_s16.json 2>&1 | tail -8

echo "=== run3 v3-style speculative-alternatives control — $(date) ==="
python systems/vqa/run_vqa_v3.py --task ag \
    --topics "$TOPICS" --frames "$FRAMES" \
    --out "$OUT/v3_style_control.csv" \
    --model qwen3-vl:8b-instruct --num-answers 10 --resume
python eval/vqa/make_submission.py --run "$OUT/v3_style_control.csv" \
    --topics "$TOPICS" --task ag \
    --out submissions/vqa2026/BiTeM_ag_v3_style_control.json 2>&1 | tail -8

echo "=== remaining runs COMPLETE $(date) ==="
ls -la submissions/vqa2026/

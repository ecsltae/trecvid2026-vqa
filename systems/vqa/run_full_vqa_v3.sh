#!/usr/bin/env bash
# run_full_vqa_v3.sh — all 6 TRECVID VQA v3 runs (3 AG + 3 MC) on qwen3-vl.
#
#   R1 — qwen3-vl:8b-instruct direct            (fast baseline)
#   R2 — qwen3-vl:30b-a3b-instruct direct       (flagship)
#   R3 — qwen3-vl:30b-a3b-instruct + few-shot   (flagship + dynamic examples)
#
# AG deadline 2026-08-29, MC deadline 2026-09-29.
# All runs are --resume-safe: rerun this script after any crash.
#
# Usage:
#   nohup bash systems/vqa/run_full_vqa_v3.sh > /tmp/vqa_v3_full.log 2>&1 &
#   bash systems/vqa/run_full_vqa_v3.sh --limit=5     # smoke test

set -e
cd .
source venv/bin/activate

LIMIT=""
for arg in "$@"; do
    case $arg in
        --limit=*) LIMIT="--limit ${arg#*=}" ;;
    esac
done

AG_TOPICS="data/vqa/topics/vqa_2025_ag_topics.csv"
MC_TOPICS="data/vqa/topics/vqa_2025_mc_topics.csv"

run_one () {  # run_one <task> <model> <fewshot:0|1> <tag>
    local task=$1 model=$2 fewshot=$3 tag=$4
    local out="runs/vqa/${task}/TREC_VQA_${tag}_$(echo $task | tr a-z A-Z).csv"
    local fs_flag=""
    [ "$fewshot" = "1" ] && fs_flag="--fewshot"
    local topics=$AG_TOPICS
    [ "$task" = "mc" ] && topics=$MC_TOPICS
    echo ""
    echo "=== [$task/$tag] model=$model fewshot=$fewshot → $out — $(date) ==="
    python systems/vqa/run_vqa_v3.py \
        --task "$task" --topics "$topics" --out "$out" \
        --model "$model" $fs_flag --resume $LIMIT
}

echo "TRECVID VQA v3 pipeline — started $(date)"

# AG first (earlier deadline), small model first (results sooner)
run_one ag qwen3-vl:8b-instruct      0 qwen3vl8b
run_one ag qwen3-vl:30b-a3b-instruct 0 qwen3vl30b
run_one ag qwen3-vl:30b-a3b-instruct 1 qwen3vl30b_fewshot
run_one mc qwen3-vl:8b-instruct      0 qwen3vl8b
run_one mc qwen3-vl:30b-a3b-instruct 0 qwen3vl30b
run_one mc qwen3-vl:30b-a3b-instruct 1 qwen3vl30b_fewshot

echo ""
echo "=== All 6 runs complete — $(date) ==="
echo "Evaluate with:"
for tag in qwen3vl8b qwen3vl30b qwen3vl30b_fewshot; do
    echo "  python eval/vqa/eval_vqa.py --pred runs/vqa/ag/TREC_VQA_${tag}_AG.csv --qrels data/vqa/qrels_ag.csv --task ag"
    echo "  python eval/vqa/eval_vqa.py --pred runs/vqa/mc/TREC_VQA_${tag}_MC.csv --qrels data/vqa/qrels_mc.csv --task mc"
done

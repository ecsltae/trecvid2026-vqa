#!/usr/bin/env bash
# run_full_vqa.sh — All 6 VQA runs (3 AG + 3 MC) with caption caching.
#
# Pipeline:
#   R1      — llava:13b direct (fastest, no qwen3 reasoning)
#   Cache   — llava:13b captions saved to JSON (runs once, shared by R2+R3)
#   R2 AG   — qwen3:32b reasoning over cached captions
#   R2 MC   — qwen3:32b CoT ranking over cached captions
#   R3 AG   — same + dynamic few-shot from 500 training examples
#   R3 MC   — same + dynamic few-shot
#
# Usage:
#   bash systems/vqa/run_full_vqa.sh [--dry-run] [--limit=N] [--skip-r1] [--skip-cache]

set -e
cd .
source venv/bin/activate

DRY=""
LIMIT=""
SKIP_R1=0
SKIP_CACHE=0
for arg in "$@"; do
    case $arg in
        --dry-run) DRY="--dry-run" ;;
        --limit=*) LIMIT="--limit=${arg#*=}" ;;
        --skip-r1) SKIP_R1=1 ;;
        --skip-cache) SKIP_CACHE=1 ;;
    esac
done

FRAMES="data/vqa/frames_full"
AG_TOPICS="data/vqa/topics/vqa_2025_ag_topics.csv"
MC_TOPICS="data/vqa/topics/vqa_2025_mc_topics.csv"
TRAINING="data/vqa/training"
CACHE="data/vqa/captions_cache.json"
CFG_V1="systems/vqa/config.yaml"
CFG_V2="systems/vqa/config_v2.yaml"

mkdir -p runs/vqa/ag runs/vqa/mc

echo "========================================"
echo "TRECVID VQA 2025 — Full Pipeline"
echo "Started: $(date)"
echo "========================================"

# ── Run 1: Baseline (llava:13b direct, no reasoning stage) ────────────────
if [ "$SKIP_R1" -eq 0 ]; then
    echo ""
    echo "[R1-AG] Baseline AG — llava:13b direct"
    python systems/vqa/run_vqa_ag.py \
        --topics "$AG_TOPICS" \
        --frames "$FRAMES" \
        --out runs/vqa/ag/TREC_VQA_llava13b_AG.csv \
        --config "$CFG_V1" \
        $DRY $LIMIT \
        2>&1 | tee /tmp/vqa_r1_ag.log
    echo "[R1-AG] Done: $(date)"

    echo ""
    echo "[R1-MC] Baseline MC — llava:13b direct"
    python systems/vqa/run_vqa_mc.py \
        --topics "$MC_TOPICS" \
        --frames "$FRAMES" \
        --out runs/vqa/mc/TREC_VQA_llava13b_MC.csv \
        --config "$CFG_V1" \
        $DRY $LIMIT \
        2>&1 | tee /tmp/vqa_r1_mc.log
    echo "[R1-MC] Done: $(date)"
fi

# ── Caption Cache: run llava ONCE, shared by R2 + R3 ─────────────────────
if [ "$SKIP_CACHE" -eq 0 ] && [ -z "$DRY" ]; then
    echo ""
    echo "[CACHE] Pre-computing llava captions for all videos..."
    CACHE_RESUME=""
    [ -f "$CACHE" ] && CACHE_RESUME="--resume"
    python systems/vqa/cache_captions.py \
        --topics "$AG_TOPICS" "$MC_TOPICS" \
        --frames "$FRAMES" \
        --out "$CACHE" \
        --max-frames 4 \
        $CACHE_RESUME \
        2>&1 | tee /tmp/vqa_cache.log
    echo "[CACHE] Done: $(date)"
elif [ -n "$DRY" ]; then
    echo "[CACHE] Skipped in dry-run mode"
fi

# ── Run 2: Two-Stage (cached captions + qwen3:32b reasoning) ──────────────
echo ""
echo "[R2-AG] Two-Stage AG — cached captions + qwen3:32b"
python systems/vqa/run_ag_v2.py \
    --topics "$AG_TOPICS" \
    --frames "$FRAMES" \
    --out runs/vqa/ag/TREC_VQA_twostage_AG.csv \
    --config "$CFG_V2" \
    --captions-cache "$CACHE" \
    --resume \
    $DRY $LIMIT \
    2>&1 | tee /tmp/vqa_r2_ag.log
echo "[R2-AG] Done: $(date)"

echo ""
echo "[R2-MC] Two-Stage MC — cached captions + qwen3:32b CoT"
python systems/vqa/run_mc_v2.py \
    --topics "$MC_TOPICS" \
    --frames "$FRAMES" \
    --out runs/vqa/mc/TREC_VQA_twostage_MC.csv \
    --config "$CFG_V2" \
    --captions-cache "$CACHE" \
    --resume \
    $DRY $LIMIT \
    2>&1 | tee /tmp/vqa_r2_mc.log
echo "[R2-MC] Done: $(date)"

# ── Run 3: Two-Stage + Few-Shot ────────────────────────────────────────────
echo ""
echo "[R3-AG] Two-Stage + Few-Shot AG"
python systems/vqa/run_ag_v2.py \
    --topics "$AG_TOPICS" \
    --frames "$FRAMES" \
    --out runs/vqa/ag/TREC_VQA_twostage_fewshot_AG.csv \
    --config "$CFG_V2" \
    --captions-cache "$CACHE" \
    --training "$TRAINING" \
    --fewshot \
    --resume \
    $DRY $LIMIT \
    2>&1 | tee /tmp/vqa_r3_ag.log
echo "[R3-AG] Done: $(date)"

echo ""
echo "[R3-MC] Two-Stage + Few-Shot MC"
python systems/vqa/run_mc_v2.py \
    --topics "$MC_TOPICS" \
    --frames "$FRAMES" \
    --out runs/vqa/mc/TREC_VQA_twostage_fewshot_MC.csv \
    --config "$CFG_V2" \
    --captions-cache "$CACHE" \
    --training "$TRAINING" \
    --fewshot \
    --resume \
    $DRY $LIMIT \
    2>&1 | tee /tmp/vqa_r3_mc.log
echo "[R3-MC] Done: $(date)"

echo ""
echo "========================================"
echo "All runs complete: $(date)"
echo "========================================"
echo ""
ls -lh runs/vqa/ag/ runs/vqa/mc/
echo ""
echo "Evaluate:"
echo "  python eval/vqa/eval_vqa.py --pred runs/vqa/ag/TREC_VQA_twostage_fewshot_AG.csv --qrels data/vqa/qrels_ag.csv --task ag"
echo "  python eval/vqa/eval_vqa.py --pred runs/vqa/mc/TREC_VQA_twostage_fewshot_MC.csv --qrels data/vqa/qrels_mc.csv --task mc"

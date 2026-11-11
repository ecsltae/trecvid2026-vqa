#!/usr/bin/env bash
# watchdog_r2r3.sh — Waits for caption cache to finish, then runs R2 and R3.
# Launch after cache_captions.py is started.
#
# Usage: nohup bash systems/vqa/watchdog_r2r3.sh > /tmp/vqa_r2r3_watchdog.log 2>&1 &

set -e
cd .
source venv/bin/activate

CACHE="data/vqa/captions_cache.json"
AG_TOPICS="data/vqa/topics/vqa_2025_ag_topics.csv"
MC_TOPICS="data/vqa/topics/vqa_2025_mc_topics.csv"
FRAMES="data/vqa/frames_full"
TRAINING="data/vqa/training"
CFG_V2="systems/vqa/config_v2.yaml"

echo "Watchdog started: $(date)"
echo "Waiting for caption cache at $CACHE..."

# Wait for cache file to exist and not be growing (check every 5 min)
prev_size=0
while true; do
    if [ -f "$CACHE" ]; then
        cur_size=$(stat -c%s "$CACHE" 2>/dev/null || echo 0)
        # Check if cache is done: process no longer running + file exists
        if ! pgrep -f "cache_captions.py" > /dev/null 2>&1; then
            echo "Cache process finished. Cache size: $(du -h $CACHE | cut -f1)"
            break
        fi
        if [ "$cur_size" -gt 0 ] && [ "$cur_size" -eq "$prev_size" ]; then
            echo "Cache size stable at $cur_size bytes but process still running..."
        fi
        prev_size=$cur_size
    fi
    sleep 300   # check every 5 minutes
done

echo ""
echo "=== Starting R2 and R3 runs: $(date) ==="
echo ""

echo "[R2-AG] Two-Stage AG"
python systems/vqa/run_ag_v2.py \
    --topics "$AG_TOPICS" \
    --frames "$FRAMES" \
    --out runs/vqa/ag/TREC_VQA_twostage_AG.csv \
    --config "$CFG_V2" \
    --captions-cache "$CACHE" \
    --resume \
    2>&1 | tee /tmp/vqa_r2_ag.log
echo "[R2-AG] Done: $(date)"

echo ""
echo "[R2-MC] Two-Stage MC"
python systems/vqa/run_mc_v2.py \
    --topics "$MC_TOPICS" \
    --frames "$FRAMES" \
    --out runs/vqa/mc/TREC_VQA_twostage_MC.csv \
    --config "$CFG_V2" \
    --captions-cache "$CACHE" \
    --resume \
    2>&1 | tee /tmp/vqa_r2_mc.log
echo "[R2-MC] Done: $(date)"

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
    2>&1 | tee /tmp/vqa_r3_mc.log
echo "[R3-MC] Done: $(date)"

echo ""
echo "=== All R2+R3 runs complete: $(date) ==="
echo ""

# Auto-evaluate when done
echo "=== Final Evaluation ==="
python eval/vqa/eval_vqa.py --pred runs/vqa/ag/TREC_VQA_twostage_AG.csv       --qrels data/vqa/qrels_ag.csv --task ag
python eval/vqa/eval_vqa.py --pred runs/vqa/mc/TREC_VQA_twostage_MC.csv       --qrels data/vqa/qrels_mc.csv --task mc
python eval/vqa/eval_vqa.py --pred runs/vqa/ag/TREC_VQA_twostage_fewshot_AG.csv --qrels data/vqa/qrels_ag.csv --task ag
python eval/vqa/eval_vqa.py --pred runs/vqa/mc/TREC_VQA_twostage_fewshot_MC.csv --qrels data/vqa/qrels_mc.csv --task mc

echo "=== Done: $(date) ==="

#!/usr/bin/env bash
# TRECVID VQA 2026 — Task 1 (Answer Generation) submission matrix.
# Deadline 2026-08-17. Max 5 runs per team.
#
# Every stage is --resume safe: rerun this script after any interruption and it
# picks up where it stopped. Runs are ordered so that a complete, submittable
# run exists as early as possible.
#
#   1  consensus-8b    ~6h   fast insurance run — a valid submission in hand
#   2  consensus-30b  ~15h   expected best: stronger perception, same method
#   3  diverse-30b    ~15h   honest-diversity CONTROL. Identical system, one
#                            answer per semantic cluster instead of consensus
#                            ranking. Isolates the aggregation effect on the
#                            official mean — the most publishable artifact here.
#   4  consensus-30b-s12 ~22h  more samples => better consensus estimate
#
# Usage:  nohup bash systems/vqa/run_2026_ag_matrix.sh > /tmp/vqa2026_matrix.log 2>&1 &
set -u
cd .
source venv/bin/activate

TOPICS=data/vqa/2026/vqa.2026.testingDataset.Task1.json
FRAMES=data/vqa/2026/frames
OUT=runs/vqa/ag/2026
mkdir -p "$OUT" submissions/vqa2026

echo "=== [prep] refresh frames for any newly downloaded YouTube videos — $(date) ==="
python systems/vqa/extract_frames_2026.py \
    --videos data/vqa/2026/videos/videos data/vqa/2026/videos_yt \
    --out "$FRAMES" --pool 16 --workers 12 2>&1 | tail -3

echo "=== [prep] downscale new frames to 512px (4.8x faster, quality-neutral) ==="
python - <<'PY'
from PIL import Image
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
MAX = 512
def shrink(p):
    try:
        im = Image.open(p); w, h = im.size
        if max(w, h) <= MAX: return 0
        s = MAX / max(w, h)
        im.convert("RGB").resize((max(1,int(w*s)), max(1,int(h*s))), Image.LANCZOS)\
          .save(p, "JPEG", quality=88)
        return 1
    except Exception:
        return 0
ps = [str(p) for p in Path("data/vqa/2026/frames").rglob("frame_*.jpg")]
with ProcessPoolExecutor(max_workers=12) as ex:
    n = sum(ex.map(shrink, ps, chunksize=200))
print(f"  resized {n} of {len(ps)} frames")
PY

run_one () {   # run_one <tag> <model> <slate> <samples>
    local tag=$1 model=$2 slate=$3 samples=$4
    local csv="$OUT/${tag}.csv"
    echo ""
    echo "=== [$tag] model=$model slate=$slate samples=$samples — start $(date) ==="
    python systems/vqa/run_vqa_v4.py \
        --topics "$TOPICS" --frames "$FRAMES" --out "$csv" \
        --model "$model" --slate "$slate" --samples "$samples" --resume
    echo "=== [$tag] inference done $(date) — building submission ==="
    python eval/vqa/make_submission.py \
        --run "$csv" --topics "$TOPICS" --task ag \
        --out "submissions/vqa2026/BiTeM_ag_${tag}.json"
    echo "=== [$tag] COMPLETE $(date) ==="
}

run_one consensus_8b     qwen3-vl:8b-instruct       consensus  8
run_one consensus_30b    qwen3-vl:30b-a3b-instruct  consensus  8
run_one diverse_30b      qwen3-vl:30b-a3b-instruct  diverse    8
run_one consensus_30b_s12 qwen3-vl:30b-a3b-instruct consensus 12

echo ""
echo "=== ALL RUNS COMPLETE $(date) ==="
ls -la submissions/vqa2026/

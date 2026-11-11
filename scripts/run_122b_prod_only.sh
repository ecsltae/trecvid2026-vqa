#!/usr/bin/env bash
# 122b production. A SEPARATE file from run_122b_final.sh -- never edit a script
# while it is running: bash reads incrementally by byte offset, so patching the
# top shifts everything and corrupts the parse mid-execution. That is exactly
# what killed the previous run after its replicate finished.
#
# FRAMING, fixed by the pre-registered rule and not negotiable afterwards:
# the two 122b replicates are 71.43% and 74.65% (spread 3.22pp, mean 73.04%)
# against 32b's 69.62/71.43/71.83/72.03 (mean 71.23%, max 72.03%). The second
# 122b run ties a 32b replicate exactly and sits below the 32b max. The
# distributions overlap. This run is a DIVERSE fifth submission from a different
# model family -- it is NOT a better system, and must not be described as one.
set -u
cd .
source "${VENV:-.venv}/bin/activate"
L=logs
D=runs/vqa/mc/2026/prod122b/chunks
mkdir -p "$D/claims"
curl -s --max-time 5 http://127.0.0.1:11438/api/tags >/dev/null || { echo "11438 down"; exit 1; }
python - <<PY
import json
d=sorted(json.load(open('data/vqa/2026/task2/vqa.2026.testingDataset.Task2_v2.json')),key=lambda x:x['Q_ID'])
CH=25; n=(len(d)+CH-1)//CH
import os
for i in range(n):
    p=f"$D/chunk_{i:03d}.json"
    if not os.path.exists(p): json.dump(d[i::n],open(p,"w"))
print(f"  {n} chunks")
PY
for c in "$D"/chunk_*.json; do
  b=$(basename "$c" .json)
  [ -f "$D/$b.done" ] && continue
  OLLAMA_HOST=127.0.0.1:11438 python systems/vqa/run_vqa_mc_v5.py \
    --recipe v3 --no-draft --parse-repair --stage2-predict 1024 --num-ctx 16384 \
    --think off --model qwen3.5:122b --call-timeout 600 \
    --adaptive --transcripts /tmp/transcripts_2026_translate.jsonl --transcript-mode routed \
    --topics "$c" --frames data/vqa/2026/frames --out "$D/$b.csv" --resume \
    >> "$L/prod122b.log" 2>&1
  want=$(python -c "import json;print(len(json.load(open('$c'))))")
  have=$(cut -d, -f1 "$D/$b.csv" 2>/dev/null | sort -u | wc -l)
  if [ "$want" -eq "$have" ]; then touch "$D/$b.done"; fi
  echo "[122b] $b $have/$want $(date '+%F %H:%M:%S')"
done
echo "[122b] PRODUCTION DONE $(date '+%F %H:%M:%S')"

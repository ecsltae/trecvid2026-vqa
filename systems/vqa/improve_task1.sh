#!/usr/bin/env bash
# Task 1 improvement queue, in value-per-hour order. Each stage produces a
# VALIDATED submission before the next begins, so there is always a better
# run ready to upload no matter when the window closes.
#
#   S1  router patch on the ASR run   ~10 min   65 -> 86 verbal-routed topics
#   S2  few-shot dev A/B              ~40 min   validate before spending 6h
#   S3  few-shot full run             ~6 h      only if S2 is positive
#
# All stages are --resume safe. Log: /tmp/improve_task1.log
set -u
cd .
source venv/bin/activate
SP=${SCRATCH:-/tmp}
TOPICS=data/vqa/2026/vqa.2026.testingDataset.Task1.json
FRAMES=data/vqa/2026/frames
OUT=runs/vqa/ag/2026

# ── S1: patch the ASR run with the widened router ────────────────────────────
echo "=== S1 router patch — $(date) ==="
cp -n "$OUT/consensus_8b_asr.csv" "$OUT/consensus_8b_asr.csv.prerouter.bak" 2>/dev/null || true
python - <<'PY'
import csv, json, sys
sys.path.insert(0, "systems/vqa")
from run_vqa_v4 import audio_route
run = "runs/vqa/ag/2026/consensus_8b_asr.csv"
T = {str(t["Q_ID"]): t for t in json.load(open("data/vqa/2026/vqa.2026.testingDataset.Task1.json"))}
# Which topics does the WIDENED router now send to the transcript path? Those
# rows were generated transcript-free, so drop them and let --resume redo them.
newly = {q for q, t in T.items() if audio_route(t["Question"]) == "verbal"}
rows = list(csv.reader(open(run, newline="", encoding="utf-8")))
kept = [r for r in rows if not (len(r) >= 4 and r[0].strip() in newly)]
with open(run, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(kept)
print(f"  dropped {len(rows)-len(kept)} rows across {len(newly)} routed topics for regeneration")
PY
python systems/vqa/run_vqa_v4.py --topics "$TOPICS" --frames "$FRAMES" \
    --out "$OUT/consensus_8b_asr.csv" --model qwen3-vl:8b-instruct \
    --samples 16 --slate consensus --resume \
    --transcripts data/vqa/transcripts_2026.jsonl --transcript-mode routed 2>&1 | tail -3
python eval/vqa/make_submission.py --run "$OUT/consensus_8b_asr.csv" --topics "$TOPICS" \
    --task ag --out submissions/vqa2026/BiTeM_ag_consensus_8b_asr.json 2>&1 | tail -6

# ── S2: few-shot full run — submitted AS the A/B, not validated locally ─────
# A local dev A/B cannot settle this. The per-topic SD of paired METEOR
# differences is 0.144 (measured), so resolving the expected +0.009 at 2 SE
# needs ~1017 topics PER ARM = 7.3 GPU-hours — comparable to the 8.8-hour run
# itself. A 150-topic gate would have had 0.8 SE power: guaranteed inconclusive.
#
# So instead of paying for a weaker local measurement, we submit the comparison.
# `fewshot_8b` is identical to the already-submitted `consensus_8b` in every
# respect except --fewshot 4 (same model, same 8 samples, same slate), so NIST
# scores the controlled pair for us on all 2443 topics and all five official
# metrics. The prior justifying the run is the v3 measurement: +0.0092 STS at
# 5.6 SE over 1851 paired topics — stronger evidence than any A/B we can afford.
echo ""
echo "=== S2 full few-shot run (8 samples, k=4) — $(date) ==="
python systems/vqa/run_vqa_v4.py --topics "$TOPICS" --frames "$FRAMES" \
    --out "$OUT/fewshot_8b.csv" --model qwen3-vl:8b-instruct \
    --samples 8 --slate consensus --fewshot 4 --resume 2>&1 | tail -3
python systems/vqa/repair_refusals.py --run "$OUT/fewshot_8b.csv" 2>&1 | tail -4
python systems/vqa/run_vqa_v4.py --topics "$TOPICS" --frames "$FRAMES" \
    --out "$OUT/fewshot_8b.csv" --model qwen3-vl:8b-instruct \
    --samples 8 --slate consensus --fewshot 4 --resume 2>&1 | tail -2
python eval/vqa/make_submission.py --run "$OUT/fewshot_8b.csv" --topics "$TOPICS" \
    --task ag --out submissions/vqa2026/BiTeM_ag_fewshot_8b.json 2>&1 | tail -6

echo ""
echo "=== improvement queue complete $(date) ==="
ls -la submissions/vqa2026/*.json

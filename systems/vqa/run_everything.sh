#!/usr/bin/env bash
# "Everything" run — every measured win in one configuration:
#   8 samples (16 vs 8 measured +0.014 at 1.2 SE — not worth 2h of deadline risk)  +  k-NN few-shot k=4  +  routed ASR with the
#   widened router  +  512px frames  +  frequency-first consensus.
#
# This is the only candidate with a real chance of BEATING consensus_8b_s16,
# which is why it is worth building rather than adding another lateral run.
# Expected over s16: ~+0.009 (few-shot prior) + ~+0.006 (audio on 86 topics)
# ≈ +0.015, though neither component is individually resolvable on dev.
#
# WHY SHARDED. We were using 1 of 3 GPUs. The box has a GRID A100D-80C (GPU1,
# already busy with the 8-sample few-shot run) and two A100D-20C slices (GPU0,
# GPU2) sitting idle — and the model needs only 9.7 GB at num_ctx 16384, so it
# fits the 20 GB cards. Measured: 22.1 s/topic on a 20C slice vs ~15 s on the
# 80C. Three shards therefore finish in ~5 h instead of ~15 h serially.
#
# Shard 3 waits for the few-shot run to release GPU1 rather than contending.
# Every shard is --resume safe; rerun this script after any interruption.
set -u
cd .
source venv/bin/activate

TOPICS=data/vqa/2026/vqa.2026.testingDataset.Task1.json
FRAMES=data/vqa/2026/frames
OUT=runs/vqa/ag/2026/everything
mkdir -p "$OUT"

python - <<'PY'
import json, math
T = json.load(open("data/vqa/2026/vqa.2026.testingDataset.Task1.json"))
n = math.ceil(len(T) / 3)
for i in range(3):
    shard = T[i*n:(i+1)*n]
    json.dump(shard, open(f"runs/vqa/ag/2026/everything/shard{i}.json", "w"))
    print(f"  shard{i}: {len(shard)} topics")
PY

common=( --frames "$FRAMES" --model qwen3-vl:8b-instruct --samples 8
         --slate consensus --fewshot 4
         --transcripts data/vqa/transcripts_2026.jsonl --transcript-mode routed --resume )

# shards 0 and 1 start immediately on the idle 20 GB cards
OLLAMA_HOST=127.0.0.1:11435 nohup python systems/vqa/run_vqa_v4.py \
    --topics "$OUT/shard0.json" --out "$OUT/shard0.csv" "${common[@]}" \
    > /tmp/everything_shard0.log 2>&1 &
OLLAMA_HOST=127.0.0.1:11436 nohup python systems/vqa/run_vqa_v4.py \
    --topics "$OUT/shard1.json" --out "$OUT/shard1.csv" "${common[@]}" \
    > /tmp/everything_shard1.log 2>&1 &
echo "shards 0,1 launched on GPU0/GPU2"

# shard 2 waits for the 80 GB card, then runs there (fastest, so it starts last
# and still finishes about the same time)
nohup bash -c '
  while pgrep -f "out runs/vqa/ag/2026/fewshot_8b.csv" >/dev/null 2>&1 \
     || pgrep -f "fewshot_8b.csv" >/dev/null 2>&1; do sleep 60; done
  cd . && source venv/bin/activate
  python systems/vqa/run_vqa_v4.py \
    --topics runs/vqa/ag/2026/everything/shard2.json \
    --out runs/vqa/ag/2026/everything/shard2.csv \
    --frames data/vqa/2026/frames --model qwen3-vl:8b-instruct --samples 8 \
    --slate consensus --fewshot 4 \
    --transcripts data/vqa/transcripts_2026.jsonl --transcript-mode routed --resume
' > /tmp/everything_shard2.log 2>&1 &
echo "shard 2 queued behind the few-shot run on GPU1"

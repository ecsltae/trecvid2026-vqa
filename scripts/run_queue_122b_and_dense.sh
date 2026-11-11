#!/usr/bin/env bash
# Master queue, in dependency order. Waits on STATE, never on process liveness.
#
#   1. everything currently in flight (E5 power arms, E7 translation A/B)
#   2. E6 -- 3x 32b replicates. HIGHEST PRIORITY: four identical-config 8b runs
#      span 4.02pp and one of their six pairs returns McNemar p=0.0169 from
#      nothing, while our headline 32b-vs-8b is +4.63pp p=0.0106. Until the 32b
#      DISTRIBUTION exists we do not know the model choice is real, and every
#      other run is built on that choice.
#   3. qwen3.5:122b with --think off on dev8. Needs 83.7 GiB, so it needs ALL
#      THREE cards and every other model unloaded. Its published 51.3% is an
#      artifact: 67.5% of its frame-bearing calls burned num_predict inside an
#      unterminated <think> block. This is the first valid measurement.
#   4. Dense-frames production run for a spare submission slot.
set -u
cd .
source "${VENV:-.venv}/bin/activate"
L=logs
say () { echo "[queue] $* $(date '+%F %H:%M:%S')"; }

# ---- 1. current work
while pgrep -f "run_vqa_mc_v5.py.*(dev8full|e7_)" >/dev/null 2>&1; do sleep 60; done
say "E5/E7 clear"

# ---- 2. E6 replicates (its own script is already armed; wait for its output)
while [ "$(ls /tmp/e6_32b_r*.csv 2>/dev/null | wc -l)" -lt 3 ] \
   || pgrep -f "run_vqa_mc_v5.py.*e6_32b" >/dev/null 2>&1; do sleep 60; done
say "E6 replicates complete"

# ---- 3. 122b: exclusive fleet. Unload everything first or it cannot fit.
for p in 11434 11435 11436 11437; do
  for m in qwen3-vl:32b-instruct qwen3-vl:30b-a3b-instruct qwen3-vl:8b-instruct; do
    OLLAMA_HOST=127.0.0.1:$p ollama stop "$m" >/dev/null 2>&1 || true
  done
done
sleep 20
cat > /tmp/start_ollama_all.sh <<'INNER'
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2
export OLLAMA_HOST=127.0.0.1:11438
export OLLAMA_MODELS=${OLLAMA_MODELS:-$HOME/.ollama}/models
export OLLAMA_KEEP_ALIVE=24h
exec ollama serve
INNER
chmod +x /tmp/start_ollama_all.sh
pkill -f "OLLAMA_HOST=127.0.0.1:11438" 2>/dev/null || true
nohup /tmp/start_ollama_all.sh > /tmp/ollama_all.log 2>&1 &
sleep 30
say "all-GPU ollama up on 11438"
OLLAMA_HOST=127.0.0.1:11438 python systems/vqa/run_vqa_mc_v5.py \
  --recipe v3 --no-draft --parse-repair --stage2-predict 1024 --num-ctx 16384 \
  --transcript-mode off --think off --model qwen3.5:122b --call-timeout 600 \
  --topics /tmp/e2_topics.json --frames data/vqa/frames_pool16_512 \
  --out /tmp/e8_122b.csv > "$L/e8_122b.log" 2>&1
say "122b done"
OLLAMA_HOST=127.0.0.1:11438 ollama stop qwen3.5:122b >/dev/null 2>&1 || true
pkill -f "OLLAMA_HOST=127.0.0.1:11438" 2>/dev/null || true
sleep 20

# ---- 4. dense-frames production run, spare slot
D=runs/vqa/mc/2026/prod32b_dense/chunks
rm -rf "$D"; mkdir -p "$D/claims"
python - <<PY
import json
d=sorted(json.load(open('data/vqa/2026/task2/vqa.2026.testingDataset.Task2_v2.json')),key=lambda x:x['Q_ID'])
CH=25; n=(len(d)+CH-1)//CH
for i in range(n): json.dump(d[i::n],open(f"$D/chunk_{i:03d}.json","w"))
print(f"  {n} chunks")
PY
# frames_dense holds ONLY the 212 long videos; symlink the rest so one --frames
# root covers the whole set and short clips keep the pool they were measured on
python - <<'PY'
from pathlib import Path
src=Path('data/vqa/2026/frames'); dst=Path('data/vqa/2026/frames_dense')
n=0
for d in src.iterdir():
    if d.is_dir() and not (dst/d.name).exists():
        (dst/d.name).symlink_to(d.resolve()); n+=1
print(f"  linked {n} short-video pools into frames_dense")
PY
sed 's|data/vqa/2026/frames|data/vqa/2026/frames_dense|; s|--adaptive|--adaptive --frames-per-sample 16|; s|prod32b_|prod32bdense_|' \
  scripts/gpu_worker_mc32.sh > scripts/gpu_worker_mc32_dense.sh
chmod +x scripts/gpu_worker_mc32_dense.sh
for t in w1 w2 w3; do
  nohup scripts/gpu_worker_mc32_dense.sh "$D" 11437 "$t" > "$L/prod32bdense_$t.out" 2>&1 &
done
wait
say "ALL QUEUED WORK DONE"

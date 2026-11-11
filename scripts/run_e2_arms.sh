#!/usr/bin/env bash
# E1+E2 in one experiment: does repairing the parser change the answer, and is
# the shipped harness (recipe v3 + draft + 3 votes) actually any good?
#
# The shipped config has NEVER been measured on dev for any model, and its
# nearest measured relative is -7.4pp against a plain single call (p=0.008).
# Arm A reproduces it exactly so the comparison is honest.
set -u
cd .
source "${VENV:-.venv}/bin/activate"
LOG=logs
T=/tmp/e2_topics.json
F=data/vqa/frames_pool16_512
M=qwen3-vl:8b-instruct

# wait for whisper to release the cards — state, not process guesswork
while pgrep -f "systems/vqa/transcribe.py" >/dev/null 2>&1; do sleep 30; done
echo "[e2] GPUs free $(date +%H:%M:%S)"

run () { # $1 label  $2 port  $3.. flags
  lbl="$1"; port="$2"; shift 2
  OLLAMA_HOST=127.0.0.1:"$port" python systems/vqa/run_vqa_mc_v5.py \
    --recipe v3 --model "$M" --num-ctx 16384 --topics "$T" --frames "$F" \
    --out "/tmp/e2_$lbl.csv" "$@" > "$LOG/e2_$lbl.log" 2>&1
  echo "[e2] done $lbl $(date +%H:%M:%S)"
}

run A_shipped   11435 --stage2-predict 512 &
run B_repaired  11437 --stage2-predict 1024 --parse-repair &
run C_nodraft   11436 --stage2-predict 1024 --parse-repair --no-draft &
wait
echo "[e2] ALL ARMS DONE $(date +%H:%M:%S)"

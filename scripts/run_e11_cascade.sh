#!/usr/bin/env bash
# E11 — the abstention cascade, as a factorial. Results decide, not intuition.
#
# THE TRIGGER IS REAL (E10, n=497): abstained items are wrong 63.1% of the time
# vs 22.0% otherwise, a 2.87x lift, firing on 41% of questions. Offering
# abstention in a SINGLE pass costs -6.64pp because it consumes items the model
# would have answered. Looping is what makes that cost go away: the final answer
# is always the argmax over RELEASED options from the accumulated tally, so the
# system never emits the abstention option.
#
# THE ARMS ISOLATE WHAT MOVES. Every escalation result this project produced
# before now confounded "retry" with "different frames" with "more votes". Here:
#   B  same frames, same model  -> PURE RESAMPLING. The control. If B matches C,
#                                  frames were never the mechanism.
#   C  new frames,  same model  -> frame effect
#   D  same frames, 32b         -> MODEL effect with frames held fixed
#   E  new frames,  32b         -> both
# Depth (1/2/3) is swept on whichever of C/D/E wins.
#
# Baseline is the four identical-config 8b runs already on disk: 65.59 / 67.40 /
# 67.81 / 69.62, mean 67.61, sd 1.65. NOTHING under 4.02pp is a result.
set -u
cd .
source "${VENV:-.venv}/bin/activate"
L=logs
B="--recipe v3 --no-draft --parse-repair --stage2-predict 1024 --num-ctx 16384 \
   --topics /tmp/e2_topics.json --frames data/vqa/frames_pool16_512 --abstain"

run () { # $1 label  $2 port  $3.. flags
  lbl="$1"; port="$2"; shift 2
  rm -f "/tmp/e11_$lbl.jsonl"
  OLLAMA_HOST=127.0.0.1:"$port" python systems/vqa/run_vqa_mc_v5.py $B \
    --loop-log "/tmp/e11_$lbl.jsonl" --out "/tmp/e11_$lbl.csv" "$@" \
    > "$L/e11_$lbl.log" 2>&1
  echo "[e11] done $lbl $(date '+%F %H:%M:%S')"
}

# ---- phase 1: 8b-only arms, GPU2, while the 122b still holds the other cards
run B_same_8b  11439 --model qwen3-vl:8b-instruct --abstain-loop 2 --loop-frames same
# moved to run_e11b_parallel.sh: run C_new_8b   11439 --model qwen3-vl:8b-instruct --abstain-loop 2 --loop-frames new
# moved to run_e11b_parallel.sh: run F_new_8b_1 11439 --model qwen3-vl:8b-instruct --abstain-loop 1 --loop-frames new
# moved to run_e11b_parallel.sh: run F_new_8b_3 11439 --model qwen3-vl:8b-instruct --abstain-loop 3 --loop-frames new
echo "[e11] phase 1 complete $(date '+%F %H:%M:%S')"

# ---- phase 2: model-switch arms. 32b is 20GB and will not fit beside the 8b on
#      a 20GB slice, so these wait for the 122b to release the 80GB card.
while [ "$(ls runs/vqa/mc/2026/prod122b/chunks/*.done 2>/dev/null | wc -l)" -lt 98 ]; do sleep 120; done
sleep 60
for p in 11438; do
  OLLAMA_HOST=127.0.0.1:$p ollama stop qwen3.5:122b >/dev/null 2>&1 || true
done
sleep 30
run D_same_32b 11437 --model qwen3-vl:8b-instruct --abstain-loop 2 --loop-frames same \
    --loop-model qwen3-vl:32b-instruct --loop-switch-at 1
run E_new_32b  11437 --model qwen3-vl:8b-instruct --abstain-loop 2 --loop-frames new \
    --loop-model qwen3-vl:32b-instruct --loop-switch-at 1
echo "[e11] ALL ARMS DONE $(date '+%F %H:%M:%S')"

#!/usr/bin/env bash
# E11b — the arms E11 has not started yet, run CONCURRENTLY on the spare VRAM
# instead of in series, plus the two --loop-all arms.
#
# WHY --loop-all EXISTS. The break-early cascade asks for a second view only when
# the first DECLINES. So it never gets a second opinion on an item the model
# answered confidently and wrongly -- and 22.0% of non-abstained items are wrong.
# Frame views do disagree: swapping them fixes 14 and breaks 14. Those 14 fixes
# are real; the net zero is a SELECTION failure, not an evidence failure. An
# oracle over two views is +2.82pp. Accumulating every view is the cheapest
# selector available, and it addresses the whole 497, not just the 41% that abstain.
set -u
cd .
source "${VENV:-.venv}/bin/activate"
L=logs
B="--recipe v3 --no-draft --parse-repair --stage2-predict 1024 --num-ctx 16384 \
   --topics /tmp/e2_topics.json --frames data/vqa/frames_pool16_512 --abstain"
run () { lbl="$1"; port="$2"; shift 2
  rm -f "/tmp/e11_$lbl.jsonl"
  OLLAMA_HOST=127.0.0.1:"$port" python systems/vqa/run_vqa_mc_v5.py $B \
    --loop-log "/tmp/e11_$lbl.jsonl" --out "/tmp/e11_$lbl.csv" "$@" \
    > "$L/e11_$lbl.log" 2>&1
  echo "[e11b] done $lbl $(date '+%F %H:%M:%S')"; }

M8=qwen3-vl:8b-instruct
# GPU1 spare VRAM
run C_new_8b    11440 --model $M8 --abstain-loop 2 --loop-frames new &
run G_all_new   11440 --model $M8 --abstain-loop 2 --loop-frames new --loop-all &
# GPU2 alongside the arm E11 is already running
run G_all_same  11439 --model $M8 --abstain-loop 2 --loop-frames same --loop-all &
wait
echo "[e11b] parallel arms done $(date '+%F %H:%M:%S')"
run F_new_8b_1 11440 --model $M8 --abstain-loop 1 --loop-frames new &
run F_new_8b_3 11440 --model $M8 --abstain-loop 3 --loop-frames new &
wait
echo "[e11b] ALL DONE $(date '+%F %H:%M:%S')"

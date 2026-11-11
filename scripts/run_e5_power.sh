#!/usr/bin/env bash
# E5: the same questions, at 3.7x the n.
#
# Not a new lever. Every remaining effect sits BELOW the 2.21pp run-to-run noise
# floor at n=497 (MDE 5.3pp), so "unresolvable" was a statement about our sample,
# not about the world. At n=1822 the MDE falls to 2.8pp, which is enough to turn
# the transcript effect (+2.62pp, p=0.0660 -- currently "direction only") into a
# result or a null.
#
# Arms 1-2 answer that on 32b, the model we ship, on the 80GB card.
# Arms 3-4 are 8b REPLICATES at identical flags: the noise floor is currently
# estimated from an ACCIDENTAL pair of runs, so it has no confidence interval at
# all. Four replicates give it one.
set -u
cd .
source "${VENV:-.venv}/bin/activate"
LOG=logs
F=data/vqa/frames_pool16_512
BASE="--recipe v3 --no-draft --parse-repair --stage2-predict 1024 --num-ctx 16384"

run () { # $1 label  $2 port  $3 model  $4 topics  $5.. extra
  lbl="$1"; port="$2"; model="$3"; top="$4"; shift 4
  OLLAMA_HOST=127.0.0.1:"$port" python systems/vqa/run_vqa_mc_v5.py $BASE \
    --model "$model" --topics "$top" --frames "$F" --out "/tmp/e5_$lbl.csv" "$@" \
    >> "$LOG/e5_$lbl.log" 2>&1
  echo "[e5] done $lbl $(date +%H:%M:%S)"
}

# 32b, both transcript conditions, sharded across the 80GB card
for s in 0 1 2; do
  run "32b_notr_s$s"  11437 qwen3-vl:32b-instruct /tmp/dev8full_s$s.json --transcript-mode off &
done
wait
for s in 0 1 2; do
  run "32b_tr_s$s"    11437 qwen3-vl:32b-instruct /tmp/dev8full_s$s.json \
      --adaptive --transcripts /tmp/dev8_translate_full.jsonl --transcript-mode routed --min-transcript-words 4 &
done
wait
echo "[e5] 32b arms complete $(date +%H:%M:%S)"

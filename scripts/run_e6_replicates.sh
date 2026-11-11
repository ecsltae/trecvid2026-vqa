#!/usr/bin/env bash
# Does 32b > 8b actually hold, or is it run-to-run noise?
#
# Four IDENTICAL-CONFIG 8b replicates span 65.59-69.62% (range 4.02pp, sd 1.65pp),
# and one of their six pairs returns McNemar p=0.0169 from nothing at all. Our
# headline 32b-vs-8b result is +4.63pp p=0.0106 — barely more extreme than a null
# pair. A single run of each cannot separate the model effect from the seed.
#
# Three 32b replicates give the 32b distribution; compared against the four 8b
# replicates we already have, the question becomes "do these two distributions
# overlap", which a single paired test cannot answer.
set -u
cd .
source "${VENV:-.venv}/bin/activate"
# Gate on the E5 arms SPECIFICALLY, via their own output files. The previous
# pattern "run_vqa_mc_v5.py.*dev8full" also matched E7 -- a job on GPU0+GPU2 that
# does not contend for the 80GB card and has a ~20h ETA -- so E6, the highest
# priority experiment we have, sat blocked for a day on nothing.
while :; do
  n_done=0
  for s in 0 1 2; do
    n=$(cut -d, -f1 "/tmp/e5_32b_tr_s$s.csv" 2>/dev/null | sort -u | wc -l)
    [ "${n:-0}" -ge 607 ] && n_done=$((n_done+1))
  done
  if [ "$n_done" -ge 3 ]; then
    # confirm the writers are gone, matching the OUTPUT path (never a bare token
    # that could match this script's own command line -- that self-match has now
    # cost this project four incidents)
    pgrep -f "out /tmp/e5_32b_tr" >/dev/null 2>&1 || break
  fi
  sleep 60
done
echo "[e6] GPU1 free $(date +%H:%M:%S)"
for r in 1 2 3; do
  OLLAMA_HOST=127.0.0.1:11437 python systems/vqa/run_vqa_mc_v5.py \
    --recipe v3 --no-draft --parse-repair --stage2-predict 1024 --num-ctx 16384 \
    --transcript-mode off --model qwen3-vl:32b-instruct --seed $((20260906 + r)) \
    --topics /tmp/e2_topics.json --frames data/vqa/frames_pool16_512 \
    --out /tmp/e6_32b_r$r.csv > logs/e6_32b_r$r.log 2>&1
  echo "[e6] done 32b replicate $r $(date +%H:%M:%S)"
done
echo "[e6] ALL DONE $(date +%H:%M:%S)"

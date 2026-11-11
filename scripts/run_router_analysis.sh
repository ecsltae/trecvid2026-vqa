#!/usr/bin/env bash
set -u
cd .
source "${VENV:-.venv}/bin/activate"
while [ "$(cat /tmp/rt8_0.jsonl /tmp/rt8_1.jsonl /tmp/rt8_2.jsonl 2>/dev/null | wc -l)" -lt 1815 ]; do sleep 120; done
sleep 60
echo "[router] data complete $(date '+%F %H:%M:%S')"
E=data/vqa/2026/evidence
cp -n /tmp/rt8_*.jsonl /tmp/rt8_*.csv $E/ 2>/dev/null
cat /tmp/e5_32b_notr_s0.csv /tmp/e5_32b_notr_s1.csv /tmp/e5_32b_notr_s2.csv > /tmp/e5_32b_notr_all.csv
cat /tmp/e5_32b_tr_s0.csv /tmp/e5_32b_tr_s1.csv /tmp/e5_32b_tr_s2.csv > /tmp/e5_32b_tr_all.csv
python eval/vqa/router_analysis.py \
  --votes /tmp/rt8_0.jsonl /tmp/rt8_1.jsonl /tmp/rt8_2.jsonl \
  --tier1 /tmp/e7_turbo.csv /tmp/e7_largev3.csv \
  --tier2 /tmp/e5_32b_notr_all.csv /tmp/e5_32b_tr_all.csv \
  2>&1 | tee logs/router_analysis.txt
echo "[router] ANALYSIS DONE $(date '+%F %H:%M:%S')"

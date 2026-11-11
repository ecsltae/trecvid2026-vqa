#!/usr/bin/env bash
# Merge every piece of the "everything" run, then build and officially validate.
#
# WAIT ON STATE, NOT ON PROCESSES. The previous version waited for shard
# processes to disappear, and fired during a two-minute window while workers
# were being restarted — merging 1630 of 2443 topics. (The build was correctly
# refused by make_submission's missing-topic gate, so nothing bad shipped, but
# the lesson stands: "no process is running" is not the same as "the work is
# finished".) This version counts .done markers instead.
set -u
cd .
source venv/bin/activate
D=runs/vqa/ag/2026/everything/chunks
TOTAL=$(ls "$D"/chunk_*.json 2>/dev/null | wc -l)

echo "waiting for $TOTAL chunks + 2 shards — $(date)"
while true; do
    done_n=$(ls "$D"/*.done 2>/dev/null | wc -l)
    [ "$done_n" -ge "$TOTAL" ] && break
    sleep 60
done
# let any in-flight flush settle, and confirm no worker is mid-chunk
while pgrep -f "run_vqa_v4.py.*chunks/chunk_" >/dev/null 2>&1; do sleep 20; done
echo "=== all $TOTAL chunks + shards complete $(date) ==="

python - <<'PY'
import csv, glob, json
from collections import defaultdict
rows = defaultdict(list)
parts = sorted(glob.glob("runs/vqa/ag/2026/everything/shard[01].csv")) + \
        sorted(glob.glob("runs/vqa/ag/2026/everything/chunks/chunk_*.csv"))
for sp in parts:
    for r in csv.reader(open(sp, newline="", encoding="utf-8")):
        if len(r) >= 4:
            rows[r[0].strip()].append(r)
T = json.load(open("data/vqa/2026/vqa.2026.testingDataset.Task1.json"))
out, missing = [], []
for t in T:                                  # emit in official topic order
    q = str(t["Q_ID"])
    if q in rows:
        out.extend(sorted(rows[q], key=lambda r: int(r[2]))[:10])
    else:
        missing.append(q)
with open("runs/vqa/ag/2026/everything.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(out)
print(f"  merged {len(parts)} parts -> {len(rows)} topics, {len(out)} rows, "
      f"{len(missing)} missing")
if missing:
    print(f"  !! MISSING {missing[:10]}{'...' if len(missing) > 10 else ''}")
PY

python systems/vqa/repair_refusals.py --run runs/vqa/ag/2026/everything.csv 2>&1 | tail -4
python eval/vqa/make_submission.py --run runs/vqa/ag/2026/everything.csv \
    --topics data/vqa/2026/vqa.2026.testingDataset.Task1.json --task ag \
    --out submissions/vqa2026/BiTeM_ag_everything.json 2>&1 | tail -8
echo "=== finish_everything done $(date) ==="

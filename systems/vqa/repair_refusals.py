"""
Strip refusal answers out of a run so they can be regenerated.

A refusal ("The video is unavailable, so no event can be described") scores a
guaranteed 0.000 on every official metric. Any plausible guess beats it, since
the metric rewards shared content words and a refusal shares none with the
reference.

This deletes every affected topic's rows from the run CSV; re-running
run_vqa_v4.py with --resume then regenerates just those topics using the fixed
no-video prompt (which no longer primes the refusal by mentioning the video).

Usage:
    python systems/vqa/repair_refusals.py --run runs/vqa/ag/2026/consensus_8b.csv
"""
import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_vqa_v4 import REFUSAL_RE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--topics", default="data/vqa/2026/vqa.2026.testingDataset.Task1.json")
    ap.add_argument("--frames", default="data/vqa/2026/frames")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = Path(args.run)
    rows = list(csv.reader(open(path, newline="", encoding="utf-8")))
    by_q = defaultdict(list)
    for r in rows:
        if len(r) >= 4:
            by_q[r[0].strip()].append(r)

    # Only topics WITHOUT frames can legitimately be refusals. Scoping the scan
    # this way means a real answer that happens to contain "no video" is never
    # thrown away.
    novideo = None
    if Path(args.topics).exists() and Path(args.frames).exists():
        import json
        have = {p.name for p in Path(args.frames).iterdir() if p.is_dir()}
        topics = json.loads(Path(args.topics).read_text(encoding="utf-8"))
        novideo = {str(t["Q_ID"]) for t in topics
                   if Path(str(t["Video_ID"])).stem not in have}
        print(f"  scoping refusal scan to {len(novideo)} topics without frames")

    bad = {q for q, rs in by_q.items()
           if (novideo is None or q in novideo)
           and any(REFUSAL_RE.search(r[3]) for r in rs)}
    short = {q for q, rs in by_q.items() if len(rs) != 10}
    drop = bad | short

    print(f"{path.name}: {len(by_q)} topics")
    print(f"  refusal-contaminated : {len(bad)}")
    print(f"  wrong answer count   : {len(short)}")
    print(f"  to regenerate        : {len(drop)}")
    for q in sorted(drop, key=lambda x: int(x) if x.isdigit() else 0)[:8]:
        print(f"    Q{q}: {by_q[q][0][3][:78]}")

    if args.dry_run or not drop:
        return

    shutil.copy(path, path.with_suffix(".csv.bak"))
    kept = [r for r in rows if not (len(r) >= 4 and r[0].strip() in drop)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(kept)
    print(f"  wrote {path} ({len(kept)} rows kept, backup at {path.name}.bak)")
    print(f"  now re-run run_vqa_v4.py with --resume to regenerate {len(drop)} topics")


if __name__ == "__main__":
    main()

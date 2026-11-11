"""
Compute the per-topic CEILING of the official STS metric: score(reference, reference).

Why this exists. The UMBC service strips a stopword list before comparing, so a
reference made entirely of stopwords reduces to the empty string and scores 0.0
against *itself*:

    score("one", "one")     -> 0.0      (unwinnable topic)
    score("two", "two")     -> 0.0      (unwinnable topic)
    score("nothing","nothing") -> 0.0   (unwinnable topic)
    score("four", "four")   -> 1.0      (winnable; a bare "four" scores perfect)

No system on earth scores above 0 on an unwinnable topic, so those topics are
dead weight in the average — for us and for every competitor. Measuring the
ceiling lets us (a) know the true achievable maximum of the whole run,
(b) exclude dead topics from model selection so they stop masking real gains,
and (c) report a normalized "share of achievable score" alongside the raw mean.

Output JSON: {q_id: {"reference": str, "ceiling": float}}

Usage:
    python eval/vqa/reference_ceiling.py \
        --qrels data/vqa/qrels_ag.csv \
        --out data/vqa/reference_ceilings.json
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from official_sts import OfficialSTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", default="data/vqa/qrels_ag.csv")
    ap.add_argument("--out", default="data/vqa/reference_ceilings.json")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    refs = defaultdict(list)
    with open(args.qrels, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = (row.get("Q_ID") or "").strip()
            a = (row.get("Answer") or "").strip()
            if q and a:
                refs[q].append(a)

    pairs, index = [], []
    for q, rs in refs.items():
        for r in rs:
            pairs.append((r, r))
            index.append((q, r))
    print(f"Scoring {len(pairs)} reference self-similarities "
          f"({len(refs)} topics) — this is a one-time, permanently cached cost")

    sts = OfficialSTS()
    scores = sts.score_many(pairs, workers=args.workers, progress_every=200)

    out = {}
    for (q, r), s in zip(index, scores):
        s = 0.0 if s != s else s
        # keep the best-scoring reference when a topic has several
        if q not in out or s > out[q]["ceiling"]:
            out[q] = {"reference": r, "ceiling": s}

    dead = [q for q, v in out.items() if v["ceiling"] <= 0.0]
    partial = [q for q, v in out.items() if 0.0 < v["ceiling"] < 0.999]
    print(f"\n  topics total      : {len(out)}")
    print(f"  UNWINNABLE (0.0)  : {len(dead)}  ({100*len(dead)/max(len(out),1):.1f}%)")
    print(f"  partial ceiling   : {len(partial)}")
    print(f"  full ceiling (1.0): {len(out)-len(dead)-len(partial)}")
    print(f"  mean ceiling      : {sum(v['ceiling'] for v in out.values())/max(len(out),1):.4f}")
    if dead:
        print("\n  examples of unwinnable references:")
        for q in dead[:15]:
            print(f"    Q{q}: {out[q]['reference']!r}")

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    print(f"  service: {sts.stats()}")


if __name__ == "__main__":
    main()

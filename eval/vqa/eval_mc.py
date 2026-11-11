"""
Score MC runs the way eval.vqa.mc.py does, plus the baselines that keep us honest.

Official scoring (eval/vqa_official_nist/eval.vqa.mc.py):
    top1 = 1 if ranked_responses[0] == correct_answer else 0
    MRR  = 1 / rank of the correct answer
Raw string equality, and it applies the same SKIP set of 18 unavailable queries
the AG scorer uses.

Two baselines are always printed alongside, because MC has shallow heuristics
that score far above chance and any real system must clear them:
  * random over each question's own option count
  * longest option (no video, no model) — measured at 47.1% on the 2025 dev set,
    nearly double chance. A system that does not beat this has learned nothing.

Accepts runs that emit either option TEXT (v5, and what a submission must
contain) or a bare A/B/C/D letter (v3's MC path).

Usage:
    python eval/vqa/eval_mc.py --runs /tmp/mc_v5_dev300.csv=v5 /tmp/mc_v3_dev300.csv=v3
"""
from __future__ import annotations

import argparse
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path

SKIP = {18, 27, 62, 222, 448, 507, 584, 629, 903, 1007, 1159, 1260, 1330,
        1381, 1554, 1691, 1755, 1822}


def load_topics(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) >= 7:
                # r[3:], NOT r[3:7]: the 2026 set has EIGHT options for 1971 of
                # 2443 queries. Truncating at 4 silently dropped every item whose
                # gold was E-H — on an 8-option dev set this resolved 293 of 600
                # golds and reported the rest as unscoreable without warning.
                # NO .strip('"'): csv.reader has already removed the field's
                # quoting, so stripping again mutilates any option whose TEXT
                # legitimately contains a double quote — 12 options in the dev8
                # subset and 38 in the 2026 test set. A mutilated option can
                # never equal a run's answer under the official scorer's raw
                # string equality, so those golds were permanently unmatchable.
                out[r[0].strip()] = [c.strip() for c in r[3:] if c.strip()]
    return out


def load_gold(path, topics):
    """qrels_mc.csv holds the correct LETTER -> resolve to option text."""
    gold = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = (row.get("Q_ID") or "").strip()
            a = (row.get("Answer") or "").strip()
            L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(topics.get(q, []))]
            if q in topics and a and a[0].upper() in L:
                gold[q] = topics[q][L.index(a[0].upper())]
    return gold


def load_run(path, topics):
    """-> {q_id: [option text in rank order]}. Letters are resolved to text."""
    rows = defaultdict(list)
    n_bad = 0
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) >= 4:
                # Parse the rank BEFORE touching `rows`. The previous version
                # indexed the defaultdict first, so a run whose rank column does
                # not parse produced an entry with an EMPTY ranking and scored a
                # silent 0.0% — the same failure class that once reported a
                # working model at 0% by comparing letters against option text.
                try:
                    rank = int(r[2])
                except ValueError:
                    n_bad += 1
                    continue
                rows[r[0].strip()].append((rank, r[3]))
    out = {}
    for q, v in rows.items():
        seq = [a for _, a in sorted(v, key=lambda x: x[0])]
        L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(topics.get(q, []))]
        if q in topics and L and all(len(s.strip()) == 1 and s.strip().upper() in L for s in seq):
            seq = [topics[q][L.index(s.strip().upper())] for s in seq]
        if not seq:
            continue                     # never hand the scorer an empty ranking
        out[q] = seq
    if n_bad:
        print(f"  WARNING {path}: {n_bad} rows had an unparseable rank column")
    if not out:
        raise SystemExit(f"REFUSING TO SCORE: {path} produced 0 usable rankings. "
                         f"Check the run format — a silent 0.0% has happened here before.")
    return out


def score(run, gold):
    top1, mrr, n = 0, 0.0, 0
    for q, ranked in run.items():
        if q not in gold or (q.isdigit() and int(q) in SKIP):
            continue
        n += 1
        if ranked and ranked[0] == gold[q]:
            top1 += 1
        for i, a in enumerate(ranked, 1):
            if a == gold[q]:
                mrr += 1.0 / i
                break
    return (top1 / n if n else 0.0), (mrr / n if n else 0.0), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="path[=label] ...")
    ap.add_argument("--topics", default="data/vqa/topics/vqa_2025_mc_topics.csv")
    ap.add_argument("--qrels", default="data/vqa/qrels_mc.csv")
    args = ap.parse_args()

    topics = load_topics(args.topics)
    gold = load_gold(args.qrels, topics)

    runs = {}
    for spec in args.runs:
        path, _, label = spec.partition("=")
        runs[label or Path(path).stem] = load_run(path, topics)
    common = set.intersection(*[set(r) for r in runs.values()]) & set(gold)
    common = {q for q in common if not (q.isdigit() and int(q) in SKIP)}
    print(f"Scoring {len(runs)} run(s) on {len(common)} shared questions\n")

    # ---- baselines on the same questions
    rng = random.Random(0)
    rnd = statistics.mean(1.0 / len(topics[q]) for q in common)
    longest = sum(1 for q in common
                  if max(topics[q], key=len) == gold[q]) / max(len(common), 1)
    print(f"{'baseline':<22} {'top-1':>8} {'MRR':>8}")
    print(f"{'random':<22} {rnd:8.1%} {'—':>8}")
    print(f"{'longest option':<22} {longest:8.1%} {'—':>8}   <- must beat this")
    print()

    print(f"{'run':<22} {'top-1':>8} {'MRR':>8} {'vs longest':>11}")
    res = {}
    for label, run in runs.items():
        sub = {q: run[q] for q in common}
        t1, mrr, n = score(sub, gold)
        res[label] = sub
        print(f"{label:<22} {t1:8.1%} {mrr:8.3f} {t1-longest:+11.1%}")

    # ---- paired McNemar between the first two runs
    labels = list(runs)
    if len(labels) >= 2:
        a, b = labels[0], labels[1]
        aw = bw = 0
        for q in common:
            ca = res[a][q] and res[a][q][0] == gold[q]
            cb = res[b][q] and res[b][q][0] == gold[q]
            aw += ca and not cb
            bw += cb and not ca
        tot = aw + bw
        z = (abs(aw - bw) - 1) / (tot ** 0.5) if tot else 0.0
        print(f"\npaired McNemar {a} vs {b}: {a} only {aw}, {b} only {bw}, "
              f"z={z:.2f} {'(significant)' if z > 1.96 else '(NOT significant)'}")


if __name__ == "__main__":
    main()

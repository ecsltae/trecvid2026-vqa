"""
Paired comparison of two or more AG runs on the OFFICIAL metric.

Reports, on the topics all runs share:
  * mean@1 / mean@5 / mean@10   (official aggregation: mean over the top-k answers)
  * ceiling-normalized score    (share of the achievable maximum — 7.5% of dev
                                 topics have an all-stopword reference and score
                                 0.0 for every system on earth, which otherwise
                                 silently dilutes every comparison)
  * paired difference vs the first run, with a standard error, so we can tell a
    real gain from sampling noise on a small dev sample.

Usage:
    python eval/vqa/compare_official.py \
        --runs runs/vqa/ag/v3.csv=v3-diverse /tmp/v4.csv=v4-consensus \
        --qrels data/vqa/qrels_ag.csv
"""
import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from official_sts import OfficialSTS
from eval_ag_official import load_run, load_qrels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="path[=label] ...")
    ap.add_argument("--qrels", default="data/vqa/qrels_ag.csv")
    ap.add_argument("--ceilings", default="data/vqa/reference_ceilings.json")
    ap.add_argument("--limit-topics", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--exclude-dead", action="store_true",
                    help="drop topics whose reference scores 0 against itself")
    # 139 of the 2000 dev videos (7.0%) are permanently gone — verified
    # 2026-08-17, all 139 re-download attempts failed: 80 unavailable, 34 DRM
    # protected, 22 private, 19 terminated accounts, 1 removed. Those topics are
    # answered from the question alone, so they contribute near-random noise to
    # BOTH arms of every paired test. Excluding them costs 7% of n and removes a
    # high-variance stratum — a net win for power, and the reason our A/Bs keep
    # landing at 1.5-2 SE. (Re-downloading is NOT the fix; it was tried.)
    ap.add_argument("--exclude-noframes", action="store_true",
                    help="drop topics whose video has no extracted frames")
    ap.add_argument("--frames", default="data/vqa/frames_pool16",
                    help="frames dir used by --exclude-noframes")
    ap.add_argument("--topics-csv", default="data/vqa/topics/vqa_2025_ag_topics.csv",
                    help="maps Q_ID -> Video_ID for --exclude-noframes")
    args = ap.parse_args()

    runs, labels = {}, []
    for spec in args.runs:
        path, _, label = spec.partition("=")
        label = label or Path(path).stem
        runs[label] = load_run(path)
        labels.append(label)

    qrels = load_qrels(args.qrels)
    ceilings = {}
    if Path(args.ceilings).exists():
        ceilings = {k: v["ceiling"] for k, v in json.loads(Path(args.ceilings).read_text()).items()}

    common = set(qrels)
    for r in runs.values():
        common &= set(r)
    common = sorted(common, key=lambda q: (len(q), q))
    n_dead = sum(1 for q in common if ceilings.get(q, 1.0) <= 0)
    if args.exclude_dead and ceilings:
        common = [q for q in common if ceilings.get(q, 1.0) > 0]

    n_noframes = 0
    if args.exclude_noframes and Path(args.frames).exists():
        vid_of = {}
        if Path(args.topics_csv).exists():
            with open(args.topics_csv, newline="", encoding="utf-8") as f:
                for row in csv.reader(f):
                    if len(row) >= 2:
                        vid_of[row[0].strip().strip('"')] = row[1].strip().strip('"')
        have = {p.name for p in Path(args.frames).iterdir()
                if p.is_dir() and any(p.glob("frame_*.jpg"))}
        before = len(common)
        common = [q for q in common
                  if Path(vid_of.get(q, "")).stem in have or vid_of.get(q, "") in have]
        n_noframes = before - len(common)
        print(f"  excluded {n_noframes} topics with no frames "
              f"(permanent link rot — answered from the question alone)")
    if args.limit_topics:
        common = common[:args.limit_topics]
    print(f"Comparing {len(labels)} runs on {len(common)} shared topics "
          f"({n_dead} unwinnable{' — excluded' if args.exclude_dead else ' — included'})")

    sts = OfficialSTS()
    pairs, keys = [], []
    for label in labels:
        for q in common:
            for rank, ans in enumerate(runs[label][q][:10]):
                for ref in qrels[q]:
                    pairs.append((ans, ref))
                    keys.append((label, q, rank))
    print(f"  {len(pairs)} answer/reference pairs (cache holds {sts.cache_size()})")
    scores = sts.score_many(pairs, workers=args.workers, progress_every=400)

    best = defaultdict(float)
    for (label, q, rank), s in zip(keys, scores):
        if s == s:
            best[(label, q, rank)] = max(best[(label, q, rank)], s)

    per = {}  # label -> {q: [score per rank]}
    for label in labels:
        per[label] = {q: [best.get((label, q, r), 0.0)
                          for r in range(len(runs[label][q][:10]))] for q in common}

    def topk(scores_list, k):
        head = scores_list[:k]
        return sum(head) / len(head) if head else 0.0

    print(f"\n{'run':<22} {'mean@1':>8} {'mean@5':>8} {'mean@10':>8} {'norm@10':>8} {'best@10':>8}")
    table = {}
    for label in labels:
        m1 = statistics.mean(topk(per[label][q], 1) for q in common)
        m5 = statistics.mean(topk(per[label][q], 5) for q in common)
        m10 = statistics.mean(topk(per[label][q], 10) for q in common)
        b10 = statistics.mean(max(per[label][q]) if per[label][q] else 0.0 for q in common)
        if ceilings:
            num = sum(topk(per[label][q], 10) for q in common)
            den = sum(ceilings.get(q, 1.0) for q in common)
            norm = num / den if den else 0.0
        else:
            norm = float("nan")
        table[label] = {"mean@1": m1, "mean@5": m5, "mean@10": m10,
                        "norm@10": norm, "best@10": b10}
        print(f"{label:<22} {m1:8.4f} {m5:8.4f} {m10:8.4f} {norm:8.4f} {b10:8.4f}")

    base = labels[0]
    if len(labels) > 1:
        print(f"\npaired differences vs '{base}' (mean@10):")
        for label in labels[1:]:
            d = [topk(per[label][q], 10) - topk(per[base][q], 10) for q in common]
            mu = statistics.mean(d)
            se = statistics.stdev(d) / len(d) ** 0.5 if len(d) > 1 else 0.0
            sig = "significant" if abs(mu) > 2 * se else "NOT significant"
            print(f"  {label:<22} {mu:+.4f}  SE {se:.4f}  ({mu/se if se else 0:+.1f} SE) {sig}")

    print(f"\n  service: {sts.stats()}")


if __name__ == "__main__":
    main()

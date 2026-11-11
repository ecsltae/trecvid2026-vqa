"""
Sweep candidate-SELECTION policies offline, against the real official metric.

Selection is the last lever with real headroom: on dev, mean@1 is ~0.28 while
best@10 is ~0.41, so the candidate pool routinely contains a better answer than
the one we rank first. Testing a new policy should not cost a GPU hour — every
run now dumps its raw candidate pool to <run>.candidates.jsonl, and every
(answer, reference) STS score is permanently cached, so a sweep is pure CPU
plus a few cache misses.

Policies compared:
  freq        exact-repeat count, then cluster mass       (current default)
  cluster     cluster mass only                           (pre-2026-08-11 behaviour)
  central     most central candidate overall
  longest     longest candidate (tests a length prior)
  shortest    shortest candidate
  medlen      candidate closest to the reference-length median for its
              question type (8 words; 1 word for counting questions)
  oracle      best possible pick — the ceiling for any reranker

Usage:
    python eval/vqa/sweep_selection.py \
        --candidates /tmp/.../v4_dev80_s16.candidates.jsonl
"""
import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from official_sts import OfficialSTS

COUNT_RE = re.compile(r"^\s*how\s+(many|much)\b", re.IGNORECASE)


def target_len(question):
    """Reference-length median: 1 word for counting, 8 words otherwise."""
    return 1 if COUNT_RE.match(question or "") else 8


def policies(rec):
    """-> {policy_name: ordered candidate list}. Order matters only for the
    top-1 metric here; mean@10 is order-invariant by construction."""
    cands = [c for c in rec.get("candidates", []) if c and c.strip()]
    ranked = [r["answer"] for r in rec.get("ranked", []) if r.get("answer")]
    if not cands:
        return {}
    freq = defaultdict(int)
    for c in cands:
        freq[c] += 1
    uniq = list(dict.fromkeys(cands))
    tl = target_len(rec.get("question") or rec.get("Question", ""))
    out = {
        "freq": sorted(uniq, key=lambda c: -freq[c]),
        "cluster": ranked or uniq,
        "central": uniq,
        "longest": sorted(uniq, key=lambda c: -len(c.split())),
        "shortest": sorted(uniq, key=lambda c: len(c.split())),
        "medlen": sorted(uniq, key=lambda c: abs(len(c.split()) - tl)),
    }
    # VLM verification order, if systems/vqa/rerank_verifier.py has been run over
    # this dump. Compared here so it is paired against every other policy on the
    # identical pool — the only way to read an effect smaller than the ~0.015
    # run-to-run SD.
    verifier = [c for c in rec.get("verifier", []) if c and c.strip()]
    if verifier:
        tail = [c for c in out["freq"] if c not in set(verifier)]
        out["verifier"] = verifier + tail
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--qrels", default="data/vqa/qrels_ag.csv")
    ap.add_argument("--ceilings", default="data/vqa/reference_ceilings.json")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    qrels = defaultdict(list)
    with open(args.qrels, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q, a = (row.get("Q_ID") or "").strip(), (row.get("Answer") or "").strip()
            if q and a:
                qrels[q].append(a)
    ceil = {}
    if Path(args.ceilings).exists():
        ceil = {k: v["ceiling"] for k, v in json.loads(Path(args.ceilings).read_text()).items()}

    recs = []
    for line in open(args.candidates, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        q = str(r.get("Q_ID", "")).strip()
        if q in qrels and ceil.get(q, 1.0) > 0:
            recs.append(r)
    print(f"{len(recs)} winnable topics with candidate pools")

    # score every unique (candidate, reference) pair once
    sts = OfficialSTS()
    pairs, keys = [], []
    for r in recs:
        q = str(r["Q_ID"]).strip()
        for c in dict.fromkeys(r.get("candidates", [])):
            if c and c.strip():
                for ref in qrels[q]:
                    pairs.append((c, ref))
                    keys.append((q, c))
    print(f"  {len(pairs)} pairs (cache holds {sts.cache_size()})")
    scores = sts.score_many(pairs, workers=args.workers, progress_every=300)
    best = defaultdict(float)
    for (q, c), s in zip(keys, scores):
        if s == s:
            best[(q, c)] = max(best[(q, c)], s)

    results = defaultdict(list)
    for r in recs:
        q = str(r["Q_ID"]).strip()
        pol = policies(r)
        if not pol:
            continue
        for name, order in pol.items():
            if order:
                results[name].append(best.get((q, order[0]), 0.0))
        allc = [c for c in dict.fromkeys(r.get("candidates", [])) if c.strip()]
        if allc:
            results["oracle"].append(max(best.get((q, c), 0.0) for c in allc))

    base = statistics.mean(results["freq"]) if results.get("freq") else 0.0
    print(f"\n{'policy':<12} {'top-1 STS':>10} {'vs freq':>9}  {'SE':>7}")
    # Preferred display order, then ANY other policy that was computed, then the
    # oracle ceiling last. Iterating a hardcoded tuple silently dropped the
    # `verifier` policy once already — new policies must not need two edits.
    preferred = ["freq", "cluster", "central", "medlen", "shortest", "longest", "verifier"]
    order = [n for n in preferred if n in results]
    order += [n for n in results if n not in preferred and n != "oracle"]
    order += ["oracle"]
    for name in order:
        v = results.get(name)
        if not v:
            continue
        d = [a - b for a, b in zip(v, results["freq"])]
        se = statistics.stdev(d) / len(d) ** 0.5 if len(d) > 1 and any(d) else 0.0
        print(f"{name:<12} {statistics.mean(v):10.4f} {statistics.mean(v)-base:+9.4f} "
              f"{se:7.4f}")
    print(f"\n  service: {sts.stats()}")


if __name__ == "__main__":
    main()

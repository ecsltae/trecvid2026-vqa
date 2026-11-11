"""
Score a TRECVID VQA AG run with the OFFICIAL metric semantics.

Two things make this different from eval/vqa/eval_vqa.py, and both invert
conclusions drawn from that script:

  1. AGGREGATION IS THE MEAN over the submitted answers, reported at top-1 /
     top-5 / top-10 cutoffs — not the best answer. eval_vqa.py reports
     max_sts_5 / max_sts_10, which rewards exactly what the official metric
     punishes: hedged, diverse alternative hypotheses at low ranks.

     Consequence worth internalizing: the arithmetic mean is order-invariant,
     so at the top-10 cutoff RERANKING CANNOT CHANGE THE SCORE. Only the
     content of the ten answers can. Reranking only moves top-1 and top-5.

  2. SIMILARITY IS THE UMBC SERVICE, not sentence-transformers (see
     official_sts.py). MiniLM cosine and UMBC STS disagree violently on
     verbose-but-correct answers.

Because every uncached answer costs a polite HTTP call, the default is to
score a deterministic stratified SAMPLE of topics; pass --all for the full set.

Usage:
    python eval/vqa/eval_ag_official.py \
        --pred runs/vqa/ag/TREC_VQA_qwen3vl30b_AG.csv \
        --qrels data/vqa/qrels_ag.csv \
        [--sample 300] [--all] [--proxy] [--offline] [--json out.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from official_sts import OfficialSTS

CUTOFFS = (1, 5, 10)


# ---------------------------------------------------------------- loading

def load_run(path):
    """Accept our CSV run format or the 2026 submission JSON. Returns
    {q_id: [answers in rank order]}."""
    path = Path(path)
    out = defaultdict(list)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data:
            q = str(item.get("Q_ID", "")).strip()
            answers = item.get("Answers", []) or []
            ranked = sorted(answers, key=lambda a: a.get("Rank", 0))
            out[q] = [str(a.get("Answer", "")) for a in ranked]
        return dict(out)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 4:
                continue
            try:
                rank = int(row[2])
            except ValueError:
                continue
            out[row[0].strip()].append((rank, row[3]))
    return {q: [a for _, a in sorted(v, key=lambda x: x[0])] for q, v in out.items()}


def load_qrels(path):
    """{q_id: [reference answers]} — supports multiple references per topic."""
    refs = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = (row.get("Q_ID") or row.get("q_id") or "").strip()
            ans = (row.get("Answer") or row.get("answer") or "").strip()
            if q and ans:
                refs[q].append(ans)
    return dict(refs)


def load_questions(path):
    if not path or not Path(path).exists():
        return {}
    q = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 3:
                q[row[0].strip()] = row[2].strip().strip('"')
    return q


def question_type(question):
    ql = (question or "").lower().strip()
    if re.match(r"^\s*how\s+many\b", ql) or re.match(r"^\s*how\s+much\b", ql):
        return "count"
    for w in ("what", "why", "how", "who", "where", "when", "which"):
        if ql.startswith(w):
            return w
    return "other"


# ---------------------------------------------------------------- metrics

def meteor_scores(pairs):
    """METEOR for (hypothesis, [references]) pairs; None if nltk missing."""
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        try:
            nltk.data.find("corpora/wordnet.zip")
        except LookupError:
            nltk.download("wordnet", quiet=True)
        return [meteor_score([r.split() for r in refs], hyp.split()) for hyp, refs in pairs]
    except Exception as e:
        print(f"WARNING: METEOR unavailable ({type(e).__name__}) — skipping")
        return None


def proxy_scores(pairs):
    """MiniLM cosine — the WRONG metric, kept only to quantify divergence."""
    from sentence_transformers import SentenceTransformer, util
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    hyps = [h for h, _ in pairs]
    flat_refs, spans = [], []
    for _, refs in pairs:
        spans.append((len(flat_refs), len(refs)))
        flat_refs.extend(refs)
    he = m.encode(hyps, convert_to_tensor=True, batch_size=256, show_progress_bar=False)
    re_ = m.encode(flat_refs, convert_to_tensor=True, batch_size=256, show_progress_bar=False)
    out = []
    for i, (start, n) in enumerate(spans):
        sims = util.cos_sim(he[i:i + 1], re_[start:start + n])
        out.append(float(sims.max()))
    return out


def aggregate(per_topic_scores, cutoffs=CUTOFFS):
    """per_topic_scores: {q_id: [score per answer in rank order]}.
    Official semantics: mean over the first k answers, then mean over topics."""
    res = {}
    for k in cutoffs:
        vals = []
        for scores in per_topic_scores.values():
            head = scores[:k]
            if head:
                vals.append(sum(head) / len(head))
        res[f"mean@{k}"] = sum(vals) / len(vals) if vals else 0.0
    # diagnostics: what a best-of-N metric would have said, and ranking quality
    best, at1, n_ans = [], [], []
    for scores in per_topic_scores.values():
        if not scores:
            continue
        best.append(max(scores[:10]))
        at1.append(1.0 if max(scores[:10]) == scores[0] else 0.0)
        n_ans.append(len(scores))
    res["best@10"] = sum(best) / len(best) if best else 0.0
    res["oracle_gap"] = res["best@10"] - res["mean@10"]
    res["best_is_rank1_rate"] = sum(at1) / len(at1) if at1 else 0.0
    res["mean_answers_per_topic"] = sum(n_ans) / len(n_ans) if n_ans else 0.0
    return res


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--qrels", default="data/vqa/qrels_ag.csv")
    ap.add_argument("--topics", default="data/vqa/topics/vqa_2025_ag_topics.csv",
                    help="used only for per-question-type breakdown")
    ap.add_argument("--sample", type=int, default=300,
                    help="deterministic stratified topic sample (0/--all = every topic)")
    ap.add_argument("--all", action="store_true", help="score every topic (expensive)")
    ap.add_argument("--proxy", action="store_true",
                    help="also compute MiniLM cosine to quantify metric divergence")
    ap.add_argument("--offline", action="store_true",
                    help="cache-only; fail on a miss instead of calling the service")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--json", default=None, help="write full results here")
    args = ap.parse_args()

    run = load_run(args.pred)
    qrels = load_qrels(args.qrels)
    questions = load_questions(args.topics)

    common = sorted(set(run) & set(qrels), key=lambda q: (len(q), q))
    if not common:
        sys.exit("No overlap between run and qrels — check Q_ID formats.")

    if not args.all and args.sample and args.sample < len(common):
        # deterministic, evenly spread over the sorted topic list
        step = len(common) / args.sample
        topics = [common[int(i * step)] for i in range(args.sample)]
    else:
        topics = common

    n_pairs = sum(min(len(run[q]), 10) for q in topics)
    print(f"Scoring {len(topics)}/{len(common)} topics ({n_pairs} answer/reference pairs) "
          f"with the OFFICIAL UMBC STS metric")

    sts = OfficialSTS(offline=args.offline)
    print(f"  cache holds {sts.cache_size()} pairs")

    # Flatten to (answer, reference) — best reference per answer when several exist.
    flat, index = [], []
    for q in topics:
        for rank, ans in enumerate(run[q][:10]):
            for ref in qrels[q]:
                flat.append((ans, ref))
                index.append((q, rank))

    scores = sts.score_many(flat, workers=args.workers, progress_every=250)

    per_topic = defaultdict(dict)
    for (q, rank), s in zip(index, scores):
        if s == s:  # skip NaN
            per_topic[q][rank] = max(per_topic[q].get(rank, 0.0), s)
    per_topic_scores = {q: [d[r] for r in sorted(d)] for q, d in per_topic.items()}

    results = {"official_sts": aggregate(per_topic_scores)}

    # METEOR on the rank-1 answer and averaged over 10, same aggregation rule
    mpairs = [(run[q][r], qrels[q]) for q in topics for r in range(min(len(run[q]), 10))]
    mscores = meteor_scores(mpairs)
    if mscores is not None:
        mt = defaultdict(list)
        for (q, _), s in zip([(q, r) for q in topics for r in range(min(len(run[q]), 10))], mscores):
            mt[q].append(s)
        results["meteor"] = aggregate(dict(mt))

    if args.proxy:
        pscores = proxy_scores(mpairs)
        pt = defaultdict(list)
        for (q, _), s in zip([(q, r) for q in topics for r in range(min(len(run[q]), 10))], pscores):
            pt[q].append(s)
        results["minilm_proxy"] = aggregate(dict(pt))

    # per-question-type breakdown on the official metric
    if questions:
        by_type = defaultdict(dict)
        for q, sc in per_topic_scores.items():
            by_type[question_type(questions.get(q, ""))][q] = sc
        results["by_question_type"] = {
            t: {"n": len(d), **{k: v for k, v in aggregate(d).items() if k.startswith("mean@")}}
            for t, d in sorted(by_type.items())
        }

    print(f"\n=== OFFICIAL AG metric — {Path(args.pred).name} ===")
    for family, vals in results.items():
        if family == "by_question_type":
            continue
        print(f"  [{family}]")
        for k in ("mean@1", "mean@5", "mean@10", "best@10", "oracle_gap",
                  "best_is_rank1_rate", "mean_answers_per_topic"):
            if k in vals:
                print(f"    {k:<22} {vals[k]:.4f}")
    if "by_question_type" in results:
        print("  [by question type — official STS]")
        for t, v in results["by_question_type"].items():
            print(f"    {t:<8} n={v['n']:<5} mean@1={v['mean@1']:.4f} "
                  f"mean@5={v['mean@5']:.4f} mean@10={v['mean@10']:.4f}")
    print(f"\n  service: {sts.stats()}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"run": str(args.pred), "n_topics": len(topics), "results": results},
            indent=2), encoding="utf-8")
        print(f"  wrote {args.json}")


if __name__ == "__main__":
    main()

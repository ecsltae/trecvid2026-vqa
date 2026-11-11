"""
Evaluate TRECVID 2025 VQA submissions.

AG metrics: STS (sentence-transformers), METEOR (nltk), BERTScore, NDCG@10
MC metrics: Top-1 accuracy, MRR

Usage:
    python eval/vqa/eval_vqa.py \\
        --pred runs/vqa/ag/run_AG.csv \\
        --qrels data/vqa/qrels.csv \\
        --task ag|mc \\
        [--verbose]
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from trec_shared.metrics import mrr


def load_run(csv_path):
    """Load submission CSV. Returns dict q_id -> list of (rank, answer/option)."""
    results = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 4:
                continue
            q_id, video_id, rank, answer = row[0], row[1], int(row[2]), row[3]
            results[q_id].append((rank, answer))
    for q_id in results:
        results[q_id].sort(key=lambda x: x[0])
    return results


def load_qrels_csv(qrels_path):
    """Load qrels CSV. Returns dict q_id -> {answer -> grade}."""
    qrels = defaultdict(dict)
    with open(qrels_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_id = row.get("Q_ID") or row.get("q_id")
            answer = row.get("Answer") or row.get("answer", "")
            grade = float(row.get("Grade") or row.get("grade", 1))
            qrels[q_id][answer] = grade
    return qrels


# ---------------------------------------------------------------------------
# AG Metrics
# ---------------------------------------------------------------------------

def compute_meteor(predictions, references):
    """METEOR score (requires nltk)."""
    try:
        from nltk.translate.meteor_score import meteor_score
        import nltk
        scores = []
        for pred, refs in zip(predictions, references):
            score = meteor_score([r.split() for r in refs], pred.split())
            scores.append(score)
        return sum(scores) / len(scores) if scores else 0.0
    except ImportError:
        print("WARNING: nltk not available — skipping METEOR")
        return None


def compute_bertscore(predictions, references):
    """BERTScore F1."""
    try:
        from bert_score import score
        refs_flat = [refs[0] if refs else "" for refs in references]
        _, _, F1 = score(predictions, refs_flat, lang="en", verbose=False)
        return float(F1.mean())
    except ImportError:
        print("WARNING: bert_score not available — skipping BERTScore")
        return None


def evaluate_ag(run, qrels, verbose=False):
    """AG metrics over the full ranked list, not just rank 1.

    Generated answers never string-match the reference, so relevance gains are
    semantic similarities (all-MiniLM-L6-v2 cosine, max over references):
      sts_1       — similarity of the rank-1 answer
      max_sts_5/10 — best answer within the top 5 / top 10
      sim_ndcg_10 — NDCG@10 with similarity gains (rewards putting the best
                    answers first); ideal = own answers re-sorted by similarity
      meteor_1 / bertscore_1 — rank-1 answer vs first reference (as before)
    """
    import math
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    topics = [(q, [a for _, a in v]) for q, v in run.items() if q in qrels and v]

    # One global batch encode: all answers + all references
    all_texts, spans = [], {}
    for q, answers in topics:
        refs = list(qrels[q].keys())
        spans[q] = (len(all_texts), len(answers), len(refs))
        all_texts.extend(answers)
        all_texts.extend(refs)
    emb = model.encode(all_texts, convert_to_tensor=True, show_progress_bar=False,
                       batch_size=256)

    def dcg(gains):
        return sum(g / math.log2(i + 2) for i, g in enumerate(gains))

    sts1, max5, max10, ndcgs = [], [], [], []
    predictions, references = [], []
    for q, answers in topics:
        start, n_ans, n_ref = spans[q]
        a_emb = emb[start:start + n_ans]
        r_emb = emb[start + n_ans:start + n_ans + n_ref]
        # per-answer similarity = max over references
        sims = util.cos_sim(a_emb, r_emb).max(dim=1).values.tolist()
        sts1.append(sims[0])
        max5.append(max(sims[:5]))
        max10.append(max(sims[:10]))
        ideal = dcg(sorted(sims[:10], reverse=True))
        ndcgs.append(dcg(sims[:10]) / ideal if ideal > 0 else 0.0)
        predictions.append(answers[0])
        references.append(list(qrels[q].keys()))

    n = max(len(topics), 1)
    results = {
        "sts_1": sum(sts1) / n,
        "max_sts_5": sum(max5) / n,
        "max_sts_10": sum(max10) / n,
        "sim_ndcg_10": sum(ndcgs) / n,
    }
    meteor = compute_meteor(predictions, references)
    if meteor is not None:
        results["meteor_1"] = meteor
    bertscore = compute_bertscore(predictions, references)
    if bertscore is not None:
        results["bertscore_1"] = bertscore

    return results


# ---------------------------------------------------------------------------
# MC Metrics
# ---------------------------------------------------------------------------

def evaluate_mc(run, qrels, verbose=False):
    top1_correct = 0
    mrr_scores = []
    total = 0

    for q_id, ranked_options in run.items():
        if q_id not in qrels:
            continue
        correct = set(qrels[q_id])  # set of correct option letters
        ranked = [opt for _, opt in ranked_options]

        top1 = ranked[0] if ranked else ""
        top1_correct += int(top1 in correct)
        mrr_scores.append(mrr(ranked, correct))
        total += 1

    return {
        "top1_accuracy": top1_correct / max(total, 1),
        "mrr": sum(mrr_scores) / max(len(mrr_scores), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--task", default="ag", choices=["ag", "mc"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run = load_run(args.pred)
    qrels = load_qrels_csv(args.qrels)
    print(f"Loaded {len(run)} topics from {args.pred}")

    if args.task == "ag":
        results = evaluate_ag(run, qrels, args.verbose)
    else:
        results = evaluate_mc(run, qrels, args.verbose)

    print(f"\n=== VQA {args.task.upper()} Evaluation ===")
    for metric, value in sorted(results.items()):
        print(f"  {metric:<20} {value:.4f}")


if __name__ == "__main__":
    main()

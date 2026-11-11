"""
k-NN few-shot retrieval over real (question -> human reference answer) pairs.

WHY THIS AND NOT `fewshot_retriever.py`. The old retriever reads
`data/vqa/training/train.2025.csv` (500 examples, and its per-video JSONs carry
MC-style correct/incorrect options). This one retrieves from the 2000 official
2025 (question, reference answer) pairs — 4x the data, and crucially the
references are the *actual answer-key style* the metric rewards. Few-shot was
originally judged useless on a number from `eval/vqa/eval_vqa.py`, which the
project's own audit marks obsolete (it aggregates by max and scores with MiniLM
instead of the official UMBC service); re-scored on the official metric the
same idea is +0.0092 STS (5.6 SE).

What the examples teach is *answer form and aspect choice*: how long a real
answer is, which aspect of a scene an annotator picks out, and the unhedged
register. That is exactly the channel our error analysis localised the deficit
to — perception is already at parity with the 2025 winner.

LEAKAGE. The 2026 test set is a different year, so retrieving from the 2025
pairs is clean. When evaluating on the 2025 dev set itself, `exclude_qid` drops
the topic's own pair (and any pair with an identical question), so the answer is
never handed to the model.

Usage:
    from fewshot_pairs import PairRetriever
    r = PairRetriever("data/vqa/topics/vqa_2025_ag_topics.csv", "data/vqa/qrels_ag.csv")
    r.block(question, k=4, exclude_qid="123")   # -> formatted prompt block or ""
"""
from __future__ import annotations

import csv
from pathlib import Path


class PairRetriever:
    def __init__(self, topics_csv, qrels_csv, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._emb = None
        self.pairs = []  # (q_id, question, reference)

        refs = {}
        with open(qrels_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                q = (row.get("Q_ID") or "").strip()
                a = (row.get("Answer") or "").strip()
                if q and a and q not in refs:
                    refs[q] = a
        with open(topics_csv, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 3:
                    continue
                qid = row[0].strip().strip('"')
                question = row[2].strip().strip('"')
                if qid in refs and question:
                    self.pairs.append((qid, question, refs[qid]))

    def __len__(self):
        return len(self.pairs)

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._emb = self._model.encode([p[1] for p in self.pairs],
                                           convert_to_tensor=True,
                                           batch_size=256, show_progress_bar=False)
        return self._model

    def retrieve(self, question, k=4, exclude_qid=None):
        if not self.pairs or not (question or "").strip():
            return []
        from sentence_transformers import util
        model = self._ensure()
        qe = model.encode([question], convert_to_tensor=True, show_progress_bar=False)
        sims = util.cos_sim(qe, self._emb)[0].tolist()
        order = sorted(range(len(self.pairs)), key=lambda i: -sims[i])
        qnorm = " ".join(question.lower().split())
        out = []
        for i in order:
            qid, q, ref = self.pairs[i]
            if exclude_qid is not None and qid == str(exclude_qid).strip():
                continue
            # also drop a duplicate question text — the dev set repeats some
            # questions across videos, which would leak the answer form verbatim
            if " ".join(q.lower().split()) == qnorm:
                continue
            out.append((q, ref))
            if len(out) >= k:
                break
        return out

    def block(self, question, k=4, exclude_qid=None):
        """Formatted few-shot block, or '' when nothing is retrievable."""
        ex = self.retrieve(question, k=k, exclude_qid=exclude_qid)
        if not ex:
            return ""
        lines = ["Examples of how these questions are answered:"]
        for q, ref in ex:
            lines.append(f"Q: {q}")
            lines.append(f"A: {ref}")
        lines.append("")
        return "\n".join(lines)

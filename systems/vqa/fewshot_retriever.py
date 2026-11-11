"""
Few-shot retriever for VQA: embeds training questions and retrieves semantically
similar examples at inference time to use as in-context demonstrations.

Training data: data/vqa/training/train.2025.csv (500 rows) + training/*.json
Each JSON has: video_url, question, correct_answer, incorrect_answers[]

Usage:
    from systems.vqa.fewshot_retriever import FewShotRetriever
    retriever = FewShotRetriever("data/vqa/training")
    examples = retriever.retrieve("What is the person doing?", k=3)
    # → [{"question": ..., "correct_answer": ..., "options": ["A)...", ...]}, ...]
"""
import csv
import json
import os
from pathlib import Path


class FewShotRetriever:
    def __init__(self, training_dir, model_name="all-MiniLM-L6-v2"):
        self.training_dir = Path(training_dir)
        self._examples = []
        self._embeddings = None
        self._model = None
        self._model_name = model_name
        self._load_training_data()

    def _load_training_data(self):
        train_csv = self.training_dir / "train.2025.csv"
        if not train_csv.exists():
            return
        with open(train_csv, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                row = [c.strip().strip('"') for c in row]
                if len(row) < 2:
                    continue
                q_id, video_id = row[0], row[1]
                question = row[2] if len(row) > 2 else ""
                json_path = self.training_dir / f"{video_id}.json"
                if json_path.exists():
                    try:
                        data = json.loads(json_path.read_text(encoding="utf-8"))
                        correct = data.get("correct_answer", "")
                        incorrect = data.get("incorrect_answers", [])
                        # Build MC-style options (1 correct + up to 3 incorrect);
                        # seeded per video so labels are stable across runs
                        options = [correct] + incorrect[:3]
                        import random
                        random.Random(video_id).shuffle(options)
                        correct_idx = options.index(correct)
                        label = chr(ord("A") + correct_idx)
                        self._examples.append({
                            "q_id": q_id,
                            "video_id": video_id,
                            "question": question,
                            "correct_answer": correct,
                            "options": options,
                            "correct_label": label,
                        })
                    except Exception:
                        pass
                else:
                    # JSON missing — add AG-only example
                    self._examples.append({
                        "q_id": q_id,
                        "video_id": video_id,
                        "question": question,
                        "correct_answer": "",
                        "options": [],
                        "correct_label": "",
                    })

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self._model = SentenceTransformer(self._model_name)
            questions = [ex["question"] for ex in self._examples]
            self._embeddings = self._model.encode(questions, convert_to_numpy=True, show_progress_bar=False)

    def retrieve(self, question, k=3):
        """Return top-k most similar training examples (excluding those with empty answers)."""
        if not self._examples:
            return []
        self._ensure_model()
        import numpy as np
        q_emb = self._model.encode([question], convert_to_numpy=True, show_progress_bar=False)[0]
        sims = self._embeddings @ q_emb / (
            (np.linalg.norm(self._embeddings, axis=1) + 1e-9) * (np.linalg.norm(q_emb) + 1e-9)
        )
        # Filter to examples with a correct answer
        candidates = [(i, sims[i]) for i, ex in enumerate(self._examples) if ex["correct_answer"]]
        candidates.sort(key=lambda x: -x[1])
        return [self._examples[i] for i, _ in candidates[:k]]

    def format_ag_fewshot(self, examples):
        """Format examples as few-shot demonstrations for AG prompt."""
        lines = []
        for ex in examples:
            lines.append(f"Example question: {ex['question']}")
            lines.append(f"Correct answer: {ex['correct_answer']}")
            lines.append("")
        return "\n".join(lines).strip()

    def format_mc_fewshot(self, examples):
        """Format examples as few-shot demonstrations for MC prompt."""
        lines = []
        for ex in examples:
            if not ex["options"] or not ex["correct_label"]:
                continue
            lines.append(f"Example question: {ex['question']}")
            for j, opt in enumerate(ex["options"]):
                lines.append(f"  {chr(ord('A') + j)}) {opt}")
            lines.append(f"Correct answer: {ex['correct_label']}) {ex['correct_answer']}")
            lines.append("")
        return "\n".join(lines).strip()

    def __len__(self):
        return len(self._examples)

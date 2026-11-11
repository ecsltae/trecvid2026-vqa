"""
MC by generate-then-match: answer the question freely, then vote for the option
closest to what the model actually said.

WHY THIS IS ALMOST FREE. The 2026 MC task is the SAME 2443 questions on the SAME
videos with the SAME Q_IDs as Task 1 — MC only adds answer options. We already
generated 16-32 independent free-text answers per question for the AG runs, and
dumped every raw candidate pool. So an MC run costs no GPU at all: each AG
sample is an independent look at the video, and we let those samples vote for
whichever option they land nearest.

WHY IT SHOULD BEAT PROMPTING THE OPTIONS DIRECTLY. Showing a model eight options
invites it to pattern-match against plausible-sounding text; asking it what it
sees, N times, and only then comparing to the options keeps the options from
biasing perception. It also turns MC into an ensemble for free — 16-32 votes
instead of one forward pass — and self-consistency is the one thing that
measurably worked in Task 1 (+0.07).

Ranking, not just top-1: the official scorer reports MRR as well as top-1
accuracy, so every option is ranked, by vote count and then by mean similarity.

Output CSV (v3 MC shape): Q_ID,Video_ID,Rank,option_text,Time
The option text is echoed BYTE-IDENTICALLY from the topics file — eval.vqa.mc.py
compares with raw `==`, no normalisation, so a retyped option scores zero.

Usage:
    python systems/vqa/mc_from_ag.py \
        --candidates runs/vqa/ag/2026/consensus_8b_s16.candidates.jsonl \
                     runs/vqa/ag/2026/consensus_8b_asr.candidates.jsonl \
        --topics data/vqa/2026/task2/vqa.2026.testingDataset.Task2.json \
        --out runs/vqa/mc/2026/from_ag.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_topics(path):
    """2026 JSON (Options = [{"option": str}]) or 2025 headerless CSV (A..D)."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        out = []
        for t in json.loads(p.read_text(encoding="utf-8")):
            opts = [o.get("option", "") if isinstance(o, dict) else str(o)
                    for o in (t.get("Options") or t.get("Answers") or [])]
            out.append({"Q_ID": str(t["Q_ID"]).strip(),
                        "Video_ID": str(t["Video_ID"]),
                        "Question": str(t.get("Question", "")),
                        "options": opts})
        return out
    rows = []
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) >= 7:
                rows.append({"Q_ID": r[0].strip(),
                             "Video_ID": r[1].strip().strip('"'),
                             "Question": r[2].strip().strip('"'),
                             "options": [c.strip().strip('"') for c in r[3:7]]})
    return rows


def load_pools(paths):
    """{q_id: [candidate answers]} merged across AG candidate dumps."""
    pool = defaultdict(list)
    for path in paths:
        if not Path(path).exists():
            print(f"  (skipping missing {path})")
            continue
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            q = str(r.get("Q_ID", "")).strip()
            for c in r.get("candidates", []):
                if c and str(c).strip():
                    pool[q].append(str(c).strip())
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--topics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    topics = load_topics(args.topics)
    if args.limit:
        topics = topics[:args.limit]
    pools = load_pools(args.candidates)
    print(f"{len(topics)} MC questions | AG pools for {len(pools)} of them")

    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer(args.model)

    # one batched encode of every option and every candidate
    texts, span = [], {}
    for t in topics:
        q = t["Q_ID"]
        cands = pools.get(q, [])
        span[q] = (len(texts), len(t["options"]), len(cands))
        texts.extend(t["options"])
        texts.extend(cands)
    emb = model.encode(texts, convert_to_tensor=True, batch_size=256,
                       show_progress_bar=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_novote = 0
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for t in topics:
            q = t["Q_ID"]
            start, n_opt, n_cand = span[q]
            opts = t["options"]
            if n_opt == 0:
                continue
            if n_cand == 0:                      # no AG pool: keep given order
                n_novote += 1
                order = list(range(n_opt))
            else:
                oe = emb[start:start + n_opt]
                ce = emb[start + n_opt:start + n_opt + n_cand]
                sim = util.cos_sim(ce, oe)       # candidates x options
                votes = [0] * n_opt
                for row in sim:                  # each sample votes for its nearest option
                    votes[int(row.argmax())] += 1
                meansim = sim.mean(dim=0).tolist()
                order = sorted(range(n_opt),
                               key=lambda i: (-votes[i], -meansim[i]))
            for rank, i in enumerate(order, 1):
                # byte-identical echo of the option text — eval.vqa.mc.py uses ==
                w.writerow([q, t["Video_ID"], rank, opts[i], "0.00"])

    print(f"Wrote {out_path} ({len(topics)} questions, {n_novote} with no AG pool)")


if __name__ == "__main__":
    main()

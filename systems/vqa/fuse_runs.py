"""
Cross-condition consensus fusion — builds a run with NO GPU and NO UMBC service.

WHY THIS IS A REASONABLE RUN 5. Two of our test-set runs dumped their raw
candidate pools (16 independent samples each), under *different* conditions:
`consensus_8b_s16` (frames only) and `consensus_8b_asr` (frames + routed speech
transcripts). Pooling them gives 32 samples per question instead of 16, and
self-consistency accuracy is limited by how well the modal answer is estimated.
Two conditions also decorrelate errors more than two draws from one condition.

WHY IT DOES NOT SIMPLY AVERAGE AWAY THE AUDIO WIN. On dev, injecting a
transcript was worth +0.1795 on the ~2.3% of questions that ask what was SAID,
and is deliberately withheld elsewhere. Blindly pooling would let 16
transcript-free samples outvote 16 transcript-informed ones on exactly the
questions where the transcript is the evidence. So topics the ASR router fired
on are passed through from the ASR run untouched; everything else is fused.

HONESTY. The UMBC service has been down since ~2026-08-12, so this run is
UNMEASURED — the fusion is motivated by the measured sample-count and
decorrelation arguments, not validated end-to-end. It is submitted as a fifth
run because TREC scores runs independently, so it cannot cost the other four
anything, and it is described as unmeasured in the notebook paper.

Usage:
    python systems/vqa/fuse_runs.py \
        --pools runs/vqa/ag/2026/consensus_8b_s16.candidates.jsonl \
                runs/vqa/ag/2026/consensus_8b_asr.candidates.jsonl \
        --asr-pool runs/vqa/ag/2026/consensus_8b_asr.candidates.jsonl \
        --out runs/vqa/ag/2026/fusion.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_vqa_v4 import Consensus, build_slate, audio_route, sanitize


def load_pool(path):
    """-> {q_id: record}"""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[str(r.get("Q_ID", "")).strip()] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", required=True,
                    help="candidate dumps to fuse")
    ap.add_argument("--asr-pool", default=None,
                    help="dump whose routed-transcript topics pass through unfused")
    ap.add_argument("--topics", default="data/vqa/2026/vqa.2026.testingDataset.Task1.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-answers", type=int, default=10)
    ap.add_argument("--cluster-threshold", type=float, default=0.72)
    args = ap.parse_args()

    topics = json.loads(Path(args.topics).read_text(encoding="utf-8"))
    pools = [load_pool(p) for p in args.pools]
    asr = load_pool(args.asr_pool) if args.asr_pool else {}
    print(f"Fusing {len(pools)} pools over {len(topics)} topics"
          + (f"; ASR passthrough from {Path(args.asr_pool).name}" if asr else ""))

    consensus = Consensus(threshold=args.cluster_threshold)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_fused = n_passthrough = n_thin = 0
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        for t in topics:
            q = str(t["Q_ID"]).strip()
            vid = str(t["Video_ID"])
            question = str(t.get("Question", ""))

            # Questions about what was SAID: keep the transcript-informed run's
            # own answer rather than letting transcript-free samples outvote it.
            if asr and audio_route(question) == "verbal" and q in asr:
                ranked = [(r["answer"], r.get("support", 1))
                          for r in asr[q].get("ranked", []) if r.get("answer")]
                n_passthrough += 1
            else:
                cands = []
                for p in pools:
                    rec = p.get(q)
                    if rec:
                        cands.extend(c for c in rec.get("candidates", []) if c and c.strip())
                if not cands:
                    n_thin += 1
                    # fall back to whatever any pool ranked first
                    for p in pools:
                        rec = p.get(q)
                        if rec and rec.get("ranked"):
                            cands = [rec["ranked"][0]["answer"]]
                            break
                ranked = consensus.rank([sanitize(c) for c in cands if c])
                n_fused += 1

            slate = build_slate(list(ranked), "consensus", args.num_answers)
            for rank, ans in enumerate(slate, 1):
                writer.writerow([q, vid, rank, ans, "0.00"])
            fh.flush()

    print(f"Wrote {out_path}: {n_fused} fused, {n_passthrough} ASR passthrough, "
          f"{n_thin} with no pooled candidates")


if __name__ == "__main__":
    main()

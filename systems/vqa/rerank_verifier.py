"""
Verifier reranking of an existing v4 candidate pool.

WHY. v4 ranks candidates by how often the model re-derived them (frequency /
cluster mass). That is a *generation* prior: it measures what the model likes to
say, not what the video shows. On dev, rank-1 is the best candidate only ~35% of
the time while best@10 ~ 0.41 against mean@1 ~ 0.28 — the pool usually already
contains a better answer than we put first. This module asks the model a
different question ("which of these does the video actually support?"), which is
a *verification* judgement over a fixed set and is therefore an easier task than
open-ended generation.

WHAT IT DOES AND DOES NOT MOVE. The official AG score is five metrics in two
families:
  - METEOR / BERTScore / STS are np.mean over the submitted answers, so at the
    top-10 cutoff they are ORDER-INVARIANT — reranking cannot change them.
  - NDCG_METEOR and NDCG_BERTScore encode the submitted order, and the reported
    mean@1 / mean@5 cutoffs also depend on order.
So reranking is worth exactly: mean@1, mean@5, and both NDCG metrics. It cannot
rescue mean@10 — only better candidate CONTENT can. Do not expect otherwise.

It never rewrites answer text: the model returns indices only. Candidate strings
are already sanitized for the scorer's perl/wget path by run_vqa_v4.sanitize(),
and re-emitting model prose here would reintroduce the verbose register the
metric punishes.

Output: the input JSONL plus a "verifier" field (ordered answer list), so it can
be scored paired against the other policies with cached STS and zero GPU:
    python eval/vqa/sweep_selection.py --candidates <out>

Usage:
    python systems/vqa/rerank_verifier.py \\
        --candidates runs/vqa/ag/dev_v4_s16_n300.candidates.jsonl \\
        --frames data/vqa/frames_pool16 \\
        --out runs/vqa/ag/dev_v4_s16_n300.verified.jsonl [--resume]
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trec_shared.ollama_client import OllamaClient
from run_vqa_v4 import frame_pool

_SYSTEM = (
    "You judge which candidate answers are supported by what is visible in a "
    "set of video frames. You never write new answers; you only choose among "
    "the ones given."
)

_TMPL = """\
Question: {question}

Candidate answers:
{numbered}

Order these candidates from the one the frames support BEST to the one they
support WORST. Judge only on what is visible; ignore how fluent an answer sounds.

Reply with the candidate numbers only, separated by commas, best first.
Example: 3,1,4,2"""


def pick_frames(pool, k=8):
    """Evenly spaced across the clip — the pool is ordered by timestamp."""
    if len(pool) <= k:
        return pool
    step = (len(pool) - 1) / (k - 1)
    return [pool[round(i * step)] for i in range(k)]


def parse_order(raw, n):
    """Model reply -> 0-based index order, deduped, missing indices appended."""
    order, seen = [], set()
    for tok in re.findall(r"\d+", raw or ""):
        i = int(tok) - 1
        if 0 <= i < n and i not in seen:
            seen.add(i)
            order.append(i)
    for i in range(n):
        if i not in seen:
            order.append(i)
    return order


def candidate_shortlist(rec, topk):
    """Distinct candidates, most-supported first (v4's own ordering, then freq)."""
    freq = defaultdict(int)
    for c in rec.get("candidates", []):
        if c and c.strip():
            freq[c.strip()] += 1
    ordered = [r["answer"].strip() for r in rec.get("ranked", [])
               if r.get("answer") and r["answer"].strip()]
    for c in sorted(freq, key=lambda x: -freq[x]):
        if c not in ordered:
            ordered.append(c)
    return [c for c in dict.fromkeys(ordered)][:topk]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3-vl:8b-instruct")
    ap.add_argument("--topk", type=int, default=10, help="candidates shown to the verifier")
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--num-predict", type=int, default=48)
    ap.add_argument("--call-timeout", type=int, default=180)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    recs = []
    for line in open(args.candidates, encoding="utf-8"):
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    if args.limit:
        recs = recs[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            line = line.strip()
            if line:
                done.add(str(json.loads(line).get("Q_ID", "")).strip())
        print(f"Resuming: {len(done)} already verified")

    client = OllamaClient(model=args.model, timeout=args.call_timeout, temperature=0.0)
    todo = [r for r in recs if str(r.get("Q_ID", "")).strip() not in done]
    print(f"Verifying {len(todo)}/{len(recs)} topics | model={args.model} | topk={args.topk}")

    n_novideo = n_singleton = n_err = 0
    t_run = time.time()
    with open(out_path, "a" if done else "w", encoding="utf-8") as fout:
        for i, rec in enumerate(todo, 1):
            shortlist = candidate_shortlist(rec, args.topk)
            pool = frame_pool(args.frames, rec.get("Video_ID", ""))
            frames = pick_frames(pool, args.max_frames)

            if len(shortlist) < 2 or not frames:
                # nothing to reorder, or nothing to look at — keep v4's order
                if not frames:
                    n_novideo += 1
                else:
                    n_singleton += 1
                rec["verifier"] = shortlist
            else:
                numbered = "\n".join(f"{j+1}. {c}" for j, c in enumerate(shortlist))
                prompt = _TMPL.format(question=rec.get("Question", ""), numbered=numbered)
                try:
                    raw = client.generate_multimodal_with_retry(
                        prompt=prompt, images=frames, system=_SYSTEM,
                        num_predict=args.num_predict, think=False)
                    order = parse_order(raw, len(shortlist))
                    rec["verifier"] = [shortlist[j] for j in order]
                except Exception as e:
                    n_err += 1
                    print(f"  !! {rec.get('Q_ID')}: {type(e).__name__}: {str(e)[:80]}", flush=True)
                    rec["verifier"] = shortlist

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if i % 25 == 0 or i == len(todo):
                rate = (time.time() - t_run) / i
                print(f"  [{i}/{len(todo)}] {rate:.1f}s/topic — ETA {rate*(len(todo)-i)/3600:.1f}h "
                      f"(novideo={n_novideo}, singleton={n_singleton}, err={n_err})", flush=True)

    print(f"Wrote {out_path} ({len(todo)} verified, {n_novideo} no-video, "
          f"{n_singleton} single-candidate, {n_err} errored)")
    print(f"Score it paired against the other policies:\n"
          f"  python eval/vqa/sweep_selection.py --candidates {out_path}")


if __name__ == "__main__":
    main()

"""
Build a TRECVID 2026 Task 2 (Multiple Choice) submission.

Written from scratch rather than extending make_submission.py's MC branch,
which is unusable here for two reasons: it is hardcoded to exactly four lettered
options (2026 has 4, 8, 9, 10 and 11), and it passes option text through
.strip().strip('"') — under the official scorer that is not a cosmetic issue.

    eval/vqa_official_nist/eval.vqa.mc.py:59
        top1_score = 1 if ranked_responses[0] == correct_answer else 0

Raw Python string equality. No normalisation, no strip, no casefold. An option
we re-typed, re-quoted or whitespace-trimmed simply scores zero, silently, and
the validator would not catch it because the run is structurally fine.

So the one hard rule here: **option strings are echoed byte-for-byte from the
released JSON.** The ranking logic only ever moves indices around; it never
touches text.

Usage:
    python eval/vqa/make_mc_submission.py \
        --run runs/vqa/mc/2026/<run>.csv \
        --topics data/vqa/2026/task2/vqa.2026.testingDataset.Task2.json \
        --out submissions/vqa2026/BiTeM_mc_<tag>.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_topics(path):
    """-> {q_id: {"Video_ID": str, "Question": str, "options": [str verbatim]}}"""
    out = {}
    for t in json.loads(Path(path).read_text(encoding="utf-8")):
        opts = [o["option"] if isinstance(o, dict) else str(o)
                for o in (t.get("Options") or t.get("Answers") or [])]
        out[str(t["Q_ID"]).strip()] = {"Video_ID": str(t["Video_ID"]),
                                       "Question": str(t.get("Question", "")),
                                       "options": opts}
    return out


def load_run(path):
    """Run CSV: Q_ID,Video_ID,Rank,option_text,Time -> {q_id: [text in rank order]}"""
    rows = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) >= 4:
                try:
                    rows[r[0].strip()].append((int(r[2]), r[3]))
                except ValueError:
                    pass
    return {q: [a for _, a in sorted(v, key=lambda x: x[0])] for q, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--topics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    topics = load_topics(args.topics)
    run = load_run(args.run)
    print(f"Topics: {len(topics)} | run covers: {len(run)}")

    payload, problems = [], []
    n_reordered = n_missing = 0
    for q, t in topics.items():
        opts = t["options"]
        # Map the run's ranking back onto option INDICES. Matching is done on a
        # normalised key purely to be forgiving about what the run file holds;
        # what gets EMITTED is always opts[i], the verbatim released string.
        key = {}
        for i, o in enumerate(opts):
            key.setdefault(" ".join(o.split()).casefold(), i)
        order, seen = [], set()
        for txt in run.get(q, []):
            i = key.get(" ".join(str(txt).split()).casefold())
            if i is not None and i not in seen:
                seen.add(i)
                order.append(i)
        if not order:
            n_missing += 1
            problems.append(f"Q_ID {q}: no option matched the run — emitted released order")
        for i in range(len(opts)):          # every option must be ranked
            if i not in seen:
                order.append(i)
                n_reordered += 1

        payload.append({
            "Q_ID": int(q) if q.isdigit() else q,
            "Video_ID": t["Video_ID"],
            "Answers": [{"Rank": r, "Answer": opts[i]}          # verbatim, no Time for MC
                        for r, i in enumerate(order, 1)],
        })

    # ---- self-check, mirroring what the scorer and a validator would demand
    for obj in payload:
        q = obj["Q_ID"]
        got = [a["Answer"] for a in obj["Answers"]]
        want = topics[str(q)]["options"]
        if sorted(got) != sorted(want):
            problems.append(f"Q_ID {q}: emitted options are not the released set")
        if [a["Rank"] for a in obj["Answers"]] != list(range(1, len(want) + 1)):
            problems.append(f"Q_ID {q}: ranks not 1..{len(want)}")
        for a, o in zip(sorted(got), sorted(want)):
            if a != o:                       # byte-level, not normalised
                problems.append(f"Q_ID {q}: option text altered")
                break

    print(f"  entries={len(payload)}  options filled in from released order={n_reordered}"
          f"  questions with no run match={n_missing}")
    if problems:
        print(f"\n!! {len(problems)} problem(s):")
        for p in problems[:15]:
            print(f"   - {p}")
        if len(problems) > 15:
            print(f"   ... and {len(problems)-15} more")
        if not args.force:
            sys.exit("\nNOT written. Fix the run before submitting.")
    else:
        print("  self-check PASSED — every option byte-identical to the released JSON")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {out} ({len(payload)} questions, {out.stat().st_size/1e6:.1f} MB)")
    print("  NB: no official MC validator has been released yet — re-poll "
          "github.com/gawad-nist/VQA before the Sept 18 deadline.")


if __name__ == "__main__":
    main()

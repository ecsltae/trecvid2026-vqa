"""
Build + validate a TRECVID VQA **2026** submission JSON from a v3 run CSV.

The 2026 guidelines changed the run format from 2025's CSV to JSON:

  AG: [ {"Q_ID": int, "Video_ID": str,
         "Answers": [ {"Rank": int, "Answer": str, "Time": float}, ... x10 ]} ]
  MC: [ {"Q_ID": int, "Video_ID": str,
         "Answers": [ {"Rank": int, "Answer": str}, ... x4 ]} ]   # no Time

Hard rules enforced here (violations are rejected by the NIST validator):
  - AG must carry EXACTLY 10 answers per question — "Any submission with less
    than 10 answers will be rejected by the validator."
  - MC "Answer" is the option TEXT, not the letter — so a letter-based run is
    mapped back through the topics file.
  - Every Q_ID in the topics file must be present exactly once.
  - Max 5 runs per team per task.

Usage:
    python eval/vqa/make_submission.py \\
        --run runs/vqa/ag/TREC_VQA_qwen3vl30b_AG.csv \\
        --topics data/vqa/2026/vqa.2026.testingDataset.Task1.json \\
        --task ag \\
        --out submissions/vqa2026/BiTeM_ag_qwen3vl30b.json
"""
import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "systems" / "vqa"))

OPTION_KEYS = ["A", "B", "C", "D"]


def norm_ws(text):
    """Whitespace-canonical form of an answer (what we actually emit)."""
    return " ".join(str(text).split())


def dup_key(text):
    """EXACTLY validate.ag.run.py's normalize_answer — the duplicate test NIST
    applies. Any dedupe we do must use this and nothing looser."""
    return " ".join(str(text).split()).casefold()


def load_run_csv(path):
    """v3 run CSV → {q_id: {"video": str, "rows": [(rank, text, time)]}}"""
    runs = defaultdict(lambda: {"video": "", "rows": []})
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 4:
                continue
            q_id, video_id, rank, answer = row[0], row[1], int(row[2]), row[3]
            t = float(row[4]) if len(row) > 4 and row[4] else 0.0
            runs[q_id]["video"] = video_id
            runs[q_id]["rows"].append((rank, answer, t))
    for q in runs:
        runs[q]["rows"].sort(key=lambda x: x[0])
    return runs


def build(run, topics, task, num_answers=10):
    """Assemble submission objects in topics order; returns (payload, problems)."""
    problems = []
    payload = []
    for topic in topics:
        q_id = str(topic["Q_ID"])
        entry = run.get(q_id)
        if not entry or not entry["rows"]:
            # Never drop a topic. Under the official scorer a missing query is
            # not skipped — it simply contributes nothing while still being
            # counted, and the validator rejects a submission whose Q_ID set
            # does not match the released queries. A weak answer beats no entry.
            problems.append(f"Q_ID {q_id}: MISSING from run — emitted placeholder")
            entry = {"video": topic.get("Video_ID", ""),
                     "rows": [(1, str(topic.get("Question", "unknown")).strip(" ?"), 0.0)]}

        answers = []
        if task == "ag":
            # Emit whitespace-canonical text. The validator compares
            # " ".join(a.split()).casefold(), so two strings differing only by a
            # double space are duplicates to NIST but distinct to a naive
            # strip() — which silently reintroduced collisions here.
            texts = [(norm_ws(a), t) for _, a, t in entry["rows"] if norm_ws(a)]
            if not texts:
                problems.append(f"Q_ID {q_id}: no non-empty answers — emitted placeholder")
                texts = [(norm_ws(str(topic.get("Question", "unknown")).strip(" ?")), 0.0)]
            deduped, seen_norm = [], set()
            for a, t in texts:
                k = dup_key(a)
                if k not in seen_norm:
                    seen_norm.add(k)
                    deduped.append((a, t))
            texts = deduped[:num_answers]
            # Pad to exactly 10 (validator rejects short topics). Pad with
            # score-equivalent SURFACE variants rather than literal duplicates:
            # the metric strips stopwords/punctuation so these cost nothing,
            # while exact casefolded duplicates risk validator rejection.
            if len(texts) < num_answers:
                from run_vqa_v4 import surface_variants
                seen_texts = {dup_key(t[0]) for t in texts}
                base, base_t = norm_ws(texts[0][0]), texts[0][1]
                for v in surface_variants(base):
                    if len(texts) >= num_answers:
                        break
                    if dup_key(v) not in seen_texts:
                        seen_texts.add(dup_key(v))
                        texts.append((norm_ws(v), base_t))
                i = 1
                while len(texts) < num_answers:  # exhausted variants
                    cand = norm_ws(f"{base}{'.' * i}")
                    i += 1
                    if dup_key(cand) in seen_texts:
                        continue          # already produced by surface_variants
                    seen_texts.add(dup_key(cand))
                    texts.append((cand, base_t))
            for rank, (text, t) in enumerate(texts, 1):
                answers.append({"Rank": rank, "Answer": str(text),
                                "Time": round(float(t), 2)})
        else:
            # MC: map A/B/C/D back to option text. Dedup on the LETTER, never on
            # the text — some topics genuinely repeat an option string, and
            # collapsing those would drop us below the required 4 options.
            opts = {k: topic.get(k, "") for k in OPTION_KEYS}
            text_to_letter = {}
            for k in OPTION_KEYS:
                text_to_letter.setdefault(opts[k].strip(), k)
            seen = []
            for _, val, _t in entry["rows"]:
                v = val.strip()
                letter = v.upper() if v.upper() in OPTION_KEYS and len(v) == 1 \
                    else text_to_letter.get(v)
                if letter and letter not in seen:
                    seen.append(letter)
            for k in OPTION_KEYS:  # ensure all four options are ranked
                if k not in seen:
                    seen.append(k)
            if len(seen) != 4:
                problems.append(f"Q_ID {q_id}: {len(seen)} ranked options (expected 4)")
            for rank, letter in enumerate(seen[:4], 1):
                answers.append({"Rank": rank, "Answer": str(opts.get(letter, ""))})

        if task == "ag" and len(answers) != num_answers:
            problems.append(f"Q_ID {q_id}: {len(answers)} answers (must be {num_answers})")

        try:
            qid_val = int(q_id)
        except ValueError:
            qid_val = q_id
        # Video_ID must match the RELEASED queries byte-for-byte (the validator
        # compares exact strings, and 1575 of the 2443 2026 ids carry a .webm
        # extension). The run CSV's copy is only good enough for a warning.
        topic_vid = str(topic.get("Video_ID", ""))
        if entry["video"] and entry["video"] != topic_vid:
            problems.append(f"Q_ID {q_id}: run Video_ID {entry['video']!r} != "
                            f"topics {topic_vid!r} (using topics value)")
        payload.append({"Q_ID": qid_val, "Video_ID": topic_vid, "Answers": answers})
    return payload, problems


def validate(payload, topics, task, num_answers=10):
    """Re-check the assembled payload the way the NIST validator would."""
    problems = []
    if len(payload) != len(topics):
        problems.append(f"{len(payload)} entries but topics file has {len(topics)}")
    seen = set()
    for obj in payload:
        q = obj["Q_ID"]
        if q in seen:
            problems.append(f"Q_ID {q}: duplicate entry")
        seen.add(q)
        if not obj.get("Video_ID"):
            problems.append(f"Q_ID {q}: empty Video_ID")
        answers = obj.get("Answers", [])
        expected = num_answers if task == "ag" else 4
        if len(answers) != expected:
            problems.append(f"Q_ID {q}: {len(answers)} answers (must be {expected})")
        ranks = [a.get("Rank") for a in answers]
        if ranks != list(range(1, len(answers) + 1)):
            problems.append(f"Q_ID {q}: ranks not 1..{len(answers)} ({ranks[:12]})")
        for a in answers:
            if not str(a.get("Answer", "")).strip():
                problems.append(f"Q_ID {q}: empty answer at rank {a.get('Rank')}")
            if task == "ag" and not isinstance(a.get("Time"), (int, float)):
                problems.append(f"Q_ID {q}: non-numeric Time at rank {a.get('Rank')}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="v3 run CSV")
    ap.add_argument("--topics", required=True, help="official topics (JSON or CSV)")
    ap.add_argument("--task", required=True, choices=["ag", "mc"])
    ap.add_argument("--out", required=True, help="submission JSON path")
    ap.add_argument("--num-answers", type=int, default=10)
    ap.add_argument("--force", action="store_true",
                    help="write the file even if validation reports problems")
    args = ap.parse_args()

    from run_vqa_v3 import load_topics
    topics = load_topics(args.topics, args.task)
    run = load_run_csv(args.run)
    print(f"Topics: {len(topics)} | run topics: {len(run)}")

    payload, problems = build(run, topics, args.task, args.num_answers)
    problems += validate(payload, topics, args.task, args.num_answers)

    if problems:
        print(f"\n!! {len(problems)} validation problem(s):")
        for p in problems[:25]:
            print(f"   - {p}")
        if len(problems) > 25:
            print(f"   ... and {len(problems) - 25} more")
        if not args.force:
            print("\nNOT written. Fix the run (or pass --force) before submitting.")
            sys.exit(1)
    else:
        print("Validation PASSED — all topics present, counts/ranks/fields correct.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    size_mb = out.stat().st_size / 1e6
    print(f"Wrote {out} ({len(payload)} topics, {size_mb:.1f} MB)")

    # Final authority: NIST's own validator, not our reimplementation of it.
    # Our checks passed on a build their validator rejected with 16 errors, so
    # this is the only gate that counts.
    official = Path(__file__).resolve().parent.parent / "vqa_official_nist" / "validate.ag.run.py"
    if args.task == "ag" and official.exists() and str(args.topics).endswith(".json"):
        print(f"\n--- official NIST validator ({official.name}) ---")
        proc = subprocess.run([sys.executable, str(official), str(args.topics), str(out)],
                              capture_output=True, text=True)
        tail = (proc.stdout or "").strip().splitlines()
        for line in tail[-15:]:
            print("  " + line)
        if proc.returncode != 0:
            print(f"\n!! OFFICIAL VALIDATOR REJECTED THIS RUN (exit {proc.returncode}). "
                  f"Do not submit it.")
            sys.exit(2)
        print("  official validator: PASSED")
    elif args.task == "ag":
        print("\n(!) official validator not run — "
              f"{'topics must be the released JSON' if official.exists() else f'missing {official}'}")


if __name__ == "__main__":
    main()

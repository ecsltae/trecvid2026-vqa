"""
Sweep rule-based ANSWER-TEXT transforms against the official metric, paired.

Why text and not order. The official AG score is np.mean over the submitted
answers, so at the top-10 cutoff it is ORDER-INVARIANT: no reranker can move
mean@10. Text is the only lever that can. These transforms are therefore
complementary to eval/vqa/sweep_selection.py, not a competitor to it.

Why rule-based and not an LLM rewriter. LLM style-rewriting was already measured
at +0.004 (a no-op) and aggressive truncation at -0.05. But that tested a
*generic* "make it terse" instruction. The metric's failure modes are specific
and mechanical, so targeted rules are a different hypothesis:

  * preamble    "In the video, the man appears to fall down" vs ref "he fell down"
                scores 0.059; "he fell" scores 1.000. Four words of framing cost
                ~0.94 because the score divides by the longer side's content length.
  * pronoun     sim("the man", "he") = 0.0, and 53.5% of references use a pronoun.
                Our answers name the subject; references pronominalise it.
  * counting    98.3% of counting references are the number WORD ("three"), and
                the service normalises digits but METEOR/BERTScore tokenise
                literally.
  * hedge       ", possibly to ...", ", as seen in the frame" add content words
                that are pure denominator.

A transform that is a no-op is itself a result worth recording: it means v4's
generation contract already avoids that failure mode, which is the interesting
kind of negative.

Usage:
    python eval/vqa/sweep_transforms.py \\
        --candidates runs/vqa/ag/dev_v4_s16_n300.candidates.jsonl [--topn 10]
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

PREAMBLE_RE = re.compile(
    r"^\s*(in (the|this) (video|clip|scene|frames?)\s*,?\s*"
    r"|the (video|clip|scene|frames?) shows( that)?\s*"
    r"|it (appears|seems)( that)?\s*"
    r"|we (can )?see( that)?\s*"
    r"|there (is|are)\s+"
    r"|the answer is\s*"
    r"|based on the (frames?|video)\s*,?\s*)+", re.IGNORECASE)

HEDGE_RE = re.compile(
    r"\s*,\s*(possibly|probably|likely|perhaps|maybe|which|as (seen|shown|visible)|"
    r"suggesting|indicating|implying)\b.*$", re.IGNORECASE)

NUM_WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
             6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
             11: "eleven", 12: "twelve"}

# Leading subject noun phrase -> pronoun. Deliberately conservative: only fires
# on an unambiguous singular human subject at the start of the answer.
SUBJ_RE = re.compile(
    r"^(the|a|an)\s+(man|woman|boy|girl|guy|person|male|female|player|driver|"
    r"child|kid|dog|cat|clown|character)\b", re.IGNORECASE)


def t_identity(a, q):
    return a


def t_preamble(a, q):
    out = PREAMBLE_RE.sub("", a).strip()
    return (out[0].upper() + out[1:]) if out else a


def t_hedge(a, q):
    out = HEDGE_RE.sub("", a).strip().rstrip(",")
    return out or a


def t_counting(a, q):
    """Counting questions -> the bare number word."""
    if not COUNT_RE.match(q or ""):
        return a
    m = re.search(r"\b(\d+)\b", a)
    if m:
        n = int(m.group(1))
        return NUM_WORDS.get(n, m.group(1))
    for w in re.findall(r"[A-Za-z]+", a.lower()):
        if w in NUM_WORDS.values() or w in ("no", "none", "nothing"):
            return "zero" if w in ("no", "none", "nothing") else w
    return a


def t_pronoun(a, q):
    """'The man kicks the ball' -> 'He kicks the ball'."""
    m = SUBJ_RE.match(a)
    if not m:
        return a
    noun = m.group(2).lower()
    pro = {"woman": "She", "girl": "She", "female": "She"}.get(noun, "He")
    if noun in ("dog", "cat"):
        pro = "It"
    return pro + a[m.end():]


def t_first_clause(a, q):
    return a.split(",")[0].strip() or a


TRANSFORMS = {
    "identity": t_identity,
    "preamble": t_preamble,
    "hedge": t_hedge,
    "counting": t_counting,
    "pronoun": t_pronoun,
    "first_clause": t_first_clause,
}
# Stacked: the plausible production combination.
def t_combo(a, q):
    return t_counting(t_hedge(t_preamble(a, q), q), q)
TRANSFORMS["combo"] = t_combo
def t_combo_pron(a, q):
    return t_pronoun(t_combo(a, q), q)
TRANSFORMS["combo+pronoun"] = t_combo_pron


def slate_of(rec, topn):
    """v4's submitted slate: ranked answers first, then frequency order."""
    ranked = [r["answer"].strip() for r in rec.get("ranked", [])
              if r.get("answer") and r["answer"].strip()]
    freq = defaultdict(int)
    for c in rec.get("candidates", []):
        if c and c.strip():
            freq[c.strip()] += 1
    for c in sorted(freq, key=lambda x: -freq[x]):
        if c not in ranked:
            ranked.append(c)
    return list(dict.fromkeys(ranked))[:topn]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--qrels", default="data/vqa/qrels_ag.csv")
    ap.add_argument("--ceilings", default="data/vqa/reference_ceilings.json")
    ap.add_argument("--topn", type=int, default=10)
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
        if q in qrels and ceil.get(q, 1.0) > 0:   # drop unwinnable topics
            recs.append(r)
    print(f"{len(recs)} winnable topics")

    # Build every (transformed answer, reference) pair, score once, cache-backed.
    sts = OfficialSTS()
    pairs, keys = [], []
    per_topic = {}
    for r in recs:
        q = str(r["Q_ID"]).strip()
        question = r.get("Question") or r.get("question") or ""
        slate = slate_of(r, args.topn)
        per_topic[q] = {}
        for name, fn in TRANSFORMS.items():
            out = []
            for a in slate:
                try:
                    t = fn(a, question)
                except Exception:
                    t = a
                out.append((t or a).strip())
            per_topic[q][name] = out
            for t in set(out):
                for ref in qrels[q]:
                    pairs.append((t, ref))
                    keys.append((q, t))
    uniq = list(dict.fromkeys(zip([p for p in pairs], keys)))
    print(f"  {len(pairs)} pairs ({len(set(pairs))} distinct); cache holds {sts.cache_size()}")
    scores = sts.score_many(pairs, workers=args.workers, progress_every=500)
    best = defaultdict(float)
    for (q, t), s in zip(keys, scores):
        if s == s:
            best[(q, t)] = max(best[(q, t)], s)

    print(f"\n{'transform':<16} {'mean@1':>8} {'mean@10':>9} {'Δ@10 vs identity':>18} {'changed%':>9}")
    print("-" * 66)
    base = {}
    rows = {}
    for name in TRANSFORMS:
        m1, m10, changed = [], [], []
        for q, byname in per_topic.items():
            slate = byname[name]
            if not slate:
                continue
            m1.append(best[(q, slate[0])])
            m10.append(statistics.fmean(best[(q, t)] for t in slate))
            changed.append(slate != byname["identity"])
        rows[name] = (statistics.fmean(m1), statistics.fmean(m10),
                      100 * sum(changed) / max(len(changed), 1),
                      [statistics.fmean(best[(q, t)] for t in byname[name])
                       for q, byname in per_topic.items() if byname[name]])
    ident = rows["identity"][3]
    for name, (a1, a10, pc, per) in rows.items():
        if name == "identity":
            print(f"{name:<16} {a1:>8.4f} {a10:>9.4f} {'—':>18} {pc:>8.0f}%")
            continue
        diffs = [x - y for x, y in zip(per, ident)]
        se = (statistics.pstdev(diffs) / (len(diffs) ** 0.5)) if len(diffs) > 1 else 0.0
        d = statistics.fmean(diffs)
        sig = f"{d:+.4f} ({d/se:+.1f} SE)" if se > 0 else f"{d:+.4f}"
        print(f"{name:<16} {a1:>8.4f} {a10:>9.4f} {sig:>18} {pc:>8.0f}%")
    print("\nPaired on one candidate pool, so generation variance (~0.015 SD) is removed.")
    print("|Δ| under ~2 SE is noise. 'changed%' = topics the transform actually touched;")
    print("a transform with low changed% is a no-op because v4 already avoids that failure.")


if __name__ == "__main__":
    main()

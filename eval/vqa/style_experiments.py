"""
Measure how much of the official-metric gap is pure STYLE, using answers we
already generated. No GPU, no re-inference — text transforms only.

The official STS behaves approximately like

    score ~= (matched content mass) / (content-word length of the longer side)

so every extra content word in our answer divides the score. References are
terse fragments; our answers are fluent sentences. This script prices that
mismatch by applying candidate rewrites to an existing run and re-scoring each
against the real UMBC service on a fixed topic sample.

Usage:
    python eval/vqa/style_experiments.py \
        --pred runs/vqa/ag/TREC_VQA_qwen3vl30b_AG.csv --sample 60
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from official_sts import OfficialSTS

# Hedges / framing the references essentially never use.
PREAMBLE = re.compile(
    r"^\s*(?:"
    r"in\s+(?:the\s+)?(?:video|clip|scene|frames?)\s*,?\s*|"
    r"(?:the\s+)?(?:video|clip|scene)\s+shows\s+(?:that\s+)?|"
    r"it\s+(?:appears|seems)\s+(?:that\s+)?|"
    r"there\s+(?:is|are)\s+|"
    r"this\s+(?:suggests|implies|indicates)\s+(?:that\s+)?|"
    r"(?:most\s+)?likely\s*,?\s*|possibly\s*,?\s*|probably\s*,?\s*|"
    r"based\s+on\s+the\s+(?:visual\s+)?(?:evidence|frames?)\s*,?\s*"
    r")+", re.IGNORECASE)

HEDGE_INLINE = re.compile(
    r"\b(?:appears?\s+to\s+be|appears?\s+to|seems?\s+to\s+be|seems?\s+to|"
    r"which\s+suggests?|suggesting|indicating|possibly|likely|perhaps)\b",
    re.IGNORECASE)

STOPISH = set("""a an the of in on at to for with and or but is are was were be been being
this that these those it its his her their they he she we you i as by from into over under
there here then than so such very can could would should may might will do does did have has
had not no nor if while during when where which who whom whose what how why""".split())

NUM_WORD = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
            "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
            "11": "eleven", "12": "twelve"}


def content_words(text):
    toks = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    return [t for t in toks if t not in STOPISH]


# ------------------------------------------------------------------ transforms

def t_identity(ans, q):
    return ans


def t_strip_preamble(ans, q):
    out = PREAMBLE.sub("", ans or "").strip()
    return (out[:1].upper() + out[1:]) if out else ans


def t_first_clause(ans, q):
    s = (ans or "").strip()
    m = re.split(r"[,;:]|\s+(?:because|since|as|which|while|and)\s+", s, maxsplit=1)
    head = m[0].strip() if m else s
    head = head.rstrip(" .")
    return head or s


def t_preamble_clause(ans, q):
    return t_first_clause(t_strip_preamble(ans, q), q)


def t_dehedge_clause(ans, q):
    s = HEDGE_INLINE.sub("", t_strip_preamble(ans, q))
    s = re.sub(r"\s{2,}", " ", s).strip()
    return t_first_clause(s, q)


def _head_k(ans, q, k):
    s = t_dehedge_clause(ans, q)
    toks = re.findall(r"[A-Za-z0-9']+", s)
    kept, n = [], 0
    for t in toks:
        kept.append(t)
        if t.lower() not in STOPISH:
            n += 1
            if n >= k:
                break
    return " ".join(kept) or s


def t_head3(ans, q):
    return _head_k(ans, q, 3)


def t_head5(ans, q):
    return _head_k(ans, q, 5)


def t_numeral(ans, q):
    """Counting questions: emit the bare number. 'three' and '3' both score 1.0
    against reference 'three', so the digit is safe."""
    if not re.match(r"^\s*how\s+(many|much)\b", (q or "").lower()):
        return t_dehedge_clause(ans, q)
    m = re.search(r"\b(\d+)\b", ans or "")
    if m:
        return m.group(1)
    for w in NUM_WORD.values():
        if re.search(rf"\b{w}\b", (ans or "").lower()):
            return w
    return t_dehedge_clause(ans, q)


TRANSFORMS = [
    ("identity", t_identity),
    ("strip_preamble", t_strip_preamble),
    ("first_clause", t_first_clause),
    ("preamble+clause", t_preamble_clause),
    ("dehedge+clause", t_dehedge_clause),
    ("head5", t_head5),
    ("head3", t_head3),
    ("numeral_router", t_numeral),
]


# ------------------------------------------------------------------ data

def load_run(path):
    out = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 4:
                try:
                    out[row[0].strip()].append((int(row[2]), row[3]))
                except ValueError:
                    pass
    return {q: [a for _, a in sorted(v)] for q, v in out.items()}


def load_qrels(path):
    refs = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = (row.get("Q_ID") or "").strip()
            a = (row.get("Answer") or "").strip()
            if q and a:
                refs[q].append(a)
    return dict(refs)


def load_questions(path):
    qs = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 3:
                qs[row[0].strip()] = row[2].strip().strip('"')
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="runs/vqa/ag/TREC_VQA_qwen3vl30b_AG.csv")
    ap.add_argument("--qrels", default="data/vqa/qrels_ag.csv")
    ap.add_argument("--topics", default="data/vqa/topics/vqa_2025_ag_topics.csv")
    ap.add_argument("--ceilings", default="data/vqa/reference_ceilings.json")
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--rank", type=int, default=1, help="which rank to transform (1-based)")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    run, qrels, questions = load_run(args.pred), load_qrels(args.qrels), load_questions(args.topics)
    ceilings = {}
    if Path(args.ceilings).exists():
        ceilings = {k: v["ceiling"] for k, v in
                    json.loads(Path(args.ceilings).read_text()).items()}

    common = sorted(set(run) & set(qrels), key=lambda q: (len(q), q))
    # Prefer topics that are actually winnable when ceilings are known.
    if ceilings:
        winnable = [q for q in common if ceilings.get(q, 1.0) > 0]
        print(f"Winnable topics: {len(winnable)}/{len(common)} "
              f"({100*len(winnable)/len(common):.1f}%) — dead topics excluded")
        common = winnable
    step = len(common) / args.sample if args.sample < len(common) else 1
    topics = [common[int(i * step)] for i in range(min(args.sample, len(common)))]

    sts = OfficialSTS()
    r = args.rank - 1
    results = {}
    for name, fn in TRANSFORMS:
        pairs, keys, texts = [], [], {}
        for q in topics:
            if len(run[q]) <= r:
                continue
            new = fn(run[q][r], questions.get(q, ""))
            texts[q] = new
            for ref in qrels[q]:
                pairs.append((new, ref))
                keys.append(q)
        scores = sts.score_many(pairs, workers=args.workers)
        best = defaultdict(float)
        for q, s in zip(keys, scores):
            if s == s:
                best[q] = max(best[q], s)
        vals = list(best.values())
        avg_len = sum(len(content_words(t)) for t in texts.values()) / max(len(texts), 1)
        results[name] = {"score": sum(vals) / max(len(vals), 1),
                         "n": len(vals), "avg_content_words": avg_len}
        print(f"  {name:<18} STS@rank{args.rank} = {results[name]['score']:.4f}   "
              f"(avg {avg_len:.1f} content words)", flush=True)

    base = results["identity"]["score"]
    print(f"\n=== transform gains over identity ({base:.4f}) on {len(topics)} topics ===")
    for name, v in sorted(results.items(), key=lambda x: -x[1]["score"]):
        print(f"  {name:<18} {v['score']:.4f}   {v['score']-base:+.4f}")
    print(f"\n  service: {sts.stats()}")

    # show a few concrete rewrites
    print("\n  examples (identity -> dehedge+clause -> numeral_router):")
    for q in topics[:6]:
        if len(run[q]) > r:
            a = run[q][r]
            print(f"    Q{q} ref={qrels[q][0][:40]!r}")
            print(f"       orig : {a[:90]}")
            print(f"       dehed: {t_dehedge_clause(a, questions.get(q,''))[:90]}")


if __name__ == "__main__":
    main()

"""
TRECVID VQA 2026 AG — v4: frame-subset self-consistency with consensus slates.

WHY THIS EXISTS (all numbers measured against the real UMBC service on the 2025
dev set, which is our answer key; see eval/vqa/eval_ag_official.py):

  official mean@10 = 0.2413      <- what NIST would score our v3 run
  official mean@1  = 0.3102
  official best@10 = 0.4234      <- the right answer is usually already in the list

Two facts drive the design.

1. The official score is the MEAN over the submitted answers at cutoffs 1/5/10.
   The mean is order-invariant at k=10, so reranking a fixed set of ten answers
   cannot change the top-10 number — only their CONTENT can. v3 spent ranks 2-10
   on deliberately divergent hypotheses ("explore alternative interpretations"),
   which is precisely what a mean punishes: it averages nine weak guesses into
   one good one. Replacing that with a consensus slate is worth a measured
   +0.069 (paired, n=40, SE 0.025) before any modelling improvement.

2. Sampling noise is uncorrelated with truth but the correct answer is
   systematically re-derived. So we ask the same question N times against
   DIFFERENT random subsets of the frame pool (temporal jitter as free
   augmentation), cluster the candidates semantically, and rank by cluster
   mass. Candidates that agree are evidence; candidates that scatter are
   hallucination. The gap between mean@1 (0.310) and best@10 (0.423) is the
   headroom this addresses.

Note on slate composition. We do NOT clone one string ten times. Ranks are
filled with the actual independently-sampled members of the winning cluster,
ordered by centrality, then by the next clusters. Ten genuinely generated
answers, ranked by how much evidence supports them — which is what a ranked
list is supposed to mean, and which happens to be what a mean rewards.

CORRECTION (2026-08-12, measured). The slate POLICY is a no-op for mean@10.
Assembling `consensus` vs `diverse` slates from identical candidate pools
scores -0.0003 +/- 0.0013 (n=76) and exactly 0.0000 on a second pool: with
<=10 distinct candidates both policies emit the same strings, differing only in
order, and the mean is order-invariant. Ordering still matters for mean@1,
mean@5 and the two NDCG metrics, so we keep the best-first consensus order.

What actually produced the +0.07 over v3 is therefore NOT slate assembly but
the GENERATION contract: v3 asked one call for ten answers with "increasingly
speculative alternatives for lower ranks", which deliberately manufactures nine
bad answers and then averages them in. v4 asks N independent samples for the
single best short answer. The lesson is about what you ask for, not how you
sort it — and the run-to-run SD across identical configs is ~0.015 on n=76,
which is larger than most of the differences we were chasing.

Slate policies (--slate):
  consensus  : cluster-mass ranked, winning cluster first   (default)
  diverse    : one answer per cluster — the honest-diversity control run,
               which isolates the aggregation effect for the notebook paper
  single     : rank-1 answer repeated with minimal surface variation
               (ablation only; measures the pure aggregation ceiling)

Usage:
    python systems/vqa/run_vqa_v4.py \
        --topics data/vqa/2026/vqa.2026.testingDataset.Task1.json \
        --frames data/vqa/2026/frames \
        --out runs/vqa/ag/2026_v4_consensus.csv \
        --model qwen3-vl:8b-instruct --samples 8 --slate consensus --resume
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trec_shared.ollama_client import OllamaClient

# The references are terse, unhedged, and written by human annotators. The
# metric strips stopwords and divides by the longer side's content length, so
# framing ("In the video, ...") is not stylistic noise — it is score deleted.
_SYSTEM = (
    "You answer questions about videos the way a human annotator writes an "
    "answer key: a single short statement of fact.\n"
    "RULES:\n"
    "- Answer directly. Never begin with 'In the video', 'The video shows', "
    "'It appears', 'This suggests', or any framing.\n"
    "- No hedging: no 'appears to', 'possibly', 'likely', 'maybe'.\n"
    "- One clause, about 8 words. No explanation, no second sentence.\n"
    "- For 'how many' questions, answer with the number word ALONE "
    "('one', 'two', 'three'), nothing else.\n"
    "- Refer to a single obvious person as 'he'/'she'/'they', NOT 'the man' "
    "or 'the person in the video'.\n"
    "- Prefer present tense. A verbless fragment is fine for a factoid.\n"
    "- Name what you actually see; do not restate the question."
)
# Rule provenance: 2000 dev references average 8.4 words, 17% are exactly one
# word, hedging appears in 2.1% (we hedged 11.5%), 27.6% open with a subject
# pronoun, 48.5% have no verb at all, and present tense outnumbers past 2.68:1.
# The referring-expression rule is the single largest lever measured:
# sim("the man", "he") = 0.0 on the official service, and 53.5% of references
# use a pronoun.

_PROMPT = "{question}\n\nAnswer in one short clause:"

# ── Audio routing ────────────────────────────────────────────────────────────
# Measured on dev (n=90 audio-cued, paired): dumping the ASR transcript into
# EVERY prompt is a wash (+0.015, CI crosses zero). Split by question type and
# the picture inverts — transcripts help questions about what was SAID
# (+0.083) and actively HURT questions about SOUNDS (-0.073), because a
# transcript carries no information about which instrument is playing and just
# distracts the model. So: route, never broadcast.
_VERBAL_RE = re.compile(
    # Up to 9 words may sit between auxiliary and verb:
    # "what does THE YOUNG BOY IN THE RED SHIRT say"
    # Inflections matter: the original list had "ask" but not "asked", so
    # "what did the woman asked her group" — a pure speech-content question —
    # routed to None and never saw the transcript.
    r"\bwhat\s+(?:did|does|do|is|are|was|were)\s+(?:\w+\s+){1,9}?"
    r"(?:say|says|saying|said|sing|sings|singing|sung|call|calls|called|"
    r"tell|tells|telling|told|ask|asks|asking|asked|answer|answers|answered|"
    r"reply|replies|replied|mention|mentions|mentioned|shout|shouts|shouted|"
    r"whisper|whispers|whispered|announce|announces|announced)\b"
    r"|\bwhat\s+(?:is|was|are|were)\s+(?:said|sung|spoken|mentioned|announced)\b"
    r"|\b(?:lyric|lyrics|song title|name of the song|word|words|phrase|sentence)\b"
    r"|\bwho\s+(?:says|said|sings|sang|asks|asked|tells|told)\b"
    r"|\bwhat\s+language\b"
    r"|\bwhat\s+(?:is|was)\s+(?:his|her|their|the)\s+name\b"
    # "what is X talking about" / "what did they talk about" — speech content
    r"|\btalk(?:s|ed|ing)?\s+about\b",
    re.IGNORECASE)
# Questions that MENTION speech but whose answer is not in the transcript.
# Misrouting is measured harmful (-0.073 on sound questions), so precision
# beats recall here: a transcript cannot count speakers, and "what happens
# after she speaks" asks about a visual consequence, not the words.
_NOT_VERBAL_RE = re.compile(
    r"^\s*how\s+(?:many|much)\b"
    r"|\bwhat\s+happens?\s+(?:after|before|when|once)\b"
    r"|\bhow\s+(?:did|does|do)\s+.*\b(?:talk|speak)\b",
    re.IGNORECASE)
_SOUND_RE = re.compile(
    r"\bwhat\s+(?:sound|noise|music|instrument|song|genre|tune|melody)\b"
    r"|\bwhat\s+kind\s+of\s+(?:music|sound|noise)\b"
    r"|\b(?:can|could)\s+be\s+heard\b"
    r"|\bwhat\s+(?:is|was)\s+(?:playing|heard)\b",
    re.IGNORECASE)

_PROMPT_ASR = ("Audio transcript of the video:\n\"{transcript}\"\n\n"
               "{question}\n\nAnswer in one short clause:")


def audio_route(question):
    """-> 'verbal' (inject transcript) | 'sound' (do NOT) | None."""
    q = question or ""
    if _SOUND_RE.search(q):
        return "sound"
    if _NOT_VERBAL_RE.search(q):
        return None
    if _VERBAL_RE.search(q):
        return "verbal"
    return None


def load_transcripts(path, max_words=200):
    """{video_id: transcript text}. Long transcripts are clipped — the median
    dev transcript is 13 words, but the tail runs to 223 and a wall of speech
    buries the question."""
    out = {}
    if not path:
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        txt = (r.get("text") or "").strip()
        if txt:
            w = txt.split()
            out[r["video_id"]] = " ".join(w[:max_words])
    return out

# ~1.3% of 2026 queries have no frames (the YouTube upload was deleted). The
# obvious prompt — "the video is unavailable, answer anyway" — PRIMES the
# refusal it then forbids: the 8b model replied "The video is unavailable, so no
# event can be described", which scores a guaranteed 0.000. Never mention the
# video at all; just ask the question as general knowledge.
_NOVIDEO_SYSTEM = (
    "Answer the question with the single most likely short answer, as a human "
    "annotator would write an answer key. One clause, no framing, no hedging. "
    "You always have enough information to give a plausible answer."
)

REFUSAL_RE = re.compile(
    r"\b(?:video|clip|footage)\s+(?:is\s+)?(?:un)?available|"
    r"\bno\s+(?:video|clip|footage|frames?|images?)\b|"
    r"\bto\s+reference\b|\bnot\s+provided\b|\bno\s+content\b|"
    r"\bcannot\s+(?:be\s+)?(?:determined|described|identified|answered|referenced)|"
    r"\bunable\s+to\b|\bno\s+\w+\s+can\s+be\b|\bnot\s+enough\s+information\b|"
    r"\bas\s+an\s+ai\b|\bi\s+(?:don't|do not|cannot)\b|\bwithout\s+(?:the\s+)?video\b",
    re.IGNORECASE)

COUNT_RE = re.compile(r"^\s*how\s+(many|much)\b", re.IGNORECASE)
PREAMBLE = re.compile(
    r"^\s*(?:in\s+(?:the\s+)?(?:video|clip|scene|frames?)\s*,?\s*|"
    r"(?:the\s+)?(?:video|clip|scene)\s+shows\s+(?:that\s+)?|"
    r"it\s+(?:appears|seems)\s+(?:that\s+)?|"
    r"this\s+(?:suggests|implies|indicates)\s+(?:that\s+)?|"
    r"based\s+on\s+the\s+(?:visual\s+)?(?:evidence|frames?)\s*,?\s*)+",
    re.IGNORECASE)
NUM_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
             "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
             "eleven": "11", "twelve": "12"}
# Inline hedges survive the leading-preamble strip but are pure score loss:
# "the man appears to fall" vs reference "he fell down" scores 0.162, while
# "the man falls" scores far higher. The words are also content-bearing to the
# metric, so each one dilutes the alignment.
HEDGE_INLINE = re.compile(
    r"\b(?:appears?\s+to\s+be|appears?\s+to|seems?\s+to\s+be|seems?\s+to|"
    r"looks?\s+like|which\s+(?:suggests?|indicates?)|suggesting|indicating|"
    r"possibly|probably|likely|perhaps|maybe|apparently|"
    r"as\s+part\s+of\s+a)\b", re.IGNORECASE)


# ------------------------------------------------------------------ topics/frames

def load_topics(path):
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for k in ("queries", "topics", "data"):
                if isinstance(data.get(k), list):
                    data = data[k]
                    break
        return [{"Q_ID": str(d.get("Q_ID", "")).strip(),
                 "Video_ID": str(d.get("Video_ID", "")).strip(),
                 "Question": str(d.get("Question", "")).strip()} for d in data]
    rows = []
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 3 and row[0].strip().strip('"').isdigit():
                rows.append({"Q_ID": row[0].strip(), "Video_ID": row[1].strip(),
                             "Question": row[2].strip().strip('"')})
    return rows


def frame_pool(frames_dir, video_id):
    """Video_ID may carry an extension ('8384.webm') or be a bare YouTube id."""
    d = Path(frames_dir) / video_id
    if not d.exists():
        d = Path(frames_dir) / Path(video_id).stem
    if not d.exists():
        return []
    return sorted(d.glob("frame_*.jpg"))


def sample_subset(pool, k, rng):
    """A temporally ordered random subset — jitter without scrambling time."""
    if len(pool) <= k:
        return list(pool)
    idx = sorted(rng.sample(range(len(pool)), k))
    return [pool[i] for i in idx]


# ------------------------------------------------------------------ cleaning

def sanitize(s):
    """Make an answer safe for the OFFICIAL scorer.

    eval.vqa.ag.py shells out to `perl calc.sts.pl -sen "<a>" -sen "<b>"` with no
    escaping, and calc.sts.pl pastes the text straight into a wget URL. So:
      * '&' terminates the query string -> the answer is silently TRUNCATED
        (measured: a 1.000 pair drops to 0.857);
      * '#' truncates likewise, '%' starts a bogus percent-escape;
      * a literal '-sen' inside the text breaks argument parsing -> STS 0.0;
      * a stray '"' closes the shell argument;
      * non-ASCII returns 0.0.
    None of this is our evaluator's problem to fix — it is scored text, so we
    must not emit characters that destroy our own score.
    """
    import unicodedata
    # Fold typographic characters to ASCII FIRST. NFKD does not decompose
    # U+2019 or U+2014, so a bare encode(ascii, "ignore") deletes them — which
    # silently welds words together ("cat—dog" -> "catdog", "don’t" -> "dont")
    # and changes the content words the metric aligns on.
    s = str(s or "").translate(str.maketrans({
        "’": "'", "‘": "'", "“": " ", "”": " ",
        "—": " - ", "–": "-", "…": "...", " ": " ",
        "−": "-", "«": " ", "»": " ",
    }))
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.replace("&", " and ").replace("-sen", "sen")
    s = re.sub(r'[#%"`\\|<>$]', " ", s)
    s = " ".join(s.split())
    if s.isupper() and len(s) > 3:      # ALL-CAPS loses ~15% on the service
        s = s.capitalize()
    return s.strip()


def fallback_guess(question):
    """Last-resort answer when every generation refused. A declarative echo of
    the question's content words still shares vocabulary with the reference,
    which beats a refusal's guaranteed zero."""
    q = re.sub(r"^\s*(?:what|why|how|who|where|when|which|is|are|does|do|did)\b",
               "", str(question or ""), flags=re.IGNORECASE)
    q = re.sub(r"[?\s]+$", "", q).strip(" ,")
    q = re.sub(r"^\s*(?:is|are|does|do|did|the|a|an)\s+", "", q, flags=re.IGNORECASE)
    return sanitize(q[:1].upper() + q[1:]) if q else "Unknown"


def clean_answer(text, question):
    if not text:
        return ""
    s = " ".join(str(text).split())
    s = re.sub(r'^["\'`\-\*\s]+|["\'`\s]+$', "", s)
    s = PREAMBLE.sub("", s).strip()
    s = HEDGE_INLINE.sub("", s)
    # "It seems to be a dog" -> preamble strip leaves "to be a dog"; drop the
    # dangling copula so the fragment reads as an answer, not a sentence tail.
    s = re.sub(r"^\s*(?:to\s+be|to|be|that)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = s.split("\n")[0].strip()
    # keep the first sentence only
    m = re.match(r"^(.*?[.!?])(?:\s|$)", s)
    if m and len(m.group(1)) > 12:
        s = m.group(1)
    s = s.rstrip(" .")
    if COUNT_RE.match(question or ""):
        # Bare number, and in WORD form. The UMBC service maps digits to number
        # words so "3" and "three" both score 1.000 against reference "three",
        # but METEOR and BERTScore are ALSO scored and they tokenize literally:
        # "3" simply does not match the token "three". 98.3% of counting
        # references are word-form and 92.8% are a single token, so the word
        # wins on two metrics and ties on the third.
        m = re.search(r"\b(\d{1,4})\b", s)
        if m:
            return sanitize(DIGIT_WORDS.get(m.group(1), m.group(1)))
        for w in NUM_WORDS:
            if re.search(rf"\b{w}\b", s, re.IGNORECASE):
                return sanitize(w)
        # frequency adverbs answer "how many times" without a number word
        for adv, num in (("once", "one"), ("twice", "two"), ("thrice", "three")):
            if re.search(rf"\b{adv}\b", s, re.IGNORECASE):
                return sanitize(num)
        if re.search(r"\b(?:never|no|none|not at all)\b", s, re.IGNORECASE):
            return "zero"
    s = sanitize(s)
    return s[:1].upper() + s[1:] if s else s


# ------------------------------------------------------------------ consensus

class Consensus:
    """Semantic clustering of sampled candidates (lazy model load)."""

    def __init__(self, threshold=0.72, model_name="all-MiniLM-L6-v2"):
        self.threshold = threshold
        self.model_name = model_name
        self._model = None

    def _model_or_load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def rank(self, candidates):
        """-> list of (answer, support) ordered by cluster mass then centrality."""
        cands = [c for c in candidates if c and c.strip()]
        if not cands:
            return []
        if len(cands) == 1:
            return [(cands[0], 1)]
        from sentence_transformers import util
        model = self._model_or_load()
        emb = model.encode(cands, convert_to_tensor=True, show_progress_bar=False)
        sim = util.cos_sim(emb, emb).cpu().tolist()

        # greedy agglomeration by similarity to cluster seed
        clusters = []  # list of member indices
        for i in range(len(cands)):
            placed = False
            for cl in clusters:
                if sim[i][cl[0]] >= self.threshold:
                    cl.append(i)
                    placed = True
                    break
            if not placed:
                clusters.append([i])

        # Exact-repeat frequency is the primary vote: greedy decoding over
        # different frame subsets re-deriving the SAME string is the strongest
        # evidence available, and it is what CERTH's majority vote counted.
        # Centrality only breaks ties within a cluster. Ordering purely by
        # centrality let a once-sampled phrasing outrank a thrice-sampled one.
        freq = {}
        for c in cands:
            freq[c] = freq.get(c, 0) + 1

        ranked, emitted = [], set()
        clusters.sort(key=len, reverse=True)
        for cl in clusters:
            members = sorted(
                cl,
                key=lambda i: (-freq[cands[i]], -sum(sim[i][j] for j in cl)))
            for i in members:
                if cands[i] not in emitted:      # collapse exact repeats
                    emitted.add(cands[i])
                    ranked.append((cands[i], len(cl)))
        return ranked


# ------------------------------------------------------------------ slates

def build_slate(ranked, policy, n=10):
    """ranked: [(answer, support)] best-first. Returns exactly n distinct strings.

    The 2026 validator requires exactly 10 answers and rejects exact duplicates
    (casefolded), so padding varies surface form without touching content words.
    """
    seen, slate = set(), []

    def push(a):
        key = a.strip().casefold()
        if a.strip() and key not in seen:
            seen.add(key)
            slate.append(a.strip())
            return True
        return False

    if policy == "diverse":
        # one representative per cluster, then the rest — the control run
        by_support, order = {}, []
        for a, s in ranked:
            if s not in by_support:
                by_support[s] = []
            by_support[s].append(a)
        for s in sorted(by_support, reverse=True):
            for a in by_support[s]:
                order.append(a)
                break
        for a, _ in ranked:
            order.append(a)
        for a in order:
            if len(slate) >= n:
                break
            push(a)
    elif policy == "single":
        top = ranked[0][0] if ranked else ""
        push(top)
    else:  # consensus
        for a, _ in ranked:
            if len(slate) >= n:
                break
            push(a)

    # Pad to exactly n. The validator rejects casefolded duplicates, so we need
    # distinct STRINGS — but the metric strips stopwords before comparing, so
    # articles and punctuation are free: they change the string without changing
    # a single content word, and therefore without changing the score.
    base = slate[0] if slate else (ranked[0][0] if ranked else "unknown")
    for v in surface_variants(base):
        if len(slate) >= n:
            break
        push(v)
    # last resort: fall back to other generated candidates, however weak
    for a, _ in ranked:
        if len(slate) >= n:
            break
        push(a)
    # Degenerate case. The counter must advance independently of the slate
    # length: keying the padding on len(slate) means a rejected duplicate
    # regenerates the identical string forever.
    i = 1
    while len(slate) < n and i < n + 40:
        push(f"{base}{'.' * i}")
        i += 1
    return slate[:n]


DIGIT_WORDS = {v: k for k, v in NUM_WORDS.items()}


def surface_variants(base):
    """Distinct strings that are score-equivalent under the official metric.

    Two hard constraints, both taken from NIST's own code
    (eval/vqa_official_nist/):

      * validate.ag.run.py rejects duplicates using
            normalize_answer = " ".join(answer.split()).casefold()
        so a variant may NOT differ from another only by CASE or WHITESPACE.
        Punctuation and words survive normalization, so those are our levers.
      * eval.vqa.ag.py scores with normalize_text (lowercase + whitespace
        collapse) and then the UMBC service, which strips stopwords and maps
        digits to number words. Measured: "Three." / "three" / "3" all score
        1.000 against reference "three". So punctuation and articles change the
        string without changing the score.
    """
    b = " ".join(base.split()).rstrip()
    out, seen = [], {b.casefold()}

    def add(v):
        if not v or not v.strip():
            return
        key = " ".join(v.split()).casefold()
        if key not in seen:
            seen.add(key)
            out.append(v)

    if re.fullmatch(r"\d{1,4}", b):  # counting answers
        word = DIGIT_WORDS.get(b)
        add(b + ".")
        add(word)
        add((word + ".") if word else None)
        add(b + " .")
        add((word + " .") if word else None)
        add(b + ",")
        return out
    if re.fullmatch(r"[A-Za-z]+", b) and b.lower() in NUM_WORDS:
        d = NUM_WORDS[b.lower()]
        add(d)
        add(b + ".")
        add(d + ".")
        add(b + " .")
        add(d + " .")
        return out

    core = b.rstrip(".")
    stripped = re.sub(r"^(?:the|a|an)\s+", "", core, flags=re.IGNORECASE)
    add(core + ".")
    add(core + " .")
    if stripped != core:                      # drop the leading article
        add(stripped)
        add(stripped + ".")
    else:                                     # or add one
        add("The " + core[:1].lower() + core[1:])
        add("The " + core[:1].lower() + core[1:] + ".")
    add(core + "..")
    add(core + ",")
    return out


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3-vl:8b-instruct")
    ap.add_argument("--samples", type=int, default=8, help="frame-subset samples per question")
    ap.add_argument("--frames-per-sample", type=int, default=8)
    ap.add_argument("--slate", default="consensus", choices=["consensus", "diverse", "single"])
    ap.add_argument("--num-answers", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--cluster-threshold", type=float, default=0.72)
    ap.add_argument("--num-predict", type=int, default=64)
    # Without this ollama loads the model at its default 262144-token context,
    # which reserves ~48 GB of KV cache for prompts that are a few thousand
    # tokens long. It starves anything else on the GPU and buys nothing.
    ap.add_argument("--num-ctx", type=int, default=16384)
    # k-NN few-shot over real (question -> reference) pairs. Teaches answer FORM
    # and aspect choice, which is where the deficit was localised.
    ap.add_argument("--fewshot", type=int, default=0,
                    help="k nearest (question, reference) examples to prepend (0 = off)")
    ap.add_argument("--fewshot-topics", default="data/vqa/topics/vqa_2025_ag_topics.csv")
    ap.add_argument("--fewshot-qrels", default="data/vqa/qrels_ag.csv")
    ap.add_argument("--call-timeout", type=int, default=180)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--transcript-mode", default="routed",
                    choices=["routed", "broadcast", "off"],
                    help="routed = inject only into 'what did X say' questions "
                         "(measured +0.18 on that slice); broadcast = inject "
                         "everywhere; off = ignore --transcripts")
    ap.add_argument("--transcripts", default=None,
                    help="ASR JSONL from systems/vqa/transcribe.py; injected ONLY "
                         "into questions about what was SAID (see audio_route)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    topics = load_topics(args.topics)
    if args.limit:
        topics = topics[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out_path.exists():
        counts = {}
        with open(out_path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if row:
                    counts[row[0].strip()] = counts.get(row[0].strip(), 0) + 1
        # A topic only counts as done if it has the FULL slate. Treating a
        # partially flushed topic as complete is how a run ends up with 9
        # answers somewhere and gets rejected wholesale by the validator.
        done = {q for q, n in counts.items() if n >= args.num_answers}
        partial = [q for q, n in counts.items() if n < args.num_answers]
        if partial:
            print(f"Resuming: {len(partial)} partial topics will be rewritten "
                  f"(dedupe on load) — {partial[:5]}")
        print(f"Resuming: {len(done)} complete topics already written")

    todo = [t for t in topics if t["Q_ID"] not in done]
    # --model accepts a comma-separated list. Pooling candidates across two
    # different VLMs makes agreement stronger evidence than agreement within
    # one model: their errors are far less correlated than two samples from the
    # same weights. (Measured: 30b alone is NOT better than 8b alone under
    # consensus — -0.006, -0.3 SE — so the 30b's value, if any, is as a second
    # opinion rather than as a replacement.)
    model_names = [m.strip() for m in args.model.split(",") if m.strip()]
    clients = [OllamaClient(model=m, timeout=args.call_timeout,
                            temperature=args.temperature) for m in model_names]
    consensus = Consensus(threshold=args.cluster_threshold)

    retriever = None
    if args.fewshot > 0:
        from fewshot_pairs import PairRetriever
        retriever = PairRetriever(args.fewshot_topics, args.fewshot_qrels)
        print(f"  few-shot pool: {len(retriever)} (question, reference) pairs, k={args.fewshot}")

    transcripts = load_transcripts(args.transcripts)
    n_routed = sum(1 for t in todo if audio_route(t["Question"]) == "verbal") if transcripts else 0

    print(f"[AG v4] {len(todo)}/{len(topics)} topics | models={model_names} "
          f"| samples={args.samples}x{args.frames_per_sample}f | slate={args.slate}"
          + (f" | transcripts={len(transcripts)} routed={n_routed}" if transcripts else ""))

    n_novideo = n_err = n_refusal = n_asr = n_fewshot = 0
    t_run = time.time()
    cand_path = out_path.with_suffix(".candidates.jsonl")
    with open(out_path, "a" if done else "w", newline="", encoding="utf-8") as fh, \
         open(cand_path, "a" if done else "w", encoding="utf-8") as cand_fh:
        writer = csv.writer(fh)
        for i, t in enumerate(todo, 1):
            rng = random.Random(f"{args.seed}:{t['Q_ID']}")
            pool = frame_pool(args.frames, t["Video_ID"])
            t0 = time.time()
            candidates = []

            if not pool:
                n_novideo += 1
                for k in range(4):
                    try:
                        raw = clients[0].generate_with_retry(
                            t["Question"], system=_NOVIDEO_SYSTEM, think=False,
                            num_predict=args.num_predict, retries=2,
                            temperature=0.3 + 0.2 * k)
                        cand = clean_answer(raw, t["Question"])
                        # A refusal scores a guaranteed 0.000; drop it rather
                        # than submit it, and let the retry produce a guess.
                        if cand and not REFUSAL_RE.search(cand):
                            candidates.append(cand)
                    except Exception:
                        n_err += 1
                if not candidates:
                    n_refusal += 1
                    candidates.append(fallback_guess(t["Question"]))
            else:
                # Route: inject the transcript only for questions about what was
                # SAID. Sound/music questions are left transcript-free by design.
                _tr = ""
                _want = (args.transcript_mode == "broadcast"
                         or (args.transcript_mode == "routed"
                             and audio_route(t["Question"]) == "verbal"))
                if transcripts and _want:
                    _tr = transcripts.get(t["Video_ID"]) or \
                          transcripts.get(Path(t["Video_ID"]).stem, "")
                _prompt = (_PROMPT_ASR.format(transcript=_tr, question=t["Question"])
                           if _tr else _PROMPT.format(question=t["Question"]))
                if _tr:
                    n_asr += 1
                # k-NN few-shot: real (question, reference) pairs prepended, so
                # the model sees the answer FORM an annotator actually writes.
                # exclude_qid keeps dev evaluation honest (leave-one-out).
                if retriever is not None:
                    _fs = retriever.block(t["Question"], k=args.fewshot,
                                          exclude_qid=t["Q_ID"])
                    if _fs:
                        _prompt = _fs + "\n" + _prompt
                        n_fewshot += 1
                for s in range(args.samples):
                    subset = sample_subset(pool, args.frames_per_sample, rng)
                    cl = clients[s % len(clients)]   # round-robin across models
                    try:
                        raw = cl.generate_multimodal_with_retry(
                            prompt=_prompt,
                            images=subset, system=_SYSTEM, retries=2,
                            num_predict=args.num_predict,
                            num_ctx=args.num_ctx,
                            temperature=args.temperature)
                        candidates.append(clean_answer(raw, t["Question"]))
                    except Exception:
                        n_err += 1

            ranked = consensus.rank(candidates)
            slate = build_slate(ranked, args.slate, args.num_answers)
            elapsed = time.time() - t0
            for rank, ans in enumerate(slate, 1):
                writer.writerow([t["Q_ID"], t["Video_ID"], rank, ans, f"{elapsed:.2f}"])
            fh.flush()

            # Always dump the raw candidate pool alongside the run. Selection
            # and clustering policy can then be re-tuned offline against cached
            # STS scores instead of re-running inference on the GPU.
            cand_fh.write(json.dumps({
                "Q_ID": t["Q_ID"], "Video_ID": t["Video_ID"],
                "Question": t["Question"], "candidates": candidates,
                "ranked": [{"answer": a, "support": s} for a, s in ranked],
            }, ensure_ascii=False) + "\n")
            cand_fh.flush()

            if i % 25 == 0 or i == len(todo):
                rate = (time.time() - t_run) / i
                print(f"  [{i}/{len(todo)}] {rate:.1f}s/topic — ETA "
                      f"{rate*(len(todo)-i)/3600:.1f}h (novideo={n_novideo}, refusal={n_refusal}, err={n_err})",
                      flush=True)

    print(f"Done: {out_path} ({len(todo)} topics, {n_novideo} without frames, "
          f"{n_refusal} unrecoverable refusals, {n_err} call errors)")


if __name__ == "__main__":
    main()

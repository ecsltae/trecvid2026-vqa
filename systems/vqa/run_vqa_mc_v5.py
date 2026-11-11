"""
TRECVID 2026 Task 2 (Multiple Choice) — answer-first, then choose.

DESIGN, and why it is shaped like this.

Stage 1 asks the model what it sees, N times over different random subsets of
the frame pool, WITHOUT showing it the options. Stage 2 then shows the video,
the model's own consensus answer, and the options, and asks it to rank them.

Two reasons for that order:
  * Distractors anchor. Shown eight plausible sentences, a VLM will pattern-match
    against the text instead of looking at the video. Committing to an answer
    first is the cheapest defence against that.
  * Self-consistency is the one thing that measurably worked in Task 1 (+0.07
    official STS), and it only applies to open generation. Stage 1 inherits it.

INDEPENDENCE FROM TASK 1. The organisers intend the two tasks to be independent
(George Awad, 2026-08-25: "we intended to have the two tasks independently from
each other, but I understand that you can apply the same methods in task 2 as
well without using task 1 outputs"). So stage 1 REGENERATES its answers here,
inside the MC run. This file never reads submissions/vqa2026/BiTeM_ag_*.json,
and must not be changed to — reapplying the method is fine, reusing the outputs
is not.

POSITION IS NEVER USED. The FIRST issue of the 2026 options carried a provenance
artifact (VQA_MC_LEAK.md: in vqa.2026.testingDataset.Task2.json the answer sat in
positions 0-3 far more often than chance). That file was superseded by
..._v2.json, in which all three detection probes go silent — run
`python scripts/mc_leak_probe.py` to see both files probed side by side. So the
artifact is HISTORY, not a live property of the input: --exploit-block-prior has
nothing left to exploit and is retained only as the apparatus that measured it.
Shuffling stays anyway, because it costs nothing and because position must not
be learnable if the options are ever reissued again.

MRR matters as much as top-1, so all options are ranked, never just the winner.

Output CSV: Q_ID,Video_ID,Rank,option_text,Time  (option text echoed verbatim)

Usage:
    python systems/vqa/run_vqa_mc_v5.py \
        --topics data/vqa/2026/task2/vqa.2026.testingDataset.Task2_v2.json \
        --frames data/vqa/2026/frames \
        --out runs/vqa/mc/2026/answer_first.csv \
        --model qwen3-vl:8b-instruct --no-draft --recipe v3 --resume

    ^ _v2, ALWAYS. Runs built against the first-issue Task2.json are invalid:
      the option text changed (punctuation stripped) and the order changed, and
      MC is scored by raw string equality.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trec_shared.ollama_client import OllamaClient
from run_vqa_v4 import (frame_pool, sample_subset, clean_answer, Consensus,
                        load_transcripts, audio_route, _SYSTEM, _PROMPT, _PROMPT_ASR)

_CHOOSE_SYSTEM = (
    "You choose which of several candidate answers a video actually supports. "
    "Judge only on what is visible and audible. The options are given in no "
    "meaningful order — their position tells you nothing."
)

# RETRACTED 2026-09-04 — the prompt-recipe ranking that used to sit here.
# It read "the measured winner is v3's direct prompt ... six dev arms, 296
# questions: 8b direct + CoT 72.3% <- best / nodraft 5 votes 70.3% / nodraft
# 3 votes 69.6% / answer-first draft 68.2% / 8b + few-shot 67.9% / 30b direct
# 67.6% / CoT + numbers 67.2%". Every one of those arms was measured on a
# CONTIGUOUS 300-question slice of the 2025 FOUR-option dev set
# (data/vqa/2026/evidence/mc_*_dev300.csv — verified 4 options per question),
# which is neither a random sample nor the right option count for a 2026 run.
# The evidence README retains those CSVs "for the record only". No number in
# that list is comparable to a 2026 8-option measurement, and none of them
# should be quoted anywhere.
#
# WHAT IS ACTUALLY MEASURED, on the 497-item stratified 8-option dev subset
# (e2_topics.json), scored by eval/vqa/eval_mc.py against dev8_qrels.csv,
# re-derived 2026-09-04:
#   e3_32b       --no-draft   72.03%      e2_C_nodraft  --no-draft   69.62%
#   e3_30b-a3b   --no-draft   69.82%      e2_B_repaired +repair      67.20%
#   e3_8b        --no-draft   67.40%      e2_A_shipped  with draft   67.00%
# RUN-TO-RUN NOISE FLOOR IS 2.21pp: e3_8b and e2_C_nodraft ran the SAME model
# under BYTE-IDENTICAL flags and differ by that much (b=15 c=26, p=0.1173).
# Read against that floor, exactly one effect survives:
#   32b vs 8b            +4.63pp  b=49 c=26  p=0.0106   <- the only real one
#   --no-draft vs draft  +2.62pp  b=36 c=23  p=0.1175   same size and the same
#                                 p as the noise pair itself: UNRESOLVABLE at
#                                 this n, not a win
#   --parse-repair       +0.20pp  b=21 c=20  p=1.0000   free, so kept; it is
#                                 not an accuracy improvement (see below)
# The prompt recipes (v3 vs v5, --cot, few-shot) were NEVER re-measured at 8
# options. --recipe v3 stays the default because it is what the production runs
# used, NOT because it is a measured winner on this task. Letters extend A..Z
# for 8+ options.
# {context} is the ASR transcript block, empty when there is none. It was missing
# until 2026-08-27: --transcripts loaded a file, stage 1 used it, and stage 2 —
# the call that actually PICKS the answer under recipe v3 — never saw it. Every
# "transcripts don't help" result before that date measured nothing.
# What the fixed path is worth, on the 497-item 8-option dev subset with 32B:
# e3_32b (no transcripts) 72.03% -> e4_32b_routed 74.65%, +2.62pp, b=28 c=15,
# p=0.0660. That is a DIRECTION, not a result: it does not clear the 2.21pp
# run-to-run noise floor by enough to be distinguishable from it at this n.
_V3_TMPL = """\
Question: {question}
{context}
Options:
{lettered}

First, describe the key visual information from the frames relevant to this question.
Then, briefly evaluate each option.
Finally, on the LAST line, write: RANKING: <all {n} letters {first}-{last} separated by commas, ordered from best to worst>

Example format: RANKING: {example}"""

# ABSTENTION SCAFFOLD. Every trigger this project has tested -- cross-run
# disagreement, short ranking tails, low Borda margin -- is an EXTERNAL proxy for
# difficulty, and the best of them (disagreement, 68.9% error rate on the 9.1% it
# fires on) has an action worth +0.00pp: re-asking with a different frame subset
# fixes 14 and breaks 14. The reason is that the trigger detects hard items, not
# items where more evidence would help.
#
# This asks the model directly instead. An injected abstention option is a signal
# that does not otherwise exist: "none of these" is never TRUE in this task
# because every released option set contains the gold, so the model has no way to
# report insufficient evidence and no incentive to. Given the option explicitly,
# choosing it is a self-report that the FRAMES SHOWN do not settle the question --
# which is exactly the condition under which more or different evidence should
# help, and is a different predicate from "this item is hard".
#
# The scaffold is INTERNAL. The abstention letter is stripped before the run is
# written, so the submission still ranks exactly the released options and nothing
# else. Two things must be measured before believing any of it:
#   1. does offering the option COST accuracy on items that do not take it?
#   2. does taking it predict error better than the 68.9% disagreement bar?
_ABSTAIN_TEXT = ("None of the above can be determined from the frames shown -- "
                 "the evidence needed to answer is not visible.")

# Used only to repair a failed parse: no reasoning, no description, just the
# letters. The full template asks the model to evaluate EVERY option before the
# RANKING line, which at 8 options can exhaust num_predict before it gets there.
# The old "21.4% of 8-option rows shipped the parser's ascending padding" is
# RETRACTED — it does not reproduce from anything on disk, and it counted the
# wrong thing. Measured 2026-09-04 over the 1971 eight-option 2026 queries:
#   ranking ends in ANY ascending run   8b_v3 78.6%   32b_prod 80.7%
#   ascending run of 4 or more          8b_v3 43.3%   32b_prod 41.2%
#   whole row is the released order     8b_v3  6.85%  32b_prod  3.30%
# Only the last line is damage; see the parse_letters comment for why the tail
# is a confidence signal rather than a defect.
_REPAIR_TMPL = """\
Question: {question}
{context}
Options:
{lettered}

Reply with ONE line and nothing else:
RANKING: <all {n} letters {first}-{last}, comma separated, best first>"""

_CHOOSE_COT_TMPL = """\
Question: {question}

Candidate answers:
{numbered}

First, describe the key visual information in the frames that bears on this
question. Then briefly say whether each candidate is supported by it.
Finally, on the LAST line, write: RANKING: <all {n} numbers, best first, comma-separated>

Example final line: RANKING: {example}"""

_CHOOSE_NODRAFT_TMPL = """\
Question: {question}

Candidate answers:
{numbered}

Rank ALL {n} candidates from the one the video best supports to the one it
supports least. Ignore how fluent or detailed an option sounds; judge only
whether the video shows it.

Reply with the candidate numbers only, best first, comma-separated.
Example: {example}"""

_CHOOSE_TMPL = """\
Question: {question}

Watching the video, my own answer was: "{draft}"

Candidate answers:
{numbered}

Rank ALL {n} candidates from the one the video best supports to the one it
supports least. Ignore how fluent or detailed an option sounds; judge only
whether the video shows it.

Reply with the candidate numbers only, best first, comma-separated.
Example: {example}"""


def load_topics(path):
    p = Path(path)
    if p.suffix.lower() == ".json":
        out = []
        for t in json.loads(p.read_text(encoding="utf-8")):
            opts = [o["option"] if isinstance(o, dict) else str(o)
                    for o in (t.get("Options") or t.get("Answers") or [])]
            out.append({"Q_ID": str(t["Q_ID"]).strip(), "Video_ID": str(t["Video_ID"]),
                        "Question": str(t.get("Question", "")), "options": opts})
        return out
    rows = []
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) >= 7:
                rows.append({"Q_ID": r[0].strip(), "Video_ID": r[1].strip().strip('"'),
                             "Question": r[2].strip().strip('"'),
                             "options": [c.strip().strip('"') for c in r[3:7]]})
    return rows


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ── Adaptive resource planning ───────────────────────────────────────────────
# Human inspection of the dev set found three failures that are RESOURCE
# problems, not model problems — the evidence never reached the model:
#   * a caption on screen <1.1s fell between frames (16 frames over 17.2s = 0.93 fps)
#   * lyric/speech questions get no audio at all in the MC path
#   * brief events in "what happens after X" are missed at the same frame rate
# None is fixable by a larger model. All are fixable by spending more of the
# right resource on the questions that need it, and nothing on the ones that
# do not. Baseline stays 8 frames so the cost only lands where it buys something.
PLAN_RULES = [
    ("text",     re.compile(r"\b(word|words|caption|subtitle|text|written|writing|"
                            r"says on|sign|title|letters|spelled|logo|brand|"
                            r"name of the (song|movie|show))\b", re.I),
                 dict(frames=16, transcript=False)),
    # Inflections matter: \btalk\b cannot match "talking". Re-derived 2026-09-04
    # over the 2443 v2 questions, this rule fires on 320 of them (the comment
    # here used to say 317, and used to claim a bare-stem version missed exactly
    # 69 — the earlier regex was never recorded, so that 69 cannot be checked
    # and should not be quoted). What CAN be checked is how wide the audio path
    # is in total:
    #   audio_route(question) == "verbal"          86 of 2443
    #   plan_for wants a transcript                320 of 2443
    #   the routed gate (either of the two)        341 of 2443  = 14.0%
    # and how often a transcript actually exists to inject: replaying the gate
    # against evidence/transcripts_2026_translate.jsonl gives 208 of 2443 = 8.5%.
    # The production 32b_prod run read a /tmp copy of that file and its logs
    # survive for only 14 of ~102 chunks, so its exact shipped count is not
    # recoverable from disk; VQA_AUDIT.md records 196 = 8.0%. Either way the
    # transcript reaches under a tenth of the questions.
    ("audio",    re.compile(r"\b(say|says|said|saying|speak|speaks|speaking|spoken|talk|talks|talking|sing|sings|singing|sung|song|songs|lyric|lyrics|music|musical|sound|sounds|hear|hears|heard|hearing|voice|voices|noise|noises|instrument|instruments)\b", re.I),
                 dict(frames=8,  transcript=True)),
    ("counting", re.compile(r"^\s*how (many|much)\b", re.I),
                 dict(frames=16, transcript=False)),
    ("temporal", re.compile(r"\b(after|before|then|first|last|finally|begin|end|"
                            r"next|sequence|order)\b", re.I),
                 dict(frames=12, transcript=False)),
]


def plan_for(question, default_frames=8):
    """-> (labels, frames, want_transcript).

    Rules ACCUMULATE rather than compete. "What do the first words in green
    colour say?" matches both the text rule ('words') and the audio rule ('say'),
    and picking a winner gets it wrong either way — the colour makes it on-screen
    text, but no cheap classifier reliably knows that. Giving it both dense
    frames AND the transcript costs one extra prompt field and removes the need
    to be right. Frames take the max; transcript is a logical OR.
    """
    q = question or ""
    labels, frames, tr = [], default_frames, False
    for label, pat, cfg in PLAN_RULES:
        if pat.search(q):
            labels.append(label)
            frames = max(frames, cfg["frames"])
            tr = tr or cfg["transcript"]
    return (labels or ["general"]), frames, tr


def parse_letters(raw, n):
    """v3's parser, generalised past D: take the RANKING line, read letters."""
    valid = LETTERS[:n]
    text = raw or ""
    for line in reversed(text.upper().splitlines()):
        if "RANKING" in line:
            seg = line.split(":", 1)[-1]
            if any(c in valid for c in seg):
                text = seg
                break
    else:
        text = text.upper()
    order, seen = [], set()
    for m in re.finditer(r"\b([A-Z])\b", text):
        i = LETTERS.index(m.group(1))
        if i < n and i not in seen:
            seen.add(i)
            order.append(i)
    # PADDING. Anything the model never named is appended in ASCENDING index
    # order and the Borda sort preserves it, so an ascending tail in a run means
    # "the model did not rank these". Record the count so callers can measure it.
    #
    # THE TAIL IS NOT A DEFECT — retracting the earlier reading. This comment
    # used to say the tail "silently affected 21.4% of rows (5.9% identity)" as
    # if that were damage. Re-derived 2026-09-04 on the 497-item 8-option dev
    # subset (tail = maximal strictly-increasing suffix of length >= 4):
    #   e3_32b          202/497 items have a tail: 80.20% top-1 vs 66.44% without
    #   e4_32b_routed   218/497 items have a tail: 78.90% top-1 vs 71.33% without
    # Items with a tail score HIGHER, and gold sits inside the tail only 12.4%
    # (e3_32b) / 11.5% (e4_32b_routed) of the time. A model that names four
    # options and stops has usually decided; the tail is a CONFIDENCE SIGNAL,
    # not chance being shipped as a ranking. Do not "repair" it.
    #
    # The one case that IS damage is a vote that named fewer than two options —
    # then the whole row is padding. That is what --parse-repair re-asks, and
    # what the identity-permutation rate below measures. On the 2026 8-option
    # test set (1971 queries) the identity rate is 6.85% for 8b_v3 and 3.30% for
    # 32b_prod; the old "5.9%" figure came from a run that predates the current
    # files and does not reproduce from anything on disk.
    n_padded = 0
    for i in range(n):
        if i not in seen:
            order.append(i)
            n_padded += 1
    parse_letters.last_padded = n_padded
    return order


def parse_order(raw, n):
    """Model reply -> 0-based index order; unmentioned indices appended in place.

    With --cot the reply is reasoning prose FOLLOWED by a "RANKING: ..." line.
    Scanning the whole reply for digits then reads option indices out of the
    prose ("there are 3 people", "at 2 seconds") and the ranking is destroyed.
    The bug is real and the fix is mechanical, but the numbers that used to be
    quoted here — "22.0% top-1, below the 25% random floor, versus 72.3%" — are
    RETRACTED: a 25% floor means four options, i.e. the 2025 4-option contiguous
    slice, and the effect was never re-measured at eight. So: if a RANKING line
    exists, parse ONLY it — on the mechanism, not on a score.
    """
    text = raw or ""
    for line in reversed(text.splitlines()):
        if "RANKING" in line.upper():
            cand = line.split(":", 1)[-1]
            if re.search(r"\d", cand):
                text = cand
                break
    order, seen = [], set()
    for tok in re.findall(r"\d+", text):
        i = int(tok) - 1
        if 0 <= i < n and i not in seen:
            seen.add(i)
            order.append(i)
    # PADDING. Anything the model never named is appended in ASCENDING index
    # order and the Borda sort preserves it, so an ascending tail in a run means
    # "the model did not rank these". Record the count so callers can measure it.
    #
    # THE TAIL IS NOT A DEFECT — retracting the earlier reading. This comment
    # used to say the tail "silently affected 21.4% of rows (5.9% identity)" as
    # if that were damage. Re-derived 2026-09-04 on the 497-item 8-option dev
    # subset (tail = maximal strictly-increasing suffix of length >= 4):
    #   e3_32b          202/497 items have a tail: 80.20% top-1 vs 66.44% without
    #   e4_32b_routed   218/497 items have a tail: 78.90% top-1 vs 71.33% without
    # Items with a tail score HIGHER, and gold sits inside the tail only 12.4%
    # (e3_32b) / 11.5% (e4_32b_routed) of the time. A model that names four
    # options and stops has usually decided; the tail is a CONFIDENCE SIGNAL,
    # not chance being shipped as a ranking. Do not "repair" it.
    #
    # The one case that IS damage is a vote that named fewer than two options —
    # then the whole row is padding. That is what --parse-repair re-asks, and
    # what the identity-permutation rate below measures. On the 2026 8-option
    # test set (1971 queries) the identity rate is 6.85% for 8b_v3 and 3.30% for
    # 32b_prod; the old "5.9%" figure came from a run that predates the current
    # files and does not reproduce from anything on disk.
    n_padded = 0
    for i in range(n):
        if i not in seen:
            order.append(i)
            n_padded += 1
    parse_order.last_padded = n_padded
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3-vl:8b-instruct")
    ap.add_argument("--samples", type=int, default=8, help="stage-1 option-blind samples")
    # The "68.2% vs 72.3% on 296 questions" claim that used to justify this flag
    # is RETRACTED: wrong dev set (4-option contiguous slice). Re-measured on the
    # 497-item 8-option subset, --no-draft is +2.62pp (67.00% -> 69.62%, b=36
    # c=23, p=0.1175) — the same magnitude and p as two byte-identical runs of
    # the same model, so it is UNRESOLVABLE at this n. The mechanism argument is
    # unchanged and is why the flag is still preferred: the draft anchors the
    # choice to the model's own free-text answer, and dropping it keeps the parts
    # that should help (option shuffling, self-consistency over the ranking).
    ap.add_argument("--no-draft", action="store_true",
                    help="skip stage 1; rank options directly with shuffling + votes")
    ap.add_argument("--cot", action="store_true",
                    help="chain-of-thought before a numbered ranking (the old "
                         "\"67.2%%\" is retracted: 4-option slice, never "
                         "re-measured at 8 options)")
    ap.add_argument("--adaptive", action="store_true",
                    help="route frames/audio by question type (see PLAN_RULES)")
    ap.add_argument("--escalate", type=int, default=0,
                    help="on low confidence, re-ask with this many frames (0 = off)")
    ap.add_argument("--recipe", choices=["v3", "v5"], default="v3",
                    help="v3 = lettered options, no shuffle, one call. Default "
                         "because it is what the production runs used; v3 vs v5 "
                         "was never measured on the 8-option set")
    ap.add_argument("--choose-votes", type=int, default=3, help="stage-2 ranking votes")
    ap.add_argument("--frames-per-sample", type=int, default=8)
    ap.add_argument("--transcripts", default=None)
    ap.add_argument("--transcript-mode", choices=["routed", "broadcast", "off"],
                    default="routed",
                    help="routed = only questions the audio gate fires on; "
                         "broadcast = every question with a usable transcript; off = none")
    ap.add_argument("--min-transcript-words", type=int, default=0,
                    help="drop transcripts shorter than this many words. A 1-3 word "
                         "transcript is usually a Whisper hallucination on non-speech "
                         "audio (one dev clip's entire transcript is \"I don\'t know.\").")
    ap.add_argument("--num-predict", type=int, default=96)
    ap.add_argument("--think", choices=["unset", "off", "on"], default="unset",
                    help="qwen3-family native thinking control. THIS RUNNER NEVER PASSED IT: "
                         "qwen3.5:122b is a thinking model, and 67.5%% of its frame-bearing "
                         "calls burned the entire token budget inside an unterminated <think> "
                         "block and emitted no RANKING line. Its recorded 51.3%% is an artifact "
                         "of that, not a measurement. Use 'off' for any thinking model. "
                         "'unset' preserves the behaviour every shipped run used.")
    ap.add_argument("--stage2-predict", type=int, default=512,
                    help="token budget for the RANKING call. 512 can be exhausted "
                         "at 8 options before the RANKING line is emitted.")
    # WHAT THIS FLAG IS WORTH, re-derived 2026-09-04 on the 497-item 8-option dev
    # subset (e2_A_shipped vs e2_B_repaired, identical but for this flag):
    #   top-1                  67.00% -> 67.20%   +0.20pp, b=21 c=20, p=1.0000
    #   identity permutation    5.63% ->  5.03%   28 items -> 25, 11% relative
    # It is NOT an accuracy improvement and it does NOT halve the identity rate.
    # It is kept because it costs one extra call on a handful of votes and
    # removes rows that are provably the released option order rather than a
    # ranking. Justify it that way, not with a score.
    ap.add_argument("--abstain", action="store_true",
                    help="inject an explicit 'cannot be determined from these frames' option "
                         "into stage 2 and record when the model ranks it first. The option is "
                         "STRIPPED before the run is written. See _ABSTAIN_TEXT.")
    ap.add_argument("--abstain-loop", type=int, default=0,
                    help="on abstention, RE-ASK up to this many extra times and accumulate the "
                         "Borda tally across attempts. The final answer is the argmax over the "
                         "RELEASED options only, so the system never outputs the abstention "
                         "option -- it is a routing signal, not an answer. Offering abstention "
                         "in a single pass costs -6.64pp (it consumes items the model would have "
                         "answered); looping is what makes the signal free.")
    ap.add_argument("--loop-all", action="store_true",
                    help="run EVERY attempt instead of stopping at the first that commits, and "
                         "let the accumulated tally arbitrate. The break-early version asks a "
                         "second view only when the first declines, so it never sees two views "
                         "of an item it answered confidently-but-wrongly. Frame views demonstrably "
                         "disagree -- swapping views fixes 14 and breaks 14 -- so the fixes are "
                         "real and the problem is SELECTION, not evidence. Accumulating every "
                         "view is the cheapest selector there is.")
    ap.add_argument("--vote-frames", choices=["random", "disjoint"], default="random",
                    help="how the choose_votes calls differ. 'random' draws a fresh subset per "
                         "vote (the behaviour of every run shipped before 2026-09-14). 'disjoint' "
                         "PARTITIONS the pool across votes, so the union is the whole pool: three "
                         "independent 8-of-16 draws cover ~14.6 frames in expectation and can miss "
                         "one entirely, while a partition guarantees all 16 are seen.")
    ap.add_argument("--loop-frames", choices=["same", "new"], default="new",
                    help="'new' resamples the frame subset each attempt; 'same' reuses it. "
                         "'same' is the CONTROL: it isolates pure resampling/temperature from "
                         "any frame effect, which every previous escalation arm in this project "
                         "confounded.")
    ap.add_argument("--loop-model", default=None,
                    help="switch to this model from attempt --loop-switch-at onward")
    ap.add_argument("--loop-switch-at", type=int, default=1,
                    help="attempt index (1-based extra attempt) at which --loop-model takes over")
    ap.add_argument("--dump-votes", default=None,
                    help="per-question JSONL with the FULL Borda tally, each vote's top-1, and the "
                         "rank-1/rank-2 margin. The run CSV keeps only the final order and an "
                         "elapsed-seconds column, so every confidence signal the system computes "
                         "is currently discarded -- vote disagreement and Borda margin have never "
                         "been observable from anything this project stored. They are the two most "
                         "promising router features after abstention.")
    ap.add_argument("--loop-log", default=None,
                    help="per-question JSON: attempts used, abstained-at-each, final source")
    ap.add_argument("--abstain-log", default=None,
                    help="write one JSON line per question with the abstention flag")
    ap.add_argument("--parse-repair", action="store_true",
                    help="on a failed parse, re-ask once with a letters-only prompt; "
                         "if it still fails, cast no vote rather than voting on "
                         "padding (+0.20pp top-1, p=1.000 — free, not a win)")
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--call-timeout", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--exploit-block-prior", action="store_true",
                    help="measurement only: bias toward options 0-3 (see VQA_MC_LEAK.md)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    _THINK = {"unset": None, "off": False, "on": True}[args.think]
    topics = load_topics(args.topics)
    if args.limit:
        topics = topics[:args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume and out_path.exists():
        counts = Counter()
        for r in csv.reader(open(out_path, newline="", encoding="utf-8")):
            if r:
                counts[r[0].strip()] += 1
        need = {t["Q_ID"]: len(t["options"]) for t in topics}
        done = {q for q, n in counts.items() if n >= need.get(q, 0) > 0}
        print(f"Resuming: {len(done)} complete questions")

    todo = [t for t in topics if t["Q_ID"] not in done]
    client = OllamaClient(model=args.model, timeout=args.call_timeout,
                          temperature=args.temperature)
    consensus = Consensus()
    transcripts = load_transcripts(args.transcripts) if args.transcripts else {}
    print(f"[MC v5] {len(todo)}/{len(topics)} questions | model={args.model} "
          f"| stage1={args.samples} blind samples | stage2={args.choose_votes} votes"
          + (f" | transcripts={len(transcripts)}" if transcripts else ""))

    n_err = n_escalated = 0
    # FRAME COVERAGE vs STAGE-1 SKIPS are two different things and used to be one
    # counter. n_novideo was incremented in the ELSE branch of
    # "if pool and not args.no_draft", so every --no-draft run reported
    # "N without frames" for all N questions even when every pool was full. That
    # produced a false "no video was read" alarm twice (e.g. logs/e3_32b.log says
    # "497 without frames" for a run in which all 497 pools held 16 frames and
    # which scored 72.03%). The counters below are disjoint and each measures one
    # thing:
    #   n_no_frames        pool was empty -> the model genuinely saw no video
    #   n_no_draft_stage1  stage 1 did not run (--no-draft, or no frames)
    #   n_framed_stage2    the ranking call was issued WITH frames attached
    #                      (this is the real frame coverage of the run)
    n_no_frames = n_no_draft_stage1 = n_framed_stage2 = 0
    n_injected = 0
    n_abstain = 0
    n_parse_fail = n_parse_fixed = 0
    plan_hits = Counter()
    t_run = time.time()
    with open(out_path, "a" if done else "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for i, t in enumerate(todo, 1):
            rng = random.Random(f"{args.seed}:{t['Q_ID']}")
            pool = frame_pool(args.frames, t["Video_ID"])
            if not pool:
                n_no_frames += 1
            opts = t["options"]
            n = len(opts)
            t0 = time.time()

            # adaptive plan: how much evidence does THIS question need?
            n_frames_q = args.frames_per_sample
            want_tr = False
            if args.adaptive:
                labels, n_frames_q, want_tr = plan_for(t["Question"],
                                                       args.frames_per_sample)
                plan_hits[",".join(labels)] += 1

            tr = ""
            # BROADCAST vs ROUTED IS NOW MEASURED, and the answer is "no
            # difference". On the 497-item 8-option dev subset with the 32B
            # model, routed 74.65% and broadcast 74.65% — the same number, 14
            # discordant items each way, p=1.0000 (e4_32b_routed.csv vs
            # e4_32b_broadcast.csv, re-derived 2026-09-04). The pre-registered
            # condition for building a broadcast run therefore FAILED and no
            # broadcast run was built. Routing is kept because it is cheaper,
            # not because it is better. The flag stays so the comparison can be
            # re-run if the transcripts or the gate change.
            gate = (args.transcript_mode == "broadcast"
                    or (args.transcript_mode == "routed"
                        and (want_tr or audio_route(t["Question"]) == "verbal")))
            if transcripts and args.transcript_mode != "off" and gate:
                tr = transcripts.get(t["Video_ID"]) or \
                     transcripts.get(Path(t["Video_ID"]).stem, "")
                if args.min_transcript_words and len(tr.split()) < args.min_transcript_words:
                    tr = ""
                if tr.strip():
                    n_injected += 1

            # ---- stage 1: what do we see, WITHOUT the options in view
            draft = ""
            if pool and not args.no_draft:
                cands = []
                p1 = (_PROMPT_ASR.format(transcript=tr, question=t["Question"])
                      if tr else _PROMPT.format(question=t["Question"]))
                for s in range(args.samples):
                    try:
                        raw = client.generate_multimodal_with_retry(
                            prompt=p1, images=sample_subset(pool, args.frames_per_sample, rng),
                            system=_SYSTEM, retries=2, num_predict=args.num_predict,
                            num_ctx=args.num_ctx, think=_THINK,
                            temperature=args.temperature)
                        cands.append(clean_answer(raw, t["Question"]))
                    except Exception:
                        n_err += 1
                ranked = consensus.rank([c for c in cands if c])
                draft = ranked[0][0] if ranked else ""
            else:
                # NOT "no video": with --no-draft this fires on every question,
                # and stage 2 below still sends frames. See the counter comment.
                n_no_draft_stage1 += 1

            # ---- stage 2: rank the options, shuffled so position cannot leak
            perm = list(range(n))
            if args.recipe == "v5":
                rng.shuffle(perm)      # v3 deliberately does NOT shuffle
            tally = [0.0] * n
            abstained = False
            if pool and (draft or args.no_draft):
                # every stage-2 call below passes sample_subset(pool, ...), so
                # reaching here means frames WERE sent to the model.
                n_framed_stage2 += 1
                numbered = "\n".join(f"{j+1}. {opts[perm[j]]}" for j in range(n))
                if args.recipe == "v3":
                    n_show = n + 1 if args.abstain else n
                    lettered = "\n".join(f"{LETTERS[j]}) {opts[perm[j]]}" for j in range(n))
                    if args.abstain:
                        lettered += f"\n{LETTERS[n]}) {_ABSTAIN_TEXT}"
                    ctx = (f"\nAudio transcript of the video:\n{tr.strip()}\n"
                           if tr and tr.strip() else "")
                    prompt = _V3_TMPL.format(
                        question=t["Question"], context=ctx, lettered=lettered,
                        n=n_show, first=LETTERS[0], last=LETTERS[n_show-1],
                        example=",".join(LETTERS[x] for x in range(min(n_show, 4))))
                else:
                    tmpl = (_CHOOSE_COT_TMPL if args.cot
                            else (_CHOOSE_NODRAFT_TMPL if args.no_draft else _CHOOSE_TMPL))
                    prompt = tmpl.format(
                        question=t["Question"], draft=draft, numbered=numbered, n=n,
                        example=",".join(str(x) for x in range(1, min(n, 4) + 1)))
                _attempts = 1 + max(0, args.abstain_loop)
                _alt_client = None
                _prev_imgs = None
                _n_att_used = 0
                for _att in range(_attempts):
                  _cli = client
                  if (args.loop_model and _att >= args.loop_switch_at):
                      if _alt_client is None:
                          _alt_client = OllamaClient(model=args.loop_model,
                                                     temperature=args.temperature)
                      _cli = _alt_client
                  _att_abstained = False
                  # PER-VOTE FRAME SAMPLING. Hoisting this out of the vote loop --
                  # which commit 486f7cc did while adding the cascade -- makes all
                  # choose_votes calls see IDENTICAL images, so self-consistency
                  # collapses to temperature jitter alone (vote 0 runs at 0.0, the
                  # rest at --temperature). Every run shipped before 2026-09-14 drew
                  # a fresh subset per vote; the cascade experiments did not, and
                  # that is a confound in them. Restored, with the cascade's
                  # same-frames option preserved as an explicit request.
                  _base = (_prev_imgs if (args.loop_frames == "same" and _att > 0
                                          and _prev_imgs is not None) else None)
                  _parts = None
                  if args.vote_frames == "disjoint" and pool and _base is None:
                      # Partition the pool across votes instead of drawing each
                      # independently. Three independent draws of 8 from 16 cover
                      # ~14.6 of 16 in expectation and can miss a frame entirely;
                      # a partition guarantees every frame is seen by some vote.
                      # Cover-then-top-up: deal the pool round-robin so every frame
                      # is owned by some vote, then refill each vote back to the
                      # full budget from the frames it does not own. Coverage is
                      # guaranteed AND no vote is starved -- a plain partition would
                      # cut 8 frames to 5 or 6, trading one confound for another.
                      _sh = list(pool)
                      rng.shuffle(_sh)
                      _k = max(1, min(n_frames_q, len(_sh)))
                      _parts = []
                      for i in range(args.choose_votes):
                          own = _sh[i::args.choose_votes]
                          rest = [x for x in _sh if x not in own]
                          rng.shuffle(rest)
                          _parts.append((own + rest)[:_k] if len(own) < _k else own[:_k])
                  for v in range(args.choose_votes):
                      if _base is not None:
                          _imgs = _base
                      elif _parts is not None:
                          _imgs = _parts[v]
                      else:
                          _imgs = sample_subset(pool, n_frames_q, rng)
                      if _prev_imgs is None or _base is None:
                          _prev_imgs = _imgs
                      try:
                          raw = _cli.generate_multimodal_with_retry(
                              prompt=prompt,
                              images=_imgs,
                              system=_CHOOSE_SYSTEM, retries=2,
                              num_predict=(args.stage2_predict
                                           if (args.cot or args.recipe == "v3") else 64),
                              num_ctx=args.num_ctx, think=_THINK,
                              temperature=0.0 if v == 0 else args.temperature)
                          _n_parse = n + 1 if (args.abstain and args.recipe == "v3") else n
                          parsed = (parse_letters(raw, _n_parse) if args.recipe == "v3"
                                    else parse_order(raw, n))
                          if args.abstain and args.recipe == "v3":
                              # index n is the injected option. Record whether the
                              # model put it FIRST, then drop it: the submission
                              # must rank the released options and nothing else.
                              if parsed and parsed[0] == n:
                                  n_abstain += 1
                                  abstained = True
                                  _att_abstained = True
                              parsed = [j for j in parsed if j != n]
                          # Read the counter off the parser ACTUALLY used. Under
                          # --recipe v5 parse_order runs and parse_letters is never
                          # called, so reading parse_letters here reported 0 parse
                          # failures forever — a lying counter of the same class as
                          # the n_novideo bug, sitting in the summary that reports it.
                          _parser = parse_letters if args.recipe == "v3" else parse_order
                          padded = getattr(_parser, "last_padded", 0)
                          # A vote that named FEWER THAN TWO options is not a
                          # ranking; the rest is ascending padding and voting on it
                          # is voting on the released option order. Re-ask once,
                          # letters only, before letting it into the tally.
                          if args.parse_repair and args.recipe == "v3" and padded >= n - 1:
                              n_parse_fail += 1
                              rprompt = _REPAIR_TMPL.format(
                                  question=t["Question"], context=ctx,
                                  lettered=lettered, n=n,
                                  first=LETTERS[0], last=LETTERS[n - 1])
                              raw_r = _cli.generate_multimodal_with_retry(
                                  prompt=rprompt,
                                  images=_imgs,
                                  system=_CHOOSE_SYSTEM, retries=1,
                                  num_predict=64, num_ctx=args.num_ctx,
                                  think=_THINK, temperature=0.0)
                              reparsed = parse_letters(raw_r, n)
                              if getattr(parse_letters, "last_padded", 0) < n - 1:
                                  parsed = reparsed
                                  n_parse_fixed += 1
                              else:
                                  continue        # still unparseable: cast NO vote
                          elif padded >= n - 1:
                              n_parse_fail += 1
                          for rank_pos, j in enumerate(parsed):
                              tally[perm[j]] += 1.0 / (rank_pos + 1)   # Borda-ish, MRR-shaped
                      except Exception:
                          n_err += 1

                  _n_att_used = _att + 1
                  if not _att_abstained and not args.loop_all:
                      break     # this attempt committed to a real option; stop paying

            if args.loop_log:
                with open(args.loop_log, "a", encoding="utf-8") as _lf:
                    _lf.write(json.dumps({"q": t["Q_ID"], "attempts": _n_att_used,
                                          "abstained_final": _att_abstained}) + "\n")

            # ESCALATION. A single ranking gives no confidence signal, so we ask
            # once more with a DIFFERENT frame subset and treat agreement on
            # top-1 as confidence. Disagreement means the two views of the video
            # support different answers — exactly the case where more evidence
            # should help — so the tie-break pass gets --escalate frames. Cost
            # lands only on the uncertain minority.
            if args.escalate and pool and n > 1 and any(tally):
                first = max(range(n), key=lambda k: tally[k])
                try:
                    raw2 = client.generate_multimodal_with_retry(
                        prompt=prompt, images=sample_subset(pool, n_frames_q, rng),
                        system=_CHOOSE_SYSTEM, retries=1,
                        num_predict=(args.stage2_predict
                                     if (args.cot or args.recipe == "v3") else 64),
                        num_ctx=args.num_ctx, think=_THINK,
                        temperature=args.temperature)
                    chk = (parse_letters(raw2, n) if args.recipe == "v3"
                           else parse_order(raw2, n))
                    if chk and perm[chk[0]] != first:          # the two views disagree
                        n_escalated += 1
                        raw3 = client.generate_multimodal_with_retry(
                            prompt=prompt,
                            images=sample_subset(pool, min(args.escalate, len(pool)), rng),
                            system=_CHOOSE_SYSTEM, retries=1,
                            num_predict=512 if (args.cot or args.recipe == "v3") else 64,
                            num_ctx=args.num_ctx, think=_THINK, temperature=0.0)
                        deep = (parse_letters(raw3, n) if args.recipe == "v3"
                                else parse_order(raw3, n))
                        for rank_pos, j in enumerate(deep):     # deep pass gets 2x weight
                            tally[perm[j]] += 2.0 / (rank_pos + 1)
                    else:
                        for rank_pos, j in enumerate(chk):
                            tally[perm[j]] += 1.0 / (rank_pos + 1)
                except Exception:
                    n_err += 1

            if args.exploit_block_prior:      # measurement only, off by default
                for k in range(n):
                    if k < 4:
                        tally[k] += 1e-3

            if args.dump_votes:
                _srt = sorted(range(n), key=lambda k: -tally[k])
                _m = (tally[_srt[0]] - tally[_srt[1]]) if n > 1 else 0.0
                with open(args.dump_votes, "a", encoding="utf-8") as _vf:
                    _vf.write(json.dumps({
                        "q": t["Q_ID"],
                        "tally": [round(x, 4) for x in tally],
                        "margin": round(_m, 4),
                        "top1": opts[_srt[0]] if n else "",
                        "abstained": bool(abstained),
                        "attempts": _n_att_used,
                        "n_opt": n,
                    }) + "\n")
            if args.abstain_log:
                with open(args.abstain_log, "a", encoding="utf-8") as _af:
                    _af.write(json.dumps({"q": t["Q_ID"], "abstained": abstained}) + "\n")
            order = sorted(range(n), key=lambda k: -tally[k])
            elapsed = time.time() - t0
            for rank, k in enumerate(order, 1):
                w.writerow([t["Q_ID"], t["Video_ID"], rank, opts[k], f"{elapsed:.2f}"])
            fh.flush()

            if i % 25 == 0 or i == len(todo):
                rate = (time.time() - t_run) / i
                print(f"  [{i}/{len(todo)}] {rate:.1f}s/q — ETA {rate*(len(todo)-i)/3600:.1f}h "
                      f"(framed={n_framed_stage2}/{i}, noframes={n_no_frames}, "
                      f"s1skip={n_no_draft_stage1}, escalated={n_escalated}, err={n_err}, "
                      f"parsefail={n_parse_fail}, fixed={n_parse_fixed}, tr={n_injected}, "
                  f"abstain={n_abstain})",
                      flush=True)

    print(f"Done: {out_path} ({len(todo)} questions, {n_err} errors)")
    print(f"  frame coverage: {n_framed_stage2}/{len(todo)} questions were ranked WITH "
          f"frames attached; {n_no_frames} had no frames on disk "
          f"({n_no_frames / len(todo) * 100:.1f}%)." if todo else "  frame coverage: nothing to do.")
    print(f"  stage 1 (option-blind draft) skipped on {n_no_draft_stage1}/{len(todo)} "
          + ("— all of them, because --no-draft is on. This is NOT a frame count."
             if args.no_draft else "— i.e. the questions with no frames."))
    print(f"  parse failures: {n_parse_fail} votes, repaired {n_parse_fixed}. "
          f"An unrepaired failure casts no vote. For reference, the identity-"
          f"permutation rate over the 1971 eight-option 2026 queries is 3.30% "
          f"for 32b_prod and 6.85% for 8b_v3 (measured 2026-09-04); a run well "
          f"above that band is emitting the released option order, not a ranking. "
          f"An ascending TAIL is a different thing and is not a defect — see the "
          f"parse_letters comment.")


if __name__ == "__main__":
    main()

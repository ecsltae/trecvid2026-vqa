"""
Build small, TARGETED dev slices so an experiment costs minutes, not days.

THE PROBLEM THIS SOLVES. A full 2443-question run is ~9h on the 8B and ~30h on
the 125B. Iterating a strategy that way means two days per idea, which is how a
project ends up shipping the first thing that worked. But a small RANDOM slice
is not the answer either: MC differences of ~5pp need n≈300 just to reach 2 SE,
so a random 100 resolves nothing.

The way out is to stop testing interventions on random samples. An intervention
usually targets a KIND of question — OCR helps on-screen text, transcripts help
speech, more frames help brief events. Measured on the slice it targets, the
effect is large and n=40-120 is plenty; measured on a random sample it is
diluted by the 80-98% of questions it cannot possibly affect, and vanishes.

Slice sizes here are chosen so that a LARGE within-slice effect (say 30pp, which
is what fixing a whole failure mode looks like) clears 2 SE, while an effect too
small to matter stays invisible. Do not use these slices to chase 2pp.

    slice      2025 dev   what it is for
    text          41      on-screen captions/signs — OCR, frame density
    audio        140      speech/music — Whisper transcripts
    counting     241      "how many" — frame density, number formatting
    temporal     449      "after/before/then" — frame density, ordering
    general      120      random control, to catch a change that hurts elsewhere

Always run the `general` slice alongside a targeted one: an intervention that
wins on its own slice and loses everywhere else is a net loss, and only the
control will show it.

Usage:
    python eval/vqa/dev_slices.py --out /tmp/slices --cap 120
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

PATTERNS = {
    "text": r"\b(word|words|caption|subtitle|text|written|writing|says on|sign|"
            r"title|letters|spelled|logo|brand|name of the (song|movie|show))\b",
    # inflected — see run_vqa_mc_v5.PLAN_RULES; the bare-stem version silently
    # defined every "audio slice" we ever measured on the wrong population
    "audio": r"\b(say|says|said|saying|speak|speaks|speaking|spoken|talk|talks|talking|sing|sings|singing|sung|song|songs|lyric|lyrics|music|musical|sound|sounds|hear|hears|heard|hearing|voice|voices|noise|noises|instrument|instruments)\b",
    "counting": r"^\s*how (many|much)\b",
    "temporal": r"\b(after|before|then|first|last|finally|begin|end|next|"
                r"sequence|order)\b",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", default="data/vqa/topics/vqa_2025_mc_topics.csv")
    ap.add_argument("--frames", default="data/vqa/frames_pool16_512")
    ap.add_argument("--qrels", default="data/vqa/qrels_mc.csv")
    ap.add_argument("--out", default="/tmp/slices")
    ap.add_argument("--cap", type=int, default=120,
                    help="max items per slice — keeps a 125B run under ~1.5h")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rows = [r for r in csv.reader(open(args.topics, newline="", encoding="utf-8"))
            if len(r) >= 7]
    have = {p.name for p in Path(args.frames).iterdir()
            if p.is_dir() and any(p.glob("frame_*.jpg"))}
    gold = {r["Q_ID"].strip() for r in csv.DictReader(open(args.qrels, newline="",
                                                           encoding="utf-8"))}
    # only keep items we can actually score AND actually show the model
    rows = [r for r in rows
            if r[0].strip() in gold and r[1].strip().strip('"') in have]
    print(f"{len(rows)} scoreable dev questions with frames")

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assigned = set()
    summary = {}

    for name, pat in PATTERNS.items():
        hits = [r for r in rows if re.search(pat, r[2].strip().strip('"'), re.I)]
        rng.shuffle(hits)
        sel = hits[:args.cap]
        assigned |= {r[0].strip() for r in sel}
        p = out / f"{name}.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(sel)
        summary[name] = len(sel)
        # power note: 2 SE on a within-slice accuracy difference
        se = (0.25 / max(len(sel), 1)) ** 0.5
        print(f"  {name:<9} {len(sel):4d} of {len(hits):4d} matching   "
              f"resolves >= {2*se*100:.0f}pp within-slice")

    general = [r for r in rows if r[0].strip() not in assigned]
    rng.shuffle(general)
    sel = general[:args.cap]
    with open(out / "general.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(sel)
    se = (0.25 / max(len(sel), 1)) ** 0.5
    print(f"  {'general':<9} {len(sel):4d} of {len(general):4d} remaining   "
          f"resolves >= {2*se*100:.0f}pp  <- always run this as the control")
    summary["general"] = len(sel)

    (out / "slices.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    total = sum(summary.values())
    print(f"\n{total} questions total across {len(summary)} slices")
    print(f"  8B  @ ~13 s/q : {total*13/3600:.1f} h for ALL slices")
    print(f"  125B @ ~45 s/q: {total*45/3600:.1f} h for ALL slices "
          f"(a single targeted slice is {args.cap*45/3600:.1f} h)")
    print(f"  full 2443-question run for comparison: "
          f"{2443*13/3600:.0f} h / {2443*45/3600:.0f} h")


if __name__ == "__main__":
    main()

"""
Score a human arbitration session, and decide whether to trust it.

CALIBRATION MODE is a gate, not a formality. It answers one question before any
2026 label is used: **on the contested items, is the human actually better than
the model?** In this project's TREC RAG round the equivalent votes turned out to
be anti-correlated with quality and indistinguishable from "prefer the longer
answer", and nobody found out until after a conclusion had been published. MC
has gold labels, so we can simply check.

Reported:
  * human accuracy on the contested items
  * the model's accuracy on the SAME items (the bar to beat — measured at ~49%)
  * the longest-option heuristic on the same items, because a human who is
    unknowingly picking the longest option is not adding information
  * a binomial test of human vs model, and the projected gain if the human
    arbitrated a given number of 2026 items

Usage:
    python systems/vqa/score_arbitration.py \
        --key curation/mc_calibration.key.json \
        --votes curation/my_votes.json \
        --runs /tmp/mc_v3_dev300.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from math import comb
from pathlib import Path


def binom_p(k, n, p=0.5):
    """Two-sided exact binomial tail."""
    if n == 0:
        return 1.0
    probs = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return min(1.0, sum(x for x in probs if x <= probs[k] + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, help="the .key.json sidecar")
    ap.add_argument("--votes", required=True, help="JSON pasted out of the page")
    ap.add_argument("--runs", nargs="*", default=[], help="model runs, to get the bar")
    ap.add_argument("--project-to", type=int, default=400,
                    help="how many 2026 items a human might arbitrate")
    ap.add_argument("--test-size", type=int, default=2443)
    args = ap.parse_args()

    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    votes = json.loads(Path(args.votes).read_text(encoding="utf-8")).get("votes", {})
    items = key["items"]

    items_early = items  # noqa: F841  (readability: items is already bound above)
    model_top = {}
    if args.runs:
        rows = defaultdict(list)
        for p in args.runs[:1]:
            for r in csv.reader(open(p, newline="", encoding="utf-8")):
                if len(r) >= 4:
                    try:
                        rows[r[0].strip()].append((int(r[2]), r[3]))
                    except ValueError:
                        pass
        for q, v in rows.items():
            top = sorted(v, key=lambda x: x[0])[0][1]
            # v3's MC path emits a bare letter; v5 emits option text. Resolve
            # letters against this item's options or every comparison below
            # silently fails and the model scores a fake 0%.
            meta = items.get(q)
            if meta and len(top.strip()) == 1 and top.strip().upper() in "ABCD":
                opts = meta["options"]
                i = "ABCD".index(top.strip().upper())
                if i < len(opts):
                    top = opts[i]
            model_top[q] = top

    n = hum = mod = longest = unsure = 0
    audio_items = []
    audio_hum = audio_mod = 0
    disagree_with_model = 0
    for qid, v in votes.items():
        meta = items.get(qid)
        if not meta or meta.get("gold") is None:
            continue
        # votes are {"p": pick, "a": needed-sound}; older sessions stored a bare int
        needed_audio = bool(v.get("a")) if isinstance(v, dict) else False
        pv = v.get("p") if isinstance(v, dict) else v
        if pv is None or int(pv) < 0:
            unsure += 1
            continue
        if needed_audio:
            audio_items.append(qid)
        # the page showed options in a shuffled order; map the click back
        picked = meta["options"][meta["perm"][int(pv)]]
        gold = meta["gold"]
        n += 1
        hum += picked == gold
        longest += max(meta["options"], key=len) == gold
        mt = model_top.get(qid)
        if mt is not None:
            mod += mt == gold
            disagree_with_model += mt != picked
            if needed_audio:
                audio_hum += picked == gold
                audio_mod += mt == gold

    if not n:
        raise SystemExit("No scoreable votes (calibration needs the gold sidecar).")

    ha, ma, la = hum / n, (mod / n if model_top else float("nan")), longest / n
    print(f"Scored {n} contested items ({unsure} marked can't-tell)\n")
    print(f"  human accuracy          {ha:6.1%}")
    if model_top:
        print(f"  model accuracy (same)   {ma:6.1%}   <- the bar")
    print(f"  longest-option heuristic{la:7.1%}   <- if the human is near this, be suspicious")
    print(f"  human differed from model on {disagree_with_model}/{n}")

    p_vs_chance = binom_p(hum, n, 1.0 / max(len(next(iter(items.values()))['options']), 2))
    print(f"\n  vs chance: p = {p_vs_chance:.4f}")

    # Decompose: an advantage that comes only from HEARING the clip is fixable by
    # giving the model a transcript (free, scales to all 2443) rather than by
    # spending human attention. Only the residual justifies arbitration.
    na = len(audio_items)
    if na and model_top:
        print(f"\n  of these, {na} were marked 'needed SOUND':")
        print(f"     human {audio_hum}/{na}   model {audio_mod}/{na}")
        rest = n - na
        if rest:
            print(f"     on the other {rest} (visual only): "
                  f"human {(hum-audio_hum)/rest:.1%}  model {(mod-audio_mod)/rest:.1%}")
        print("     -> audio-driven gap is fixable with transcripts; the visual-only "
              "gap is what actually needs a human.")

    if model_top:
        gain_per = ha - ma
        proj = gain_per * min(args.project_to, args.test_size) / args.test_size
        print(f"\n  gain per arbitrated item : {gain_per:+.1%}")
        print(f"  projected on {args.project_to} of {args.test_size} 2026 items: "
              f"{proj:+.2%} top-1 overall")
        if ha <= ma:
            print("\n  VERDICT: DO NOT USE. The human is not beating the model on the "
                  "items that matter; arbitration would cost accuracy.")
        elif ha - la < 0.05:
            print("\n  VERDICT: SUSPECT. Human accuracy is close to the longest-option "
                  "heuristic — the votes may carry no visual information. Check a few by hand.")
        else:
            print("\n  VERDICT: USE IT. Arbitrate the most contested 2026 items and let "
                  "the human pick override the model.")


if __name__ == "__main__":
    main()

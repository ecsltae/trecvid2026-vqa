#!/usr/bin/env python3
"""
Build a STRATIFIED 8-OPTION dev set from the 2025 MC data.

Why this exists
---------------
Every MC architecture decision we have made was measured on `Q_ID 1..300` of the
2025 set: a CONTIGUOUS head slice, 4 options per item. Two things are wrong with
that as an instrument for choosing a 2026 system:

  1. Option count. The 2026 test set is 8-option for 1971 of 2443 queries
     (4-option for the other 472). Chance is 12.5%, not 25%, and the shape of the
     decision changes with it.
  2. The slice. Measured here: the 30b-8b delta on Q<=300 sits at the 0.00
     percentile of all 1683 contiguous 296-item windows -- it is the single most
     8b-favouring window in the data. 8b "wins" there (-1.35pp for 30b) and loses
     by 3.98pp on the other 1682 items (McNemar p=1.4e-4).

So: sample from ALL scoreable items, stratified to the 2026 question-type mix,
and pad every item from 4 options to 8.

Distractor policy
-----------------
Random distractors from unrelated questions make the task trivially easy and the
measurement worthless. Candidates are drawn, in priority order, from
  (a) options of OTHER questions about the SAME video,
  (b) options of other questions of the SAME question type,
  (c) options of other questions (fallback),
and are LENGTH-MATCHED to the item's existing options, because the longest-option
heuristic scores 47.2% at 4 options and a set whose distractors are systematically
short would simply re-encode that heuristic.

Guards: never equal to an existing option (casefolded), never token-Jaccard > 0.6
with the gold, never a duplicate within the item.
"""
import argparse, csv, json, random, re
from collections import defaultdict, Counter

SKIP = {18,27,62,222,448,507,584,629,903,1007,1159,1260,1330,1381,1554,1691,1755,1822}

def qtype(q):
    s = q.lower().strip()
    if re.search(r'\bhow many\b|\bhow much\b|\bcount\b|\bnumber of\b', s): return 'count'
    if re.search(r'\bsay|said|hear|heard|music|song|sound|speak|talk|voice|sing|audio|lyric', s): return 'audio'
    if re.search(r'\bbefore\b|\bafter\b|\bwhen\b|\bfirst\b|\bthen\b|\bnext\b|\bend of\b|\bbegin', s): return 'temporal'
    if s.startswith('why') or ' why ' in s: return 'why'
    if re.match(r'^(is|are|does|do|did|was|were|can|could|will|would|has|have|had)\b', s): return 'yesno'
    if s.startswith('what') or ' what ' in s: return 'what'
    if s.startswith('how'): return 'how'
    return 'other'

def norm(s):  return " ".join(s.split()).casefold()
def toks(s):  return set(re.findall(r"[a-z0-9]+", s.lower()))
def jac(a, b):
    A, B = toks(a), toks(b)
    return len(A & B) / max(len(A | B), 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", default="data/vqa/testing_mc_2025.csv")
    ap.add_argument("--qrels",  default="data/vqa/qrels_mc.csv")
    ap.add_argument("--mc2026", default="data/vqa/2026/task2/vqa.2026.testingDataset.Task2_v2.json")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--out-prefix", default="data/vqa/dev8/dev8")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    items = {}
    for r in csv.reader(open(args.topics, newline="", encoding="utf-8")):
        if len(r) >= 7:
            items[r[0].strip()] = {"vid": r[1].strip(), "q": r[2].strip(),
                                   "opts": [c.strip() for c in r[3:7]]}
    gold = {}
    for row in csv.DictReader(open(args.qrels, newline="", encoding="utf-8")):
        q, a = (row.get("Q_ID") or "").strip(), (row.get("Answer") or "").strip()
        if q in items and a and a[0].upper() in "ABCD":
            gold[q] = items[q]["opts"]["ABCD".index(a[0].upper())]

    universe = [q for q in items
                if q in gold and q.isdigit() and int(q) not in SKIP
                and len(set(map(norm, items[q]["opts"]))) == 4]
    print(f"universe: {len(universe)} scoreable 4-distinct-option items "
          f"(of {len(items)} topics, {len(gold)} with gold)")

    # --- target mix from the real 2026 MC test set ---
    T26 = json.load(open(args.mc2026))
    mix = Counter(qtype(t["Question"]) for t in T26)
    tot26 = sum(mix.values())

    by_type = defaultdict(list)
    for q in universe:
        by_type[qtype(items[q]["q"])].append(q)
    for v in by_type.values():
        rng.shuffle(v)

    # largest-remainder allocation, capped by availability
    quota, rem = {}, []
    for t in sorted(mix):
        exact = args.n * mix[t] / tot26
        quota[t] = min(int(exact), len(by_type[t]))
        rem.append((exact - int(exact), t))
    for _, t in sorted(rem, reverse=True):
        if sum(quota.values()) >= args.n: break
        if quota[t] < len(by_type[t]): quota[t] += 1
    chosen = [q for t in sorted(quota) for q in by_type[t][:quota[t]]]
    print(f"stratified sample: {len(chosen)} items")

    # --- distractor pools ---
    by_video, by_qtype, allopts = defaultdict(list), defaultdict(list), []
    for q in universe:
        for o in items[q]["opts"]:
            by_video[items[q]["vid"]].append((q, o))
            by_qtype[qtype(items[q]["q"])].append((q, o))
            allopts.append((q, o))

    out, stats = [], Counter()
    for q in chosen:
        it, g = items[q], gold[q]
        have = {norm(o) for o in it["opts"]}
        lens = [len(o) for o in it["opts"]]
        # gold's current rank by length among the 4 originals (1 = longest)
        r0 = sorted((len(o) for o in it["opts"]), reverse=True).index(len(g)) + 1
        # Only ADDING options, so the gold's final length-rank can only be r0..r0+4.
        # Draw uniformly on that feasible range (rejection on 1..8) — the marginal
        # cannot be exactly uniform with 4 additions; the residual is reported.
        need_long = rng.randint(0, 4)
        for _ in range(50):
            R = rng.randint(1, 8)
            if r0 <= R <= r0 + 4:
                need_long = R - r0
                break
        n_long = 0
        picked = []
        for tier, pool in (("same_video", by_video[it["vid"]]),
                           ("same_qtype", by_qtype[qtype(it["q"])]),
                           ("global",     allopts)):
            if len(picked) == 4: break
            cands = [o for (src, o) in pool if src != q and norm(o) not in have]
            rng.shuffle(cands)
            for o in cands:
                if len(picked) == 4: break
                if norm(o) in have:            # re-check: cands was built before picks
                    continue
                if jac(o, g) > 0.6:            # never a paraphrase of the gold
                    continue
                # Length policy. The gold's rank by length must be UNIFORM over the
                # 8 options. Left alone, the gold is the longest option 48% of the
                # time (the longest-option heuristic scores 47-48% at 4 options);
                # forcing every distractor longer inverts the leak just as badly.
                # So: draw a target rank, and take exactly `need_long` distractors
                # longer than the gold and the rest shorter, length-matched.
                longer = len(o) > len(g)
                if longer and n_long >= need_long:
                    continue
                if not longer and (len(picked) - n_long) >= (4 - need_long):
                    continue
                if not longer:
                    target = rng.choice(lens)
                    if not (0.5 * target <= len(o) <= 1.7 * target):
                        continue
                picked.append(o); have.add(norm(o)); stats[tier] += 1
                n_long += longer
        if len(picked) < 4:                     # relax length match rather than emit <8
            cands = [o for (src, o) in allopts if src != q and norm(o) not in have
                     and jac(o, g) <= 0.6]
            rng.shuffle(cands)
            for o in cands:
                if len(picked) == 4: break
                if norm(o) in have: continue
                picked.append(o); have.add(norm(o)); stats["relaxed"] += 1
        if len(picked) < 4:
            stats["DROPPED"] += 1; continue
        opts8 = it["opts"] + picked
        rng.shuffle(opts8)                      # gold lands in a random position
        out.append({"Q_ID": int(q), "Video_ID": it["vid"], "Question": it["q"],
                    "Options": opts8, "gold": g, "qtype": qtype(it["q"])})

    out.sort(key=lambda d: d["Q_ID"])
    with open(f"{args.out_prefix}_topics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for d in out:
            w.writerow([d["Q_ID"], d["Video_ID"], d["Question"]] + d["Options"])
    with open(f"{args.out_prefix}_qrels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["Q_ID", "Answer", "Grade"])
        for d in out:
            w.writerow([d["Q_ID"], "ABCDEFGH"[d["Options"].index(d["gold"])], 1])
    json.dump(out, open(f"{args.out_prefix}.json", "w"), indent=1)
    print(f"distractor sources: {dict(stats)}")
    print(f"wrote {len(out)} items -> {args.out_prefix}_{{topics.csv,qrels.csv}}, {args.out_prefix}.json")

if __name__ == "__main__":
    main()

"""
Learned escalation router: can a cheap model's own uncertainty decide when to
call an expensive one?

THE ARCHITECTURE. Tier 1 answers everything with an 8B. A router reads signals
that tier 1 already produced -- whether it took an injected abstention option,
its Borda rank1-rank2 margin, and how much independent 8B runs disagree -- and
escalates a subset to a 32B (and optionally a 122B). The answer for an escalated
item comes from the larger model; everything else keeps tier 1's answer.

WHY THE ABSTENTION OPTION MUST NOT BE IN THE ANSWERING PASS. Offering it costs
-6.64pp, because it competes with the real options and consumes items the model
would otherwise have answered. The run that produces the ROUTING SIGNAL and the
run that produces the TIER-1 ANSWER are therefore different runs. Conflating
them is what made the first cascade lose to plain baseline on every arm.

WHY THIS SCRIPT IS STRICT ABOUT FITTING. A hand-picked conjunction
("abstain AND disagree>=2") looked like it matched full-32B at 16% of the cost.
A split-half test -- choose the policy on one half, score on the other -- showed
that picking the policy per-dataset costs -0.51pp, i.e. the thresholds were
fitted to the same 497 items they were scored on. Everything here is therefore
fitted on a train split and reported on a test split it never saw, and every
policy is compared against RANDOM ESCALATION OF THE SAME COUNT, which is the
control that decides whether the signal carries information at all.

Usage:
    python eval/vqa/router_analysis.py --votes /tmp/rt8_{0,1,2}.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
from collections import defaultdict
from math import comb


def binom_p(k, n):
    if n == 0:
        return 1.0
    pr = [comb(n, i) * 0.5 ** n for i in range(n + 1)]
    return min(1.0, sum(x for x in pr if x <= pr[k] + 1e-12))


def load_run(path, topics):
    rows = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) >= 4:
                try:
                    rank = int(r[2])
                except ValueError:
                    continue
                rows[r[0].strip()].append((rank, r[3]))
    return {q: [a for _, a in sorted(v)] for q, v in rows.items() if v}


def opts_of(t):
    return [o["option"] if isinstance(o, dict) else o for o in t["Options"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", default="/tmp/dev8full_topics.json")
    ap.add_argument("--qrels", default="data/vqa/dev8/dev8full_qrels.csv")
    ap.add_argument("--votes", nargs="+", required=True, help="--dump-votes JSONL from the tier-1 signal run")
    ap.add_argument("--tier1", nargs="+", required=True, help="CLEAN 8b runs (no abstention option)")
    ap.add_argument("--tier2", nargs="+", required=True, help="32b runs")
    ap.add_argument("--tier3", nargs="*", default=[], help="optional 122b runs")
    ap.add_argument("--sec1", type=float, default=11.8)
    ap.add_argument("--sec2", type=float, default=35.2)
    ap.add_argument("--sec3", type=float, default=43.6)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    topics = {str(x["Q_ID"]): x for x in json.load(open(args.topics))}
    gold = {}
    for r in csv.DictReader(open(args.qrels, newline="", encoding="utf-8")):
        q = (r.get("Q_ID") or "").strip()
        a = (r.get("Answer") or "").strip()
        if q in topics:
            oo = opts_of(topics[q])
            L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(oo)]
            if a and a[0].upper() in L:
                gold[q] = oo[L.index(a[0].upper())]
            elif a in oo:
                gold[q] = a

    sig = {}
    for p in args.votes:
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            sig[str(d["q"])] = d

    t1 = [load_run(p, topics) for p in args.tier1]
    t2 = [load_run(p, topics) for p in args.tier2]
    t3 = [load_run(p, topics) for p in args.tier3] if args.tier3 else []

    common = sorted(
        [q for q in set.intersection(*[set(r) for r in t1 + t2 + (t3 or [t1[0]])])
         if q in gold and q in sig],
        key=lambda x: int(x) if x.isdigit() else 0)
    if len(common) < 50:
        raise SystemExit(f"only {len(common)} usable questions — nothing to fit")

    # ---- features, all from tier 1's own output
    feats = {}
    for q in common:
        d = sig[q]
        picks = [r[q][0] for r in t1] + [d.get("top1", "")]
        picks = [p for p in picks if p]
        modal = max(set(picks), key=picks.count) if picks else ""
        tal = sorted(d.get("tally", []), reverse=True)
        tot = sum(tal) or 1.0
        feats[q] = {
            "abstain": 1.0 if d.get("abstained") else 0.0,
            "margin": float(d.get("margin", 0.0)),
            "margin_norm": (tal[0] - tal[1]) / tot if len(tal) > 1 else 0.0,
            "top_share": (tal[0] / tot) if tal else 0.0,
            "disagree": float(len(set(picks)) - 1),
            "n_opt": float(d.get("n_opt", len(opts_of(topics[q])))),
        }
    FN = ["abstain", "margin", "margin_norm", "top_share", "disagree", "n_opt"]

    def correct(run, q):
        return run[q][0] == gold[q]

    def score(sel2, sel3, items, i1, i2, i3=0):
        r1, r2 = t1[i1 % len(t1)], t2[i2 % len(t2)]
        r3 = t3[i3 % len(t3)] if t3 else None
        n = 0
        for q in items:
            r = r3 if (r3 and q in sel3) else (r2 if q in sel2 else r1)
            n += correct(r, q)
        return n / len(items)

    def cost(n2, n3, n):
        return (args.sec1 * n + args.sec2 * n2 + args.sec3 * n3) / 3600.0

    rng = random.Random(args.seed)

    # ---- logistic regression on "will tier 1 be wrong", fitted per fold
    def fit_logistic(train, ycol):
        w = {k: 0.0 for k in FN}
        b = 0.0
        mu = {k: st.mean([feats[q][k] for q in train]) for k in FN}
        sd = {k: (st.pstdev([feats[q][k] for q in train]) or 1.0) for k in FN}
        for _ in range(400):
            gw = {k: 0.0 for k in FN}
            gb = 0.0
            for q in train:
                z = b + sum(w[k] * (feats[q][k] - mu[k]) / sd[k] for k in FN)
                p = 1 / (1 + pow(2.718281828, -max(-30, min(30, z))))
                e = p - ycol[q]
                for k in FN:
                    gw[k] += e * (feats[q][k] - mu[k]) / sd[k]
                gb += e
            for k in FN:
                w[k] -= 0.05 * gw[k] / len(train)
            b -= 0.05 * gb / len(train)
        return lambda q: b + sum(w[k] * (feats[q][k] - mu[k]) / sd[k] for k in FN), w

    folds = [[] for _ in range(args.folds)]
    sh = common[:]
    rng.shuffle(sh)
    for i, q in enumerate(sh):
        folds[i % args.folds].append(q)

    print(f"n={len(common)}  tier1 runs={len(t1)}  tier2={len(t2)}  tier3={len(t3)}")
    base1 = st.mean([score(set(), set(), common, i, 0) for i in range(len(t1))])
    base2 = st.mean([score(set(common), set(), common, 0, i) for i in range(len(t2))])
    print(f"\n  tier1 everywhere {base1:7.2%}   {cost(0,0,len(common)):.2f} GPU-h")
    print(f"  tier2 everywhere {base2:7.2%}   {cost(len(common),0,len(common)):.2f} GPU-h\n")

    # ---- 1. LEARNED ROUTER, cross-validated
    print("1. LEARNED ROUTER — fitted on 4 folds, scored on the held-out fold")
    print(f"   {'budget':>7}{'routed':>9}{'random@k':>10}{'lift':>8}{'GPU-h':>8}")
    for frac in (0.10, 0.20, 0.30, 0.40, 0.50):
        accs, rands, costs = [], [], []
        for fi in range(args.folds):
            test = folds[fi]
            train = [q for j, f in enumerate(folds) if j != fi for q in f]
            y = {q: 0.0 if correct(t1[0], q) else 1.0 for q in train}
            sc, _ = fit_logistic(train, y)
            k = max(1, int(round(frac * len(test))))
            sel = set(sorted(test, key=lambda q: -sc(q))[:k])
            accs.append(score(sel, set(), test, 0, 0))
            rands.append(st.mean([score(set(rng.sample(test, k)), set(), test, 0, 0) for _ in range(40)]))
            costs.append(cost(k, 0, len(test)))
        print(f"   {frac:>6.0%}{st.mean(accs):>9.2%}{st.mean(rands):>10.2%}"
              f"{st.mean(accs)-st.mean(rands):>+8.2%}{sum(costs):>8.2f}")

    # ---- 2. MARGIN THRESHOLD, a continuous knob
    print("\n2. MARGIN THRESHOLD — escalate the lowest-margin fraction")
    print(f"   {'budget':>7}{'routed':>9}{'random@k':>10}{'lift':>8}")
    order = sorted(common, key=lambda q: feats[q]["margin"])
    for frac in (0.10, 0.20, 0.30, 0.40, 0.50):
        k = int(round(frac * len(common)))
        sel = set(order[:k])
        a = st.mean([score(sel, set(), common, i, j) for i in range(len(t1)) for j in range(len(t2))])
        r = st.mean([score(set(rng.sample(common, k)), set(), common, 0, 0) for _ in range(60)])
        print(f"   {frac:>6.0%}{a:>9.2%}{r:>10.2%}{a-r:>+8.2%}")

    # ---- 3. THREE TIERS
    if t3:
        print("\n3. THREE TIERS — tier2 on the low-margin fraction, tier3 on the lowest of those")
        print(f"   {'t2':>5}{'t3':>5}{'acc':>9}{'GPU-h':>8}{'vs tier2-all':>14}")
        for f2 in (0.20, 0.30, 0.40):
            for f3 in (0.05, 0.10):
                k2 = int(round(f2 * len(common)))
                k3 = int(round(f3 * len(common)))
                sel2 = set(order[:k2])
                sel3 = set(order[:k3])
                a = st.mean([score(sel2, sel3, common, i, j, k)
                             for i in range(len(t1)) for j in range(len(t2)) for k in range(len(t3))])
                ch = cost(len(sel2 - sel3), len(sel3), len(common))
                print(f"   {f2:>4.0%}{f3:>5.0%}{a:>9.2%}{ch:>8.2f}{a-base2:>+14.2%}")

    # ---- oracle ceiling
    orc = set(q for q in common if not correct(t1[0], q))
    print(f"\n  ORACLE (escalate exactly tier1's errors, {len(orc)/len(common):.0%}): "
          f"{score(orc, set(), common, 0, 0):.2%}  <- ceiling for any router at this budget")


if __name__ == "__main__":
    main()

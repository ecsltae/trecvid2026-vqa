"""
Assemble a two-tier routed run from artefacts that already exist.

No inference happens here. Tier 1's answers come from a CLEAN run (one that never
saw the abstention option, because offering it in the answering pass costs
-6.64pp); tier 2's answers come from an existing large-model run; and a logistic
router fitted on dev decides which questions take tier 2's answer.

The policy is fixed in data/vqa/2026/evidence/ROUTER_PREREG.md, written before the
2026 signal run finished. Nothing here may deviate from it: this project has
already lost one result to threshold-shopping, where a hand-picked policy looked
like it matched the big model at 16% of the cost and a split-half test showed the
per-dataset choice was worth -0.51pp.

    python eval/vqa/build_routed_run.py \
        --dev-votes  <dev --dump-votes jsonl ...>   --dev-tier1 <clean 8b dev runs> \
        --dev-tier2  <32b dev runs>                 --dev-qrels data/vqa/dev8/dev8full_qrels.csv \
        --dev-topics /tmp/dev8full_topics.json \
        --test-votes <2026 --dump-votes jsonl ...>  --test-tier1 <clean 8b 2026 run> \
        --test-tier2 <32b 2026 run>                 --test-topics <Task2_v2.json> \
        --budget 0.50 --out runs/vqa/mc/2026/mc2026_routed.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

FN = ["abstain", "margin", "margin_norm", "top_share", "disagree", "n_opt"]


def opts_of(t):
    return [o["option"] if isinstance(o, dict) else o for o in t["Options"]]


def load_run(path):
    rows = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) >= 4:
                try:
                    rank = int(r[2])
                except ValueError:
                    continue
                rows[r[0].strip()].append((rank, r[3], r[1].strip()))
    out = {}
    for q, v in rows.items():
        v.sort(key=lambda x: x[0])
        out[q] = {"order": [a for _, a, _ in v], "video": v[0][2]}
    return out


def load_votes(paths):
    sig = {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            sig[str(d["q"])] = d
    return sig


def featurise(sig, tier1_runs, topics):
    feats = {}
    for q, d in sig.items():
        if q not in topics:
            continue
        picks = [r[q]["order"][0] for r in tier1_runs if q in r]
        if d.get("top1"):
            picks.append(d["top1"])
        tal = sorted(d.get("tally", []), reverse=True)
        tot = sum(tal) or 1.0
        feats[q] = {
            "abstain": 1.0 if d.get("abstained") else 0.0,
            "margin": float(d.get("margin", 0.0)),
            "margin_norm": (tal[0] - tal[1]) / tot if len(tal) > 1 else 0.0,
            "top_share": (tal[0] / tot) if tal else 0.0,
            "disagree": float(len(set(picks)) - 1) if picks else 0.0,
            "n_opt": float(d.get("n_opt", len(opts_of(topics[q])))),
        }
    return feats


def fit_logistic(feats, y, iters=600, lr=0.05):
    keys = list(y)
    mu = {k: st.mean([feats[q][k] for q in keys]) for k in FN}
    sd = {k: (st.pstdev([feats[q][k] for q in keys]) or 1.0) for k in FN}
    w = {k: 0.0 for k in FN}
    b = 0.0
    for _ in range(iters):
        gw = {k: 0.0 for k in FN}
        gb = 0.0
        for q in keys:
            z = b + sum(w[k] * (feats[q][k] - mu[k]) / sd[k] for k in FN)
            p = 1 / (1 + pow(2.718281828459045, -max(-30.0, min(30.0, z))))
            e = p - y[q]
            for k in FN:
                gw[k] += e * (feats[q][k] - mu[k]) / sd[k]
            gb += e
        for k in FN:
            w[k] -= lr * gw[k] / len(keys)
        b -= lr * gb / len(keys)
    return w, b, mu, sd


def score_fn(w, b, mu, sd):
    return lambda f: b + sum(w[k] * (f[k] - mu[k]) / sd[k] for k in FN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-votes", nargs="+", required=True)
    ap.add_argument("--dev-tier1", nargs="+", required=True)
    ap.add_argument("--dev-tier2", nargs="+", required=True)
    ap.add_argument("--dev-topics", required=True)
    ap.add_argument("--dev-qrels", required=True)
    ap.add_argument("--test-votes", nargs="+", required=True)
    ap.add_argument("--test-tier1", required=True)
    ap.add_argument("--test-tier2", required=True)
    ap.add_argument("--test-topics", required=True)
    ap.add_argument("--budget", type=float, default=0.50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="fit and report on dev only; write nothing")
    args = ap.parse_args()

    dtop = {str(x["Q_ID"]): x for x in json.load(open(args.dev_topics))}
    gold = {}
    for r in csv.DictReader(open(args.dev_qrels, newline="", encoding="utf-8")):
        q = (r.get("Q_ID") or "").strip()
        a = (r.get("Answer") or "").strip()
        if q in dtop:
            oo = opts_of(dtop[q])
            L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(oo)]
            if a and a[0].upper() in L:
                gold[q] = oo[L.index(a[0].upper())]
            elif a in oo:
                gold[q] = a

    d1 = [load_run(p) for p in args.dev_tier1]
    d2 = [load_run(p) for p in args.dev_tier2]
    dsig = load_votes(args.dev_votes)
    dfe = featurise(dsig, d1, dtop)
    fit_on = [q for q in dfe if q in gold and all(q in r for r in d1 + d2)]
    y = {q: 0.0 if d1[0][q]["order"][0] == gold[q] else 1.0 for q in fit_on}
    print(f"dev: fitting the router on {len(fit_on)} items "
          f"({sum(y.values())/len(y):.0%} of them tier-1 errors)")
    w, b, mu, sd = fit_logistic(dfe, y)
    sc = score_fn(w, b, mu, sd)
    print("  weights: " + "  ".join(f"{k}={w[k]:+.3f}" for k in FN))

    k = max(1, int(round(args.budget * len(fit_on))))
    sel = set(sorted(fit_on, key=lambda q: -sc(dfe[q]))[:k])
    def acc(selection):
        return st.mean([sum(1 for q in fit_on
                            if (r2 if q in selection else r1)[q]["order"][0] == gold[q]) / len(fit_on)
                        for r1 in d1 for r2 in d2])
    print(f"  dev tier1 {acc(set()):.2%} | tier2 {acc(set(fit_on)):.2%} | "
          f"routed@{args.budget:.0%} {acc(sel):.2%}")
    if args.dry_run:
        print("  --dry-run: nothing written")
        return

    ttop = {str(x["Q_ID"]): x for x in json.load(open(args.test_topics))}
    t1 = load_run(args.test_tier1)
    t2 = load_run(args.test_tier2)
    tsig = load_votes(args.test_votes)
    tfe = featurise(tsig, [t1], ttop)
    usable = [q for q in ttop if q in t1 and q in t2]
    missing = [q for q in usable if q not in tfe]
    if missing:
        print(f"  WARNING {len(missing)} test questions have no routing signal — they keep tier 1")
    ranked = sorted([q for q in usable if q in tfe], key=lambda q: -sc(tfe[q]))
    kk = max(1, int(round(args.budget * len(usable))))
    tsel = set(ranked[:kk])
    print(f"test: {len(usable)} questions, escalating {len(tsel)} = {len(tsel)/len(usable):.0%}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    changed = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        for q in sorted(usable, key=lambda x: int(x) if x.isdigit() else 0):
            src = t2 if q in tsel else t1
            if t2[q]["order"][0] != t1[q]["order"][0] and q in tsel:
                changed += 1
            off = opts_of(ttop[q])
            if sorted(src[q]["order"]) != sorted(off):
                raise SystemExit(f"REFUSING: {q}'s ranking is not a permutation of its options")
            for rank, a in enumerate(src[q]["order"], 1):
                wri.writerow([q, src[q]["video"], rank, a, "0.00"])
    print(f"  wrote {out}: {len(usable)} questions, {changed} top-1 answers changed by routing "
          f"({changed/len(usable):.1%} of the run)")


if __name__ == "__main__":
    main()

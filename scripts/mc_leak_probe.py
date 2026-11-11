#!/usr/bin/env python3
"""
TRECVID 2026 VQA Task 2 — option-block provenance leak: the three detection probes.

This is the reproduction command cited by VQA_MC_LEAK.md. It rebuilds, from the
files on disk and nothing else, the three independent lines of evidence that
detected the leak in the v1 MC test set, and re-runs the same three probes
against the reissued v2 file to show the artifact is gone.

WHAT THE LEAK WAS. `vqa.2026.testingDataset.Task2.json` (v1, 2443 queries) is
described as one multiple-choice task with up to 11 options. For the 1952
queries carrying 8 options it behaved like two: positions 0-3 looked like the
original human-authored 4-option item, positions 4-7 like appended
machine-written filler. A system that simply never picked positions 4-7 would
have gained substantially without doing any additional video understanding.

THE THREE PROBES (each stands alone; the leak is the fact that all three agree):

  probe 1  PUNCTUATION SPLITS AT INDEX 4.  Terminal-full-stop rate per option
           position over the 8-option queries. In v1 positions 0-3 sit within
           .003 of each other (mean .534) and 4-6 within .001 of each other at
           a rate 6.5x lower (mean .082); position 7 differs again (.376),
           which is why position 7 is reported apart throughout. 308 queries
           have "every option in 0-3 ends with a full stop, none in 4-7 does"
           and 0 have the reverse.
           (VQA_MC_LEAK.md says 0-3 "agree to three decimals". They agree to
           two: .535 .533 .534 .533. The split is the same size either way.)

  probe 2  THE 4-OPTION QUERIES ARE A FLAT CONTROL.  The 472 queries that were
           never extended show no split. This is a SPECIFICITY control, not a
           detector: it is what rules out "options are just ordered that way".

  probe 3  OPTION-BLIND ANSWERS LAND IN BLOCK A.  Our Task-1 AG answers were
           generated from the video alone and never saw the MC options.
           Matching each to its nearest option by MiniLM cosine puts the match
           in positions 0-3 for 65.8% of the 8-option queries, against a 50%
           null.

           "Option-blind" is checkable, not asserted: the AG generator
           systems/vqa/run_vqa_v4.py contains no reference to Task2, to
           "Options", or to the word "option" anywhere in the file.

           *** This is the ONE place in the codebase that reads an AG run file,
           and it is DIAGNOSTIC ANALYSIS, never a submitted run. The organisers
           intend the two tasks to be independent (George Awad, 2026-08-25); no
           MC run reads AG output, and run_vqa_mc_v5.py regenerates its own
           free-text answer inside the MC run. Do not repurpose this loader. ***

           WHICH AG RUN. --ag defaults to BiTeM_ag_everything.json because it
           is the file whose numbers VQA_MC_LEAK.md reports, and it reproduces
           them to the decimal. The finding does not depend on that choice; all
           five AG runs on disk were measured on 2026-09-04 and every one of
           them fires on v1, is silent on v2, and separates the two files by
           13-16 points:
             consensus_8b       v1 65.5%   v2 51.5%
             consensus_8b_s16   v1 66.5%   v2 50.6%
             everything         v1 65.8%   v2 50.8%   <- default
             fewshot_8b         v1 64.4%   v2 50.2%
             v3_style_control   v1 64.5%   v2 51.4%
           Pass --ag to re-check any of them.

WHAT COUNTS AS DETECTION. Probes 1 and 3 are the detectors; probe 2 tells you
whether to believe probe 1. The thresholds are constants below, each with the
measured v1 and v2 value beside it so a reader can see the margin rather than
trust a number. The verdict is on the ACTIVE file (v2 by default) — the one runs
are actually built against. v1 is run as a POSITIVE CONTROL: if the probes stop
firing on v1, the probes are broken, not the data.

EXIT CODES
  0  no leak in the active file, and the positive control reproduced
  1  LEAK DETECTED in the active file
  2  the probes could not be trusted (positive control did not reproduce, or a
     required input was missing) — this is a probe failure, not a data verdict

USAGE
  python scripts/mc_leak_probe.py                 # v1 control + v2 verdict, all 3 probes
  python scripts/mc_leak_probe.py --no-embed      # probes 1+2 only (no model download/GPU)
  python scripts/mc_leak_probe.py --only <file>   # verdict on one file, no control

OUTPUT OF THE 2026-09-04 RUN (`python scripts/mc_leak_probe.py`, exit 0). Probe 3
encodes 1952 answers + 15616 options for v1 and 1971 + 15768 for v2; whole script
14 s wall on the A100, and byte-stable across repeat runs on this machine.

    ================================================================================
    TRECVID 2026 VQA Task 2 — option-block provenance leak probe
    ================================================================================
    control (v1) : data/vqa/2026/task2/vqa.2026.testingDataset.Task2.json
    ACTIVE       : data/vqa/2026/task2/vqa.2026.testingDataset.Task2_v2.json
    AG (probe 3) : submissions/vqa2026/BiTeM_ag_everything.json
                   ^ read as DIAGNOSTIC ONLY. No MC run reads AG output.

    --------------------------------------------------------------------------------
    FILE: v1  (control)  data/vqa/2026/task2/vqa.2026.testingDataset.Task2.json
    --------------------------------------------------------------------------------
    probe 0 — option-count census over 2443 queries
      4 options : 472
      8 options : 1952
      9 options : 10
      10 options: 5
      11 options: 4
      anomalous (not 4 and not 8): 19 queries   <-- data-hygiene flag

    probe 1 — terminal full stop by option position, 1952 eight-option queries
      pos     0     1     2     3  |     4     5     6     7
      rate .535 .533 .534 .533  | .082 .082 .081 .376
      block A (0-3) mean 0.5339   block B (4-7) mean 0.1552   gap 0.3787
      2x2 chi-square (block x ends-with-stop): chi2=2477.6  p=0.00e+00
      spread within A 0.003   within B 0.294
      "all of 0-3 punctuated, none of 4-7":     308 queries
      reverse ("all of 4-7, none of 0-3"):        0 queries
      => PROBE 1 FIRES: gap 0.3787 >= 0.050 and p < 1e-06; forward/reverse 308/0

    probe 2 — specificity control: the 472 never-extended 4-option queries
      pos     0     1     2     3
      rate .386 .388 .383 .371
      spread 0.017 (<= 0.050 means flat: no block structure where none was added)
      => CONTROL HOLDS: the probe-1 signal is specific to extended queries

    probe 3 — option-blind AG answers, nearest option by MiniLM cosine
      1952 eight-option queries matched
      pos      0     1     2     3  |     4     5     6     7
      share 16.5% 16.4% 16.6% 16.2%  | 11.8%  8.6%  6.2%  7.6%
      block A 65.8%  vs block B 34.2%   (null 50.0%)
      binomial test vs 50%: 1284/1952  p=9.03e-45
      within-A spread 0.5pp (flat, as a shuffled 4-option item should be)
      block B 4-6 decays monotonically: True  (position 7 apart: 7.6%)
      => PROBE 3 FIRES: block A 65.8% >= 55.0% and p < 1e-06

      VERDICT for v1: LEAK DETECTED (probe 1 fires, probe 3 fires)

    --------------------------------------------------------------------------------
    FILE: v2  (ACTIVE)  data/vqa/2026/task2/vqa.2026.testingDataset.Task2_v2.json
    --------------------------------------------------------------------------------
    probe 0 — option-count census over 2443 queries
      4 options : 472
      8 options : 1971
      anomalous (not 4 and not 8): 0 queries

    probe 1 — terminal full stop by option position, 1971 eight-option queries
      pos     0     1     2     3  |     4     5     6     7
      rate .000 .000 .000 .000  | .000 .000 .000 .000
      block A (0-3) mean 0.0000   block B (4-7) mean 0.0000   gap 0.0000
      2x2 chi-square (block x ends-with-stop): degenerate (no option ends with a stop)
      spread within A 0.000   within B 0.000
      "all of 0-3 punctuated, none of 4-7":       0 queries
      reverse ("all of 4-7, none of 0-3"):        0 queries
      => probe 1 silent: gap 0.0000 < 0.050

    probe 2 — specificity control: the 472 never-extended 4-option queries
      pos     0     1     2     3
      rate .000 .000 .000 .000
      spread 0.000 (<= 0.050 means flat: no block structure where none was added)
      => CONTROL HOLDS: the probe-1 signal is specific to extended queries

    probe 3 — option-blind AG answers, nearest option by MiniLM cosine
      1971 eight-option queries matched
      pos      0     1     2     3  |     4     5     6     7
      share 12.9% 13.5% 12.8% 11.5%  | 11.9% 11.1% 13.0% 13.2%
      block A 50.8%  vs block B 49.2%   (null 50.0%)
      binomial test vs 50%: 1001/1971  p=4.99e-01
      within-A spread 2.1pp (flat, as a shuffled 4-option item should be)
      block B 4-6 decays monotonically: False  (position 7 apart: 13.2%)
      => probe 3 silent: block A 50.8% < 55.0% (p=4.99e-01)

      VERDICT for v2: clean (no probe fires)

    ================================================================================
    SUMMARY
    ================================================================================
      probe                                    v1 (control)        v2 (ACTIVE)
      0  anomalous option counts                    19                    0
      1  block A/B full-stop gap                0.3787               0.0000
      1  "A punctuated, B not" / reverse         308/0                  0/0
      2  4-option control spread                 0.017                0.000
      3  option-blind landing in block A         65.8%                50.8%
      3  binomial p vs 50%                    9.03e-45             4.99e-01

    positive control: PASSED — the probes still fire on v1, so a clean v2 is a
    statement about the data, not about a broken probe.
    ACTIVE FILE vqa.2026.testingDataset.Task2_v2.json: NO LEAK DETECTED.
    Per-position landing in v2 is flat at 11.1-13.5% against a 12.5% expectation:
    the artifact is not masked, it is absent.

    exit 0
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

V1 = "data/vqa/2026/task2/vqa.2026.testingDataset.Task2.json"
V2 = "data/vqa/2026/task2/vqa.2026.testingDataset.Task2_v2.json"
AG = "submissions/vqa2026/BiTeM_ag_everything.json"

# ---- detection thresholds -------------------------------------------------
# Each carries the measured v1 / v2 value so the margin is visible. They are
# deliberately loose: every one of them separates the two files by more than an
# order of magnitude, so nothing here turns on where exactly the line is drawn.
GAP_MIN = 0.05        # probe 1 block A/B full-stop gap.   v1 .3787   v2 .0000
CHI_P_MAX = 1e-6      # probe 1 chi-square p.              v1 0.0     v2 n/a
FWD_MIN = 20          # probe 1 "A punctuated, B not".     v1 308     v2 0
FLAT_MAX = 0.05       # probe 2 control spread.            v1 .017    v2 .000
BLOCK_A_MIN = 0.55    # probe 3 landing share in 0-3.      v1 65.8%   v2 50.8%
BINOM_P_MAX = 1e-6    # probe 3 binomial p vs 50%.         v1 9.0e-45 v2 5.0e-01

EMBED_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------- loading

def load_options(path):
    """-> {Q_ID: [option text, ...]} preserving released option order."""
    out = {}
    for q in json.loads(Path(path).read_text(encoding="utf-8")):
        opts = [o["option"] if isinstance(o, dict) else str(o)
                for o in (q.get("Options") or q.get("Answers") or [])]
        out[str(q["Q_ID"]).strip()] = opts
    return out


def load_ag_top1(path):
    """-> {Q_ID: rank-1 free-text answer}.

    DIAGNOSTIC ONLY. See the module docstring: the two tasks are intended to be
    independent and no MC run may read this. Probe 3 uses it as a measuring
    instrument for the option ordering, which is analysis, not a submission.
    """
    out = {}
    for q in json.loads(Path(path).read_text(encoding="utf-8")):
        answers = sorted(q.get("Answers") or [], key=lambda r: r.get("Rank", 1))
        if answers:
            out[str(q["Q_ID"]).strip()] = answers[0].get("Answer", "")
    return out


def ends_with_stop(s):
    return s.strip().endswith(".")


def fmt_rates(rates, split=None):
    cells = []
    for i, r in enumerate(rates):
        if split is not None and i == split:
            cells.append(" |")
        cells.append(f"{r:.3f}".lstrip("0") if r else ".000")
    return " ".join(cells)


# ---------------------------------------------------------------- probes

def probe0_counts(opts):
    counts = Counter(len(v) for v in opts.values())
    print(f"probe 0 — option-count census over {len(opts)} queries")
    for k in sorted(counts):
        print(f"  {k} options{'' if k >= 10 else ' '}: {counts[k]}")
    anomalous = sum(n for k, n in counts.items() if k not in (4, 8))
    flag = "   <-- data-hygiene flag" if anomalous else ""
    print(f"  anomalous (not 4 and not 8): {anomalous} queries{flag}")
    return {"counts": dict(counts), "anomalous": anomalous}


def probe1_punctuation(opts):
    eight = [v for v in opts.values() if len(v) == 8]
    print()
    print(f"probe 1 — terminal full stop by option position, "
          f"{len(eight)} eight-option queries")
    if not eight:
        print("  no 8-option queries: probe 1 does not apply")
        return {"applies": False, "fires": False}
    rates = [sum(ends_with_stop(v[p]) for v in eight) / len(eight) for p in range(8)]
    print("  pos     0     1     2     3  |     4     5     6     7")
    print("  rate " + fmt_rates(rates, split=4))
    a_mean = sum(rates[:4]) / 4
    b_mean = sum(rates[4:]) / 4
    gap = abs(a_mean - b_mean)
    print(f"  block A (0-3) mean {a_mean:.4f}   block B (4-7) mean {b_mean:.4f}   "
          f"gap {gap:.4f}")

    a_stop = sum(sum(ends_with_stop(v[p]) for p in range(4)) for v in eight)
    b_stop = sum(sum(ends_with_stop(v[p]) for p in range(4, 8)) for v in eight)
    cell = len(eight) * 4
    p_chi = None
    if 0 < a_stop + b_stop < 2 * cell:
        from scipy.stats import chi2_contingency
        chi2, p_chi, _, _ = chi2_contingency(
            [[a_stop, cell - a_stop], [b_stop, cell - b_stop]])
        print(f"  2x2 chi-square (block x ends-with-stop): chi2={chi2:.1f}  p={p_chi:.2e}")
    else:
        print("  2x2 chi-square (block x ends-with-stop): degenerate "
              "(no option ends with a stop)" if a_stop + b_stop == 0 else
              "  2x2 chi-square (block x ends-with-stop): degenerate "
              "(every option ends with a stop)")

    print(f"  spread within A {max(rates[:4]) - min(rates[:4]):.3f}   "
          f"within B {max(rates[4:]) - min(rates[4:]):.3f}")
    fwd = sum(1 for v in eight
              if all(ends_with_stop(v[p]) for p in range(4))
              and not any(ends_with_stop(v[p]) for p in range(4, 8)))
    rev = sum(1 for v in eight
              if all(ends_with_stop(v[p]) for p in range(4, 8))
              and not any(ends_with_stop(v[p]) for p in range(4)))
    print(f'  "all of 0-3 punctuated, none of 4-7":  {fwd:>6} queries')
    print(f'  reverse ("all of 4-7, none of 0-3"):   {rev:>6} queries')

    fires = (gap >= GAP_MIN and p_chi is not None and p_chi < CHI_P_MAX) or \
            (fwd >= FWD_MIN and fwd > 5 * max(rev, 1))
    if fires:
        print(f"  => PROBE 1 FIRES: gap {gap:.4f} >= {GAP_MIN:.3f} and "
              f"p < {CHI_P_MAX:g}; forward/reverse {fwd}/{rev}")
    else:
        print(f"  => probe 1 silent: gap {gap:.4f} < {GAP_MIN:.3f}")
    return {"applies": True, "fires": fires, "gap": gap, "fwd": fwd, "rev": rev,
            "rates": rates}


def probe2_control(opts):
    four = [v for v in opts.values() if len(v) == 4]
    print()
    print(f"probe 2 — specificity control: the {len(four)} never-extended "
          f"4-option queries")
    if not four:
        print("  no 4-option queries: control unavailable")
        return {"applies": False, "holds": False}
    rates = [sum(ends_with_stop(v[p]) for v in four) / len(four) for p in range(4)]
    print("  pos     0     1     2     3")
    print("  rate " + fmt_rates(rates))
    spread = max(rates) - min(rates)
    print(f"  spread {spread:.3f} (<= {FLAT_MAX:.3f} means flat: no block "
          f"structure where none was added)")
    holds = spread <= FLAT_MAX
    print("  => CONTROL HOLDS: the probe-1 signal is specific to extended queries"
          if holds else
          "  => CONTROL BROKEN: the 4-option set is ALSO split — probe 1 is not "
          "diagnostic on this file")
    return {"applies": True, "holds": holds, "spread": spread}


def probe3_option_blind(opts, ag, model_name=EMBED_MODEL):
    print()
    print("probe 3 — option-blind AG answers, nearest option by MiniLM cosine")
    qids = sorted(q for q, v in opts.items() if len(v) == 8 and ag.get(q, "").strip())
    if not qids:
        print("  no 8-option queries with an AG answer: probe 3 does not apply")
        return {"applies": False, "fires": False}
    import numpy as np
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    ans = m.encode([ag[q] for q in qids], normalize_embeddings=True,
                   batch_size=256, show_progress_bar=False)
    flat = [o for q in qids for o in opts[q]]
    oe = m.encode(flat, normalize_embeddings=True, batch_size=256,
                  show_progress_bar=False).reshape(len(qids), 8, -1)
    pos = np.argmax(np.einsum("qd,qod->qo", ans, oe), axis=1)
    share = [float((pos == p).mean()) for p in range(8)]
    a_hits = int((pos < 4).sum())
    a_share = a_hits / len(qids)

    print(f"  {len(qids)} eight-option queries matched")
    print("  pos      0     1     2     3  |     4     5     6     7")
    row = []
    for i, s in enumerate(share):
        if i == 4:
            row.append(" |")
        row.append(f"{s*100:4.1f}%")
    print("  share " + " ".join(row))
    print(f"  block A {a_share*100:.1f}%  vs block B {(1-a_share)*100:.1f}%   "
          f"(null 50.0%)")
    from scipy.stats import binomtest
    p_bin = binomtest(a_hits, len(qids), 0.5).pvalue
    print(f"  binomial test vs 50%: {a_hits}/{len(qids)}  p={p_bin:.2e}")
    within_a = (max(share[:4]) - min(share[:4])) * 100
    # Positions 4-6 are the filler proper; position 7 behaves differently in
    # probe 1 too (rate .376 against .082 for 4-6), so it is reported apart
    # rather than folded into a single "block B decays" claim. VQA_MC_LEAK.md
    # says block B "decays monotonically"; over all four positions it does not
    # (v1 is 11.8 / 8.6 / 6.2 / 7.6 — position 7 rises). The monotone decay is
    # real over 4-6 only, and that is what is asserted here.
    decays_46 = all(share[i] >= share[i + 1] for i in range(4, 6))
    print(f"  within-A spread {within_a:.1f}pp (flat, as a shuffled 4-option "
          f"item should be)")
    print(f"  block B 4-6 decays monotonically: {decays_46}"
          f"  (position 7 apart: {share[7]*100:.1f}%)")

    fires = a_share >= BLOCK_A_MIN and p_bin < BINOM_P_MAX
    if fires:
        print(f"  => PROBE 3 FIRES: block A {a_share*100:.1f}% >= "
              f"{BLOCK_A_MIN*100:.1f}% and p < {BINOM_P_MAX:g}")
    else:
        print(f"  => probe 3 silent: block A {a_share*100:.1f}% < "
              f"{BLOCK_A_MIN*100:.1f}% (p={p_bin:.2e})")
    return {"applies": True, "fires": fires, "a_share": a_share, "p": p_bin,
            "share": share}


def run_file(label, path, ag, embed):
    print()
    print("-" * 80)
    print(f"FILE: {label}  {path}")
    print("-" * 80)
    opts = load_options(ROOT / path if not Path(path).is_absolute() else path)
    r = {"p0": probe0_counts(opts)}
    r["p1"] = probe1_punctuation(opts)
    r["p2"] = probe2_control(opts)
    r["p3"] = (probe3_option_blind(opts, ag) if embed and ag
               else {"applies": False, "fires": False, "skipped": True})
    if not r["p3"]["applies"]:
        print()
        print("probe 3 — SKIPPED (--no-embed, or no AG file). Probes 1 and 2 do "
              "not depend on it.")
    fired = [n for n in ("p1", "p3") if r[n]["fires"]]
    r["leak"] = bool(fired)
    print()
    names = {"p1": "probe 1 fires", "p3": "probe 3 fires"}
    print(f"  VERDICT for {label.split()[0]}: "
          + (f"LEAK DETECTED ({', '.join(names[f] for f in fired)})"
             if fired else "clean (no probe fires)"))
    return r


def main():
    ap = argparse.ArgumentParser(
        description="Reproduce the three probes that detected the Task 2 "
                    "option-block leak, and verify v2 fixes it.")
    ap.add_argument("--v1", default=V1, help="positive control (leaky) file")
    ap.add_argument("--active", default=V2,
                    help="the file runs are built against; the verdict is on this")
    ap.add_argument("--ag", default=AG,
                    help="AG run used by probe 3 as a DIAGNOSTIC ONLY")
    ap.add_argument("--only", default=None,
                    help="probe a single file and exit on its verdict; no control")
    ap.add_argument("--no-embed", action="store_true",
                    help="skip probe 3 (no sentence-transformers model needed)")
    args = ap.parse_args()

    print("=" * 80)
    print("TRECVID 2026 VQA Task 2 — option-block provenance leak probe")
    print("=" * 80)

    ag = {}
    if not args.no_embed:
        ag_path = ROOT / args.ag if not Path(args.ag).is_absolute() else Path(args.ag)
        if not ag_path.exists():
            print(f"probe 3 unavailable: {args.ag} not found. Re-run with "
                  f"--no-embed to run probes 1 and 2 only.", file=sys.stderr)
            return 2
        ag = load_ag_top1(ag_path)

    if args.only:
        print(f"single file  : {args.only}")
        if ag:
            print(f"AG (probe 3) : {args.ag}")
            print("               ^ read as DIAGNOSTIC ONLY. No MC run reads AG output.")
        r = run_file(f"{Path(args.only).name}  (verdict)", args.only, ag,
                     not args.no_embed)
        print()
        print("exit", 1 if r["leak"] else 0)
        return 1 if r["leak"] else 0

    print(f"control (v1) : {args.v1}")
    print(f"ACTIVE       : {args.active}")
    if ag:
        print(f"AG (probe 3) : {args.ag}")
        print("               ^ read as DIAGNOSTIC ONLY. No MC run reads AG output.")

    c = run_file("v1  (control)", args.v1, ag, not args.no_embed)
    a = run_file("v2  (ACTIVE)", args.active, ag, not args.no_embed)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("  probe                                    v1 (control)        v2 (ACTIVE)")
    print(f"  0  anomalous option counts            {c['p0']['anomalous']:>10}"
          f"{a['p0']['anomalous']:>21}")
    print(f"  1  block A/B full-stop gap            {c['p1']['gap']:>10.4f}"
          f"{a['p1']['gap']:>21.4f}")
    c_fr = "{}/{}".format(c["p1"]["fwd"], c["p1"]["rev"])
    a_fr = "{}/{}".format(a["p1"]["fwd"], a["p1"]["rev"])
    print(f"  1  \"A punctuated, B not\" / reverse    {c_fr:>10}{a_fr:>21}")
    print(f"  2  4-option control spread            {c['p2']['spread']:>10.3f}"
          f"{a['p2']['spread']:>21.3f}")
    if c["p3"]["applies"] and a["p3"]["applies"]:
        print(f"  3  option-blind landing in block A    "
              f"{c['p3']['a_share']*100:>9.1f}%{a['p3']['a_share']*100:>20.1f}%")
        print(f"  3  binomial p vs 50%                  "
              f"{c['p3']['p']:>10.2e}{a['p3']['p']:>21.2e}")
    else:
        print("  3  option-blind landing in block A          skipped (--no-embed)")

    print()
    if not c["leak"]:
        print("positive control: FAILED — the probes no longer fire on the KNOWN-LEAKY "
              "v1 file.\nDo not read anything into the verdict on the active file: the "
              "probes are broken\n(wrong file passed as --v1, or a loader/threshold "
              "regression).")
        print()
        print("exit 2")
        return 2
    print("positive control: PASSED — the probes still fire on v1, so a clean v2 is a")
    print("statement about the data, not about a broken probe.")

    if a["leak"]:
        print(f"ACTIVE FILE {Path(args.active).name}: *** LEAK DETECTED ***")
        print("Runs built against this file inherit an option-ordering artifact. Report "
              "it to the\norganisers before submitting anything (see "
              "submissions/vqa2026/NIST_mc_leak_report.md).")
        print()
        print("exit 1")
        return 1

    print(f"ACTIVE FILE {Path(args.active).name}: NO LEAK DETECTED.")
    if a["p3"]["applies"]:
        s = a["p3"]["share"]
        print(f"Per-position landing in v2 is flat at {min(s)*100:.1f}-{max(s)*100:.1f}% "
              f"against a 12.5% expectation:\nthe artifact is not masked, it is absent.")
    print()
    print("exit 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())

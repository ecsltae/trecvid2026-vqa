"""
Build a human arbitration queue + a self-contained offline voting page.

WHY ARBITRATION AND NOT LABELLING-FOR-TRAINING. Measured on the 2025 MC dev set:
where our runs agree, top-1 accuracy is 85.9%; where they disagree (30.7% of
items) it collapses to 49.3%. Perfect arbitration of only the disagreements is
worth +9.5pp — larger than every architecture change we have tested. By
contrast one 8-option label carries log2(8)=3 bits, so 500 hand labels are
~188 bytes of supervision against the 1996 gold labels we already hold free.
Human attention is worth far more spent deciding contested items than
generating training data.

WHY CALIBRATION FIRST — the lesson from this project's own TREC RAG round.
There, a human voted 119/119 topics in a very similar UI, and the votes turned
out to be significantly ANTI-correlated with answer quality (44/115 = 38.3%,
p = 0.015) and statistically indistinguishable from "prefer the longer answer".
A conclusion was published internally and had to be retracted. The designed
calibration queue was never actually voted, so nobody found out until later.

MC does not have that excuse: we have gold labels. So the queue is built in two
modes, and calibration is not optional.

  --mode calibration : 2025 items WITH known answers. The page never shows the
                       gold, but it is stored alongside, so the human's accuracy
                       can be measured before a single 2026 label is trusted.
  --mode arbitrate   : 2026 items, no gold, ranked by how contested they are.

DESIGN OF THE PAGE, inherited from the RAG page that actually got used
(curation-pages/index.html): one file, no server, no network,
frames inlined as data URIs, progress saved to localStorage so it survives a
reload, keyboard-driven, and a copy-to-clipboard export. It is deliberately
BLIND — it never shows which system picked what, and options are shuffled per
item, so the human cannot anchor on a model or on position.

Usage:
    python systems/vqa/build_arbitration.py --mode calibration \
        --runs /tmp/mc_v3_dev300.csv /tmp/mc_nodraft_dev300.csv /tmp/mc_v5_dev300.csv \
        --topics data/vqa/topics/vqa_2025_mc_topics.csv \
        --frames data/vqa/frames_pool16_512 \
        --limit 60 --out curation/mc_calibration.html
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path


def load_topics(path):
    p = Path(path)
    if p.suffix.lower() == ".json":
        out = {}
        for t in json.loads(p.read_text(encoding="utf-8")):
            out[str(t["Q_ID"]).strip()] = {
                "video": str(t["Video_ID"]), "q": str(t.get("Question", "")),
                "options": [o["option"] if isinstance(o, dict) else str(o)
                            for o in (t.get("Options") or t.get("Answers") or [])]}
        return out
    out = {}
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) >= 7:
                out[r[0].strip()] = {"video": r[1].strip().strip('"'),
                                     "q": r[2].strip().strip('"'),
                                     "options": [c.strip().strip('"') for c in r[3:7]]}
    return out


def load_run(path, topics):
    rows = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) >= 4:
                try:
                    rows[r[0].strip()].append((int(r[2]), r[3]))
                except ValueError:
                    pass
    out = {}
    for q, v in rows.items():
        seq = [a for _, a in sorted(v, key=lambda x: x[0])]
        if q in topics and all(len(s.strip()) == 1 and s.strip().upper() in "ABCD" for s in seq):
            opts = topics[q]["options"]
            seq = [opts["ABCD".index(s.strip().upper())] for s in seq]
        out[q] = seq
    return out


def frame_b64(frames_dir, video, k=4, px=224, quality=58):
    """k evenly-spaced frames, downscaled, as data URIs. Size matters: the whole
    queue ships as ONE html file, so ~15KB/frame keeps 300 items under 20MB."""
    from PIL import Image
    d = Path(frames_dir) / video
    if not d.exists():
        d = Path(frames_dir) / Path(video).stem
    if not d.exists():
        return []
    fr = sorted(d.glob("frame_*.jpg"))
    if not fr:
        return []
    step = (len(fr) - 1) / (k - 1) if k > 1 and len(fr) > 1 else 0
    picks = [fr[round(i * step)] for i in range(min(k, len(fr)))]
    out = []
    for p in picks:
        try:
            im = Image.open(p).convert("RGB")
            w, h = im.size
            s = px / max(w, h)
            if s < 1:
                im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=quality)
            out.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
        except Exception:
            pass
    return out


PAGE = """<!doctype html><meta charset=utf-8><title>VQA arbitration</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
 body{font:15px/1.45 system-ui,sans-serif;margin:0;background:#111;color:#eee}
 header{position:sticky;top:0;background:#181818;padding:10px 14px;border-bottom:1px solid #333}
 #bar{height:4px;background:#2a2a2a;border-radius:2px;overflow:hidden;margin-top:8px}
 #fill{height:100%;background:#4ea1ff;width:0}
 main{padding:14px;max-width:900px;margin:0 auto}
 .frames{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}
 .frames img{height:104px;border-radius:5px;background:#000}
 video{width:100%;max-height:420px;background:#000;border-radius:8px}
 .hint{color:#888;font-size:13px;margin:6px 0 0}
 .q{font-size:17px;font-weight:600;margin:12px 0}
 button.opt{display:block;width:100%;text-align:left;margin:6px 0;padding:11px 13px;
   background:#232323;color:#eee;border:1px solid #3a3a3a;border-radius:8px;cursor:pointer;font-size:15px}
 button.opt:hover{background:#2c2c2c;border-color:#4ea1ff}
 .k{display:inline-block;min-width:20px;color:#4ea1ff;font-weight:700}
 .row{display:flex;gap:8px;margin-top:14px;flex-wrap:wrap}
 .row button{padding:9px 14px;background:#2a2a2a;color:#ccc;border:1px solid #3a3a3a;
   border-radius:8px;cursor:pointer}
 #done{text-align:center;padding:40px}
 textarea{width:100%;height:180px;background:#0c0c0c;color:#8f8;border:1px solid #333;border-radius:8px}
 small{color:#999}
</style>
<header>
  <b>VQA arbitration</b> &nbsp;<small id=meta></small>
  <button onclick=exportNow() style="float:right;margin-left:6px">Save progress</button>
  <button onclick=importNow() style="float:right">Restore</button>
  <div id=bar><div id=fill></div></div>
</header>
<main id=app></main>
<script>
const D=__DATA__, KEY="vqa_arb_"+D.tag;
let votes=JSON.parse(localStorage.getItem(KEY)||"{}"), i=0, hist=[];
function save(){localStorage.setItem(KEY,JSON.stringify(votes))}
function next(){ while(i<D.items.length && votes[D.items[i].id]!==undefined) i++; render() }
let aud=false;
function toggleAudio(){ aud=!aud; const b=document.getElementById('ab');
  if(b){ b.textContent=(aud?'\u2611':'\u2610')+' answer needed SOUND (a)';
         b.style.borderColor=aud?'#4ea1ff':'#3a3a3a'; } }
function pick(v){ const it=D.items[i]; if(!it) return;
  votes[it.id]={p:v,a:aud}; hist.push(it.id); aud=false; save(); i++; next() }
function undo(){ const id=hist.pop(); if(id===undefined) return; delete votes[id]; save();
  i=D.items.findIndex(x=>x.id===id); render() }
function render(){
  const n=Object.keys(votes).length, tot=D.items.length;
  document.getElementById('meta').textContent=n+" / "+tot+" decided";
  document.getElementById('fill').style.width=(100*n/tot)+"%";
  const app=document.getElementById('app');
  if(i>=tot){ app.innerHTML='<div id=done><h2>All done — '+n+' decided</h2>'
    +'<p>Copy this and send it back.</p><textarea id=out readonly></textarea>'
    +'<div class=row><button onclick="copy()">Copy to clipboard</button></div></div>';
    document.getElementById('out').value=JSON.stringify({tag:D.tag,votes:votes},null,1); return; }
  const it=D.items[i];
  const vid = it.v ? '<video src="'+it.v+'" controls loop playsinline preload=metadata></video>'
      +'<p class=hint>Full clip WITH SOUND. The model sees '+D.model_frames
      +' still frames and no audio — if the answer came from the SOUND, press a before choosing.</p>' : '';
  app.innerHTML=vid+'<div class=frames>'+it.f.map(s=>'<img src="'+s+'">').join('')+'</div>'
    +'<div class=q>'+it.q+'</div>'
    +it.o.map((o,j)=>'<button class=opt onclick="pick('+j+')"><span class=k>'+(j+1)+'</span> '+o+'</button>').join('')
    +'<div class=row><button id=ab onclick="toggleAudio()">\u2610 answer needed SOUND (a)</button></div>'
    +'<div class=row><button onclick="pick(-1)">Can\\'t tell (u)</button>'
    +'<button onclick="undo()">Undo (z)</button>'
    +'<span style="flex:1"></span><small>'+(i+1)+' of '+tot+'</small></div>';
}
function copy(){ const t=document.getElementById('out'); t.select();
  document.execCommand('copy'); alert('Copied.'); }
// Votes live in localStorage, which is per-ORIGIN: file:// and http://127.0.0.1
// are different stores, so switching how you open the page silently loses them.
// These two buttons make progress portable, and are always visible rather than
// only appearing on the last item.
function exportNow(){
  const p=JSON.stringify({tag:D.tag,votes:votes});
  navigator.clipboard.writeText(p).then(
    ()=>alert('Progress copied: '+Object.keys(votes).length+' items. Paste it into a file to be safe.'),
    ()=>{ window.prompt('Copy this:',p); });
}
function importNow(){
  const t=window.prompt('Paste previously saved progress JSON:');
  if(!t) return;
  try{ const o=JSON.parse(t); const v=o.votes||o;
    let k=0; for(const id in v){ votes[id]=v[id]; k++; }
    save(); alert('Restored '+k+' items.'); i=0; next();
  }catch(e){ alert('Could not parse that.'); }
}
document.addEventListener('keydown',e=>{
  if(e.key==='z'){undo();return} if(e.key==='u'){pick(-1);return}
  if(e.key==='a'){toggleAudio();return}
  const n=parseInt(e.key); const it=D.items[i];
  if(it&&n>=1&&n<=it.o.length) pick(n-1);
});
next();
</script>"""


def main():
    ap = argparse.ArgumentParser()
    # control mode samples items where every run AGREES. Those are NOT the items
    # arbitration targets; the point is to measure the base rate of defective
    # items so the contested set's rate can be read as an enrichment rather than
    # an absolute. Without it, "20% of contested items are broken" is unreadable.
    ap.add_argument("--mode", choices=["calibration", "arbitrate", "control"],
                    required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--topics", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--qrels", default="data/vqa/qrels_mc.csv")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    # The model sees 8 frames. Showing the human FEWER makes a poor showing
    # uninterpretable — is the person unreliable, or just under-informed? So the
    # human gets the whole pool, and where the clip is on disk, the actual video
    # with sound. Several questions ask what was SAID or heard, which no still
    # frame can answer at all.
    ap.add_argument("--video-dirs", nargs="*", default=[],
                    help="dirs holding the source clips; enables real playback")
    # Show exactly what the model gets: v3's get_frames takes 8 EVENLY SPACED
    # frames from the pool, so the strip IS the model's literal input. The video
    # above is the human's extra information; the strip is the model's.
    ap.add_argument("--frames-shown", type=int, default=8,
                    help="filmstrip frames — keep equal to what the model receives")
    ap.add_argument("--model-frames", type=int, default=8)
    ap.add_argument("--shuffle-contested", action="store_true",
                    help="representative random order, so any prefix is unbiased")
    ap.add_argument("--take", type=int, default=0,
                    help="ship at most N items — keeps the inline script small")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N built items — for resuming a session")
    ap.add_argument("--video-subdir", default="small",
                    help="which prepared clip dir to embed from")
    ap.add_argument("--embed-video", action="store_true",
                    help="inline compressed clips as data URIs — one portable file")
    args = ap.parse_args()

    topics = load_topics(args.topics)
    runs = [load_run(p, topics) for p in args.runs]

    gold = {}
    if args.mode in ("calibration", "control") and Path(args.qrels).exists():
        with open(args.qrels, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                q = (row.get("Q_ID") or "").strip()
                a = (row.get("Answer") or "").strip()
                if q in topics and a and a[0].upper() in "ABCD":
                    opts = topics[q]["options"]
                    if len(opts) > "ABCD".index(a[0].upper()):
                        gold[q] = opts["ABCD".index(a[0].upper())]

    # contested = the runs' top-1 picks disagree. That is exactly the population
    # where model accuracy collapses (49.3% vs 85.9%), so it is where a human
    # decision changes an outcome instead of confirming one.
    common = set.intersection(*[set(r) for r in runs])
    if args.mode == "calibration":
        common &= set(gold)
    # sorted(), not the raw set: string-set iteration order varies per process
    # (hash randomisation), so an unsorted pool makes the seeded shuffle below
    # non-reproducible and --offset/--take slice a DIFFERENT list every build.
    contested, agreed = [], []
    for q in sorted(common):
        tops = [r[q][0] for r in runs if r.get(q)]
        if len(set(tops)) > 1:
            contested.append((q, len(set(tops))))
        else:
            agreed.append(q)
    # Sorting most-contested-first makes a partially completed session
    # unusable: the 3-way disagreements all land at the front, so whatever is
    # left is the easy tail and any accuracy measured on it is optimistic.
    # A representative random sample keeps every prefix unbiased.
    if args.shuffle_contested:
        random.Random(args.seed).shuffle(contested)
    else:
        contested.sort(key=lambda x: (-x[1], x[0]))
    if args.mode == "control":
        pool = sorted(agreed)
        random.Random(args.seed).shuffle(pool)
        chosen = pool[:args.limit]
        print(f"{len(common)} shared questions | {len(agreed)} AGREED "
              f"({100*len(agreed)/max(len(common),1):.1f}%) | taking {len(chosen)} as control")
    else:
        chosen = [q for q, _ in contested[:args.limit]]
        print(f"{len(common)} shared questions | {len(contested)} contested "
              f"({100*len(contested)/max(len(common),1):.1f}%) | taking {len(chosen)}")

    out_dir_media = Path(args.out).parent / "media"
    out_dir_media.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    items, skipped = [], 0
    for q in chosen:
        f = frame_b64(args.frames, topics[q]["video"], k=args.frames_shown)
        if not f:
            skipped += 1
            continue
        opts = list(topics[q]["options"])
        order = list(range(len(opts)))
        rng.shuffle(order)                     # blind to position AND to model picks
        rel = ""
        # .mkv first-class is a trap: browsers cannot decode Matroska, so an
        # .mkv item renders with a dead <video> and the human is left with only
        # the frames — or, worse, the page appears to hang. Prefer an already
        # transcoded .mp4 sitting next to it in media/.
        small = Path(args.out).parent / args.video_subdir / (topics[q]["video"] + ".mp4")
        if args.embed_video and small.exists():
            rel = ("data:video/mp4;base64,"
                   + base64.b64encode(small.read_bytes()).decode())
        pre = Path(args.out).parent / "media"
        for cand in ([] if rel else [pre / (topics[q]["video"] + ".mp4")]):
            if cand.exists():
                rel = "media/" + cand.name
                break
        for vd in ([] if rel else args.video_dirs):
            for ext in (".mp4", ".webm", ".mkv"):
                cand = Path(vd) / (topics[q]["video"] + ext)
                if cand.exists():
                    dest = out_dir_media / cand.name
                    if not dest.exists():
                        dest.symlink_to(cand.resolve())
                    rel = "media/" + cand.name
                    break
            if rel:
                break
        items.append({"id": q, "q": html.escape(topics[q]["q"]),
                      "o": [html.escape(opts[j]) for j in order],
                      "f": f, "v": rel, "perm": order})
    print(f"  built {len(items)} items ({skipped} skipped for missing frames)")

    if args.take:
        items = items[:args.offset + args.take] if args.offset else items[:args.take]
    if args.offset:
        # A 37.9M-character inline script stalls the browser, so a resumed
        # session ships only the items that are left.
        items = items[args.offset:]
        print(f"  offset {args.offset} -> shipping {len(items)} remaining items")
    tag = f"{args.mode}_{Path(args.out).stem}"
    data = {"tag": tag, "items": items, "model_frames": args.model_frames}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    page = PAGE.replace("__DATA__", json.dumps(data))
    # A single unterminated JS string silently kills the whole script and the
    # page renders as a bare header — which looks exactly like "no data" and
    # cost a session to diagnose. Check the script block before writing.
    body = page.split("<script>", 1)[1].rsplit("</script>", 1)[0]
    for ln, line in enumerate(body.splitlines(), 1):
        stripped = re.sub(r"\\.", "", line)
        if stripped.count("'") % 2 or stripped.count('"') % 2:
            raise SystemExit(f"REFUSING TO WRITE: unbalanced quote in script line {ln}:\n  {line[:120]}")
    out.write_text(page, encoding="utf-8")

    # the key/gold sidecar stays OUT of the page so the human cannot see it
    side = out.with_suffix(".key.json")
    side.write_text(json.dumps({
        "tag": tag, "mode": args.mode,
        "items": {it["id"]: {"perm": it["perm"],
                             "options": topics[it["id"]]["options"],
                             "gold": gold.get(it["id"])} for it in items}}, indent=1),
        encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size/1e6:.1f} MB) and {side.name}")
    print(f"  open it in a browser, decide, then Copy to clipboard and save the JSON")


if __name__ == "__main__":
    main()

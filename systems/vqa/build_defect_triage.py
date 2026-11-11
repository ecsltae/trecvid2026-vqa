"""
Build a DEFECT-TRIAGE page: is this item's gold answer defensible at all?

WHY THIS PROTOCOL AND NOT MORE ARBITRATION. The 22-item arbitration round measured
the human at +0.0pp against qwen3-vl:32b (11/15 vs 11/15, McNemar p=1.0000), so
asking a human to out-pick the model is worth nothing measurable. What the same
round DID establish, at p=0.000094 (Fisher), is that the human's "can't tell" is
a DEFECT DETECTOR: 6 of 7 can't-tells were annotation defects, 0 of 15 decided
items were. So point the human at the thing they are demonstrably good at.

Triage is also far cheaper per item. Arbitration asks "which of these is right?"
(measured 1.8 min/item). Triage shows ONE proposed answer and asks "is this
defensible?" — a judgement, not a search.

TWO QUESTIONS, NOT ONE. v1 asked only "is this defensible?" and the sham arm
immediately exposed the flaw: on a sham whose real gold was "Dancing." the rater
wrote "i would have said dancing... but i feel like she is crying also" and
marked the sham defensible. That is a CORRECT answer to the question asked, and
a useless one for measuring gold quality, because "defensible" and "is the gold"
are different predicates. So the page now also asks whether a BETTER option
exists, which separates gold-is-wrong from gold-is-ambiguous from gold-is-fine
and gives the sham arm something to detect.

THE SHAM ARM IS NOT OPTIONAL. Showing someone a proposed answer invites
acquiescence: say yes, move on. So a hidden fraction of items are shown a
DISTRACTOR in place of the gold. The false-accept rate on shams calibrates the
defect rate on real golds; without it a low defect rate is unreadable, because
"the golds are fine" and "the rater rubber-stamped everything" predict the same
number.

STRATA. The 2025 defect estimate is currently one-sided: the defect rate on
CONTESTED items is ~27%, and the rate on AGREED items — 64% of the weight in the
ceiling estimate — has never been measured. Agreed items are therefore sampled
hardest. Counting is sampled too: it is the worst category (50.8%) and the human
already disagreed with gold on 2 of 3 counting items they decided.

Usage:
    python systems/vqa/build_defect_triage.py --runs a.csv b.csv c.csv \
        --out curation/triage.html --offset 0 --take 10
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
import subprocess
from collections import defaultdict
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_topics(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) >= 7:
                out[r[0].strip()] = dict(
                    video=r[1].strip().strip('"'),
                    q=r[2].strip().strip('"'),
                    options=[c.strip().strip('"') for c in r[3:] if c.strip()],
                )
    return out


def load_gold(path, topics):
    gold = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = (row.get("Q_ID") or "").strip()
            a = (row.get("Answer") or "").strip()
            if q in topics and a and a[0].upper() in LETTERS:
                i = LETTERS.index(a[0].upper())
                if i < len(topics[q]["options"]):
                    gold[q] = topics[q]["options"][i]
    return gold


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
    out = {}
    for q, v in rows.items():
        seq = [a for _, a in sorted(v, key=lambda x: x[0])]
        if not seq:
            continue
        opts = topics.get(q, {}).get("options", [])
        L = LETTERS[: len(opts)]
        if opts and all(len(s.strip()) == 1 and s.strip().upper() in L for s in seq):
            seq = [opts[L.index(s.strip().upper())] for s in seq]
        out[q] = seq
    return out


def frames_for(frames_root, video, k=8):
    d = Path(frames_root) / video
    fs = sorted(d.glob("frame_*.jpg"))
    if not fs:
        return []
    # the 8 the model actually sees: an even stride over the pool
    if len(fs) > k:
        step = len(fs) / k
        fs = [fs[int(i * step)] for i in range(k)]
    out = []
    for p in fs:
        try:
            from PIL import Image
            im = Image.open(p).convert("RGB")
            im.thumbnail((360, 360))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=70)
            out.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
        except Exception:
            pass
    return out


def tiny_video(video, video_dirs, tiny_dir):
    tiny_dir.mkdir(parents=True, exist_ok=True)
    dst = tiny_dir / f"{video}.mp4"
    if not (dst.exists() and dst.stat().st_size > 1000):
        src = None
        for d in video_dirs:
            for ext in (".webm", ".mkv", ".mp4", ".m4v", ".mov"):
                c = Path(d) / (video + ext)
                if c.is_file():
                    src = c
                    break
            if src:
                break
        if not src:
            return ""
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src),
             "-vf", "scale='min(288,iw)':-2", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "36", "-r", "10", "-c:a", "aac", "-b:a", "32k", "-ac", "1",
             "-movflags", "+faststart", str(dst)],
            capture_output=True, timeout=300)
    if dst.exists() and dst.stat().st_size > 1000:
        return "data:video/mp4;base64," + base64.b64encode(dst.read_bytes()).decode()
    return ""


PAGE = """<meta charset="utf-8"><title>VQA defect triage</title>
<style>
 body{font:15px/1.5 system-ui,sans-serif;max-width:900px;margin:0 auto;padding:18px;background:#111;color:#eee}
 .item{border:1px solid #333;border-radius:9px;padding:16px;margin:22px 0;background:#181818}
 .q{font-size:18px;font-weight:600;margin:10px 0}
 .prop{border-left:4px solid #d9a441;background:#231d10;padding:11px 13px;margin:12px 0;border-radius:5px;font-size:17px}
 .frames img{height:104px;margin:2px;border-radius:4px;vertical-align:top}
 video{max-width:100%;border-radius:6px;background:#000}
 .opts{color:#9a9a9a;font-size:13px;margin:10px 0 4px}
 .better button.v{display:block;text-align:left;width:100%;margin:4px 0;font-size:14px;white-space:normal}
 .better button.v.sel{background:#2f4f7a;border-color:#4a72a8}
 button.v{font-size:15px;padding:9px 16px;margin:5px 7px 0 0;border-radius:6px;border:1px solid #444;background:#222;color:#eee;cursor:pointer}
 button.v.sel{background:#2d6a3f;border-color:#3f8f57;color:#fff}
 button.v.no.sel{background:#7a2b2b;border-color:#a63a3a}
 button.v.ct.sel{background:#6a5a1f;border-color:#8f7a2a}
 .bar{position:sticky;top:0;background:#111;padding:9px 0;border-bottom:1px solid #333;z-index:9}
 textarea{width:100%;background:#0d0d0d;color:#eee;border:1px solid #333;border-radius:5px;padding:7px;font:13px system-ui}
 .done{color:#6bbf7b}
</style>
<div class="bar">
 <b>Defect triage</b> &mdash; <span id="prog">0</span> of <span id="tot">0</span>
 <button class="v" id="cp">Copy progress</button>
 <button class="v" id="rs">Restore</button>
</div>
<p style="color:#aaa">For each clip you are shown <b>one proposed answer</b>. Judge only this:
<b>is that answer defensible from the video?</b> Not "is it the best of the four" &mdash; just
whether it is a fair answer to the question as asked. Then say whether any option in the list is
a <b>better</b> answer &mdash; that is a separate question, and an answer can be defensible while
another is better. If you pick <b>Can&rsquo;t tell</b> the item is finished, no second answer needed. Some proposed answers are deliberately wrong; that is the calibration, so please
do not assume the answer shown is correct.</p>
<div id="app"></div>
<script>
const D=__DATA__;
function isDone(k){var w=votes[k];return !!(w&&w.v&&(w.v==='ct'||w.b!==undefined&&w.b!==null))}
let votes={};
try{votes=JSON.parse(localStorage.getItem(D.tag)||'{}')}catch(e){votes={}}
function save(){try{localStorage.setItem(D.tag,JSON.stringify(votes))}catch(e){}
 document.getElementById('prog').textContent=Object.keys(votes).filter(isDone).length}
function pick(id,val){
 votes[id]=Object.assign(votes[id]||{},{v:val});
 if(val==='ct'){votes[id].b=null}            // cannot-tell needs no second answer
 save();draw()}
function note(id,t){votes[id]=Object.assign(votes[id]||{},{n:t});save()}
function better(id,x){votes[id]=Object.assign(votes[id]||{},{b:x});save();draw()}
function draw(){
 const app=document.getElementById('app');app.innerHTML='';
 D.items.forEach(function(it,ix){
  const cur=(votes[it.id]||{}).v;
  const curb=(votes[it.id]||{}).b;
  const d=document.createElement('div');d.className='item';
  let h='<div style="color:#777;font-size:13px">item '+(ix+1)+' of '+D.items.length+
    (isDone(it.id)?' <span class="done">&#10003; done</span>':
     ' <span style="color:#c88">&mdash; needs both answers</span>')+'</div>';
  h+='<div class="q">'+it.q+'</div>';
  if(it.v){h+='<video controls preload="metadata" src="'+it.v+'"></video>';}
  h+='<div class="frames">'+it.f.map(function(s){return '<img src="'+s+'">'}).join('')+'</div>';
  h+='<div class="prop"><b>Proposed answer:</b> '+it.a+'</div>';
  h+='<div class="opts">Is any of these a BETTER answer than the one proposed?</div>';
  h+='<div class="better"><button class="v b'+(curb===-1?' sel':'')+'" data-b="'+it.id+'" data-x="-1">None is better</button>';
  it.o.forEach(function(s,oi){
   h+='<button class="v b'+(curb===oi?' sel':'')+'" data-b="'+it.id+'" data-x="'+oi+'">'+s+'</button>';
  });
  h+='</div>';
  h+='<button class="v yes'+(cur==='ok'?' sel':'')+'" data-i="'+it.id+'" data-v="ok">Defensible</button>';
  h+='<button class="v no'+(cur==='bad'?' sel':'')+'" data-i="'+it.id+'" data-v="bad">Not defensible</button>';
  h+='<button class="v ct'+(cur==='ct'?' sel':'')+'" data-i="'+it.id+'" data-v="ct">Can’t tell</button>';
  h+='<div style="margin-top:9px"><textarea rows="2" placeholder="why? (optional, but very useful when you say not defensible)" data-n="'+it.id+'">'+
     ((votes[it.id]||{}).n||'')+'</textarea></div>';
  d.innerHTML=h;app.appendChild(d);
 });
 document.querySelectorAll('button.v[data-i]').forEach(function(b){
  b.onclick=function(){pick(b.getAttribute('data-i'),b.getAttribute('data-v'))}});
 document.querySelectorAll('button.v[data-b]').forEach(function(b){
  b.onclick=function(){better(b.getAttribute('data-b'),parseInt(b.getAttribute('data-x'),10))}});
 document.querySelectorAll('textarea[data-n]').forEach(function(t){
  t.onchange=function(){note(t.getAttribute('data-n'),t.value)}});
 document.getElementById('tot').textContent=D.items.length;
 document.getElementById('prog').textContent=Object.keys(votes).filter(isDone).length;
}
document.getElementById('cp').onclick=function(){
 const s=JSON.stringify({tag:D.tag,votes:votes},null,1);
 navigator.clipboard.writeText(s).then(
  function(){alert('Copied '+Object.keys(votes).length+' judgements. Paste into a file.')},
  function(){const w=window.open('');w.document.write('<pre>'+s.replace(/</g,'&lt;')+'</pre>')});
};
document.getElementById('rs').onclick=function(){
 const s=prompt('Paste a saved progress JSON');
 if(!s)return;try{votes=JSON.parse(s).votes||{};save();draw()}catch(e){alert('could not parse')}
};
draw();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--topics", default="data/vqa/topics/vqa_2025_mc_topics.csv")
    ap.add_argument("--qrels", default="data/vqa/qrels_mc.csv")
    ap.add_argument("--frames", default="data/vqa/frames_pool16_512")
    ap.add_argument("--video-dirs", nargs="*", default=["data/vqa/videos_full"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-agreed", type=int, default=30)
    ap.add_argument("--n-contested", type=int, default=12)
    ap.add_argument("--n-counting", type=int, default=8)
    ap.add_argument("--n-sham", type=int, default=10)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--take", type=int, default=0)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()

    topics = load_topics(args.topics)
    gold = load_gold(args.qrels, topics)
    runs = [load_run(p, topics) for p in args.runs]
    have = {p.name for p in Path(args.frames).iterdir()
            if p.is_dir() and any(p.glob("frame_*.jpg"))}
    common = sorted(set.intersection(*[set(r) for r in runs]) & set(gold))
    common = [q for q in common if topics[q]["video"] in have]

    contested, agreed = [], []
    for q in common:
        tops = {r[q][0] for r in runs if r.get(q)}
        (contested if len(tops) > 1 else agreed).append(q)
    counting = [q for q in common if re.match(r"^\s*how (many|much)\b", topics[q]["q"], re.I)]

    rng = random.Random(args.seed)
    def take(pool, n, used):
        c = [q for q in pool if q not in used]
        rng.shuffle(c)
        return c[:n]

    used = set()
    sel = []
    for strat, n in (("agreed", args.n_agreed), ("contested", args.n_contested),
                     ("counting", args.n_counting)):
        pool = {"agreed": agreed, "contested": contested, "counting": counting}[strat]
        got = take(pool, n, used)
        used |= set(got)
        sel += [(q, strat) for q in got]
    print(f"pools: agreed {len(agreed)} | contested {len(contested)} | counting {len(counting)}")

    # sham: show a DISTRACTOR as the proposed answer. Without this the defect
    # rate is unreadable — rubber-stamping and clean golds look identical.
    sham_ids = set(rng.sample([q for q, _ in sel], min(args.n_sham, len(sel))))

    rng.shuffle(sel)
    key, items = {}, []
    tiny = Path(args.out).parent / "tiny"
    for q, strat in sel:
        opts = topics[q]["options"]
        g = gold[q]
        is_sham = q in sham_ids
        if is_sham:
            alts = [o for o in opts if o != g]
            if not alts:
                is_sham = False
                proposed = g
            else:
                proposed = rng.choice(alts)
        else:
            proposed = g
        f = frames_for(args.frames, topics[q]["video"])
        if not f:
            continue
        v = tiny_video(topics[q]["video"], args.video_dirs, tiny)
        items.append(dict(id=q, q=html.escape(topics[q]["q"]),
                          a=html.escape(proposed),
                          o=[html.escape(o) for o in opts], f=f, v=v))
        key[q] = dict(stratum=strat, sham=is_sham, proposed=proposed, gold=g,
                      question=topics[q]["q"], video=topics[q]["video"])

    if args.take:
        items = items[args.offset:args.offset + args.take]
        key = {k: v for k, v in key.items() if k in {i["id"] for i in items}}

    tag = "triage_" + Path(args.out).stem
    page = PAGE.replace("__DATA__", json.dumps({"tag": tag, "items": items}))

    body = page.split("<script>", 1)[1].rsplit("</script>", 1)[0]
    for ln, line in enumerate(body.splitlines(), 1):
        stripped = re.sub(r"\\.", "", line)
        if stripped.count("'") % 2 or stripped.count('"') % 2:
            raise SystemExit(f"REFUSING TO WRITE: unbalanced quote in script line {ln}:\n  {line[:120]}")

    Path(args.out).write_text(page, encoding="utf-8")
    Path(args.out).with_suffix(".key.json").write_text(
        json.dumps({"tag": tag, "items": key}, indent=1), encoding="utf-8")
    nsham = sum(1 for v in key.values() if v["sham"])
    print(f"Wrote {args.out} ({Path(args.out).stat().st_size/1e6:.1f} MB) — "
          f"{len(items)} items, {nsham} sham, video on "
          f"{sum(1 for i in items if i['v'])}/{len(items)}")


if __name__ == "__main__":
    main()

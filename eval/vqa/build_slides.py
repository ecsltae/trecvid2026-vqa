"""
Generate the TRECVID VQA 2026 explainer deck (self-contained HTML).

A long-form deck: it explains the task, the evaluation, how the system was
actually built step by step, every experiment we ran, and what went wrong in the
process. Written so someone who has never heard of VQA can follow it end to end.

Videos and frames are embedded as base64 so it works offline with no network.

LICENSING: only YouTube-sourced videos from the 2025 development set are shown.
The 2026 test set's 1088 .webm clips come from the KINO LORBER / MCOLL
collection under a NIST data agreement restricting redistribution.

Transcode the clips first (audio matters — one example is about sound):
    mkdir -p /tmp/slidevid
    for v in WvT2Ba7IGaY pEO3-LtxNxw tjDArsnB6Ik; do
      src=$(ls data/vqa/videos_full/$v.* | head -1)
      ffmpeg -y -loglevel error -i "$src" -t 20 -vf "scale=400:-2,fps=15" \
        -c:v libx264 -crf 32 -preset veryfast -c:a aac -b:a 40k \
        -movflags +faststart /tmp/slidevid/$v.mp4
    done
    python eval/vqa/build_slides.py --out vqa2026_slides.html
"""
import argparse
import base64
import io
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
VIDDIR = Path("/tmp/slidevid")


def video_b64(vid):
    p = VIDDIR / f"{vid}.mp4"
    return ("data:video/mp4;base64," + base64.b64encode(p.read_bytes()).decode()
            if p.exists() else None)


def player(vid, caption=""):
    uri = video_b64(vid)
    if not uri:
        return filmstrip(vid, caption or "[clip not transcoded — frames shown]")
    cap = f'<div class="dim small vidcap">{caption}</div>' if caption else ""
    return (f'<div class="vidwrap"><video controls preload="metadata" playsinline '
            f'src="{uri}"></video>{cap}</div>')


def frames_b64(vid, n=4, width=240, quality=70):
    d = ROOT / "data/vqa/frames_pool16" / vid
    fs = sorted(d.glob("frame_*.jpg"))
    if not fs:
        return []
    step = (len(fs) - 1) / (n - 1) if n > 1 and len(fs) > 1 else 0
    out = []
    for i in range(min(n, len(fs))):
        im = Image.open(fs[round(i * step)]).convert("RGB")
        im.thumbnail((width, width))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
        out.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
    return out


def filmstrip(vid, caption=""):
    imgs = frames_b64(vid)
    if not imgs:
        return '<div class="dim small">[frames unavailable]</div>'
    tiles = "".join(f'<div class="fr"><img src="{u}"><span>t{i+1}</span></div>'
                    for i, u in enumerate(imgs))
    cap = f'<div class="dim small vidcap">{caption}</div>' if caption else ""
    return f'<div class="strip">{tiles}</div>{cap}'


CSS = """
:root{--bg:#0d1117;--panel:#161b22;--panel2:#1c2330;--ink:#e6edf3;--dim:#8b949e;
--line:#30363d;--accent:#58a6ff;--good:#3fb950;--warn:#d29922;--bad:#f85149;
--purple:#bc8cff;--cyan:#39c5cf;--orange:#f0883e}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{background:var(--bg);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
overflow:hidden}
#deck{height:100vh;width:100vw;position:relative}
.slide{position:absolute;inset:0;display:none;flex-direction:column;
padding:28px 44px 16px;overflow-y:auto;overflow-x:hidden;animation:fade .25s ease}
.slide.active{display:flex}
.slide::-webkit-scrollbar{width:9px}
.slide::-webkit-scrollbar-thumb{background:var(--line);border-radius:5px}
@keyframes fade{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
h1{font-size:34px;line-height:1.15;margin-bottom:6px}
h2{font-size:23px;margin-bottom:10px;border-left:4px solid var(--accent);padding-left:13px;flex:none}
h3{font-size:14.5px;color:var(--cyan);margin:7px 0 4px}
p,li{font-size:14px;line-height:1.44}
.dim{color:var(--dim)}.small{font-size:12.5px}
.sub{font-size:15.5px;color:var(--dim);margin-bottom:10px}
ul,ol{margin-left:18px}li{margin:3px 0}
.tag{display:inline-block;background:var(--panel2);border:1px solid var(--line);
border-radius:20px;padding:3px 12px;font-size:12px;color:var(--accent);margin:2px 4px 2px 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:12px 14px}
.kpi{font-size:26px;font-weight:700;line-height:1.1}
.good{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.acc{color:var(--accent)}.pur{color:var(--purple)}.cy{color:var(--cyan)}.org{color:var(--orange)}
footer{margin-top:auto;padding-top:8px;border-top:1px solid var(--line);
display:flex;justify-content:space-between;font-size:11px;color:var(--dim);flex:none}
.progress{position:absolute;top:0;left:0;height:3px;background:var(--accent);
transition:width .3s;z-index:5}
.two{display:grid;grid-template-columns:1fr 1fr;gap:15px}
.three{display:grid;grid-template-columns:1fr 1fr 1fr;gap:11px}
.vid2{display:grid;grid-template-columns:minmax(0,330px) 1fr;gap:17px;align-items:start}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid var(--line);padding:5px 9px;font-size:13px;text-align:left;
vertical-align:top}
th{background:var(--panel2);color:var(--cyan)}
code{background:var(--panel2);padding:2px 5px;border-radius:4px;font-size:12px;color:var(--cyan)}
pre{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:10px 12px;
font-size:11.5px;line-height:1.4;overflow-x:auto;color:var(--ink)}
.quote{background:var(--panel2);border-left:3px solid var(--purple);padding:8px 12px;
border-radius:6px;font-size:13px;margin:6px 0}
.big-center{justify-content:center}
.vidwrap{width:100%}
.vidwrap video{width:100%;max-height:38vh;border-radius:9px;border:1px solid var(--line);
background:#000;display:block}
.vidcap{margin-top:5px}
.strip{display:flex;gap:7px}
.fr{position:relative;flex:1}
.fr img{width:100%;border-radius:6px;border:1px solid var(--line);display:block}
.fr span{position:absolute;bottom:3px;left:5px;font-size:10px;background:rgba(0,0,0,.65);
padding:1px 5px;border-radius:3px;color:#fff}
.qa{background:var(--panel);border:1px solid var(--line);border-radius:9px;
padding:9px 12px;margin-top:7px}
.scorepill{display:inline-block;border-radius:14px;padding:1px 9px;font-size:12px;
font-weight:700;margin-left:7px;white-space:nowrap}
.s-good{background:rgba(63,185,80,.16);color:var(--good);border:1px solid var(--good)}
.s-mid{background:rgba(210,153,34,.16);color:var(--warn);border:1px solid var(--warn)}
.s-bad{background:rgba(248,81,73,.16);color:var(--bad);border:1px solid var(--bad)}
.flow{display:flex;align-items:stretch;gap:7px;flex-wrap:wrap}
.step{background:var(--panel);border:1px solid var(--line);border-radius:9px;
padding:8px 10px;font-size:12.5px;flex:1;min-width:112px}
.runrow{display:grid;grid-template-columns:74px 1fr;gap:11px;align-items:start;
padding:8px 0;border-bottom:1px solid var(--line)}
.runrow:last-child{border-bottom:none}
.runbadge{background:var(--panel2);border:1px solid var(--line);border-radius:8px;
padding:5px 6px;text-align:center;font-weight:700;font-size:12.5px}
.sect{font-size:12px;letter-spacing:2px;color:var(--purple);text-transform:uppercase;
margin-bottom:4px}
.divider h1{font-size:40px}
"""

JS = """
const slides=[...document.querySelectorAll('.slide')];let i=0;
function show(n){slides[i].classList.remove('active');i=(n+slides.length)%slides.length;
const s=slides[i];s.classList.add('active');s.scrollTop=0;
document.getElementById('prog').style.width=((i+1)/slides.length*100)+'%';
document.querySelectorAll('.pageno').forEach(e=>e.textContent=(i+1)+' / '+slides.length);
slides.forEach(sl=>sl.querySelectorAll('video').forEach(v=>{if(!sl.classList.contains('active'))v.pause()}));}
document.addEventListener('keydown',e=>{
if(e.target.tagName==='VIDEO')return;
if(['ArrowRight','PageDown','Enter'].includes(e.key)){e.preventDefault();show(i+1)}
if(['ArrowLeft','PageUp','Backspace'].includes(e.key)){e.preventDefault();show(i-1)}
if(e.key==='Home')show(0); if(e.key==='End')show(slides.length-1);});
show(0);
"""

SUBMISSION_JSON = """[
  {
    "Q_ID": 1,
    "Video_ID": "8384.webm",
    "Answers": [
      { "Rank": 1,  "Answer": "Image fades into mist",                "Time": 31.82 },
      { "Rank": 2,  "Answer": "Image fades to mist",                  "Time": 31.82 },
      { "Rank": 3,  "Answer": "Image fades into mist then disappears", "Time": 31.82 },
      { "Rank": 4,  "Answer": "Image fades to mist as they walk past", "Time": 31.82 },
      ...
      { "Rank": 10, "Answer": "They walk toward the camera",          "Time": 31.82 }
    ]
  },
  ...  2443 entries, one per question
]"""


def pill(s):
    cls = "s-good" if s >= 0.6 else ("s-mid" if s >= 0.3 else "s-bad")
    return f'<span class="scorepill {cls}">STS {s:.3f}</span>'


def foot():
    return ('<footer><span>TRECVID VQA 2026 · Team BiTeM</span>'
            '<span>← → navigate</span><span class="pageno"></span></footer>')


def divider(kicker, title, blurb):
    return (f'<section class="slide big-center divider"><div style="max-width:900px">'
            f'<div class="sect">{kicker}</div><h1>{title}</h1>'
            f'<p class="sub" style="margin-top:8px">{blurb}</p></div>{foot()}</section>')


def build(out):
    S = []
    A = S.append

    # ══════════ TITLE ══════════
    A(f"""<section class="slide active big-center"><div style="max-width:950px;width:100%">
<div class="tag">TRECVID 2026 · Team BiTeM</div>
<h1>Video Question Answering <span class="acc">explained</span></h1>
<p class="sub">The task, the evaluation, how we built a system for it, and everything we
measured — including what failed</p>
<div class="three" style="margin-top:18px">
<div class="card"><div class="kpi acc">2,443</div><div class="dim small">questions in the 2026 test set</div></div>
<div class="card"><div class="kpi pur">1,956</div><div class="dim small">videos to understand</div></div>
<div class="card"><div class="kpi good">4</div><div class="dim small">validated runs submitted</div></div>
</div>
<p class="dim small" style="margin-top:16px">Clips are embedded and playable. All illustrations use
publicly available YouTube videos from the 2025 development set; the 2026 movie clips are under a
NIST data agreement and are not reproduced here.</p>
</div>{foot()}</section>""")

    # ══════════ PART 1 · THE TASK ══════════
    A(divider("Part one", "The task",
              "What TRECVID VQA asks for, what the data looks like, and why it is hard"))

    A(f"""<section class="slide"><h2>What is the VQA track?</h2>
<p class="sub">TRECVID is NIST's long-running video evaluation. VQA joined in 2025.</p>
<div class="two"><div>
<div class="card"><h3>The task in one sentence</h3>
<p>Given a <b>video</b> and a <b>question in plain English</b>, answer the question by actually
watching the video — not by guessing from the question alone.</p></div>
<div class="card" style="margin-top:11px"><h3>What a "shared task" means</h3>
<p>NIST ships identical videos and questions to every team, collects the runs, scores them against
one hidden answer key, and publishes a leaderboard. Teams then write a paper describing how their
runs were produced — <b>including what failed</b>.</p>
<p style="margin-top:6px">Nobody can quietly tune on the test set, and nobody can report only their
best result. It is measurement, not demonstration.</p></div></div><div>
<div class="card"><h3>Why video is harder than images</h3>
<p>A still image has no <i>before</i> and <i>after</i>. Video questions routinely require ordering
events, inferring causes, reading on-screen text, listening to speech, and combining several
observations into one answer.</p>
<p style="margin-top:6px">Our system does not watch video at all, strictly speaking: it sees
<b>16 still frames</b> plus, where relevant, a speech transcript. Motion between frames is lost.</p></div>
<div class="card" style="margin-top:11px"><h3>The field is small</h3>
<p>2025 drew <b>7 teams / 20 runs</b> for answer generation. The winner scored 0.351 on the main
metric; most systems sat between 0.20 and 0.30. This is a reachable bar, not a saturated benchmark.</p></div>
</div></div>{foot()}</section>""")

    A(f"""<section class="slide"><h2>Two tasks, same videos, different deadlines</h2>
<div class="two">
<div class="card"><h3 class="acc">Task 1 · Answer Generation (AG)</h3>
<p>Write <b>exactly 10 answers</b>, ranked best-first, in free text, plus the time each took.</p>
<div class="quote">"What happens when the clown gets on his knees?"<br>
&rarr; <b>1.</b> The dancers behind him start dancing<br>&rarr; <b>2.</b> …</div>
<p class="small dim">Fewer than 10 answers, or any duplicate, and the run is <b class="bad">rejected
outright</b>. Test data 17 Jul, deadline <b>17 Aug 2026</b>.</p></div>
<div class="card"><h3 class="pur">Task 2 · Multiple Choice (MC)</h3>
<p>Four options supplied; rank them most- to least-likely correct.</p>
<div class="quote">A) the 3 dancers behind start to dance ✓<br>B) he stops dancing<br>
C) he cries in pain<br>D) he gets back up</div>
<p class="small dim">Scored on top-1 accuracy and MRR. Test data 21 Aug, deadline
<b>18 Sep 2026</b>.</p></div></div>
<div class="card" style="margin-top:12px"><h3>Why the split matters enormously</h3>
<p>The same system scores ~<b class="good">75%</b> on multiple choice but ~<b class="warn">0.27</b>
on answer generation. <i>Recognising</i> a correct answer among four candidates is a fundamentally
easier problem than <i>producing</i> one — and, as the next section shows, the generation metric is
unforgiving about how you phrase it even when you are right.</p></div>
<p class="small dim" style="margin-top:8px">Up to 5 runs per team per task. Results released to
teams 28 September; workshop at NIST 30 Nov – 3 Dec.</p>
{foot()}</section>""")

    A(f"""<section class="slide"><h2>The 2026 data — what actually arrived</h2>
<div class="three" style="margin-bottom:12px">
<div class="card"><div class="kpi acc">2,443</div><div class="dim small">questions</div></div>
<div class="card"><div class="kpi pur">1,956</div><div class="dim small">unique videos</div></div>
<div class="card"><div class="kpi cy">9.6</div><div class="dim small">mean words per question</div></div>
</div>
<div class="two">
<div class="card"><h3>Two very different video sources</h3>
<table style="margin-top:5px">
<tr><th>Source</th><th>Videos</th><th>Questions</th></tr>
<tr><td>Movie clips (<code>.webm</code>)</td><td>1,088</td><td>1,575 (64%)</td></tr>
<tr><td>YouTube IDs</td><td>868</td><td>868 (36%)</td></tr>
</table>
<p class="small dim" style="margin-top:7px">The movie clips ship in a zip and come from the KINO
LORBER collection — they require a signed data agreement, which is why NIST gates the download.
The YouTube half you must fetch yourself, and some of it has since been deleted.</p></div>
<div class="card"><h3>Duration is bimodal, and that matters</h3>
<p>YouTube clips: median <b>15s</b>, none over 30s. Movie clips: median <b>9s</b> but a long tail —
<b>20% exceed a minute</b>, up to 3.4 minutes.</p>
<p style="margin-top:6px">A fixed frame budget is therefore wrong. Eight frames on a 206-second clip
is one glimpse every 26 seconds. We sample a <b>duration-aware</b> number of frames instead.</p></div>
</div>
<div class="card" style="margin-top:12px"><h3>What a question looks like</h3>
<p class="small">Questions are short and blunt — 3 to 29 words. <i>"What happens to the image of the
bride and groom?"</i> · <i>"How many times does the young man cut himself?"</i> ·
<i>"Why might the field be verdant?"</i></p></div>
{foot()}</section>""")

    A(f"""<section class="slide"><h2>Seven question categories</h2>
<p class="sub">The organisers define seven types. Percentages below are our own keyword estimate over
the 2026 test set — indicative, not the official split.</p>
<table>
<tr><th style="width:150px">Category</th><th style="width:56px">~Share</th><th>Real examples from the 2026 test set</th></tr>
<tr><td><b class="acc">Causal</b><br><span class="small dim">why something happened</span></td><td>15%</td>
<td class="small">"Why is the young man so happy to get his GED."<br>
"Why did the women stop their car out in a flat scrub land area?"</td></tr>
<tr><td><b class="acc">Temporal</b><br><span class="small dim">order / sequence</span></td><td>17%</td>
<td class="small">"What does the text say after 'Seoul's past, present, future'?"<br>
"What prop is the first young man wearing as he reads a script?"</td></tr>
<tr><td><b class="acc">Counting</b><br><span class="small dim">quantities</span></td><td>13%</td>
<td class="small">"How many guitarists are there in the band?"<br>
"How many stacks of boxes are there in the room?"</td></tr>
<tr><td><b class="acc">Audio / multimodal</b><br><span class="small dim">speech, music, on-screen text</span></td><td>7%</td>
<td class="small">"what was the sound that followed the boys when they entered the discotek"<br>
"Why did the boy, with the yellow shirt, sing at the end of the video?"</td></tr>
<tr><td><b class="acc">People–object</b><br><span class="small dim">interaction</span></td><td>2%</td>
<td class="small">"To whom does the woman give her face mask?"<br>
"How did the man use his knife and fork, when he ate in the restaurant?"</td></tr>
<tr><td><b class="acc">Multi-hop</b> · <b class="acc">Common-sense</b></td><td>rest</td>
<td class="small">Chaining several observations, or applying world knowledge not visible in frame.</td></tr>
</table>
<p class="small dim" style="margin-top:9px">Categories are not evenly weighted and the mix changed
between 2025 and 2026 — the 2025 set was balanced 20% per category, so a system tuned on it is
tuned to the wrong distribution.</p>
{foot()}</section>""")

    # ══════════ PART 2 · EXAMPLES ══════════
    A(divider("Part two", "Three worked examples",
              "One the system gets right, one it gets wrong, and one that exposed a whole missing sense"))

    A(f"""<section class="slide"><h2>Example 1 — a question we answer correctly</h2>
<div class="vid2">{player('WvT2Ba7IGaY', 'Press play. The system never sees this motion — only 16 stills.')}
<div>
<div class="qa"><b class="cy">Question</b><br>what is the breed of the third dog in the video?</div>
<div class="qa"><b class="dim">Human reference answer</b><br>Doberman Pincher</div>
<div class="qa"><b class="acc">Our answer</b><br>Doberman Pinscher {pill(1.0)}</div>
<p class="small dim" style="margin-top:8px">To answer this the model must count dogs, identify
<i>the third one specifically</i>, and know dog breeds. Note the <i>reference</i> misspells
"Pinscher" — the metric compares meaning-bearing words, so the typo costs nothing. A system that
"corrected" the human would score identically.</p>
</div></div>{foot()}</section>""")

    A(f"""<section class="slide"><h2>Example 2 — a question we get wrong</h2>
<div class="vid2">{player('pEO3-LtxNxw', 'Crowds and religious architecture dominate every frame.')}
<div>
<div class="qa"><b class="cy">Question</b><br>What are the monks in the background doing?</div>
<div class="qa"><b class="dim">Human reference answer</b><br>Sweeping the floor with whisk brooms.</div>
<div class="qa"><b class="bad">Our answer</b><br>They are walking around the Kaaba {pill(0.0)}</div>
<p class="small dim" style="margin-top:8px">This is a <b>perception</b> failure, not a wording
problem. The question asks about a small <i>background</i> detail; the model describes the salient
foreground instead — a very human mistake, and a completely wrong answer.</p>
<p class="small dim" style="margin-top:6px">About <b>30%</b> of our lowest-scoring answers look like
this. No amount of clever post-processing repairs them; only better perception would.</p>
</div></div>{foot()}</section>""")

    A(f"""<section class="slide"><h2>Example 3 — the question that revealed a missing sense</h2>
<div class="vid2">{player('tjDArsnB6Ik', '⚠ Turn the sound ON — the answer exists only in the audio.')}
<div>
<div class="qa"><b class="cy">Question</b><br>What are the lyrics heard at the beginning of this video?</div>
<div class="qa"><b class="dim">Human reference</b><br>I got guns in my head and they won't go</div>
<div class="qa"><b class="bad">Frames only</b><br>"No lyrics heard at beginning" {pill(0.0)}</div>
<div class="qa"><b class="good">Frames + speech transcript</b><br>
"I got guns in my head and they won't go" {pill(1.0)}</div>
<p class="small dim" style="margin-top:7px">The model was not wrong so much as <b>deaf</b>. It
answered honestly that it heard nothing — because it was never given anything to hear. Adding an
automatic transcript moved this class of question from <b>0.35 to 0.53</b>.</p>
</div></div>{foot()}</section>""")

    A(f"""<section class="slide"><h2>What a submission actually is</h2>
<p class="sub">A single JSON file per run. 2443 entries, exactly 10 ranked answers each, plus the
wall-clock seconds spent generating them.</p>
<pre>{SUBMISSION_JSON}</pre>
<div class="two" style="margin-top:10px">
<div class="card"><h3 class="bad">Three ways to get rejected</h3>
<ul class="small"><li><b>Fewer than 10 answers</b> on any question</li>
<li><b>Duplicate answer text</b> within a question — compared after whitespace collapsing and
case-folding, so "Image fades" and "image  fades" collide</li>
<li>Missing or extra <code>Q_ID</code>s</li></ul></div>
<div class="card"><h3 class="good">How we guarantee it</h3>
<p class="small">Our builder assembles the JSON, then <b>shells out to NIST's own validator</b> and
fails the build if it reports a single error. That is stricter than checking it ourselves, and it
caught two real bugs our own checks had passed.</p></div></div>
{foot()}</section>""")

    # ══════════ PART 3 · HOW VIDEO QA WORKS ══════════
    A(divider("Part three", "How video question answering works",
              "The mechanics underneath any such system, and what the state of the art looks like"))

    A(f"""<section class="slide"><h2>From pixels to tokens — what the model actually receives</h2>
<p class="sub">A vision-language model does not "watch" anything. It reads a sequence of numbers
that happen to have been derived from images.</p>
<div class="flow" style="margin:10px 0">
<div class="step"><b class="cy">Frame</b><br><span class="small dim">one still image, e.g. 512×512</span></div>
<div class="step"><b class="cy">Patches</b><br><span class="small dim">cut into a grid of small tiles (typically 14×14 px)</span></div>
<div class="step"><b class="cy">Vision encoder</b><br><span class="small dim">a ViT turns each patch into a vector</span></div>
<div class="step"><b class="cy">Projector</b><br><span class="small dim">maps those vectors into the language model's token space</span></div>
<div class="step"><b class="cy">LLM</b><br><span class="small dim">reads them interleaved with your question, as if they were words</span></div>
</div>
<div class="two">
<div class="card"><h3>Why this framing matters</h3>
<p>Once projected, image-derived vectors sit in the <i>same space</i> as word tokens. The language
model is doing what it always does — predicting the next token given a context — except part of
that context came from a camera.</p>
<p style="margin-top:6px">This is why the same model can answer, refuse, hedge, or hallucinate about
a video exactly the way it does about text. Everything you know about prompting still applies.</p></div>
<div class="card"><h3>And why "16 frames" is a hard architectural limit, not laziness</h3>
<p>Each frame costs <b>hundreds of tokens</b> after pooling. Sixteen frames can occupy several
thousand tokens of context before the question is even added.</p>
<p style="margin-top:6px">Frames are not free to add. They compete directly with the context the
model needs to reason, and they cost time and memory linearly.</p></div></div>
{foot()}</section>""")

    A(f"""<section class="slide"><h2>The central trade-off — you are always throwing video away</h2>
<div class="two"><div>
<div class="card"><h3>The arithmetic</h3>
<p>A two-minute clip at 30 fps is <b>3,600 frames</b>. We look at <b>16</b>.</p>
<p style="margin-top:6px">That is <b>0.4%</b> of the footage. Every video QA system is answering
questions about a video it has mostly not seen — the design question is only <i>which</i> 0.4%.</p></div>
<div class="card" style="margin-top:11px"><h3>Three dials, and they fight</h3>
<ul><li><b>More frames</b> → better temporal coverage, more tokens, slower</li>
<li><b>Higher resolution</b> → finer detail per frame, far more tokens</li>
<li><b>Longer context</b> → room for both, but slower and often worse at using it</li></ul>
<p class="small dim" style="margin-top:6px">Our own measurement: dropping to 512px cost nothing
measurable and ran <b>4.8× faster</b>. Detail we were paying for was not being used.</p></div></div>
<div>
<div class="card"><h3>What gets lost</h3>
<p><b>Motion.</b> Sixteen stills contain no movement. "How did he fall?" becomes a question about
before-and-after snapshots, and anything happening between them is invisible.</p>
<p style="margin-top:6px"><b>Short events.</b> A one-second action in a two-minute clip has roughly a
one-in-eight chance of appearing at all under uniform sampling.</p>
<p style="margin-top:6px"><b>Sound</b>, unless you explicitly add it — most systems are deaf by
default, and one of the seven official question categories is about audio.</p></div>
<div class="card" style="margin-top:11px"><h3>The honest framing</h3>
<p class="small">Almost every "video" QA system, ours included, is really a <b>multi-image</b> QA
system with a temporal ordering hint. Genuinely video-native architectures exist, but the
frame-sampling approach still dominates because it reuses image models directly.</p></div></div></div>
{foot()}</section>""")

    A(f"""<section class="slide"><h2>Four ways people build these systems</h2>
<table>
<tr><th style="width:170px">Approach</th><th>How it works</th><th style="width:230px">Trade-off</th></tr>
<tr><td><b class="acc">Uniform sampling</b><br><span class="small dim">what we do, and most teams</span></td>
<td class="small">Take N evenly spaced frames, hand them all to the model with the question.</td>
<td class="small">Simple, robust, no extra components. Blind to anything between samples; wastes
budget on redundant frames in static scenes.</td></tr>
<tr><td><b class="acc">Query-conditioned selection</b></td>
<td class="small">Score every frame for relevance to <i>this</i> question (e.g. with CLIP), then feed
only the top-scoring ones. A retrieval step before the QA step.</td>
<td class="small">Much better on long video, where the answer is in a few seconds out of many
minutes. Adds a component that can itself be wrong — and if it drops the right frame, nothing
downstream can recover.</td></tr>
<tr><td><b class="acc">Caption-then-reason</b></td>
<td class="small">A vision model writes a description of the frames; a text-only LLM answers from
that description.</td>
<td class="small">Lets you pair a small vision model with a strong reasoner. But the caption is
written <i>without knowing the question</i>, so it discards exactly the detail being asked about.
<b>We measured this: it lost to asking directly.</b></td></tr>
<tr><td><b class="acc">Video-native / agentic</b></td>
<td class="small">Either genuine temporal modelling across frames, or a model that iteratively
requests more frames, zooms, or calls tools until it can answer.</td>
<td class="small">The strongest results on long-video benchmarks. Needs either a purpose-built
architecture or many sequential model calls.</td></tr>
</table>
<div class="card" style="margin-top:11px"><h3>Where self-consistency fits</h3>
<p>Orthogonal to all four. Instead of changing <i>what</i> the model sees, run it several times and
keep what recurs. It costs linear compute and needs no new components — which is why it was the
right choice for a one-week local build, and why it produced our largest gain.</p></div>
{foot()}</section>""")

    A(f"""<section class="slide"><h2>What the state of the art looks like</h2>
<p class="sub">The reference benchmark is <b>Video-MME</b>: 900 videos, 254 hours, 2,700 human
annotated questions, clip lengths from 11 seconds to an hour, with subtitles and audio available.</p>
<div class="two"><div>
<div class="card"><h3>Frontier scores (multiple choice)</h3>
<table style="margin-top:5px">
<tr><th>Model</th><th>Video-MME</th></tr>
<tr><td>Seed 2.1 Pro (ByteDance)</td><td class="good"><b>89.2%</b></td></tr>
<tr><td>Gemini 2.5 Pro</td><td class="good"><b>84.8%</b></td></tr>
</table>
<p class="small dim" style="margin-top:6px">Both are large proprietary systems with long context
windows and native audio-visual input. Open-weight families — Qwen-VL, InternVL, LLaVA-OneVision —
trail but are closing, and are what most academic teams actually use.</p></div>
<div class="card" style="margin-top:11px"><h3>Where the research effort goes</h3>
<ul class="small"><li><b>Token compression</b> — fit more frames in the same context
(e.g. selective-scan compression)</li>
<li><b>Long-context scaling</b> — architectures built for hour-long input</li>
<li><b>Temporal search</b> — find <i>when</i> the answer happens, then look closely</li>
<li><b>Memory</b> — streaming video without re-reading everything</li></ul></div></div>
<div>
<div class="card"><h3 class="warn">Why those numbers are not comparable to ours</h3>
<p>Video-MME is <b>multiple choice</b>: pick A/B/C/D, scored as accuracy. TRECVID AG asks for
<b>free-text generation</b>, scored by string similarity against one human reference.</p>
<p style="margin-top:6px">An 89% on Video-MME and a 0.26 on TRECVID AG are not the same axis. The
generation task additionally penalises phrasing, length, and word choice — none of which
multiple choice tests at all.</p></div>
<div class="card" style="margin-top:11px"><h3 class="cy">The useful comparison</h3>
<p>TRECVID runs <b>both</b> formats on the same videos. That makes it one of the few settings where
you can separate "did the system see correctly?" from "did it say it well?" — which is exactly the
measurement on the next slide.</p></div></div></div>
{foot()}</section>""")

    A(f"""<section class="slide"><h2>Where our system actually sits</h2>
<p class="sub">An 8-billion-parameter open-weight model, run locally, against a fine-tuned
competition winner — separated into perception and expression.</p>
<div class="two">
<div class="card"><h3 class="good">Perception: at parity</h3>
<table style="margin-top:5px">
<tr><th>Multiple choice (same videos)</th><th>Top-1</th></tr>
<tr><td>Our system, off-the-shelf 8B</td><td class="good"><b>74.7%</b></td></tr>
<tr><td>2025 winner, fine-tuned</td><td><b>74.4%</b></td></tr>
</table>
<p class="small dim" style="margin-top:7px">On the format that tests <i>seeing</i> with phrasing
factored out, we match a system that was fine-tuned for this task. Our eyes are not the problem.</p></div>
<div class="card"><h3 class="bad">Expression: well behind</h3>
<table style="margin-top:5px">
<tr><th>Answer generation</th><th>METEOR</th></tr>
<tr><td>Our system</td><td class="bad"><b>0.159</b></td></tr>
<tr><td>2025 winner</td><td class="good"><b>0.260</b></td></tr>
</table>
<p class="small dim" style="margin-top:7px">A 39% relative deficit on the same videos, with
equivalent perception. The gap is in <b>which aspect of the scene we choose to describe</b> and how
we word it — not in what we can see.</p></div></div>
<div class="card" style="margin-top:12px"><h3 class="cy">Why this is the single most useful number we have</h3>
<p>It redirects all remaining effort. Every instinct says "the model is too small, get a bigger one"
— and we tested that: a 4× larger model scored <b>−0.006</b>. The multiple-choice parity explains
why. Perception was never the bottleneck, so buying more perception bought nothing.</p>
<p style="margin-top:6px">It also tells you what a genuine step change would require: not better
eyes, but learning which aspect a human annotator would have picked out, and saying it the way they
would have said it.</p></div>
{foot()}</section>""")

    # ══════════ PART 4 · THE METRIC ══════════
    A(divider("Part four", "How answers are scored",
              "The most surprising part of the task — and where most of the work went"))

    A(f"""<section class="slide"><h2>Five metrics in two families</h2>
<div class="two">
<div class="card"><h3 class="acc">Order-invariant (three)</h3>
<p><b>STS</b> (semantic similarity), <b>METEOR</b> (word overlap with stemming and synonyms), and
<b>BERTScore</b> (neural embedding overlap). Each is computed for all 10 answers and then
<b>averaged</b>, reported at the top-1 / top-5 / top-10 cutoffs.</p>
<p class="small dim" style="margin-top:6px">Because it is a <i>mean</i>, one brilliant answer is
dragged down by nine weak ones. At the top-10 cutoff, reordering the same ten answers cannot change
the number at all — that is arithmetic.</p></div>
<div class="card"><h3 class="pur">Order-sensitive (two)</h3>
<p><b>NDCG-METEOR</b> and <b>NDCG-BERTScore</b> reward putting your best answer first.</p>
<p class="small dim" style="margin-top:6px">So ranking is not free to ignore — it just cannot rescue
the three headline averages at the full cutoff. Ranking moves top-1, top-5, and these two.</p>
<p class="small dim" style="margin-top:6px">A quirk: with only one reference answer per question,
NDCG is close to degenerate — a system whose ten answers are all equally good scores a perfect 1.0.</p></div></div>
<div class="card" style="margin-top:12px"><h3 class="warn">Design consequence</h3>
<p>If your score is the <i>mean</i> of ten answers, then producing "one good answer plus nine
speculative alternatives" is actively self-harming. This single observation was worth more to us
than any change of model — see Part four.</p></div>
{foot()}</section>""")

    A(f"""<section class="slide"><h2>The aggregation trap — and how 2025 exposed it</h2>
<div class="two"><div>
<div class="card"><h3>What the 2025 winner noticed</h3>
<p>The scoring script averages each metric across <i>every answer you submit</i>. So submitting
fewer answers raises your average.</p>
<p style="margin-top:6px">They tested it: their run with a <b>single</b> answer per question scored
<b>0.351</b> — higher than their own 10-answer run at <b>0.348</b>, and the best STS of the entire
track.</p>
<p class="small dim" style="margin-top:6px">They documented this openly in their paper: <i>"the
presence of additional responses has a significant negative impact."</i></p></div>
<div class="card" style="margin-top:11px"><h3>NIST's response for 2026</h3>
<p>Exactly 10 answers are now <b>mandatory</b>, and duplicates are a hard validator error. You can
no longer opt out of the average by submitting less.</p></div></div>
<div>
<div class="card"><h3 class="warn">Why this still matters</h3>
<p>The loophole is closed, but the underlying incentive is not: <b>all ten of your answers are
averaged into your score</b>. Answers 2 through 10 are no longer throwaways — they are 90% of the
number.</p>
<p style="margin-top:7px">Most teams will still optimise answer 1 and let the rest fall where they
may. That is the single most exploitable fact about this evaluation, and it is publicly documented.</p></div>
<div class="card" style="margin-top:11px"><h3>What we did about it</h3>
<p class="small">Stopped asking the model for "ten ranked alternatives" and started asking it the
same question sixteen times, keeping what it agreed on. Detailed in Part four — it was our largest
single gain.</p></div></div></div>
{foot()}</section>""")

    A(f"""<section class="slide"><h2>The metric is strange (1) — same meaning, different score</h2>
<p class="sub">"STS" is not a neural model. It is a web service published in 2013 that compares
content words. Every number below was measured against the live service.</p>
<div class="card"><b class="cy">Reference:</b> &nbsp;"The girl fell to her knees."</div>
<table style="margin-top:9px">
<tr><th>Candidate answer</th><th style="width:82px">Score</th><th>Why</th></tr>
<tr><td>The girl fell to her knees.</td><td class="good"><b>1.000</b></td><td class="small dim">identical</td></tr>
<tr><td>She fell to her knees</td><td class="good"><b>0.880</b></td><td class="small dim">pronoun swap costs a little</td></tr>
<tr><td>The girl fell to her knees, possibly because she lost her balance while rollerblading</td><td class="good"><b>0.784</b></td><td class="small dim">padding dilutes, content still matches</td></tr>
<tr><td>She fell</td><td class="good"><b>0.761</b></td><td class="small dim">dropping words is cheap</td></tr>
<tr><td>In the video, the girl appears to fall to her knees</td><td class="warn"><b>0.758</b></td><td class="small dim">preamble is dead weight</td></tr>
<tr><td>Girl falls while rollerblading</td><td class="warn"><b>0.592</b></td><td class="small dim">true, same event, fewer shared words</td></tr>
</table>
<p class="small dim" style="margin-top:8px">Every one of these is arguably <b>correct</b>. The spread
from 1.000 to 0.592 is almost entirely phrasing. Roughly, the score is <i>matched content ÷ length
of the longer side</i> — so extra words divide your score, and the penalty is harshest when the
reference is short. Against a terse reference like "he fell down", that same preamble collapses the
score to <b>0.059</b>.</p>
{foot()}</section>""")

    A(f"""<section class="slide"><h2>The metric is strange (2) — unanswerable questions</h2>
<p class="sub">The scorer deletes common words before comparing. If the reference <i>is</i> a common
word, it becomes an empty string and matches nothing — including itself.</p>
<div class="two"><div>
<table>
<tr><th>Reference</th><th>Our answer</th><th>Score</th></tr>
<tr><td>one</td><td>one</td><td class="bad"><b>0.000</b></td></tr>
<tr><td>two</td><td>two</td><td class="bad"><b>0.000</b></td></tr>
<tr><td>yes</td><td>yes</td><td class="bad"><b>0.000</b></td></tr>
<tr><td>nothing</td><td>nothing</td><td class="bad"><b>0.000</b></td></tr>
<tr><td>first</td><td>first</td><td class="bad"><b>0.000</b></td></tr>
<tr><td class="dim">four</td><td class="dim">four</td><td class="good"><b>1.000</b></td></tr>
</table>
<p class="small dim" style="margin-top:8px">"four" is not on the stopword list; "one" and "two" are.
The difference is an accident of a 2013 word list, not a judgement about answer quality.</p></div>
<div><div class="card"><h3 class="warn">How much of the test set is unwinnable</h3>
<p>We scored every reference answer <b>against itself</b> to find each question's ceiling.
<b>7.5%</b> score zero no matter what — for us and every competing team alike.</p>
<p style="margin-top:7px">The best achievable average is therefore <b>0.925</b>, not 1.000.</p></div>
<div class="card" style="margin-top:11px"><h3>Why we now exclude them</h3>
<p class="small">Those dead questions add a fixed block of zeros to every system's average. When
comparing two of our own systems, that constant shrinks the visible difference between them and
makes real improvements look smaller than they are. All our internal numbers exclude them.</p></div>
<div class="card" style="margin-top:11px"><h3>The general lesson</h3>
<p class="small">Before optimising against any benchmark, compute what a <b>perfect</b> system would
score. If it is not 1.0, find out where the missing part went.</p></div></div></div>
{foot()}</section>""")

    A(f"""<section class="slide"><h2>How we learned all this</h2>
<p class="sub">None of the previous three slides is documented in the track guidelines. Finding it
out was most of the project.</p>
<div class="two"><div>
<div class="card"><h3 class="good">1 · The official scorer is public</h3>
<p>NIST publishes its evaluation code. We vendored it into the repository and read it line by line.
That is how we learned the aggregation is <code>np.mean</code> over all answers, that there are five
metrics rather than three, and that STS is a <code>wget</code> to a 2013 web service.</p></div>
<div class="card" style="margin-top:11px"><h3 class="good">2 · Their ground truth is our dev set</h3>
<p>The scorer ships with the 2025 answer key — and it is <b>identical</b> to the development data we
already had. So our 2000-question dev set <i>is</i> the official 2025 test set. We can reproduce the
published leaderboard locally and know where we would have placed.</p></div></div>
<div>
<div class="card"><h3 class="good">3 · We probed the live service</h3>
<p>Once we knew STS was an HTTP endpoint, we could ask it questions directly: how it treats
preambles, pronouns, digits, length. Those probes produced the tables you just saw, and a permanent
local cache of <b>10,703</b> scored pairs.</p></div>
<div class="card" style="margin-top:11px"><h3 class="warn">4 · Our own eval was measuring the wrong thing</h3>
<p>Our original harness reported the <b>best</b> of the ten answers using a neural similarity model.
The official metric takes the <b>mean</b> of ten answers using a 2013 web service. Those are not the
same number, and they reward opposite behaviour.</p>
<p class="small dim" style="margin-top:6px">Every conclusion drawn before we noticed had to be
re-tested. Several reversed.</p></div></div></div>
{foot()}</section>""")

    # ══════════ PART 4 · BUILDING ══════════
    A(divider("Part five", "Building the system",
              "Every stage, in the order we built it, and what went wrong at each"))

    A(f"""<section class="slide"><h2>The pipeline end to end</h2>
<div class="flow" style="margin:10px 0">
<div class="step"><b class="cy">1 · Get data</b><br><span class="small dim">2443 questions, 1956 videos, credential-gated</span></div>
<div class="step"><b class="cy">2 · Frames</b><br><span class="small dim">16 per video, duration-aware, 512px</span></div>
<div class="step"><b class="cy">3 · Transcribe</b><br><span class="small dim">Whisper over every video</span></div>
<div class="step"><b class="cy">4 · Generate ×16</b><br><span class="small dim">same question, different frame subsets</span></div>
<div class="step"><b class="cy">5 · Cluster</b><br><span class="small dim">group answers meaning the same thing</span></div>
<div class="step"><b class="cy">6 · Rank &amp; build</b><br><span class="small dim">slate of 10, validated JSON</span></div>
</div>
<div class="two">
<div class="card"><h3>Everything runs locally</h3>
<p>One machine, three GPUs, open-weight models through Ollama. No external API is called at any
point — a hard project constraint. The vision model is <code>qwen3-vl:8b-instruct</code>;
speech is <code>faster-whisper large-v3-turbo</code>.</p></div>
<div class="card"><h3>Everything is resumable</h3>
<p>Each stage writes progress as it goes and skips completed work on restart. Over a five-day build
with jobs running overnight, crash-safety mattered more than speed: no single failure ever cost more
than the topic it was working on.</p></div></div>
<p class="small dim" style="margin-top:9px">Total compute for one full run over 2443 questions:
roughly 10 hours on a single GPU, or about 15 seconds per question for 16 samples each.</p>
{foot()}</section>""")

    A(f"""<section class="slide"><h2>Stage 1 — getting the data</h2>
<p class="sub">Harder than it sounds, and the source of the project's most useful save.</p>
<div class="two"><div>
<div class="card"><h3>Three obstacles</h3>
<ol><li><b>Credentials.</b> The test set sits behind HTTP authentication for registered participants.</li>
<li><b>A data agreement.</b> The 1088 movie clips are licensed; NIST will not score any run from a
team that has not signed and returned the form.</li>
<li><b>YouTube rot.</b> 868 questions point at YouTube IDs you must fetch yourself.</li></ol></div>
<div class="card" style="margin-top:11px"><h3 class="warn">The 42% that wasn't real</h3>
<p>Our first download attempt failed on <b>371 of 868</b> videos — a 42.7% loss that would have
crippled the run.</p>
<p style="margin-top:6px">Rather than accept it, we sampled the errors. Most were genuine deletions
("account terminated"), but several were <i>"Sign in to confirm you're not a bot"</i>, and several
returned no error at all on a retry.</p></div></div>
<div>
<div class="card"><h3 class="good">The fix, and what it bought</h3>
<p>Retrying with alternate player clients and request throttling recovered <b>339 of the 371</b>.</p>
<table style="margin-top:7px">
<tr><th></th><th>First pass</th><th>After retry</th></tr>
<tr><td>YouTube videos</td><td>497 / 868</td><td class="win"><b>836 / 868</b></td></tr>
<tr><td>Questions with video</td><td>84.8%</td><td class="good"><b>98.7%</b></td></tr>
</table>
<p class="small dim" style="margin-top:7px">Only <b>32 questions</b> (1.3%) have no video at all —
genuinely deleted uploads. Those fall back to answering from the question alone.</p></div>
<div class="card" style="margin-top:11px"><h3>The lesson</h3>
<p class="small">A 42% failure rate that is really a 4% failure rate plus a misconfigured client is
the kind of thing you only find by <b>reading the errors instead of counting them</b>.</p></div></div></div>
{foot()}</section>""")

    A(f"""<section class="slide"><h2>Stage 2 — turning video into something a model can read</h2>
<div class="two"><div>
<div class="card"><h3>Duration-aware sampling</h3>
<p>We extract a pool of up to <b>16 frames</b> per video, evenly spaced across its whole duration,
then sample subsets from that pool at generation time.</p>
<p style="margin-top:6px">A fixed count would be wrong in both directions: wasteful on a 9-second
clip, hopelessly sparse on a 206-second one.</p></div>
<div class="card" style="margin-top:11px"><h3 class="good">512 pixels is free speed</h3>
<p>Downscaling frames to 512px on the long side made generation <b>4.8× faster</b> with no
measurable quality loss (+0.013, not significant). Larger frames buy nothing on this task and cost
five times the compute.</p></div></div>
<div>
<div class="card"><h3 class="bad">A bug worth describing</h3>
<p>The frame selector originally took the <b>first</b> N frames from the pool:</p>
<pre>sorted(frames)[:8]   # WRONG</pre>
<p>With a 16-frame pool on a long movie clip, that fed the model only the <b>opening quarter</b> of
the video and silently ignored the rest. Every question about a later event was unanswerable, and
nothing in the output looked broken.</p>
<pre>step = (len(frames)-1)/(n-1)
[frames[round(i*step)] for i in range(n)]   # spans the clip</pre>
<p class="small dim" style="margin-top:6px">Found by testing the function on a synthetic 32-frame
directory and printing which indices came back — not by looking at scores, where it would have hidden
indefinitely as "the model is bad at long videos".</p></div></div></div>
{foot()}</section>""")

    A(f"""<section class="slide"><h2>Stage 3 — giving the system ears</h2>
<div class="two"><div>
<div class="card"><h3>Scale</h3>
<p>Every video transcribed with Whisper large-v3-turbo: <b>1861 dev + 1925 test</b> videos at
<b>0.67 s/video</b> — the entire corpus in about 21 minutes on an otherwise idle GPU.</p>
<p class="small dim" style="margin-top:6px">Zero silent outputs, zero errors. Cheap enough that
there was never a resource argument against trying it.</p></div>
<div class="card" style="margin-top:11px"><h3 class="bad">The one-flag trap</h3>
<p>Whisper's voice-activity filter, commonly left on by default, returns an <b>empty transcript on
sung vocals</b>.</p>
<p style="margin-top:6px">That silently destroys exactly the questions audio was meant to answer —
lyrics, songs, music. With <code>vad_filter=False</code> the gold lyric line comes back verbatim;
with it on, nothing at all.</p></div></div>
<div>
<div class="card"><h3 class="good">Route, do not broadcast</h3>
<p>Dumping the transcript into every prompt is a wash. Split by question type and the picture
inverts:</p>
<table style="margin-top:6px">
<tr><th>Question is about…</th><th>Effect of transcript</th></tr>
<tr><td>what was <b>said</b> / sung</td><td class="good"><b>+0.083</b></td></tr>
<tr><td>what was <b>heard</b> (music, effects)</td><td class="bad"><b>−0.073</b></td></tr>
</table>
<p class="small dim" style="margin-top:7px">A transcript says nothing about which instrument is
playing — it just adds distracting text. So we classify the question first and inject only where it
helps.</p></div>
<div class="card" style="margin-top:11px"><h3>An awkward finding</h3>
<p class="small"><b>38%</b> of the videos are not in English — Hindi, Japanese, Russian, Korean.
A plausible reason audio helps us less than the literature suggests.</p></div></div></div>
{foot()}</section>""")

    A(f"""<section class="slide"><h2>Stage 4 — the change that mattered most</h2>
<p class="sub">Same model, same frames, same everything. Only the request changed.</p>
<div class="two">
<div class="card"><h3 class="bad">Before — one call, ten answers</h3>
<pre>"Provide 10 answers, ranked.
 Explore alternative interpretations.
 For lower ranks, give increasingly
 speculative alternatives."</pre>
<p class="small" style="margin-top:6px">This does exactly what it says: it <b>manufactures nine bad
answers</b>. And since the score is the mean of all ten, those nine are averaged straight into the
result. The instruction was actively engineering a lower score.</p></div>
<div class="card"><h3 class="good">After — sixteen calls, one answer each</h3>
<pre>"Answer in one short clause."
   × 16 independent samples
   × different random frame subsets</pre>
<p class="small" style="margin-top:6px">Then cluster the sixteen answers by meaning and rank by how
often the model re-derived each one. Ten genuinely generated answers, ordered by how much evidence
supports them.</p></div></div>
<div class="card" style="margin-top:12px"><h3>Why repeated sampling works</h3>
<p>Random variation in generation is uncorrelated with the truth, but a <b>correct</b> answer keeps
being re-derived from different views of the video. Answers that agree are evidence; answers that
scatter are hallucination. Using a different random subset of frames each time turns the video
itself into free data augmentation.</p>
<p style="margin-top:6px" class="good"><b>Measured gain: +0.070</b> — larger than every other change
we made, combined.</p></div>
<p class="small dim" style="margin-top:8px">A correction worth recording: we first credited this win
to the clustering step. Testing both ranking policies on identical candidate pools showed they score
the same to three decimals. <b>It is what you ask for, not how you sort it.</b></p>
{foot()}</section>""")

    A(f"""<section class="slide"><h2>Stage 5 — building a submission that will not be rejected</h2>
<div class="two"><div>
<div class="card"><h3>The rules are unforgiving</h3>
<ul class="small"><li>Exactly 10 answers per question — 9 is a rejection</li>
<li>No duplicate answer text after case-folding and whitespace collapsing</li>
<li>Every question present, exactly once</li>
<li>Ranks 1..10, numeric timing field</li></ul>
<p class="small dim" style="margin-top:6px">These two rules interact nastily: you must always produce
ten, and they must all be <b>distinct strings</b>. When a model only produces six distinct answers,
padding is required — but padding with copies is illegal.</p></div>
<div class="card" style="margin-top:11px"><h3>Sanitising for a 2013 scorer</h3>
<p class="small">The official scorer pastes answer text into a shell command and a URL without
escaping. So <code>&amp;</code> and <code>#</code> truncate your answer, <code>%</code> corrupts it,
and non-ASCII scores zero. Answers are sanitised before submission — silently losing points to an
unescaped ampersand is a miserable way to lose.</p></div></div>
<div>
<div class="card"><h3 class="good">Validate with their code, not yours</h3>
<p>Our builder writes the JSON and then <b>runs NIST's own validator</b> on it, failing the build on
any error. Checking your work with your own understanding of the rules only verifies that you are
self-consistent.</p>
<p class="small dim" style="margin-top:6px">It caught two real defects that our own checks had
passed.</p></div>
<div class="card" style="margin-top:11px"><h3>Final state of all four runs</h3>
<table style="margin-top:5px">
<tr><th>Check</th><th>Result</th></tr>
<tr><td>Questions answered</td><td class="good">2443 / 2443</td></tr>
<tr><td>Answers per question</td><td class="good">exactly 10</td></tr>
<tr><td>Duplicate answers</td><td class="good">0</td></tr>
<tr><td>Validator errors</td><td class="good">0</td></tr>
</table></div></div></div>
{foot()}</section>""")

    # ══════════ PART 5 · EXPERIMENTS ══════════
    A(divider("Part six", "What we tried",
              "Eleven strategies, measured the same way — two worked"))

    A(f"""<section class="slide"><h2>How we measure, and why it is easy to fool yourself</h2>
<div class="two"><div>
<div class="card"><h3 class="bad">The noise floor</h3>
<p>Running the <b>identical configuration twice</b> produces scores that differ by about
<b>0.015</b>. That is larger than most of the effects anyone reports on this task.</p>
<p style="margin-top:6px">So any comparison between two independently generated runs differing by
less than ~0.03 is indistinguishable from chance. Several published-sounding claims we started from
were inside that band.</p></div>
<div class="card" style="margin-top:11px"><h3 class="good">Paired testing</h3>
<p>Instead, every run dumps its raw candidate pool. Post-processing changes are then tested on the
<b>same fixed pool</b>, which removes generation randomness entirely and makes a 0.004 difference
readable.</p></div></div>
<div>
<div class="card"><h3>The three things that make this possible</h3>
<ul><li><b>A real answer key.</b> Our dev set is the official 2025 test set, so we score against
genuine human references, not a proxy.</li>
<li><b>The official metric.</b> We call the same service NIST calls, so a dev gain is a real gain.</li>
<li><b>A score cache.</b> 10,703 scored pairs stored locally — re-running an analysis is free and
survives the service going down.</li></ul></div>
<div class="card" style="margin-top:11px"><h3 class="warn">Excluding the dead questions</h3>
<p class="small">All comparisons drop the 7.5% of questions that are unwinnable. Leaving them in adds
a constant to both sides that compresses every difference toward zero.</p></div></div></div>
{foot()}</section>""")

    A(f"""<section class="slide"><h2>The full results table</h2>
<p class="sub">All measured against the official metric, paired where possible, dead questions
excluded. Anything under ~0.015 is noise.</p>
<table>
<tr><th>What we changed</th><th style="width:112px">Effect</th><th style="width:150px">Verdict</th></tr>
<tr><td>Ask for one best answer 16× instead of "10 ranked answers"</td><td class="good"><b>+0.070</b></td><td class="good">Works — biggest win</td></tr>
<tr><td>Speech transcripts, routed to "what was said" questions</td><td class="good"><b>+0.180</b></td><td class="good">Works — on 2.3% of questions</td></tr>
<tr><td>Downscale frames to 512px</td><td class="good"><b>4.8× faster</b></td><td class="good">Free speed</td></tr>
<tr><td>16 samples instead of 8</td><td class="warn">+0.014</td><td class="warn">Below noise floor</td></tr>
<tr><td>Show the model its candidates, ask which the video supports</td><td class="bad">−0.005</td><td class="bad">No effect</td></tr>
<tr><td>Rank by cluster mass instead of repeat count</td><td class="bad">−0.000</td><td class="bad">No effect</td></tr>
<tr><td>Use the 4× larger 30B model</td><td class="bad">−0.006</td><td class="bad">No effect, 1.35× slower</td></tr>
<tr><td>Mix candidates from two different models</td><td class="bad">+0.004</td><td class="bad">No effect</td></tr>
<tr><td>Rewrite answers into terser "annotator style"</td><td class="bad">+0.004</td><td class="bad">No effect</td></tr>
<tr><td>Pad the 10 slots with paraphrases of the best answer</td><td class="bad">+0.005</td><td class="bad">No effect — and illegal</td></tr>
<tr><td>Strip preambles / hedges / fix counting answers</td><td class="bad">0.000</td><td class="bad">Nothing left to fix</td></tr>
<tr><td>Replace "the man" with "he" to match reference style</td><td class="bad">−0.004</td><td class="bad">Significantly <b>harmful</b></td></tr>
<tr><td>Aggressively truncate answers</td><td class="bad">−0.050</td><td class="bad">Actively harmful</td></tr>
</table>
<p class="small dim" style="margin-top:8px">Two wins out of thirteen. Both are about <b>what the
model is given or asked</b>; every single post-processing idea measured zero.</p>
{foot()}</section>""")

    A(f"""<section class="slide"><h2>Failure worth understanding (1) — every selection rule loses</h2>
<div class="two"><div>
<div class="card"><h3 class="bad">The result</h3>
<table style="margin-top:5px">
<tr><th>Selection policy</th><th>Score</th><th>vs baseline</th></tr>
<tr><td><b>repeat count</b> (ours)</td><td><b>0.3342</b></td><td>—</td></tr>
<tr><td>cluster mass</td><td>0.3340</td><td class="dim">−0.000</td></tr>
<tr><td><b>ask the model</b></td><td>0.3294</td><td class="bad">−0.005</td></tr>
<tr><td>longest answer</td><td>0.3273</td><td class="bad">−0.007</td></tr>
<tr><td>closest to typical length</td><td>0.3267</td><td class="bad">−0.008</td></tr>
<tr><td>most central answer</td><td>0.3196</td><td class="bad">−0.015</td></tr>
<tr><td>shortest answer</td><td>0.2929</td><td class="bad">−0.041</td></tr>
</table>
<p class="small dim" style="margin-top:7px">We showed the model the video <i>and</i> its own ten
candidates and asked which the footage supports. Verification over a fixed set ought to be easier
than open-ended generation. It lost. So did every other rule — nothing beat simply counting which
answer the model repeated most.</p></div></div>
<div>
<div class="card"><h3 class="warn">The trap we nearly published</h3>
<p>The obvious next line is: the <b>best</b> of 16 candidates scores 0.468 against 0.334 for our
top pick, so a perfect selector would gain <b>+0.134</b> — four times our biggest real improvement.</p>
<p style="margin-top:7px"><b class="bad">That number is almost entirely an artifact.</b> Taking the
maximum of 16 noisy measurements beats a random draw <i>even when there is no signal at all</i>.
A null model with zero real selection ability reproduces <b>+0.132 of the +0.134</b>.</p></div>
<div class="card" style="margin-top:11px"><h3>Two further problems with it</h3>
<ul class="small"><li>It is a <b>top-1</b> quantity, but our runs are scored on <b>mean@10</b>, where
the real selection headroom is <b>+0.0095</b> — fourteen times smaller, because only 28.5% of
questions even have more than 10 distinct candidates to choose between.</li>
<li>"Rank-1 is best only 35% of the time" is the <b>expected</b> rate for 16 near-tied candidates.
It is arithmetic, not a defect.</li></ul></div>
<div class="card" style="margin-top:11px"><h3 class="cy">What survives</h3>
<p class="small">The <b>negative</b> finding, which is the publishable one: every selection policy we
tried failed to beat plain repeat-counting. "There is +0.134 lying on the table" is not a finding —
it is a statistician's oldest trap, selecting on the outcome.</p></div></div></div>
{foot()}</section>""")

    A(f"""<section class="slide"><h2>Failure worth understanding (2) — a fix that backfires</h2>
<div class="two"><div>
<div class="card"><h3>The reasoning, which sounds airtight</h3>
<p><b>53.5%</b> of human reference answers use a pronoun ("he", "she", "they"). And the metric scores
<code>sim("the man", "he") = 0.0</code> — a total mismatch.</p>
<p style="margin-top:6px">So rewriting our answer "<i>The man kicks the ball</i>" as "<i>He kicks the
ball</i>" should align us with how annotators actually write. Cheap, rule-based, obviously correct.</p></div>
<div class="card" style="margin-top:11px"><h3 class="bad">It measurably hurts</h3>
<p><b>−0.0035</b>, at <b>−2.9 standard errors</b>. Not noise — a reliable, repeatable loss.</p></div></div>
<div>
<div class="card"><h3>Why</h3>
<p>Pronouns are <b>stopwords</b>. The scorer strips them before comparing. So:</p>
<pre>"The man kicks the ball"  →  man kicks ball
"He kicks the ball"       →  kicks ball</pre>
<p style="margin-top:6px">The rewrite does not swap one referring expression for another — it
<b>deletes a content word</b> and replaces it with nothing. "man" was contributing match mass;
"He" contributes zero.</p></div>
<div class="card" style="margin-top:11px"><h3 class="warn">The general lesson</h3>
<p class="small">The premise was a real measured fact. The conclusion drawn from it was exactly
backwards, because the mechanism was never traced end to end. A plausible causal story plus a real
statistic is still not evidence.</p></div></div></div>
{foot()}</section>""")

    A(f"""<section class="slide"><h2>The quiet negative — nothing left to clean</h2>
<div class="two"><div>
<div class="card"><h3>What we expected to gain</h3>
<p>The metric punishes preambles, hedges, and verbose counting answers. We built rule-based fixes for
all three, expecting a solid gain:</p>
<ul class="small"><li>strip "In the video, …" framing</li>
<li>drop ", possibly because …" hedges</li>
<li>reduce "There are five people" to "five"</li></ul></div>
<div class="card" style="margin-top:11px"><h3 class="bad">What happened</h3>
<table style="margin-top:5px">
<tr><th>Transform</th><th>Answers touched</th><th>Gain</th></tr>
<tr><td>preamble strip</td><td>3%</td><td>0.000</td></tr>
<tr><td>hedge removal</td><td><b>0%</b></td><td>0.000</td></tr>
<tr><td>counting fix</td><td><b>0%</b></td><td>0.000</td></tr>
</table></div></div>
<div>
<div class="card"><h3 class="good">Why they did nothing</h3>
<p>Not because the rules were broken — they were unit-tested and work correctly on the failure cases
they target. They found nothing because <b>the generation contract had already eliminated those
failures upstream</b>.</p>
<p style="margin-top:7px">Asking for "one short clause, no framing, no hedging, number word alone for
counting" produced answers with no preambles to strip and no hedges to remove.</p></div>
<div class="card" style="margin-top:11px"><h3 class="cy">The most transferable finding here</h3>
<p><b>Fix the request and the cleanup has nothing left to clean.</b> We could have shipped an elegant
post-processing layer, measured it against an unfixed baseline, and reported a healthy gain — for a
component that is worthless once the prompt is right.</p></div></div></div>
{foot()}</section>""")

    # ══════════ PART 6 · RUNS AND CLOSING ══════════
    A(divider("Part seven", "What we submitted",
              "Five run slots, used as an experiment rather than five attempts at the same thing"))

    A(f"""<section class="slide"><h2>Run strategy — what each submission is for</h2>
<p class="sub">Five runs are allowed per team. Used well, the leaderboard itself becomes a
measurement you could not otherwise make.</p>
<div class="card">
<div class="runrow"><div class="runbadge acc">RUN 1</div><div>
<b>consensus · 8 samples</b> — the safe baseline.
<span class="dim small">Built and validated first, before any experiment could fail, so a complete
submission always existed. Insurance against the rest of the week going wrong.</span></div></div>
<div class="runrow"><div class="runbadge acc">RUN 2</div><div>
<b>consensus · 16 samples</b> — the value of test-time compute.
<span class="dim small">Doubles the sampling budget, everything else identical to Run 1. Isolates one
variable: does thinking longer help? Dev estimate +0.014, inside the noise band — the leaderboard
gives a second, independent reading.</span></div></div>
<div class="runrow"><div class="runbadge pur">RUN 3</div><div>
<b>v3-style control</b> — deliberately <i>not</i> our best system.
<span class="dim small">Reproduces the old "ten increasingly speculative answers" contract. We expect
it to score worst. Submitting it measures the cost of the aggregation trap on the official metric,
with everything else held constant — the single result we most want to publish.</span></div></div>
<div class="runrow"><div class="runbadge good">RUN 4</div><div>
<b>16 samples + routed speech transcripts</b> — the audio system.
<span class="dim small">Whisper transcripts injected only into questions about what was said. Worth
+0.18 on that slice; those are ~2.3% of questions, so a small global effect but a large categorical
one, and the first system of ours that can hear.</span></div></div>
<div class="runrow"><div class="runbadge warn">RUN 5</div><div>
<b>not submitted</b> — and that is a decision, not an omission.
<span class="dim small">The candidate was broadcasting transcripts to every question. The scoring
service went down before we could measure it, and submitting an unmeasured run on deadline day risks
nothing useful. Four measured runs beat five where one is a guess.</span></div></div>
</div>
{foot()}</section>""")

    A(f"""<section class="slide"><h2>What went wrong in the process</h2>
<p class="sub">The engineering failures are as instructive as the experimental ones, and are usually
left out of write-ups.</p>
<div class="two"><div>
<div class="card"><h3 class="bad">A result computed and silently discarded</h3>
<p>The verifier experiment ran correctly and its number was computed — then dropped, because the
reporting loop iterated a <b>hardcoded list</b> of policy names that did not include the new one.</p>
<p class="small dim" style="margin-top:6px">It would have been reported as "still pending"
indefinitely. Fixed by iterating whatever was actually computed.</p></div>
<div class="card" style="margin-top:11px"><h3 class="bad">A job that waited on itself</h3>
<p>A chained script waited for a previous stage using a process-name match — which matched
<b>its own command line</b>. It waited seven hours for itself to finish.</p></div></div>
<div>
<div class="card"><h3 class="bad">Two GPUs left idle for days</h3>
<p>The machine has three GPUs. We used one, and queued jobs behind each other — a dev run and a full
run halving each other's speed — because an early check misread the device list.</p>
<p class="small dim" style="margin-top:6px">Once noticed, the audio work ran on an idle card in
parallel at zero cost to anything else.</p></div>
<div class="card" style="margin-top:11px"><h3 class="warn">The scorer went offline</h3>
<p>The UMBC service that computes the official STS metric became unreachable on <b>12 August</b> and
stayed down through the deadline. Our runs did not need it — but the two final experiments could not
be scored, which is why there is no Run 5.</p>
<p class="small dim" style="margin-top:6px">Reported to the track organisers, since their scorer
depends on the same service.</p></div></div></div>
{foot()}</section>""")

    A(f"""<section class="slide"><h2>Where we stand</h2>
<div class="three" style="margin-bottom:13px">
<div class="card"><div class="kpi good">4 / 5</div><div class="dim small">runs submitted, all validator-clean</div></div>
<div class="card"><div class="kpi acc">98.7%</div><div class="dim small">of questions have their video</div></div>
<div class="card"><div class="kpi warn">0.259</div><div class="dim small">our STS vs 0.351 for the 2025 winner</div></div>
</div>
<div class="two">
<div class="card"><h3>Honest assessment</h3>
<p>We are <b>not</b> ahead of the 2025 winner. The comparison is softer than it looks — their
"10-answer" run averaged 2.84 answers once duplicates collapsed, which the 2026 rules now forbid, so
every team's number will move — but we should not pretend the gap is an artifact.</p>
<p style="margin-top:6px">What we do have is four complete, valid, measured runs and an unusually
clear account of why each one is what it is.</p></div>
<div class="card"><h3>What would actually move us</h3>
<ol><li><b>Better perception.</b> 30% of our worst answers are simply wrong about what is on screen.
No post-processing touches that.</li>
<li><b>Aspect selection.</b> Our perception matches the 2025 winner on multiple choice (74.7% vs 74.4%), yet our generated answers score 39% worse — we describe the wrong thing, not the wrong scene.</li>
<li><b>Audio for sound questions</b>, which needs an audio captioner rather than speech
recognition.</li></ol></div></div>
<div class="card" style="margin-top:12px"><h3 class="cy">Next: Task 2</h3>
<p>Multiple choice, deadline 18 September. Same videos — so the frames and transcripts are already
built — and a far friendlier metric: ~75% accuracy instead of 0.259, no 2013 web service, no
unwinnable questions.</p></div>
{foot()}</section>""")

    A(f"""<section class="slide"><h2>What we would tell the next team</h2>
<div class="two"><div>
<div class="card"><h3 class="good">1 · Measure the evaluator first</h3>
<p>More than half this project was discovering what the score actually rewards: that it averages
rather than takes your best, that it is a 2013 web service with a stopword list, that 7.5% of
questions are unwinnable, that four words of preamble can cost 94% of a correct answer.</p>
<p class="small dim" style="margin-top:6px">None of this is in the guidelines. All of it is in the
published scorer, which anyone can read.</p></div>
<div class="card" style="margin-top:10px"><h3 class="good">2 · Fix the request before the output</h3>
<p>Our one large win was changing what we asked the model for. Every downstream cleanup we then
built measured exactly zero, because there was nothing left to clean.</p></div></div>
<div>
<div class="card"><h3 class="good">3 · Know your noise floor</h3>
<p>Identical configurations differ by 0.015 here. Without paired testing on fixed candidate pools we
would have "confirmed" at least three effects that do not exist — and one earlier conclusion in this
project was exactly that mistake.</p></div>
<div class="card" style="margin-top:10px"><h3 class="good">4 · A bigger model is not a plan</h3>
<p>Four times the parameters bought <b>−0.006</b> and ran 1.35× slower. The 2025 winner ran a 7B
model on a single consumer GPU. Method beat scale on this task, consistently.</p></div>
<div class="card" style="margin-top:10px"><h3 class="cy">5 · Publish the negatives</h3>
<p class="small">Eleven of our thirteen experiments failed. Two of those failures — a model that
cannot recognise its own best answer, and a well-motivated rewrite that backfires because pronouns
are stopwords — are more useful to the next team than our score is.</p></div></div></div>
{foot()}</section>""")

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BiTeM · TRECVID VQA 2026 — Video Question Answering explained</title>
<style>{CSS}</style></head><body>
<div id="deck"><div class="progress" id="prog"></div>
{''.join(S)}
</div><script>{JS}</script></body></html>"""
    Path(out).write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({len(S)} slides, {Path(out).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="vqa2026_slides.html")
    build(ap.parse_args().out)

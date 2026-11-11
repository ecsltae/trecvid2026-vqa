#!/usr/bin/env bash
# qwen3.5:122b — replicate first, then production if it holds.
#
# WHY A REPLICATE BEFORE 30 HOURS. The single 122b measurement is 74.65%, above
# every 32b replicate (max 72.03%) and +3.42pp over the 32b mean of 71.23%. But
# this project has now been burned repeatedly by single runs: four identical-
# config 8b runs span 4.02pp and one of their six pairs returns McNemar p=0.0169
# from nothing. 6 GPU-hours to check the point estimate is cheap against 30.
#
# 122b needs 83.7 GiB, so it takes all three cards with every other model
# unloaded. Nothing else can run while it does.
#
# DECISION RULE, fixed before the data: run production if the replicate lands
# within 2.0pp of 74.65% AND above the 32b maximum of 72.03%. If it lands below
# the 32b max, 122b is inside the 32b distribution and the 30 hours buy a
# diverse run rather than a better one -- still worth a spare slot, but say so.
set -u
cd .
source "${VENV:-.venv}/bin/activate"
L=logs
say () { echo "[122b] $* $(date '+%F %H:%M:%S')"; }

for p in 11434 11435 11436 11437; do
  for m in qwen3-vl:32b-instruct qwen3-vl:30b-a3b-instruct qwen3-vl:8b-instruct; do
    OLLAMA_HOST=127.0.0.1:$p ollama stop "$m" >/dev/null 2>&1 || true
  done
done
sleep 20
cat > /tmp/start_ollama_all.sh <<'INNER'
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2
export OLLAMA_HOST=127.0.0.1:11438
export OLLAMA_MODELS=${OLLAMA_MODELS:-$HOME/.ollama}/models
export OLLAMA_KEEP_ALIVE=24h
exec ollama serve
INNER
chmod +x /tmp/start_ollama_all.sh
# Kill only a PREVIOUS server, never this process tree. Scanning /proc for a
# literal string matches THIS script and the shell that launched it, because the
# heredoc above puts that string in their command lines -- the same self-match
# that killed a shell via pgrep four times before. Match the running binary and
# its port, and skip our own ancestry.
_self=$$; _ppid=$PPID
for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
  [ "$p" = "$_self" ] || [ "$p" = "$_ppid" ] && continue
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null || true)
  case "$c" in
    *"ollama serve"*)
      e=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -c '^OLLAMA_HOST=127.0.0.1:11438$' || true)
      [ "${e:-0}" -gt 0 ] && kill "$p" 2>/dev/null
      ;;
  esac
done
nohup /tmp/start_ollama_all.sh > /tmp/ollama_all.log 2>&1 &
sleep 30
say "all-GPU ollama up"

BASE="--recipe v3 --no-draft --parse-repair --stage2-predict 1024 --num-ctx 16384 \
      --think off --model qwen3.5:122b --call-timeout 600"

# ---- replicate
OLLAMA_HOST=127.0.0.1:11438 python systems/vqa/run_vqa_mc_v5.py $BASE \
  --transcript-mode off --seed 20260910 \
  --topics /tmp/e2_topics.json --frames data/vqa/frames_pool16_512 \
  --out /tmp/e9_122b_r2.csv > "$L/e9_122b_r2.log" 2>&1
say "replicate done"

python - <<'PY' | tee /tmp/e9_verdict.txt
import csv,json
from collections import defaultdict
T={str(x['Q_ID']):x for x in json.load(open('/tmp/e2_topics.json'))}
def o(x): return [q['option'] if isinstance(q,dict) else q for q in x['Options']]
g={}
for r in csv.DictReader(open('data/vqa/dev8/dev8_qrels.csv',newline='',encoding='utf-8')):
    q=(r.get('Q_ID') or '').strip(); a=(r.get('Answer') or '').strip()
    if q in T:
        L="ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(o(T[q]))]
        if a and a[0].upper() in L: g[q]=o(T[q])[L.index(a[0].upper())]
        elif a in o(T[q]): g[q]=a
def load(p):
    rows=defaultdict(list); out={}
    for r in csv.reader(open(p,newline='',encoding='utf-8')):
        if len(r)>=4:
            try: rk=int(r[2])
            except ValueError: continue
            rows[r[0].strip()].append((rk,r[3]))
    for q,v in rows.items(): out[q]=[a for _,a in sorted(v)]
    return out
A=load('/tmp/e8_122b.csv'); B=load('/tmp/e9_122b_r2.csv')
c=[q for q in set(A)&set(B) if q in g]
a1=sum(1 for q in c if A[q][0]==g[q])/len(c); a2=sum(1 for q in c if B[q][0]==g[q])/len(c)
print(f"122b r1 {a1:.2%}  r2 {a2:.2%}  mean {(a1+a2)/2:.2%}  spread {abs(a1-a2):.2%}")
print(f"32b max was 72.03%, 32b mean 71.23%")
go = a2 >= 0.7203 and abs(a2-a1) <= 0.02
print("GO" if go else "NOGO")
PY

if grep -q "^GO" /tmp/e9_verdict.txt; then
  say "replicate holds -> production"
else
  say "replicate did NOT hold -> running production anyway as a DIVERSE spare-slot run, not as a better one"
fi

D=runs/vqa/mc/2026/prod122b/chunks
rm -rf "$D"; mkdir -p "$D/claims"
python - <<PY
import json
d=sorted(json.load(open('data/vqa/2026/task2/vqa.2026.testingDataset.Task2_v2.json')),key=lambda x:x['Q_ID'])
CH=25; n=(len(d)+CH-1)//CH
for i in range(n): json.dump(d[i::n],open(f"$D/chunk_{i:03d}.json","w"))
print(f"  {n} chunks")
PY
for c in "$D"/chunk_*.json; do
  b=$(basename "$c" .json)
  [ -f "$D/$b.done" ] && continue
  OLLAMA_HOST=127.0.0.1:11438 python systems/vqa/run_vqa_mc_v5.py $BASE \
    --adaptive --transcripts /tmp/transcripts_2026_translate.jsonl --transcript-mode routed \
    --topics "$c" --frames data/vqa/2026/frames --out "$D/$b.csv" --resume \
    >> "$L/prod122b.log" 2>&1
  want=$(python -c "import json;print(len(json.load(open('$c'))))")
  have=$(cut -d, -f1 "$D/$b.csv" 2>/dev/null | sort -u | wc -l)
  [ "$want" -eq "$have" ] && touch "$D/$b.done"
  echo "[122b] $b $have/$want $(date +%H:%M:%S)"
done
say "PRODUCTION DONE"

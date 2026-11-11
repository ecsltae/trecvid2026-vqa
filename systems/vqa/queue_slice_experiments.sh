#!/usr/bin/env bash
# Runs the targeted-slice experiments once the 2026 baseline pool is empty.
#
# Waits on STATE (chunk .done count), never on process liveness — a worker
# restarting for two minutes previously looked like "finished" and triggered a
# premature merge of 1630/2443 topics.
set -u
cd .
source venv/bin/activate
D=runs/vqa/mc/2026/baseline8b/chunks
while [ "$(ls "$D"/*.done 2>/dev/null | wc -l)" -lt 98 ]; do sleep 120; done
echo "[slices] baseline pool empty, starting $(date +%H:%M:%S)"

run () {  # $1 label, $2 slice csv, $3.. extra args
    lbl="$1"; slice="$2"; shift 2
    OLLAMA_HOST=127.0.0.1:11434 python systems/vqa/run_vqa_mc_v5.py \
        --recipe v3 --model qwen3-vl:8b-instruct --num-ctx 16384 \
        --topics "$slice" --frames data/vqa/frames_pool16_512 \
        --out "/tmp/slice_$lbl.csv" "$@" >> logs/slice_experiments.log 2>&1
    echo "[slices] done $lbl $(date +%H:%M:%S)"
}

# --- TEMPORAL: the only significant win so far (+7.5pp, p=0.022, at 12 frames).
#     Find where it saturates; 16 is the pool size so 20 is not available.
run temporal_f8  /tmp/slices/temporal.csv --frames-per-sample 8
run temporal_f12 /tmp/slices/temporal.csv --frames-per-sample 12
run temporal_f16 /tmp/slices/temporal.csv --frames-per-sample 16

# --- AUDIO: two arms. Transcripts can only address the 33% speech subset, but
#     MORE FRAMES may answer sound-EVENT questions visually — you can often see
#     the instrument. That hypothesis has never been tested and is the cheaper one.
run audio_f8    /tmp/slices/audio.csv --frames-per-sample 8
run audio_f16   /tmp/slices/audio.csv --frames-per-sample 16
run audio_asr   /tmp/slices/audio.csv --frames-per-sample 8 --transcripts /tmp/transcripts_audio_en.jsonl
run audio_f16asr /tmp/slices/audio.csv --frames-per-sample 16 --transcripts /tmp/transcripts_audio_en.jsonl

# --- CONTROL: an intervention that wins on its slice and loses elsewhere is a
#     net loss, and only the control shows it.
run general_f16 /tmp/slices/general.csv --frames-per-sample 16
echo "[slices] ALL DONE $(date +%H:%M:%S)"

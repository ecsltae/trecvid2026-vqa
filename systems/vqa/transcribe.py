"""
Transcribe VQA videos with faster-whisper, pinned to a specific GPU.

WHY THIS EXISTS. One of the seven official question categories is
Audio/Multimodal, and our system is completely deaf. Both 2025 top teams used
audio; every team without it named that as a limitation in their write-up.

THE CRITICAL FLAG. `vad_filter=True` (a common default) returns an EMPTY
transcript on sung vocals and truncates song openings — precisely the questions
audio is supposed to answer. Measured on dev: with VAD on, a video whose gold
answer is "Jeffrey Bezos" transcribes to nothing; with VAD off it comes back
verbatim. We therefore force vad_filter=False.

GPU PINNING. Ollama holds GPU 1 for the VLM runs. Whisper is plain PyTorch, so
--device-index puts it on an otherwise idle card and costs the running VLM jobs
nothing. Check `nvidia-smi` first and pick a card at 0%.

Usage:
    python systems/vqa/transcribe.py \\
        --videos data/vqa/videos_full \\
        --out data/vqa/transcripts_dev.jsonl \\
        --device-index 2 [--model large-v3-turbo] [--resume]
"""
import argparse
import json
import os
import time
from pathlib import Path

VIDEO_EXT = {".webm", ".mp4", ".mkv", ".mov", ".avi", ".m4a", ".wav"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--device-index", type=int, default=0,
                    help="CUDA device index — pick an idle one, Ollama usually holds another")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--beam-size", type=int, default=1)
    ap.add_argument("--language", default=None,
                    help="force a language. Auto-detect hallucinates a language on "
                         "NON-SPEECH audio (music, engine noise), which produced 30.6%% "
                         "mostly-non-ASCII garbage on the dev set — hi/ja/ru/ko/zh on "
                         "clips with no speech at all. Forcing 'en' removes that failure "
                         "mode at the cost of genuinely non-English speech.")
    ap.add_argument("--task", default="transcribe", choices=["transcribe","translate"],
                    help="'translate' emits ENGLISH regardless of source language. Forcing "
                         "--language en on Hindi/Chinese speech produces transliterated "
                         "nonsense; auto-detect + translate is the correct handling.")
    ap.add_argument("--max-no-speech", type=float, default=1.0,
                    help="drop segments whose no_speech_prob exceeds this (1.0 = keep all). "
                         "Whisper marks its own hallucinations here; 0.6 is a good cut.")
    ap.add_argument("--min-logprob", type=float, default=-99.0,
                    help="drop segments below this avg_logprob (-1.0 is a good cut)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    files = []
    for d in args.videos:
        for p in sorted(Path(d).rglob("*")):
            if p.is_file() and p.suffix.lower() in VIDEO_EXT:
                files.append(p)
    if args.limit:
        files = files[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            line = line.strip()
            if line:
                done.add(json.loads(line)["video_id"])
        print(f"Resuming: {len(done)} already transcribed")

    todo = [p for p in files if p.stem not in done]
    print(f"Transcribing {len(todo)}/{len(files)} videos | model={args.model} "
          f"| cuda:{args.device_index} | vad_filter=False")

    model = WhisperModel(args.model, device="cuda", device_index=args.device_index,
                         compute_type=args.compute_type)

    n_empty = n_err = 0
    t0 = time.time()
    with open(out_path, "a" if done else "w", encoding="utf-8") as fout:
        for i, p in enumerate(todo, 1):
            rec = {"video_id": p.stem, "text": "", "segments": [], "language": None}
            try:
                segs, info = model.transcribe(
                    str(p),
                    beam_size=args.beam_size,
                    language=args.language,
                    task=args.task,
                    vad_filter=False,          # never True: kills sung vocals
                    condition_on_previous_text=False,
                )
                # Whisper reports its own confidence per segment. On non-speech
                # audio it emits fluent nonsense with no_speech_prob near 1.0, so
                # filtering here is what separates a transcript from a hallucination.
                segs = [s_ for s_ in segs
                        if getattr(s_, "no_speech_prob", 0.0) <= args.max_no_speech
                        and getattr(s_, "avg_logprob", 0.0) >= args.min_logprob]
                rec["dropped"] = 0
                rec["text"] = " ".join(s.text.strip() for s in segs).strip()
                rec["segments"] = [{"start": round(s.start, 2), "end": round(s.end, 2),
                                    "text": s.text.strip()} for s in segs]
                rec["language"] = getattr(info, "language", None)
                if not rec["text"]:
                    n_empty += 1
            except Exception as e:
                n_err += 1
                rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if i % 100 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                print(f"  [{i}/{len(todo)}] {rate:.2f}s/video — ETA "
                      f"{rate*(len(todo)-i)/60:.0f}min (empty={n_empty}, err={n_err})",
                      flush=True)

    print(f"Wrote {out_path} ({len(todo)} transcribed, {n_empty} silent, {n_err} errored)")


if __name__ == "__main__":
    main()

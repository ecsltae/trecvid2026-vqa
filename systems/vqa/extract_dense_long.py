"""
Denser frame pools for the LONG clips only.

WHY THIS EXISTS. `--pool` is a flat 16 regardless of duration, and the 2026
collection is bimodal: the YouTube third is ~15s (0.94s between pool frames)
while the movie two-thirds runs to 389s. 212 videos exceed 40s -- median 110s --
and they carry 479 questions, 19.6% of the submission. At 16 frames those get a
6.9s step at the median and 24s at the worst.

WHY EVERY FRAME EXPERIMENT MISSED IT. All of them ran on dev, whose longest video
is 30.4s. We concluded "frames do not matter" from a population where the pool
was already dense. That conclusion does not transfer to the movie corpus, and
CANNOT be tested on dev -- there is no long-video labelled data anywhere.

WHAT THIS CHANGES, HONESTLY. Pool size alone is not the binding constraint: the
model still sees only `--frames-per-sample` images per call. A denser pool helps
only when paired with a larger per-call budget. Used as intended --
`--frames data/vqa/2026/frames_dense --frames-per-sample 16` -- a 110s clip goes
from 13.8s between the frames the model actually sees to 6.9s. That is the whole
mechanism; it is unvalidated and unvalidatable before the deadline, which is why
it belongs in a spare submission slot rather than replacing a measured run.

    python systems/vqa/extract_dense_long.py --long /tmp/long_videos.json \
        --out data/vqa/2026/frames_dense --sec-per-frame 2 --min-pool 32 --max-pool 64
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def cap_evenly(out_dir: Path, pool: int):
    frames = sorted(out_dir.glob("frame_*.jpg"))
    if len(frames) <= pool:
        return len(frames)
    keep = {frames[round(i * (len(frames) - 1) / (pool - 1))] for i in range(pool)}
    for f in frames:
        if f not in keep:
            f.unlink()
    return len(list(out_dir.glob("frame_*.jpg")))


def one(job):
    vid, src, dur, out_root, spf, lo, hi = job
    out_dir = Path(out_root) / vid
    if out_dir.exists() and len(list(out_dir.glob("frame_*.jpg"))) >= lo:
        return (vid, "skip", len(list(out_dir.glob("frame_*.jpg"))))
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = max(lo, min(hi, math.ceil(dur / spf)))
    # a single direct fps, not the two-stage fps*1.5 + thin: that produced a PAIR
    # grid (8 tight pairs separated by 7 double-width holes), which wastes one
    # frame in eight of the budget on a near-duplicate
    fps = min(max(pool / dur, 0.02), 8.0)
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src),
           "-vf", f"fps={fps:.6f},scale='min(512,iw)':-2",
           "-q:v", "3", str(out_dir / "frame_%04d.jpg")]
    try:
        subprocess.run(cmd, capture_output=True, timeout=900, check=False)
    except subprocess.TimeoutExpired:
        return (vid, "TIMEOUT", 0)
    n = cap_evenly(out_dir, pool)
    return (vid, "ok" if n else "EMPTY", n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", required=True, help="JSON [[video_id, src_path, duration], ...]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sec-per-frame", type=float, default=2.0)
    ap.add_argument("--min-pool", type=int, default=32)
    ap.add_argument("--max-pool", type=int, default=64)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    long = json.loads(Path(args.long).read_text())
    jobs = [(v, s, d, args.out, args.sec_per_frame, args.min_pool, args.max_pool)
            for v, s, d in long]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    counts, done = {"ok": 0, "skip": 0, "EMPTY": 0, "TIMEOUT": 0}, 0
    with ProcessPoolExecutor(args.workers) as ex:
        for vid, st, n in ex.map(one, jobs):
            counts[st] = counts.get(st, 0) + 1
            done += 1
            if done % 25 == 0:
                print(f"  [{done}/{len(jobs)}] {counts}", flush=True)
    print(f"Done. {counts}")

    sizes = sorted(len(list((Path(args.out) / v).glob("frame_*.jpg"))) for v, _, _ in long)
    if sizes:
        print(f"  pool sizes: min {sizes[0]}, median {sizes[len(sizes)//2]}, max {sizes[-1]}")


if __name__ == "__main__":
    main()

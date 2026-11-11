"""
Parallel frame extraction for the TRECVID 2026 AG test set.

Differs from extract_frames.py in three ways that matter here:
  * runs ffmpeg across all cores instead of one video at a time (1956 videos);
  * extracts a POOL of frames (default 16) rather than the 8 we feed the model,
    because the self-consistency ensemble samples random 8-frame subsets from
    the pool — temporal jitter as free augmentation, which needs spare frames;
  * handles the 2026 mix of sources: local .webm movie clips (Video_ID carries
    the extension, e.g. "8384.webm") and downloaded YouTube videos of assorted
    containers. Frames are keyed by the stem so get_frames() finds either.

Usage:
    python systems/vqa/extract_frames_2026.py \
        --videos data/vqa/2026/videos/videos data/vqa/2026/videos_yt \
        --out data/vqa/2026/frames --pool 16 --workers 12
"""
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

VIDEO_EXTS = (".webm", ".mp4", ".mkv", ".m4v", ".avi", ".mov")


def probe_duration(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60)
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def cap_evenly(out_dir, pool):
    frames = sorted(out_dir.glob("frame_*.jpg"))
    if len(frames) <= pool:
        return len(frames)
    step = len(frames) / pool
    keep = {frames[int(i * step)] for i in range(pool)}
    for f in frames:
        if f not in keep:
            f.unlink()
    # renumber so frame_0001..frame_000N are contiguous
    for i, f in enumerate(sorted(keep), 1):
        target = out_dir / f"frame_{i:04d}.jpg"
        if f != target:
            f.rename(target)
    return pool


def extract_one(args):
    video_path, out_root, pool = args
    video_path = Path(video_path)
    out_dir = Path(out_root) / video_path.stem
    if out_dir.exists() and list(out_dir.glob("frame_*.jpg")):
        return (video_path.stem, "skip", len(list(out_dir.glob("frame_*.jpg"))))
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = probe_duration(video_path)
    # Aim slightly above the pool size, then thin evenly — cheaper and more
    # robust than trying to land exactly N frames with a single fps value.
    fps = (pool * 1.5) / duration if duration and duration > 0 else 2.0
    fps = min(max(fps, 0.05), 8.0)
    try:
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(video_path),
             "-vf", f"fps={fps:.4f}", "-q:v", "3",
             str(out_dir / "frame_%04d.jpg")],
            check=True, capture_output=True, timeout=300)
    except Exception as e:
        # Fall back to grabbing whatever decodes in the first pass
        if not list(out_dir.glob("frame_*.jpg")):
            return (video_path.stem, f"FAIL:{type(e).__name__}", 0)
    n = cap_evenly(out_dir, pool)
    return (video_path.stem, "ok" if n else "EMPTY", n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    videos = []
    for d in args.videos:
        for p in Path(d).rglob("*"):
            if p.suffix.lower() in VIDEO_EXTS and p.is_file():
                videos.append(p)
    videos = sorted(set(videos))
    print(f"Found {len(videos)} videos; extracting up to {args.pool} frames each "
          f"with {args.workers} workers", flush=True)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    tasks = [(str(v), args.out, args.pool) for v in videos]
    counts = {"ok": 0, "skip": 0, "EMPTY": 0}
    fails = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool_ex:
        futs = [pool_ex.submit(extract_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            stem, status, n = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if status.startswith("FAIL") or status == "EMPTY":
                fails.append((stem, status))
            if i % 200 == 0:
                print(f"  [{i}/{len(tasks)}] ok={counts.get('ok',0)} "
                      f"skip={counts.get('skip',0)} fail={len(fails)}", flush=True)

    print(f"\nDone. {counts}")
    if fails:
        print(f"Failures ({len(fails)}):")
        for stem, status in fails[:25]:
            print(f"  {stem}: {status}")


if __name__ == "__main__":
    main()

"""
Download TRECVID 2025 VQA YouTube Shorts via yt-dlp.

Usage:
    python download_videos.py \\
        --topics /path/to/topics.csv \\
        --out /path/to/videos/ \\
        [--limit 10]
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path


def load_video_ids(topics_csv):
    """Extract unique Video_IDs from topics CSV."""
    video_ids = set()
    with open(topics_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("Video_ID") or row.get("video_id") or row.get("VideoID")
            if vid:
                video_ids.add(vid.strip())
    return sorted(video_ids)


def download_video(video_id, out_dir, fmt="bestvideo[height<=720]+bestaudio/best[height<=720]"):
    """Download a single YouTube video. Returns True on success."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_template = str(Path(out_dir) / f"{video_id}.%(ext)s")
    existing = list(Path(out_dir).glob(f"{video_id}.*"))
    if existing:
        return True  # already downloaded

    cmd = [
        "yt-dlp",
        "--format", fmt,
        "--output", output_template,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    video_ids = load_video_ids(args.topics)
    if args.limit:
        video_ids = video_ids[:args.limit]

    print(f"Downloading {len(video_ids)} videos to {args.out}...")
    succeeded = failed = skipped = 0
    failures = []

    for i, vid_id in enumerate(video_ids, 1):
        existing = list(Path(args.out).glob(f"{vid_id}.*"))
        if existing:
            skipped += 1
            continue
        ok = download_video(vid_id, args.out)
        if ok:
            succeeded += 1
        else:
            failed += 1
            failures.append(vid_id)
        if i % 50 == 0:
            print(f"  [{i}/{len(video_ids)}] ok={succeeded} fail={failed} skip={skipped}", flush=True)

    print(f"\nDone: {succeeded} downloaded, {skipped} skipped, {failed} failed")
    if failures:
        fail_log = Path(args.out).parent / "download_failures.txt"
        fail_log.write_text("\n".join(failures))
        print(f"Failed IDs written to {fail_log}")


if __name__ == "__main__":
    main()

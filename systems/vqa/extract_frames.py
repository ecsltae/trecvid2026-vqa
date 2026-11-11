"""
Extract frames from videos for VLM inference.

Usage:
    python extract_frames.py \\
        --videos /path/to/videos/ \\
        --out /path/to/frames/ \\
        [--mode keyframe|uniform] \\
        [--fps 1] \\
        [--max-frames 8]
"""
import argparse
import subprocess
import sys
from pathlib import Path


def extract_keyframes(video_path, out_dir, max_frames=8):
    """Extract I-frames using ffmpeg."""
    out_pattern = str(out_dir / "frame_%04d.jpg")
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-skip_frame", "noref",
        "-i", str(video_path),
        "-vsync", "vfr",
        "-q:v", "2",
        out_pattern,
    ]
    subprocess.run(cmd, check=True)
    return _cap_frames(out_dir, max_frames)


def extract_uniform(video_path, out_dir, fps=1, max_frames=8):
    """Extract frames at uniform intervals."""
    out_pattern = str(out_dir / "frame_%04d.jpg")
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        out_pattern,
    ]
    subprocess.run(cmd, check=True)
    return _cap_frames(out_dir, max_frames)


def _cap_frames(out_dir, max_frames):
    """Keep only evenly-spaced subset up to max_frames."""
    frames = sorted(out_dir.glob("frame_*.jpg"))
    if len(frames) <= max_frames:
        return frames
    # Keep evenly-spaced subset
    step = len(frames) / max_frames
    keep = {frames[int(i * step)] for i in range(max_frames)}
    for f in frames:
        if f not in keep:
            f.unlink()
    return sorted(keep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", default="keyframe", choices=["keyframe", "uniform"])
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=8)
    args = parser.parse_args()

    videos_dir = Path(args.videos)
    out_root = Path(args.out)
    video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.webm")) + list(videos_dir.glob("*.mkv"))
    print(f"Processing {len(video_files)} videos...")

    for video_path in video_files:
        video_id = video_path.stem
        out_dir = out_root / video_id
        if out_dir.exists() and list(out_dir.glob("frame_*.jpg")):
            continue  # already extracted
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            if args.mode == "keyframe":
                frames = extract_keyframes(video_path, out_dir, args.max_frames)
            else:
                frames = extract_uniform(video_path, out_dir, args.fps, args.max_frames)
        except subprocess.CalledProcessError as e:
            print(f"  FAILED: {video_id}: {e}", flush=True)

    print("Frame extraction complete.")


if __name__ == "__main__":
    main()

"""
Pre-compute Stage-1 llava:13b captions for all videos once.
R2 and R3 both read from this cache, so llava runs only once total.

Output: JSON file  {video_id: caption_string}

Usage:
    python cache_captions.py \\
        --topics data/vqa/topics/vqa_2025_ag_topics.csv \\
                 data/vqa/topics/vqa_2025_mc_topics.csv \\
        --frames data/vqa/frames_full/ \\
        --out data/vqa/captions_cache.json \\
        [--max-frames 4] [--resume]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from trec_shared.ollama_client import OllamaClient

_CAPTION_SYSTEM = (
    "You are a precise video analyst. Describe what you observe in the provided video frames. "
    "Focus on: the main subject(s), their actions and movements, the setting/environment, "
    "any text or objects visible, and how the scene evolves across frames. "
    "Be factual and specific. Do not guess — only describe what is clearly visible."
)

_CAPTION_PROMPT = (
    "Describe the visual content of these video frames in detail. "
    "Organize your description as:\n"
    "1. Main subject(s) and their appearance\n"
    "2. Actions and events occurring\n"
    "3. Setting and background\n"
    "4. Temporal progression (what changes frame to frame)\n"
    "Be specific and factual."
)


def collect_video_ids(topic_csvs):
    ids = set()
    for path in topic_csvs:
        with open(path, newline="", encoding="utf-8") as f:
            first = f.readline(); f.seek(0)
            first_field = first.split(",")[0].strip().strip('"')
            has_header = not first_field.lstrip().isdigit()
            reader = csv.reader(f)
            if has_header:
                next(reader)
            for row in reader:
                if len(row) >= 2:
                    ids.add(row[1].strip().strip('"'))
    return sorted(ids)


def get_frames(frames_dir, video_id, max_frames=4):
    vid_dir = Path(frames_dir) / video_id
    if not vid_dir.exists():
        return []
    return sorted(vid_dir.glob("frame_*.jpg"))[:max_frames]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", nargs="+", required=True)
    parser.add_argument("--frames", default="data/vqa/frames_full")
    parser.add_argument("--out", default="data/vqa/captions_cache.json")
    parser.add_argument("--model", default="llava:13b")
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    video_ids = collect_video_ids(args.topics)
    print(f"Unique videos: {len(video_ids)}")

    cache = {}
    if args.resume and Path(args.out).exists():
        cache = json.loads(Path(args.out).read_text(encoding="utf-8"))
        print(f"Resuming: {len(cache)} captions already cached")

    client = OllamaClient(model=args.model, temperature=0.1)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    to_process = [v for v in video_ids if v not in cache]
    print(f"To process: {len(to_process)} videos")

    for i, video_id in enumerate(to_process, 1):
        frame_paths = get_frames(args.frames, video_id, args.max_frames)
        if not frame_paths:
            cache[video_id] = ""
        else:
            t0 = time.time()
            caption = client.generate_multimodal(
                prompt=_CAPTION_PROMPT,
                images=frame_paths,
                system=_CAPTION_SYSTEM,
            )
            elapsed = time.time() - t0
            cache[video_id] = caption
            if i % 100 == 0:
                print(f"  [{i}/{len(to_process)}] {video_id} ({elapsed:.1f}s)", flush=True)
        # Save checkpoint every 50 videos
        if i % 50 == 0:
            out_path.write_text(json.dumps(cache, ensure_ascii=False, indent=None), encoding="utf-8")

    out_path.write_text(json.dumps(cache, ensure_ascii=False, indent=None), encoding="utf-8")
    print(f"Caption cache saved to {out_path} ({len(cache)} entries)")


if __name__ == "__main__":
    main()

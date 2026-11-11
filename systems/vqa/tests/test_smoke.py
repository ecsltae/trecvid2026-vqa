"""
VQA sandbox smoke tests.

Run with: python -m pytest systems/vqa/tests/test_smoke.py -v
from 
"""
import csv
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent.parent
FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(ROOT))


def test_load_topics():
    from systems.vqa.download_videos import load_video_ids
    video_ids = load_video_ids(str(FIXTURES / "topics_toy.csv"))
    assert len(video_ids) == 2
    assert "dQw4w9WgXcQ" in video_ids


def test_ag_dry_run():
    """Run AG with --dry-run flag (no Ollama, no videos needed)."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        out = f.name

    result = subprocess.run(
        [sys.executable, str(ROOT / "systems/vqa/run_vqa_ag.py"),
         "--topics", str(FIXTURES / "topics_toy.csv"),
         "--frames", str(FIXTURES),   # empty dir — dry-run doesn't need frames
         "--out", out,
         "--limit", "2",
         "--dry-run"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr + result.stdout

    rows = list(csv.reader(open(out)))
    assert len(rows) > 0
    # Each row: Q_ID, Video_ID, Rank, Answer, Time
    assert len(rows[0]) == 5
    assert rows[0][2] == "1"  # first rank


def test_mc_dry_run():
    """Run MC with --dry-run flag."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        out = f.name

    result = subprocess.run(
        [sys.executable, str(ROOT / "systems/vqa/run_vqa_mc.py"),
         "--topics", str(FIXTURES / "topics_toy.csv"),
         "--frames", str(FIXTURES),
         "--out", out,
         "--limit", "2",
         "--dry-run"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr + result.stdout

    rows = list(csv.reader(open(out)))
    assert len(rows) == 8  # 2 topics × 4 options each
    assert rows[0][2] == "1"  # first rank


def test_parse_mc_ranking():
    from systems.vqa.run_vqa_mc import parse_mc_ranking
    # Normal case
    assert parse_mc_ranking("B, A, D, C") == ["B", "A", "D", "C"]
    # Model is verbose but includes ranking
    assert parse_mc_ranking("The answer is most likely B, then A, then C, then D.") == ["B", "A", "C", "D"]
    # Missing options filled in
    ranking = parse_mc_ranking("A, B")
    assert len(ranking) == 4
    assert ranking[:2] == ["A", "B"]


def test_frame_cap():
    from systems.vqa.extract_frames import _cap_frames
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # Create 10 fake frame files
        for i in range(1, 11):
            (tmpdir / f"frame_{i:04d}.jpg").touch()
        frames = _cap_frames(tmpdir, max_frames=4)
        assert len(frames) == 4

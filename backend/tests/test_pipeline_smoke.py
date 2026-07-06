"""End-to-end smoke test on the sample squat clip.

Skipped when the clip is absent — test videos are gitignored, so a fresh
checkout runs the unit tests only. Takes ~15s (real MediaPipe inference).
"""
import json
from pathlib import Path

import pytest

from backend.pipeline.runner import run_squat_pipeline, RunOptions

CLIP = Path(__file__).resolve().parents[2] / "data" / "test_videos" / "squat.mp4"


@pytest.mark.skipif(not CLIP.exists(),
                    reason="sample clip not present (test videos are gitignored)")
def test_pipeline_end_to_end(tmp_path):
    result = run_squat_pipeline(CLIP, tmp_path, RunOptions(coach=False))

    payload = json.loads(result.angles_path.read_text())
    assert payload["status"] == "ok"
    assert len(payload["reps"]) >= 1
    assert len(payload["angles_per_frame"]) == payload["frame_count"]
    assert len(payload["phase_per_frame"]) == payload["frame_count"]

    # a real squat flexes well below parallel and returns to standing
    assert result.summary["knee_min"] < 100
    assert result.summary["knee_max"] > 150

    assert result.annotated_video_path.exists()
    assert result.worst_frame_path is not None and result.worst_frame_path.exists()
    assert result.best_frame_path is not None and result.best_frame_path.exists()

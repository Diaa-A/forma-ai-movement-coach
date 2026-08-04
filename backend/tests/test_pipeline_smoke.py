"""End-to-end smoke test on the sample squat clip.

Skipped when the clip is absent — test videos are gitignored, so a fresh
checkout runs the unit tests only. Takes ~15s (real MediaPipe inference).
"""
import json
from pathlib import Path

import pytest

from backend.pipeline import whisper_wrapper
from backend.pipeline.probe import probe_clip
from backend.pipeline.runner import run_squat_pipeline, RunOptions

VIDEOS = Path(__file__).resolve().parents[2] / "data" / "test_videos"
CLIP = VIDEOS / "squat.mp4"
# an audio-only .mp4 — the voice note used for manual testing. It is the exact
# shape of file someone picks by mistake out of a phone gallery.
AUDIO_ONLY = VIDEOS / "notefortest.mp4"


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


@pytest.mark.skipif(not CLIP.exists(),
                    reason="sample clip not present (test videos are gitignored)")
def test_the_probe_agrees_with_the_clip_it_is_looking_at():
    """20.3 s, 608 frames at 30 fps — the reference clip every Chapter 4 figure
    comes from. Metadata is approximate but this is the accuracy the gate needs.
    """
    clip = probe_clip(CLIP)
    assert clip.readable
    assert 20.0 < clip.seconds < 20.5


@pytest.mark.skipif(not AUDIO_ONLY.exists(),
                    reason="audio fixture not present (test videos are gitignored)")
def test_the_probe_refuses_a_file_with_no_video_stream():
    """The reason readable is a separate question from length. This file opens
    fine and reports no duration, so a length-only check waves it straight
    through to the pipeline."""
    clip = probe_clip(AUDIO_ONLY)
    assert not clip.readable


def test_a_non_video_is_refused_under_every_extension_the_api_accepts(tmp_path):
    """Uploads are staged as input<suffix>, so the extension is always one of
    these. Worth testing all four rather than one: OpenCV chooses its backend
    partly from the extension, and the same bytes named .txt fall through to a
    backend that cheerfully reports 640x400 at 25 fps. That is why the check
    reads frame size instead of trusting the file to be what it is called.
    """
    source = (Path(__file__).resolve().parents[2] / "requirements.txt").read_bytes()
    for suffix in (".mp4", ".mov", ".webm", ".m4v"):
        staged = tmp_path / f"input{suffix}"
        staged.write_bytes(source)
        assert not probe_clip(staged).readable, f"{suffix} was accepted as video"


def test_a_file_opencv_cannot_open_at_all_is_not_readable():
    readme = Path(__file__).resolve().parents[2] / "README.md"
    assert not probe_clip(readme).readable


@pytest.mark.skipif(not CLIP.exists(),
                    reason="sample clip not present (test videos are gitignored)")
def test_a_failed_transcription_warns_without_quoting_the_provider(tmp_path,
                                                                   monkeypatch):
    """warnings[] is rendered verbatim by the PWA, in a banner above the report.

    It used to carry f"voice transcription failed: {e}", and with an invalid key
    that exception is the provider's own error body — so "Groq transcription
    error 401: {...}" appeared on the user's screen. The warning still has to
    appear (a voice note that quietly did nothing is exactly the sort of silent
    drop this project keeps refusing to make), just in the user's terms.

    Coaching runs in dry-run here so nothing goes over the network; transcription
    is attempted before the LLM either way, which is the path under test.
    """
    def refuse(*args, **kwargs):
        raise RuntimeError("Groq transcription error 401: {\"message\": \"Invalid API Key\"}")

    monkeypatch.setattr(whisper_wrapper, "transcribe", refuse)

    note = tmp_path / "note.m4a"
    note.write_bytes(b"not real audio, but transcribe() never gets that far")

    result = run_squat_pipeline(CLIP, tmp_path, RunOptions(
        coach=True, force_dry_run_coach=True, voice_audio_path=str(note),
    ))

    warnings = result.summary.get("warnings", [])
    assert warnings, "a voice note that failed to transcribe was dropped silently"
    joined = " ".join(warnings)
    assert "401" not in joined
    assert "Groq" not in joined
    assert "voice note" in joined.lower()

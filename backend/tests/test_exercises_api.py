"""GET /exercises — the endpoint the PWA builds its picker and filming guidance from.

The property worth protecting here isn't the JSON shape, it's that the filming
guidance the user reads before recording is the *same string* the cue layer uses.
Two copies would drift, and the one in the coaching output is the one that gets
quoted in the report.
"""
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.api.routes import ALLOWED_EXERCISES
from backend.exercises.registry import PROFILES
from backend.exercises.squat_cues import SQUAT_PROFILE
from backend.exercises.base import SAGITTAL, FRONTAL
from backend.pipeline.probe import ClipProbe


client = TestClient(app)


def test_returns_every_registered_exercise():
    r = client.get("/exercises")
    assert r.status_code == 200
    ids = [e["id"] for e in r.json()["exercises"]]
    assert ids == sorted(PROFILES)


def test_squat_guidance_is_the_same_string_the_cue_layer_uses():
    """Not 'looks similar' — identical. This is the whole point of the endpoint."""
    r = client.get("/exercises")
    squat = next(e for e in r.json()["exercises"] if e["id"] == "squat")
    assert squat["filming_guide"] == SQUAT_PROFILE.filming_guide
    assert squat["view_label"] == SQUAT_PROFILE.view_label


def test_exposes_what_each_view_can_assess():
    """The UI needs this to explain that a side-on clip can't judge symmetry,
    without the frontend knowing anything about anatomy."""
    r = client.get("/exercises")
    squat = next(e for e in r.json()["exercises"] if e["id"] == "squat")
    assert "squat depth" in squat["assesses"][SAGITTAL]
    assert "left/right symmetry" in squat["assesses"][FRONTAL]


def test_display_name_is_capitalised():
    r = client.get("/exercises")
    squat = next(e for e in r.json()["exercises"] if e["id"] == "squat")
    assert squat["name"] == "Squat"


def test_allowlist_is_derived_from_the_registry():
    """The bug this prevents: adding an exercise to one list and not the other,
    so the picker offers something /analyze then rejects with a 400."""
    assert ALLOWED_EXERCISES == set(PROFILES)


def test_analyze_rejects_an_exercise_that_is_not_registered():
    r = client.post("/analyze", data={"exercise_type": "deadlift"},
                    files={"video": ("x.mp4", b"not really a video", "video/mp4")})
    assert r.status_code == 400
    assert "deadlift" in r.json()["detail"]


def test_analyze_runs_the_exercise_that_was_asked_for(monkeypatch):
    """The API must pass exercise_type to the pipeline, not just into the job name.

    It previously did not. `analyze()` called the squat-shaped alias, so a push-up
    upload was validated as a push-up, written to pushup_<timestamp>/, and echoed
    back with exercise_type "pushup" -- while the analysis ran the squat pipeline.
    Every surface looked correct except the measurements, which came back as knee
    angles and trunk lean for a person doing push-ups.

    Nothing caught it: the endpoint tests covered /exercises and the rejection
    branch, and the CLI was verified separately and was fine. This asserts the one
    link that was missing, without needing a real video.
    """
    seen = {}

    def fake_run(video_path, output_root, exercise="squat", options=None, job_id=None):
        seen["exercise"] = exercise
        seen["job_id"] = job_id
        raise RuntimeError("stop here - we only care which exercise was requested")

    monkeypatch.setattr("backend.api.routes.run_pipeline", fake_run)
    # the upload has to survive the readability and length checks to reach the
    # pipeline at all, and these bytes are not a video
    monkeypatch.setattr("backend.api.routes.probe_clip",
                        lambda path: ClipProbe(readable=True, seconds=12.0))

    for requested in ("pushup", "squat"):
        seen.clear()
        client.post("/analyze",
                    data={"exercise_type": requested},
                    files={"video": ("clip.mp4", b"not a real video", "video/mp4")})
        assert seen.get("exercise") == requested, (
            f"asked for {requested}, pipeline was told {seen.get('exercise')}")
        assert seen["job_id"].startswith(requested)


def test_no_feedback_form_is_offered_unless_one_is_configured(monkeypatch):
    """The link is round-scoped: outside a testing round the variable is unset
    and the results screen must have nothing to render."""
    monkeypatch.delenv("FEEDBACK_FORM_URL", raising=False)
    r = client.get("/exercises")
    assert r.json()["feedback_form_url"] is None


def test_the_feedback_form_url_is_served_verbatim(monkeypatch):
    """Placeholder included — substituting {job_id} is the client's job, because
    only the client knows which analysis the participant is looking at."""
    url = "https://docs.google.com/forms/d/e/abc/viewform?entry.7=%7Bjob_id%7D"
    monkeypatch.setenv("FEEDBACK_FORM_URL", url)
    r = client.get("/exercises")
    assert r.json()["feedback_form_url"] == url

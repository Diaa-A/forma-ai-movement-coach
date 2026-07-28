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

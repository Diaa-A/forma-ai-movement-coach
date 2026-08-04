"""The rejection branches, and the things /analyze must not do to a stranger.

Everything here is about the endpoint rather than the analysis: what it accepts,
what it refuses and how it says so, and what it puts in a response when something
goes wrong. None of it needs a real video, which is the point — these are the
paths the pose tests never touch and the ones a user hits first.

The one that would have caught a real defect on its own is the job-id test. Two
uploads in the same second resolved to the same directory, and since /analyze is
a sync handler FastAPI runs in a threadpool, that is two people on one link
during a testing session, not a race you have to contrive.
"""
import logging

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.api import routes
from backend.main import app
from backend.pipeline.probe import ClipProbe


client = TestClient(app)

FAKE_CLIP = ("clip.mp4", b"not really a video", "video/mp4")


@pytest.fixture(autouse=True)
def staging(tmp_path, monkeypatch):
    """Stage uploads under tmp_path so the tests never write into data/."""
    upload_root = tmp_path / "uploads"
    monkeypatch.setattr(routes, "UPLOAD_ROOT", upload_root)
    return upload_root


@pytest.fixture
def readable_duration(monkeypatch):
    """A normal 12-second clip, for tests that are about something else."""
    monkeypatch.setattr(routes, "probe_clip",
                        lambda path: ClipProbe(readable=True, seconds=12.0))


def probes_as(monkeypatch, **kwargs):
    """Make the probe report whatever a test needs it to."""
    kwargs.setdefault("readable", True)
    kwargs.setdefault("seconds", 12.0)
    monkeypatch.setattr(routes, "probe_clip", lambda path: ClipProbe(**kwargs))


@pytest.fixture
def captured_runs(monkeypatch):
    """Stop at the pipeline boundary and record what it was asked to do.

    Raising means the request comes back as a 500, which is correct and not what
    these tests look at — they assert on what reached the pipeline, because that
    is the boundary being tested.
    """
    calls = []

    def fake_run(video_path, output_root, exercise="squat", options=None, job_id=None):
        calls.append({"exercise": exercise, "job_id": job_id,
                      "video": video_path, "options": options})
        raise RuntimeError("stopped deliberately - the run itself isn't under test")

    monkeypatch.setattr(routes, "run_pipeline", fake_run)
    return calls


def post(files=None, **form):
    form.setdefault("exercise_type", "squat")
    return client.post("/analyze", data=form,
                       files=files or {"video": FAKE_CLIP})


# ---------------------------------------------------------------------------
# what gets refused, and whether the message is any use
# ---------------------------------------------------------------------------

def test_unknown_exercise_is_named_in_the_message():
    r = post(exercise_type="deadlift")
    assert r.status_code == 400
    assert "deadlift" in r.json()["detail"]


def test_unsupported_video_format_lists_what_would_work():
    r = post(files={"video": ("clip.avi", b"x", "video/avi")})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert ".avi" in detail and ".mp4" in detail


def test_unsupported_audio_format_is_refused_before_the_video_is_written(staging):
    """The audio suffix used to be checked after the clip had been saved, so a
    mistyped voice-note extension cost a full upload on mobile data first."""
    r = post(files={"video": FAKE_CLIP,
                    "voice_note": ("note.aiff", b"x", "audio/aiff")})
    assert r.status_code == 400
    assert ".aiff" in r.json()["detail"]
    assert not staging.exists(), "the clip was written before the audio was checked"


def test_ios_voice_notes_are_accepted(readable_duration, captured_runs):
    """Safari's MediaRecorder produced MP4/AAC only until 18.4, so an iOS voice
    note arrives as .mp4. It was missing from the allowlist, and the frontend was
    renaming the blob to .m4a to get around it."""
    post(files={"video": FAKE_CLIP,
                "voice_note": ("note.mp4", b"x", "audio/mp4")})
    assert captured_runs, "an iOS voice note was refused at the allowlist"
    assert captured_runs[0]["options"].voice_audio_path.endswith(".mp4")


def test_a_missing_pose_model_is_not_reported_as_a_bad_request(monkeypatch,
                                                               readable_duration):
    """FileNotFoundError was mapped to 400 with str(e) in the body. The pose
    model raises exactly that when it is absent, and its message carries the
    server's filesystem path and the curl command to fix it — so a deployment
    with a missing model told users where its files live."""
    def missing_model(*args, **kwargs):
        raise FileNotFoundError(
            "pose model not found at /srv/app/data/models/x.task. Download with: curl ...")

    monkeypatch.setattr(routes, "run_pipeline", missing_model)
    r = post()
    assert r.status_code == 500
    assert "/srv/app" not in r.text
    assert "curl" not in r.text


def test_oversize_clip_is_rejected_in_units_a_person_reads(monkeypatch):
    monkeypatch.setattr(routes, "MAX_VIDEO_BYTES", 8)
    r = post(files={"video": ("clip.mp4", b"more than eight bytes", "video/mp4")})
    assert r.status_code == 413
    assert "MB" in r.json()["detail"]


def test_oversize_voice_note_says_which_file_it_means(readable_duration, monkeypatch):
    """Both caps produced the same sentence, so being told an upload was too big
    did not tell you whether to trim the clip or the voice note."""
    monkeypatch.setattr(routes, "MAX_AUDIO_BYTES", 4)
    r = post(files={"video": FAKE_CLIP,
                    "voice_note": ("note.m4a", b"more than four bytes", "audio/m4a")})
    assert r.status_code == 413
    assert "voice note" in r.json()["detail"]


# ---------------------------------------------------------------------------
# clip length
# ---------------------------------------------------------------------------

def test_over_long_clip_is_refused_before_the_pipeline_runs(monkeypatch, captured_runs):
    probes_as(monkeypatch, seconds=300.0)
    r = post()
    assert r.status_code == 400
    assert "45" in r.json()["detail"]
    assert not captured_runs, "five minutes of pose inference was started anyway"


def test_too_short_clip_is_refused_with_the_length_that_would_work(monkeypatch):
    probes_as(monkeypatch, seconds=1.2)
    r = post()
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "1.2" in detail and "5" in detail


def test_a_rejected_clip_is_removed_rather_than_left_on_disk(monkeypatch, staging):
    """Nothing has been analysed at this point, and until WP-07 exists nothing
    else would ever delete it."""
    probes_as(monkeypatch, seconds=300.0)
    post()
    staged = list(staging.rglob("input.*")) if staging.exists() else []
    assert staged == []


def test_an_unknown_duration_lets_the_clip_through(monkeypatch, captured_runs):
    """Some containers will not report a duration. Refusing on a number we do not
    have would reject valid files for a reason the user cannot act on, and the
    byte cap already bounds the bad case."""
    probes_as(monkeypatch, seconds=None)
    post()
    assert captured_runs, "an unknown duration was treated as a rejection"


def test_a_file_with_no_video_in_it_is_a_bad_request_not_a_crash(monkeypatch,
                                                                 captured_runs):
    """An audio-only .mp4 is what you get by picking a voice memo out of the
    gallery. It reports no duration, so it passed the length check, and then died
    in extract_landmarks with "no frames decoded" — which reached the user as a
    500 for something they did, not something we did.
    """
    probes_as(monkeypatch, readable=False, seconds=None)
    r = post()
    assert r.status_code == 400
    assert "voice memo" in r.json()["detail"]
    assert not captured_runs


# ---------------------------------------------------------------------------
# job identity
# ---------------------------------------------------------------------------

def test_two_uploads_in_the_same_second_get_different_job_ids(readable_duration,
                                                              captured_runs):
    """The id was exercise plus a timestamp to the second. Two people uploading
    at once resolved to one directory: the second run overwrote the first, and
    the first person's /results URL then served the second person's video.
    """
    post()
    post()
    ids = [c["job_id"] for c in captured_runs]
    assert len(ids) == 2
    assert ids[0] != ids[1], f"both uploads got {ids[0]}"


def test_the_job_id_still_says_which_exercise_it_was(readable_duration, captured_runs):
    post(exercise_type="pushup")
    assert captured_runs[0]["job_id"].startswith("pushup_")


# ---------------------------------------------------------------------------
# what a failure tells the caller
# ---------------------------------------------------------------------------

def test_a_pipeline_crash_does_not_hand_back_the_exception(monkeypatch, readable_duration):
    """It used to return f"pipeline error: {e}", so internal paths and module
    names went to whoever sent the request."""
    def explode(*args, **kwargs):
        raise RuntimeError("C:/secret/path/internals.py exploded at line 12")

    monkeypatch.setattr(routes, "run_pipeline", explode)
    r = post()
    assert r.status_code == 500
    assert "internals.py" not in r.text
    assert "secret" not in r.text


def test_a_pipeline_crash_returns_a_reference_that_matches_the_log(
        monkeypatch, readable_duration, caplog):
    """A tester who hits a 500 can read the reference back, and it points at the
    traceback rather than at whichever run we guess they meant."""
    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(routes, "run_pipeline", explode)
    with caplog.at_level(logging.ERROR, logger="coach.api"):
        r = post()

    ref = r.headers.get("X-Error-Reference")
    assert ref, "no reference for the user to quote"
    assert ref in caplog.text
    assert "boom" in caplog.text, "the detail has to survive somewhere"


# ---------------------------------------------------------------------------
# cross-origin access
# ---------------------------------------------------------------------------

def test_no_cross_origin_access_by_default():
    """allow_origins was ["*"], which let any page a user visited POST a video
    here from their browser and read the analysis back. The PWA is served by this
    same app, so nothing legitimate needs the header."""
    r = client.get("/exercises", headers={"Origin": "https://somewhere.example"})
    assert r.status_code == 200
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


def test_cross_origin_is_opt_in_through_the_environment():
    """The escape hatch has to actually work, or the next person hosting the
    frontend separately will put the wildcard back."""
    assert "CORS_ALLOW_ORIGINS" in main.__doc__ or hasattr(main, "_cors_origins")
    assert main._cors_origins == [], "a test run should not have origins configured"


# ---------------------------------------------------------------------------
# the limits the frontend reads
# ---------------------------------------------------------------------------

def test_served_limits_are_the_ones_actually_enforced():
    """The frontend checks a file before uploading it. It kept its own copy of
    these numbers, and the copy had already drifted — it documented a 3-45 second
    server gate that did not exist. Serving them is only worth anything if they
    are read off the same constants the route uses.
    """
    limits = client.get("/exercises").json()["limits"]
    assert limits["max_video_bytes"] == routes.MAX_VIDEO_BYTES
    assert limits["max_audio_bytes"] == routes.MAX_AUDIO_BYTES
    assert set(limits["video_suffixes"]) == routes.ALLOWED_VIDEO_SUFFIXES
    assert set(limits["audio_suffixes"]) == routes.ALLOWED_AUDIO_SUFFIXES
    assert limits["min_seconds"] == routes.MIN_CLIP_SECONDS
    assert limits["max_seconds"] == routes.MAX_CLIP_SECONDS


def test_the_ideal_band_sits_inside_the_band_that_is_enforced():
    """The UI warns inside the ideal range and blocks outside the hard one. If
    they crossed, it would warn about a clip it was about to refuse."""
    limits = client.get("/exercises").json()["limits"]
    assert limits["min_seconds"] <= limits["ideal_min_seconds"]
    assert limits["ideal_max_seconds"] <= limits["max_seconds"]

"""The TTL sweep and the delete-now path (WP-07).

The thing being guarded against is a sweep that looks like it works. Deleting the
output directory and leaving the uploaded video is worse than deleting nothing,
because the consent form then states a retention period the code does not keep -
so the tests below assert on both roots every time rather than on a return value.
"""
import json
import os
import time

import pytest
from fastapi.testclient import TestClient

from backend.api import retention, routes


JOB = "squat_20260812_101500_ab12cd"
OTHER = "pushup_20260812_101501_ff9911"


@pytest.fixture
def roots(tmp_path, monkeypatch):
    up = tmp_path / "uploads"
    out = tmp_path / "outputs"
    up.mkdir()
    out.mkdir()
    monkeypatch.setattr(routes, "UPLOAD_ROOT", up)
    monkeypatch.setattr(routes, "OUTPUT_ROOT", out)
    return up, out


def make_job(roots, job_id=JOB, age_hours=0.0):
    """A job with files in both roots, aged by moving its mtime backwards."""
    up, out = roots
    (up / job_id).mkdir()
    (up / job_id / "input.mp4").write_bytes(b"video")
    (up / job_id / "voice.m4a").write_bytes(b"audio")
    (out / job_id).mkdir()
    for name, blob in (("annotated.mp4", b"mp4"), ("worst.jpg", b"jpg"),
                       ("best.jpg", b"jpg"), ("angles.json", b"{}"),
                       ("coaching.json", b"{}")):
        (out / job_id / name).write_bytes(blob)
    if age_hours:
        when = time.time() - age_hours * 3600
        for d in (up / job_id, out / job_id):
            os.utime(d, (when, when))
    return up / job_id, out / job_id


def residue(roots, job_id=JOB):
    """Anything at all still on disk for this job, in either root."""
    left = []
    for root in roots:
        d = root / job_id
        if d.exists():
            left.append(str(d))
            left += [str(p) for p in d.rglob("*")]
    return left


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------

def test_an_expired_job_leaves_nothing_in_either_root(roots):
    make_job(roots, age_hours=30)
    result = retention.sweep(roots, hours=24)
    assert result.deleted == [JOB]
    assert residue(roots) == [], "files survived a sweep that reported success"


def test_the_upload_is_deleted_and_not_just_the_output(roots):
    """The failure this is here for: outputs are the visible half, so a sweep
    that only clears OUTPUT_ROOT looks correct from /results while the original
    video - the part a consent form is actually about - stays on disk."""
    up_dir, out_dir = make_job(roots, age_hours=48)
    retention.sweep(roots, hours=24)
    assert not out_dir.exists()
    assert not up_dir.exists(), "the uploaded video outlived its results"


def test_a_job_inside_the_period_is_untouched(roots):
    up, out = roots
    make_job(roots, age_hours=23)
    result = retention.sweep(roots, hours=24)
    assert result.deleted == []
    assert (up / JOB / "input.mp4").exists()
    assert (up / JOB / "voice.m4a").exists()
    assert (out / JOB / "annotated.mp4").exists()
    assert (out / JOB / "coaching.json").exists()


def test_only_the_expired_job_goes(roots):
    make_job(roots, JOB, age_hours=30)
    make_job(roots, OTHER, age_hours=1)
    retention.sweep(roots, hours=24)
    assert residue(roots, JOB) == []
    assert residue(roots, OTHER), "a fresh job was swept with an expired one"


def test_the_cutoff_is_measured_from_now_not_from_process_start(roots):
    """`now` is injected, so this pins that the boundary moves with the clock."""
    make_job(roots, age_hours=10)
    assert retention.sweep(roots, hours=24).deleted == []
    later = time.time() + 20 * 3600
    assert retention.sweep(roots, now=later, hours=24).deleted == [JOB]


def test_directories_that_are_not_jobs_are_never_swept(roots):
    """The first run of this sweep deleted data/outputs/penn_eval/ and
    penn_eval_pushup/ along with 116 real jobs. OUTPUT_ROOT is data/outputs/, the
    benchmark and the faithfulness harness write in there too, and nothing in the
    sweep said those were not jobs - they were just directories older than a day.

    Recoverable, because the benchmark reproduces. Still the wrong direction of
    failure for a sweep to have: leaving a clip behind costs a promise, deleting
    evidence costs the evidence.
    """
    up, out = roots
    for name in ("penn_eval", "penn_eval_pushup", "faithfulness"):
        d = out / name
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps({"aggregate": {}}))
        old = time.time() - 400 * 24 * 3600
        os.utime(d, (old, old))
    make_job(roots, age_hours=30)

    result = retention.sweep(roots, hours=24)

    assert result.deleted == [JOB]
    for name in ("penn_eval", "penn_eval_pushup", "faithfulness"):
        assert (out / name / "metrics.json").exists(), f"{name} was swept"


def test_a_sweep_over_empty_roots_is_quiet(roots):
    result = retention.sweep(roots, hours=24)
    assert result.deleted == [] and result.ok and result.scanned == 0


def test_one_undeletable_job_does_not_stop_the_rest(roots, monkeypatch):
    make_job(roots, JOB, age_hours=30)
    make_job(roots, OTHER, age_hours=30)
    real = retention.shutil.rmtree

    def explode(path, *a, **kw):
        if JOB in str(path):
            raise OSError("in use")
        return real(path, *a, **kw)

    monkeypatch.setattr(retention.shutil, "rmtree", explode)
    result = retention.sweep(roots, hours=24)
    assert OTHER in result.deleted
    assert result.failed and not result.ok
    assert residue(roots, OTHER) == []


# ---------------------------------------------------------------------------
# the thread that actually runs it
# ---------------------------------------------------------------------------
# The sweep function was tested and the startup pass was observed on real data.
# The 15-minute thread was neither, which is the half that keeps a long-running
# deployment clean -- a TTL nothing fires is a retention policy on paper.

def test_the_sweep_thread_keeps_firing_and_stops_when_told(monkeypatch):
    import threading
    from backend import main

    calls = []
    monkeypatch.setattr(main.retention, "SWEEP_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(main.retention, "sweep", lambda roots: calls.append(roots))

    main._sweep_stop.clear()
    t = threading.Thread(target=main._sweep_forever, daemon=True)
    t.start()
    for _ in range(200):                      # up to ~2s, exits as soon as it can
        if len(calls) >= 3:
            break
        time.sleep(0.01)
    main._sweep_stop.set()
    t.join(timeout=2)

    assert len(calls) >= 3, f"thread fired {len(calls)} times, expected to repeat"
    assert not t.is_alive(), "the thread ignored the stop event"
    assert calls[0], "swept an empty list of roots"


def test_a_failing_sweep_does_not_kill_the_thread(monkeypatch):
    """Otherwise one transient disk error turns into retention silently never
    running again for the life of the process, and nothing says so."""
    import threading
    from backend import main

    calls = []

    def flaky(roots):
        calls.append(roots)
        if len(calls) == 1:
            raise OSError("disk busy")

    monkeypatch.setattr(main.retention, "SWEEP_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(main.retention, "sweep", flaky)

    main._sweep_stop.clear()
    t = threading.Thread(target=main._sweep_forever, daemon=True)
    t.start()
    for _ in range(200):
        if len(calls) >= 3:
            break
        time.sleep(0.01)
    main._sweep_stop.set()
    t.join(timeout=2)

    assert len(calls) >= 3, "the thread died on the first failure"


# ---------------------------------------------------------------------------
# delete now
# ---------------------------------------------------------------------------

def test_delete_now_removes_a_job_that_is_nowhere_near_expiry(roots):
    make_job(roots, age_hours=0)
    result = retention.delete_job(roots, JOB)
    assert result.deleted == [JOB]
    assert residue(roots) == []


def test_delete_now_refuses_anything_that_is_not_a_job_id(roots):
    """The job id goes on the end of a filesystem path. Matched against the
    format runner.new_job_id emits rather than stripped of bad characters,
    because a sanitiser is a thing you can get subtly wrong."""
    make_job(roots, age_hours=0)
    for bad in ("../../etc", "..", "squat_20260812_101500_ab12cd/../..",
                "squat", "", "SQUAT_20260812_101500_AB12CD",
                "squat_20260812_101500_ab12cd ", "a/b"):
        with pytest.raises(ValueError):
            retention.delete_job(roots, bad)
    assert residue(roots), "a rejected id still managed to delete something"


def test_deleting_a_job_that_is_already_gone_is_not_an_error(roots):
    result = retention.delete_job(roots, JOB)
    assert result.deleted == [] and result.ok


# ---------------------------------------------------------------------------
# what the user is told
# ---------------------------------------------------------------------------

def test_the_served_period_is_the_one_the_sweep_uses(roots):
    """The acceptance criterion: one number, not two. A frontend that states 24
    hours while the sweep keeps 72 is the failure being designed out."""
    limits = routes.current_limits()
    assert limits.retention_hours == retention.RETENTION_HOURS
    assert str(int(retention.RETENTION_HOURS)) in limits.retention_note


def test_the_note_reads_as_a_sentence_at_the_periods_we_might_pick():
    assert "24 hours" in retention.retention_note(24)
    assert "3 days" in retention.retention_note(72)
    assert "1.5 hours" in retention.retention_note(1.5)


def test_exercises_serves_the_retention_period(roots, monkeypatch):
    monkeypatch.setenv("DISABLE_RETENTION_SWEEP", "1")
    from backend.main import app
    with TestClient(app) as client:
        body = client.get("/exercises").json()
    assert body["limits"]["retention_hours"] == retention.RETENTION_HOURS
    assert body["limits"]["retention_note"]


def test_delete_endpoint_removes_both_roots_and_rejects_a_bad_id(roots, monkeypatch):
    monkeypatch.setenv("DISABLE_RETENTION_SWEEP", "1")
    make_job(roots, age_hours=0)
    from backend.main import app
    with TestClient(app) as client:
        assert client.delete(f"/jobs/{JOB}").json()["deleted"] is True
        assert residue(roots) == []
        # already gone, same answer
        again = client.delete(f"/jobs/{JOB}").json()
        assert again["deleted"] is False
        assert client.delete("/jobs/not-a-job-id").status_code == 400

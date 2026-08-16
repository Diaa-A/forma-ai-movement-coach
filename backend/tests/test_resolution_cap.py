"""The working-resolution cap, and what /health can tell you.

Both exist because of one incident. The container idles at 447 MB of a 1000 MB
limit, a 4K analysis wants about 690 MB on top, and an iPhone films 4K by
default -- so the upload was killed three seconds in, before the overlay render
was reached. Nothing outside the container could see the memory, which is why it
took a day to find.
"""
import numpy as np

from backend.pipeline.pose import MAX_ANALYSIS_EDGE, fit_dims, fit_within


def frame(w, h):
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# the cap
# ---------------------------------------------------------------------------

def test_a_frame_already_within_the_cap_is_returned_untouched():
    """The whole safety argument for shipping this rests on it. Every fixture in
    the repo is smaller than the cap, so if this is an identity below the
    threshold then no published angle, figure or threshold can move."""
    for w, h in ((1024, 576), (576, 1024), (1280, 720), (464, 640), (1920, 1080)):
        f = frame(w, h)
        assert fit_dims(w, h) == (w, h)
        assert fit_within(f) is f, f"{w}x{h} was resized and should not have been"


def test_4k_comes_down_to_the_cap_on_its_long_edge():
    assert fit_dims(3840, 2160) == (1920, 1080)
    assert fit_within(frame(3840, 2160)).shape[:2] == (1080, 1920)


def test_portrait_is_capped_on_its_long_edge_too():
    """A phone held upright is the common case, and the long edge is the height."""
    w, h = fit_dims(2160, 3840)
    assert max(w, h) == MAX_ANALYSIS_EDGE
    assert (w, h) == (1080, 1920)


def test_aspect_ratio_survives():
    for w, h in ((3840, 2160), (2160, 3840), (4096, 2160), (2560, 1440)):
        tw, th = fit_dims(w, h)
        assert abs((tw / th) - (w / h)) < 0.01, f"{w}x{h} -> {tw}x{th}"
        assert max(tw, th) == MAX_ANALYSIS_EDGE


def test_a_degenerate_frame_does_not_produce_a_zero_dimension():
    """cv2.resize raises on a zero side, and a 1-pixel-tall frame is exactly the
    sort of thing a corrupt upload produces."""
    assert fit_dims(4000, 1) == (1920, 1)
    assert fit_dims(0, 0) == (0, 0)


def test_the_cap_is_the_resolution_the_filming_guides_promise():
    """Both guides tell people 1080p is plenty. If the cap and the advice drift
    apart, one of them is lying to the user."""
    from backend.exercises.registry import PROFILES
    assert MAX_ANALYSIS_EDGE == 1920
    for profile in PROFILES.values():
        assert "1080p" in profile.filming_guide


# ---------------------------------------------------------------------------
# what /health reports
# ---------------------------------------------------------------------------

def test_memory_is_absent_rather_than_invented_when_there_is_no_cgroup():
    """Off Linux there is nothing to read. A zero would read as "no memory left",
    which is worse than saying nothing."""
    from backend.main import _container_memory
    mem = _container_memory()
    assert mem is None or set(mem) == {"limit_mb", "usage_mb", "headroom_mb"}


def test_health_still_answers_without_a_cgroup(monkeypatch):
    monkeypatch.setenv("DISABLE_RETENTION_SWEEP", "1")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["degraded"] is False
    if "memory" in body:
        assert body["memory"]["usage_mb"] > 0

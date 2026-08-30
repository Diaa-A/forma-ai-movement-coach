"""The rotation guard: a sideways-decoded file must be declared, not analysed.

The failure this covers was found regenerating a figure, not by any test. An MP4
whose pixels are stored rotated normally carries a display-matrix flag and
decodes upright; strip the flag (messaging-app re-encodes do) and the pipeline
measures a tipped-over person without noticing. The reference clip with its flag
removed came back 1 rep instead of 7, with the best frame of the set reported as
a forward-lean fault — status still ok. Confidently wrong rather than visibly
broken, which is the exact state the coverage rule forbids.

The guard reads the body's own axis, because there is no metadata left to read.
Thresholds are calibrated in mechanics.py next to the cuts themselves.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from backend.exercises.mechanics import (axis_looks_rotated, body_axis_tilt,
                                         TILT_ROTATED_ABOVE, TILT_ROTATED_BELOW)
from backend.exercises.pushup import PUSHUP
from backend.exercises.pushup_cues import PUSHUP_PROFILE
from backend.exercises.squat import SQUAT
from backend.exercises.squat_cues import SQUAT_PROFILE
from backend.pipeline.coaching import not_analyzed_report
from backend.pipeline.pose import LM, N_LANDMARKS
from backend.pipeline.runner import run_squat_pipeline, RunOptions

VIDEOS = Path(__file__).resolve().parents[2] / "data" / "test_videos"
# the reference squat clip with its rotation flag stripped and nothing else
# changed: ffmpeg -display_rotation 0 -i squat.mp4 -c copy _squat_noflag.mp4
NOFLAG = VIDEOS / "_squat_noflag.mp4"


def _frames(n=30, axis="vertical", visibility=0.9, sides=("left", "right")):
    """Synthetic landmark array with a body axis pointing the requested way."""
    lm = np.zeros((n, N_LANDMARKS, 4))
    for side in sides:
        sh, an = LM[f"{side}_shoulder"], LM[f"{side}_ankle"]
        if axis == "vertical":
            lm[:, sh, :2] = [0.5, 0.3]
            lm[:, an, :2] = [0.5, 0.8]
        else:
            lm[:, sh, :2] = [0.3, 0.5]
            lm[:, an, :2] = [0.8, 0.5]
        lm[:, sh, 3] = visibility
        lm[:, an, 3] = visibility
    return lm


def test_an_upright_body_reads_near_vertical():
    assert body_axis_tilt(_frames(axis="vertical")) == pytest.approx(0.0, abs=1.0)


def test_a_sideways_body_reads_near_horizontal():
    assert body_axis_tilt(_frames(axis="horizontal")) == pytest.approx(90.0, abs=1.0)


def test_the_better_seen_side_wins():
    """Side-on footage rarely clears the visibility gate on both sides at once,
    so requiring both starves the measure of frames — the first calibration pass
    got zero usable frames from every push-up clip that way."""
    lm = _frames(axis="vertical", sides=("left",))
    # the far side is present but below the gate, with junk coordinates
    lm[:, LM["right_shoulder"]] = [0.9, 0.1, 0.0, 0.3]
    lm[:, LM["right_ankle"]] = [0.1, 0.9, 0.0, 0.3]
    assert body_axis_tilt(lm) == pytest.approx(0.0, abs=1.0)


def test_too_few_qualifying_frames_means_no_verdict():
    lm = _frames(n=30, axis="horizontal")
    lm[5:, :, 3] = 0.0        # only five frames clear the gate
    assert body_axis_tilt(lm) is None


def test_nan_coordinates_do_not_poison_the_median():
    """extract_landmarks emits NaN for frames with no detection; a qualifying
    visibility next to a NaN coordinate made the whole median NaN in the
    calibration script before the finite check went in."""
    lm = _frames(axis="vertical")
    lm[::2, LM["left_shoulder"], 0] = np.nan
    lm[::2, LM["right_shoulder"], 0] = np.nan
    tilt = body_axis_tilt(lm)
    assert tilt is not None and np.isfinite(tilt)


def test_each_movement_flags_the_axis_it_does_not_expect():
    assert axis_looks_rotated(SQUAT, 81.3)          # the stripped-flag squat
    assert not axis_looks_rotated(SQUAT, 10.5)      # the worst correct squat
    assert axis_looks_rotated(PUSHUP, 18.6)         # the sideways-stored push-up
    assert not axis_looks_rotated(PUSHUP, 54.5)     # the worst correct push-up


def test_the_dead_band_refuses_nobody():
    """Between the two cuts the evidence is ambiguous, and refusing someone's
    upload on ambiguous evidence is worse than analysing it."""
    for tilt in (TILT_ROTATED_BELOW, 45.0, TILT_ROTATED_ABOVE):
        assert not axis_looks_rotated(SQUAT, tilt)
        assert not axis_looks_rotated(PUSHUP, tilt)


def test_no_measurement_is_never_refused():
    assert not axis_looks_rotated(SQUAT, None)
    assert not axis_looks_rotated(PUSHUP, None)


@pytest.mark.parametrize("profile", [SQUAT_PROFILE, PUSHUP_PROFILE])
def test_the_rotated_report_blames_the_file_and_not_the_form(profile):
    report = not_analyzed_report("rotated", profile)
    assert "sideways" in report.primary_issue
    assert any("original clip" in c for c in report.corrective_cues)
    # the person did nothing wrong, so the re-record advice would be misleading
    assert "Re-record" not in report.next_session_focus


@pytest.mark.skipif(not NOFLAG.exists(),
                    reason="stripped-flag fixture not present (test videos are gitignored)")
def test_a_stripped_flag_clip_is_refused_end_to_end(tmp_path):
    """The regression the guard exists for, on the real artefact. Before it, this
    exact file produced status ok, 1 rep, and a best frame flagged as a fault."""
    result = run_squat_pipeline(NOFLAG, tmp_path, RunOptions(coach=True,
                                                            force_dry_run_coach=True))
    payload = json.loads(result.angles_path.read_text())
    assert payload["status"] == "rotated"
    assert payload["body_axis_tilt"] > TILT_ROTATED_ABOVE

    coaching = json.loads(result.coaching_path.read_text())
    assert coaching["report"]["source"] == "not_analyzed"
    assert "sideways" in coaching["report"]["primary_issue"]

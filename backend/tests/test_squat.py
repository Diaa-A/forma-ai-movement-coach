"""Squat analysis: rep validity, frame gating, worst/best selection."""
import numpy as np
import pytest

from backend.exercises import squat
from backend.pipeline.pose import LM


def _frame(knee=90.0, spine=30.0, shin=30.0):
    return {
        "knee_left": knee, "knee_right": knee,
        "hip_left": 90.0, "hip_right": 90.0,
        "ankle_left": 80.0, "ankle_right": 80.0,
        "shin_left": shin, "shin_right": shin,
        "spine": spine,
    }


# --- keep_real_reps: hip-travel rep filter -----------------------------------

def test_jitter_rep_is_dropped():
    hip_y = np.full(60, 0.5)
    hip_y[10:31] += 0.20 * np.bartlett(21)   # genuine squat: half a leg length
    hip_y[40:51] += 0.01 * np.bartlett(11)   # tracking wobble
    reps = [(10, 20, 30), (40, 45, 50)]
    assert squat.keep_real_reps(reps, hip_y, scale=0.4) == [(10, 20, 30)]


def test_relative_floor_drops_partial_dips():
    # a quarter of the clip's own biggest travel is not a rep, even though it
    # clears the absolute floor
    hip_y = np.full(60, 0.5)
    hip_y[10:31] += 0.20 * np.bartlett(21)
    hip_y[40:51] += 0.05 * np.bartlett(11)
    reps = [(10, 20, 30), (40, 45, 50)]
    assert squat.keep_real_reps(reps, hip_y, scale=0.4) == [(10, 20, 30)]


def test_all_jitter_clip_keeps_nothing():
    hip_y = np.full(60, 0.5)
    hip_y[10:21] += 0.010 * np.bartlett(11)
    hip_y[40:51] += 0.012 * np.bartlett(11)
    reps = [(10, 15, 20), (40, 45, 50)]
    assert squat.keep_real_reps(reps, hip_y, scale=0.4) == []


def test_shallow_but_real_rep_survives():
    # the defect that motivated this filter: a shallow squat is still a rep
    hip_y = np.full(60, 0.5)
    hip_y[10:31] += 0.08 * np.bartlett(21)   # 20% of leg length
    assert squat.keep_real_reps([(10, 20, 30)], hip_y, scale=0.4) == [(10, 20, 30)]


# --- frame validity gate ------------------------------------------------------

def test_frame_valid_accepts_normal_frame():
    assert squat.frame_valid(_frame(), "left")


def test_frame_valid_rejects_missing_knee():
    assert not squat.frame_valid(_frame(knee=float("nan")), "left")


def test_frame_valid_rejects_implausible_lean():
    # landmark scramble: spine reading past the plausibility ceiling
    assert not squat.frame_valid(_frame(spine=100.0), "left")


def test_deepest_frame_skips_glitched_frames():
    frames = [_frame(knee=120), _frame(knee=40, spine=100), _frame(knee=95)]
    assert squat.deepest_frame(frames, 0, 2, "left") == 2


def test_deepest_frame_none_when_window_unusable():
    frames = [_frame(spine=100), _frame(knee=float("nan"))]
    assert squat.deepest_frame(frames, 0, 1, "left") is None


# --- scoring and worst/best selection ------------------------------------------

def test_worst_and_best_pick_the_right_reps():
    frames = [_frame(knee=140) for _ in range(31)] + [_frame(knee=90) for _ in range(31)]
    reps = [(0, 15, 30), (31, 46, 61)]
    worst = squat.worst_frame(frames, reps, "left", fps=30.0)
    best = squat.best_frame(frames, reps, "left", fps=30.0)
    assert 0 <= worst <= 30    # the shallow rep
    assert 31 <= best <= 61    # the deep rep


def test_lean_penalty_can_dominate():
    # deep but back-dominant rep scores worse than a clean deep one
    leaning = [_frame(knee=90, spine=65, shin=35) for _ in range(31)]
    clean = [_frame(knee=90, spine=40, shin=35) for _ in range(31)]
    reps = [(0, 15, 30), (31, 46, 61)]
    worst = squat.worst_frame(leaning + clean, reps, "left", fps=30.0)
    assert 0 <= worst <= 30


# --- body scale and side selection ---------------------------------------------

def _landmark_stack(n=10):
    lms = np.full((n, 33, 4), np.nan)
    for name, y, vis in [
        ("left_hip", 0.5, 0.95), ("left_knee", 0.7, 0.95), ("left_ankle", 0.9, 0.95),
        ("right_hip", 0.5, 0.40), ("right_knee", 0.7, 0.40), ("right_ankle", 0.9, 0.40),
    ]:
        lms[:, LM[name]] = [0.5, y, 0.0, vis]
    return lms


def test_body_scale_is_hip_to_ankle_length():
    assert squat.body_scale(_landmark_stack()) == pytest.approx(0.4)


def test_pick_side_prefers_the_visible_leg():
    assert squat.pick_side(_landmark_stack(), [(0, 5, 9)]) == "left"


# ---------------------------------------------------------------------------
# Regressions found on a real phone upload, 29 Jul 2026
# ---------------------------------------------------------------------------

def test_one_impossible_travel_does_not_bin_every_real_rep():
    """The bug: subject walks back toward the camera at the end of the clip, the
    hips 'travel' 2.9x leg length in that segment, and because the relative floor
    calibrates against the biggest travel in the clip, all six genuine reps
    measure ~0.15 against the artefact and get discarded as noise. The system
    then reported the artefact as the only rep it found."""
    scale = 0.35
    # six real reps at ~0.45 of leg length, then one impossible one
    reps = [(i * 100, i * 100 + 40, i * 100 + 90) for i in range(7)]
    hip_y = np.zeros(700)
    for i in range(6):
        hip_y[i * 100 + 40] = 0.16          # a real squat
    hip_y[640] = 1.02                        # tracking coming apart

    kept = squat.keep_real_reps(reps, hip_y, scale)

    assert len(kept) == 6, "real reps were binned because of the outlier"
    assert (600, 640, 690) not in kept, "the impossible travel was kept as a rep"


def test_travel_ceiling_scales_with_the_person():
    """It's a multiple of leg length, not a pixel number — someone filmed close up
    and someone filmed across a gym must be judged the same way."""
    reps = [(0, 40, 90)]
    hip_y = np.zeros(100)
    hip_y[40] = 0.5

    # 0.5 travel on a 0.6 leg is fine; the same travel on a 0.2 leg is not a squat
    assert len(squat.keep_real_reps(reps, hip_y, scale=0.6)) == 1
    assert len(squat.keep_real_reps(reps, hip_y, scale=0.2)) == 0


def test_all_travels_implausible_falls_back_rather_than_dividing_by_nothing():
    reps = [(0, 40, 90), (100, 140, 190)]
    hip_y = np.zeros(200)
    hip_y[40] = 2.0
    hip_y[140] = 2.2
    # shouldn't raise; the status machinery deals with the resulting mess
    squat.keep_real_reps(reps, hip_y, scale=0.35)


def _lean_frame(spine, shin, knee=90.0):
    return {"knee_left": knee, "knee_right": knee, "spine": spine,
            "shin_left": shin, "shin_right": shin}


def test_flag_frames_marks_the_torso_while_the_lean_is_happening():
    """Previously only the single worst frame was ever flagged — one frame in six
    hundred, which at 30fps is invisible, so the overlay looked all-green however
    the set went."""
    angles = ([_lean_frame(20, 18)] * 10          # upright, trunk near shin
              + [_lean_frame(60, 25)] * 10        # +35 excess — well over the limit
              + [_lean_frame(20, 18)] * 10)
    flagged = squat.flag_frames(angles, reps=[], side="left")

    assert all(not f for f in flagged[:10]), "flagged a frame that was upright"
    assert all("left_shoulder" in f and "left_hip" in f for f in flagged[10:20])
    assert all(not f for f in flagged[20:]), "lean flag outlived the lean"


def test_flag_frames_leaves_invalid_frames_alone():
    """A frame that fails the validity gate is a tracking glitch — withhold the
    judgement rather than colouring it, same principle as the cue layer."""
    angles = [_lean_frame(200, 10)] * 5      # spine past MAX_PLAUSIBLE_LEAN
    assert all(not f for f in squat.flag_frames(angles, reps=[], side="left"))


def test_flag_frames_marks_a_shallow_rep_only_near_the_bottom():
    """Depth is a per-rep property. The descent itself isn't the fault, so the
    whole rep shouldn't turn red."""
    deep = _lean_frame(20, 18, knee=170.0)
    shallow_bottom = _lean_frame(20, 18, knee=140.0)   # never got below 110
    angles = [deep] * 20 + [shallow_bottom] + [deep] * 20

    flagged = squat.flag_frames(angles, reps=[(0, 20, 40)], side="left",
                                depth_window=3)

    assert "left_knee" in flagged[20]
    assert "left_knee" in flagged[18] and "left_knee" in flagged[22]
    assert not flagged[0] and not flagged[40], "flagged the whole rep, not the bottom"


def test_flag_frames_leaves_a_deep_rep_unflagged():
    deep_bottom = _lean_frame(20, 18, knee=70.0)
    angles = [_lean_frame(20, 18, knee=170.0)] * 20 + [deep_bottom]
    flagged = squat.flag_frames(angles, reps=[(0, 20, 20)], side="left")
    assert all(not f for f in flagged)


def _standing(knee=170.0):
    return {"knee_left": knee, "knee_right": knee, "spine": 8.0,
            "shin_left": 10.0, "shin_right": 10.0}


def test_a_dip_where_the_knee_never_bends_is_not_a_rep():
    """Hip travel says something moved, not what. Someone settling into position
    dropped their hips ~0.2 of a leg length — over the travel floor — while the
    knee stayed at 154 degrees. It counted as a rep, scored least-bad, and so
    became the 'worst form' key frame: the headline image was a photo of someone
    standing still."""
    reps = [(0, 40, 90)]
    hip_y = np.zeros(100)
    hip_y[40] = 0.08
    angles = [_standing(154.0)] * 100

    assert squat.keep_real_reps(reps, hip_y, 0.35, angles_per_frame=angles,
                                side="left", fps=30.0) == []


def test_a_genuinely_shallow_squat_still_counts():
    """The point of judging by travel was that shallow reps survive to be counted
    and then flagged shallow. A flexion gate must not undo that."""
    reps = [(0, 40, 90)]
    hip_y = np.zeros(100)
    hip_y[40] = 0.16
    angles = [_standing(170.0)] * 100
    angles[40] = _standing(135.0)      # shallow, but unmistakably a squat

    kept = squat.keep_real_reps(reps, hip_y, 0.35, angles_per_frame=angles,
                                side="left", fps=30.0)
    assert kept == [(0, 40, 90)]


def test_flexion_gate_is_skipped_when_no_angles_are_given():
    """Travel-only behaviour stays available; the gate is additive."""
    reps = [(0, 40, 90)]
    hip_y = np.zeros(100)
    hip_y[40] = 0.16
    assert squat.keep_real_reps(reps, hip_y, 0.35) == [(0, 40, 90)]


def _rep_frames(n, deepest_at, deepest_knee, knee_top=175.0):
    """A rep's worth of frames with a single deepest point."""
    frames = []
    for i in range(n):
        knee = deepest_knee if i == deepest_at else knee_top
        frames.append({"knee_left": knee, "knee_right": knee, "spine": 20.0,
                       "shin_left": 18.0, "shin_right": 18.0})
    return frames


def test_settling_into_position_is_not_counted_as_a_rep():
    """From a 4K phone upload: the first detected bottom was the subject settling
    before starting. Knee reached 137.8 degrees (about 42 of bend) against 34-48
    degrees (132-146 of bend) for the four genuine reps. It was counted, scored
    least-bad, and became the 'worst form' key frame — so the headline image was
    the user standing still."""
    angles = (_rep_frames(100, 50, 137.8) + _rep_frames(100, 50, 34.2)
              + _rep_frames(100, 50, 47.8) + _rep_frames(100, 50, 39.9))
    reps = [(i * 100, i * 100 + 50, i * 100 + 99) for i in range(4)]
    hip_y = np.zeros(400)
    for i in range(4):
        hip_y[i * 100 + 50] = 0.16

    kept = squat.keep_real_reps(reps, hip_y, 0.35, angles_per_frame=angles,
                                side="left", fps=30.0)

    assert (0, 50, 99) not in kept, "the settling movement was counted as a rep"
    assert len(kept) == 3


def test_a_set_of_uniformly_shallow_squats_all_survive():
    """The ratio is against the deepest rep in the SAME clip, so someone who only
    ever quarter-squats still gets counted — and then told they were shallow.
    Rejecting them outright would be the system deciding they didn't exercise."""
    angles = (_rep_frames(100, 50, 130.0) + _rep_frames(100, 50, 128.0)
              + _rep_frames(100, 50, 133.0))
    reps = [(i * 100, i * 100 + 50, i * 100 + 99) for i in range(3)]
    hip_y = np.zeros(300)
    for i in range(3):
        hip_y[i * 100 + 50] = 0.16

    kept = squat.keep_real_reps(reps, hip_y, 0.35, angles_per_frame=angles,
                                side="left", fps=30.0)
    assert len(kept) == 3

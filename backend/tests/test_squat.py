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

"""Angle maths against known geometry. The first thing that had to be right."""
import math

import numpy as np
import pytest

from backend.pipeline.angles import (
    joint_angle, angle_from_vertical, squat_angles_per_frame,
)
from backend.pipeline.pose import LM, N_LANDMARKS


def test_right_angle():
    assert joint_angle((0, 1), (0, 0), (1, 0)) == pytest.approx(90.0)


def test_straight_line_is_180():
    assert joint_angle((0, 1), (0, 0), (0, -1)) == pytest.approx(180.0)


def test_fully_folded_is_0():
    assert joint_angle((1, 0), (0, 0), (2, 0)) == pytest.approx(0.0)


def test_sixty_degrees():
    c = (math.cos(math.radians(60)), math.sin(math.radians(60)))
    assert joint_angle((1, 0), (0, 0), c) == pytest.approx(60.0)


def test_translation_invariance():
    # same triangle shifted away from the origin gives the same angle
    assert joint_angle((5, 4), (5, 3), (6, 3)) == pytest.approx(90.0)


def test_degenerate_vertex_is_nan():
    assert math.isnan(joint_angle((0, 0), (0, 0), (1, 0)))


def test_vertical_segment_has_no_lean():
    assert angle_from_vertical((0, 0), (0, 1)) == pytest.approx(0.0)


def test_horizontal_segment_leans_90():
    assert angle_from_vertical((0, 0), (1, 0)) == pytest.approx(90.0)


def test_diagonal_leans_45():
    assert angle_from_vertical((0, 0), (1, 1)) == pytest.approx(45.0)


def test_zero_length_segment_is_nan():
    assert math.isnan(angle_from_vertical((0.5, 0.5), (0.5, 0.5)))


# --- per-frame extraction on a constructed pose ------------------------------

def _pose_frame(points):
    """Build a (33, 4) landmark frame from {name: (x, y)}. Image coords, y down."""
    fr = np.full((N_LANDMARKS, 4), np.nan)
    for name, (x, y) in points.items():
        fr[LM[name]] = [x, y, 0.0, 1.0]
    return fr


def test_standing_pose_angles():
    # straight vertical body: knee fully extended, no trunk or shin lean
    fr = _pose_frame({
        "left_shoulder": (0.45, 0.2), "right_shoulder": (0.55, 0.2),
        "left_hip": (0.45, 0.5), "right_hip": (0.55, 0.5),
        "left_knee": (0.45, 0.7), "right_knee": (0.55, 0.7),
        "left_ankle": (0.45, 0.9), "right_ankle": (0.55, 0.9),
        "left_foot_index": (0.5, 0.92), "right_foot_index": (0.6, 0.92),
    })
    angles = squat_angles_per_frame(np.stack([fr]))[0]
    assert angles["knee_left"] == pytest.approx(180.0)
    assert angles["spine"] == pytest.approx(0.0)
    assert angles["shin_left"] == pytest.approx(0.0)


def test_bent_knee_pose():
    # horizontal thigh + vertical shin = 90 degree knee
    fr = _pose_frame({
        "left_shoulder": (0.3, 0.4), "right_shoulder": (0.3, 0.4),
        "left_hip": (0.3, 0.7), "right_hip": (0.3, 0.7),
        "left_knee": (0.5, 0.7), "right_knee": (0.5, 0.7),
        "left_ankle": (0.5, 0.9), "right_ankle": (0.5, 0.9),
        "left_foot_index": (0.55, 0.92), "right_foot_index": (0.55, 0.92),
    })
    angles = squat_angles_per_frame(np.stack([fr]))[0]
    assert angles["knee_left"] == pytest.approx(90.0)


def test_missing_landmarks_give_nan():
    fr = np.full((N_LANDMARKS, 4), np.nan)
    angles = squat_angles_per_frame(np.stack([fr]))[0]
    assert math.isnan(angles["knee_left"])
    assert math.isnan(angles["spine"])

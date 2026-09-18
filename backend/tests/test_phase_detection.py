"""Velocity zero-crossing phase detection on synthetic hip trajectories."""
import numpy as np

from backend.pipeline.phase_detection import (
    detect_bottoms, segment_reps, label_phases,
    PHASE_STANDING, PHASE_DESCENT, PHASE_BOTTOM, PHASE_ASCENT,
)


def _squat_wave(n, period, depth=0.2):
    """Hip y over n frames: dips of `depth` every `period` frames (y grows down)."""
    i = np.arange(n)
    return 0.5 + depth * 0.5 * (1 - np.cos(2 * np.pi * i / period))


def test_finds_one_bottom_per_dip():
    hip_y = _squat_wave(90, period=30)   # dips bottom out at 15, 45, 75
    bottoms = detect_bottoms(hip_y)
    assert len(bottoms) == 3
    for found, expected in zip(bottoms, (15, 45, 75)):
        assert abs(found - expected) <= 2


def test_flat_signal_has_no_bottoms():
    assert detect_bottoms(np.full(60, 0.5)) == []


def test_pure_descent_has_no_bottoms():
    # sinking without coming back up never crosses descent -> ascent
    assert detect_bottoms(np.linspace(0.4, 0.8, 60)) == []


def test_min_separation_merges_jitter_crossings():
    hip_y = _squat_wave(90, period=30)
    hip_y[17] += 0.01   # wobble just past a bottom -> extra zero-crossing pair
    assert len(detect_bottoms(hip_y, min_separation=1)) > 3
    assert len(detect_bottoms(hip_y, min_separation=5)) == 3


def test_segment_reps_tiles_the_clip():
    reps = segment_reps([15, 45, 75], n_frames=90)
    assert reps == [(0, 15, 30), (30, 45, 60), (60, 75, 89)]


def test_the_last_rep_does_not_run_to_the_end_of_the_clip():
    """Recording keeps going while the lifter drops off the bar and walks back to the
    phone. Bounded by the set's own rhythm, the last rep stops where the next one would
    have started: on a real pull-up set the unbounded version measured 4.2 body lengths
    of travel and was thrown out as impossible."""
    reps = segment_reps([100, 160, 220], n_frames=600)

    assert reps[-1] == (190, 220, 250)
    assert reps[0] == (70, 100, 130)


def test_label_phases_ordering_within_a_rep():
    phases = label_phases([(0, 15, 30)], n_frames=31)
    assert phases[5] == PHASE_DESCENT
    assert phases[15] == PHASE_BOTTOM
    assert phases[25] == PHASE_ASCENT


def test_frames_outside_reps_are_standing():
    phases = label_phases([(30, 45, 60)], n_frames=90)
    assert phases[10] == PHASE_STANDING
    assert phases[75] == PHASE_STANDING
    assert len(phases) == 90

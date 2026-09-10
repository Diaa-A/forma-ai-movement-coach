"""Pull-up mechanics.

Only what is specific to the pull-up. The shared rep machinery is covered by the
squat and push-up suites and is not re-tested here.

The first three tests are about one thing: a pull-up puts the hard part of the
rep at the top of the body's travel, and every other exercise puts it at the
bottom. That is the failure this exercise introduced, and it is the kind that
produces a sensible-looking rep count built entirely on the wrong frames.
"""
import numpy as np
import pytest

from backend.exercises import mechanics, pullup, pushup, squat
from backend.pipeline.phase_detection import detect_bottoms


def _hanging_body(n_reps=3, frames_per_rep=40):
    """Synthetic pull-up travel: hanging low, pulled high, lowered again.

    Image y increases downward, so a dead hang is a LARGE y and the chin at the
    bar is a small one. Returns (travel_y, top_indices).
    """
    y = []
    tops = []
    for r in range(n_reps):
        base = r * frames_per_rep
        half = frames_per_rep // 2
        # down to up
        y.extend(np.linspace(0.80, 0.30, half))
        tops.append(base + half - 1)
        # back down
        y.extend(np.linspace(0.30, 0.80, frames_per_rep - half))
    return np.array(y), tops


def test_detector_without_the_flag_finds_the_hang_not_the_pull():
    """The defect the flag exists to prevent, pinned so it cannot come back.

    Run the raw signal through the detector and it returns the dead hang between
    reps. Nothing about the output says so -- the count is plausible and the reps
    are evenly spaced, which is exactly why this needed a test rather than a
    reading of the results.
    """
    y, tops = _hanging_body()
    bottoms = detect_bottoms(y, min_separation=5)

    assert bottoms, "expected the detector to find something"
    for b in bottoms:
        # every returned frame sits at the bottom of the travel, not the top
        assert y[b] > 0.7, f"frame {b} has y={y[b]:.2f}, which is not a hang"
        assert all(abs(b - t) > 10 for t in tops), \
            f"frame {b} landed near a pull-up top, which the raw signal cannot do"


def test_flag_moves_the_detected_frame_to_the_top_of_the_pull():
    y, tops = _hanging_body()
    flipped = mechanics.travel_for_phase(pullup.PULLUP, y)
    bottoms = detect_bottoms(flipped, min_separation=5)

    assert len(bottoms) >= len(tops) - 1, \
        f"expected about {len(tops)} reps, got {len(bottoms)}"
    for b in bottoms:
        assert y[b] < 0.4, f"frame {b} has y={y[b]:.2f}, which is not the top of a pull"
        assert min(abs(b - t) for t in tops) <= 2, \
            f"frame {b} is not within two frames of a pull-up top"


def test_only_the_pullup_inverts_its_travel():
    """A guard, not a behaviour check.

    Setting effort_at_top on a squat or a push-up would silently invert the one
    thing their whole cue set is built around, so the value is asserted rather
    than left to review.
    """
    assert pullup.PULLUP.effort_at_top is True
    assert squat.SQUAT.effort_at_top is False
    assert pushup.PUSHUP.effort_at_top is False

    y = np.array([0.1, 0.5, 0.9])
    assert np.allclose(mechanics.travel_for_phase(squat.SQUAT, y), y)
    assert np.allclose(mechanics.travel_for_phase(pushup.PUSHUP, y), y)
    assert np.allclose(mechanics.travel_for_phase(pullup.PULLUP, y), -y)


@pytest.mark.parametrize("elbow", [8.0, 14.9, 17.0, 27.4])
def test_a_deeply_folded_elbow_is_a_measurement_not_a_glitch(elbow):
    """The floor was 15 degrees and that was wrong.

    It rejected the top of the pull on sequence 1172, which reads 8 degrees, and
    with the effort frame gone the flexion gate saw a straight arm and discarded
    the whole repetition. Three of 25 clips scored nothing because of it.

    Penn Action ground truth settled it: over 2709 measurements the 0-30 band has
    a median signed error of -2.1 degrees and is the most accurate of the six,
    and ground truth holds 206 readings in it. These are projected 2D angles, and
    a front-on pull-up overlaps the upper arm with the forearm at the top.
    """
    assert pullup.frame_valid({"elbow_left": elbow, "trunk": 5.0}, "left")


def test_a_degenerate_zero_is_still_rejected():
    """All the floor is for now: coincident landmarks, not a hard pull."""
    assert not pullup.frame_valid({"elbow_left": 0.0, "trunk": 5.0}, "left")


def test_a_trunk_past_anything_a_hanging_body_does_is_not_scored():
    assert not pullup.frame_valid({"elbow_left": 90.0, "trunk": 75.0}, "left")
    assert pullup.frame_valid({"elbow_left": 90.0, "trunk": 20.0}, "left")


def test_travel_ceiling_admits_a_rep_that_moves_a_whole_torso_length():
    """The shared default would have thrown away every genuine rep.

    Body scale for a pull-up is the torso, and the body travels roughly 0.9-1.3
    times that per rep. `Movement.max_travel` defaults to 1.0, which is right for
    a squat measured against leg length and would have discarded a correct
    pull-up as implausible tracking.
    """
    assert pullup.PULLUP.max_travel > 1.3
    assert squat.SQUAT.max_travel <= 1.0

    scale = 0.25
    travel = np.array([0.0, 0.30])          # 1.2x scale, a normal rep
    reps = [(0, 1, 1)]
    angles = [{"elbow_left": 170.0, "trunk": 2.0},
              {"elbow_left": 40.0, "trunk": 2.0}]
    kept = pullup.keep_real_reps(reps, travel, scale,
                                 angles_per_frame=angles, side="left")
    assert kept == reps, "a rep moving 1.2 torso lengths should survive the gate"


def test_pulling_higher_than_the_target_is_not_a_fault():
    """One-sided, like the push-up's depth penalty.

    Someone who gets their chest to the bar has done more than asked, and a
    two-sided penalty would score them worse than someone who stopped at the
    target.
    """
    at_target, _ = pullup._score_frame({"elbow_left": pullup.ELBOW_TARGET_TOP}, "left")
    higher, _ = pullup._score_frame({"elbow_left": 25.0}, "left")
    short, _ = pullup._score_frame({"elbow_left": 110.0}, "left")

    assert at_target == 0.0
    assert higher == 0.0
    assert short > 0.0


def test_scale_is_the_torso_so_tucked_legs_do_not_change_it():
    """Most people hang with the knees bent, which shortens shoulder-to-ankle by
    an amount that has nothing to do with how big they are."""
    pairs = set(pullup.PULLUP.scale_pairs)
    assert pairs == {("left_shoulder", "left_hip"), ("right_shoulder", "right_hip")}
    assert all("ankle" not in a and "ankle" not in b for a, b in pullup.PULLUP.scale_pairs)


def test_the_overlay_asserts_no_fault_while_no_cue_is_calibrated():
    """Silence is the deliberate choice here, so it is pinned.

    A red joint is a diagnosis. Step 2 established that Penn Action has no
    labelled incomplete repetitions, so there is no threshold behind a height
    claim, so the overlay makes none. If a cue gets calibrated this test should
    fail and be rewritten -- that is the point of it.
    """
    angles = [{"elbow_left": 170.0, "trunk": 2.0},
              {"elbow_left": 120.0, "trunk": 2.0},   # a rep that came up short
              {"elbow_left": 165.0, "trunk": 2.0}]
    flags = pullup.flag_frames(angles, [(0, 1, 2)], "left")

    assert len(flags) == len(angles)
    assert all(f == set() for f in flags), \
        "the overlay claimed a fault with no calibrated threshold behind it"


def test_the_score_orders_reps_even_though_it_cannot_judge_them():
    """The split step 2 turned on: ordering needs monotonicity, a cue needs a
    defensible boundary. Only the first is available."""
    poor, _ = pullup._score_frame({"elbow_left": 95.0}, "left")
    mid, _ = pullup._score_frame({"elbow_left": 60.0}, "left")
    good, _ = pullup._score_frame({"elbow_left": 20.0}, "left")

    assert poor > mid > good

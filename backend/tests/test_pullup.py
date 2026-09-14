"""Pull-up mechanics.

Only what is specific to the pull-up; the shared rep machinery is tested in the
squat and push-up suites. The first three tests cover the effort-at-top flip,
where a mistake would still produce a believable rep count.
"""
import numpy as np
import pytest

from backend.exercises import mechanics, pullup, pushup, squat
from backend.pipeline.angles import pullup_angles_per_frame
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
    """The defect the flag exists for. On the raw signal the detector returns the
    dead hang between reps, and the count and spacing look normal."""
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
    """Guard: setting it on the squat or push-up would invert what their cues
    are built around."""
    assert pullup.PULLUP.effort_at_top is True
    assert squat.SQUAT.effort_at_top is False
    assert pushup.PUSHUP.effort_at_top is False

    y = np.array([0.1, 0.5, 0.9])
    assert np.allclose(mechanics.travel_for_phase(squat.SQUAT, y), y)
    assert np.allclose(mechanics.travel_for_phase(pushup.PUSHUP, y), y)
    assert np.allclose(mechanics.travel_for_phase(pullup.PULLUP, y), -y)


def _two_tops(dip):
    """Travel as phase detection sees it, higher meaning the body is higher: a
    pull, a dip of `dip` between two tops, and the lowering."""
    rise = np.linspace(0.0, 1.0, 30)
    between = 1.0 - dip * (0.5 - 0.5 * np.cos(np.linspace(0.0, 2 * np.pi, 60)))
    lower = np.linspace(1.0, 0.0, 30)
    return np.concatenate([rise, between, lower])


def test_a_hold_at_the_top_counts_once():
    """Held at the top, the hips wobble and the detector finds two turns."""
    y = _two_tops(dip=0.002)
    bottoms = detect_bottoms(y, min_separation=5)
    assert len(bottoms) == 2
    assert len(mechanics.merge_bottoms_without_return(pullup.PULLUP, bottoms, y, 1.0)) == 1


def test_lowering_between_two_pulls_still_counts_two():
    y = _two_tops(dip=0.9)
    bottoms = detect_bottoms(y, min_separation=5)
    assert len(bottoms) == 2
    assert mechanics.merge_bottoms_without_return(pullup.PULLUP, bottoms, y, 1.0) == bottoms


def test_only_the_pullup_merges_bottoms_without_a_return():
    """Guard: on the push-up clips the rule merged real reps, because the hips
    barely move between them."""
    assert pullup.PULLUP.min_return > 0
    assert squat.SQUAT.min_return == 0
    assert pushup.PUSHUP.min_return == 0


@pytest.mark.parametrize("elbow", [8.0, 14.9, 17.0, 27.4])
def test_a_deeply_folded_elbow_is_a_measurement_not_a_glitch(elbow):
    """8 degrees is the top of the pull on sequence 1172. The old 15-degree floor
    rejected it and three clips lost their reps; ground truth has 206 elbow
    readings under 30 degrees."""
    assert pullup.frame_valid({"elbow": elbow, "trunk": 5.0}, "left")


def test_a_degenerate_zero_is_still_rejected():
    """Coincident landmarks, not a hard pull."""
    assert not pullup.frame_valid({"elbow": 0.0, "trunk": 5.0}, "left")


def test_a_trunk_past_anything_a_hanging_body_does_is_not_scored():
    assert not pullup.frame_valid({"elbow": 90.0, "trunk": 75.0}, "left")
    assert pullup.frame_valid({"elbow": 90.0, "trunk": 20.0}, "left")


def test_travel_ceiling_admits_a_rep_that_moves_a_whole_torso_length():
    """Against the torso a real rep travels up to 1.3, so the shared default
    ceiling of 1.0 would drop it."""
    assert pullup.PULLUP.max_travel > 1.3
    assert squat.SQUAT.max_travel <= 1.0

    scale = 0.25
    travel = np.array([0.0, 0.30])          # 1.2x scale, a normal rep
    reps = [(0, 1, 1)]
    angles = [{"elbow": 170.0, "trunk": 2.0},
              {"elbow": 40.0, "trunk": 2.0}]
    kept = pullup.keep_real_reps(reps, travel, scale,
                                 angles_per_frame=angles, side="left")
    assert kept == reps, "a rep moving 1.2 torso lengths should survive the gate"


def test_the_score_is_one_sided_and_keeps_the_order():
    """Only used to rank reps for the key frames. Pulling higher than the
    reference costs nothing; coming up shorter costs more."""
    at_ref, _ = pullup._score_frame({"elbow": pullup.ELBOW_TARGET_TOP}, "left")
    higher, _ = pullup._score_frame({"elbow": 25.0}, "left")
    short, _ = pullup._score_frame({"elbow": 60.0}, "left")
    shorter, _ = pullup._score_frame({"elbow": 95.0}, "left")

    assert at_ref == higher == 0.0
    assert shorter > short > 0.0


def test_scale_is_the_torso_so_tucked_legs_do_not_change_it():
    """Most people hang with bent knees, which shortens shoulder-to-ankle."""
    pairs = set(pullup.PULLUP.scale_pairs)
    assert pairs == {("left_shoulder", "left_hip"), ("right_shoulder", "right_hip")}
    assert all("ankle" not in a and "ankle" not in b
               for a, b in pullup.PULLUP.scale_pairs)


def test_the_overlay_asserts_no_fault_while_no_cue_is_calibrated():
    """No calibrated height threshold, so the overlay marks nothing. When a cue
    gets one, this test should fail and change with it."""
    angles = [{"elbow": 170.0, "trunk": 2.0},
              {"elbow": 120.0, "trunk": 2.0},   # a rep that came up short
              {"elbow": 165.0, "trunk": 2.0}]
    flags = pullup.flag_frames(angles, [(0, 1, 2)], "left")

    assert len(flags) == len(angles)
    assert all(f == set() for f in flags), \
        "the overlay claimed a fault with no calibrated threshold behind it"


def _arms(left_deg, right_deg, left_vis, right_vis):
    """One frame of landmarks with each elbow bent to a known angle.

    Shoulder straight above the elbow and the wrist swung out by the angle, so
    joint_angle gives back exactly what was asked for. Visibility is set per arm
    on all three points of its triplet.
    """
    from backend.pipeline.pose import LM
    fr = np.zeros((33, 4))
    fr[:, 3] = 0.9
    for side, deg, vis, x in (("left", left_deg, left_vis, 0.4),
                              ("right", right_deg, right_vis, 0.6)):
        t = np.radians(deg)
        fr[LM[f"{side}_shoulder"]] = [x, 0.3, 0.0, vis]
        fr[LM[f"{side}_elbow"]] = [x, 0.5, 0.0, vis]
        fr[LM[f"{side}_wrist"]] = [x + 0.2 * np.sin(t),
                                   0.5 - 0.2 * np.cos(t), 0.0, vis]
        fr[LM[f"{side}_hip"]] = [x, 0.8, 0.0, 0.9]
    return np.array([fr])


def test_both_arms_tracked_means_both_arms_count():
    elbow = pullup_angles_per_frame(_arms(30.0, 40.0, 0.9, 0.9))[0]["elbow"]
    assert elbow == pytest.approx(35.0, abs=0.5)


def test_an_arm_below_the_confidence_gate_is_left_out():
    """MediaPipe still returns landmarks for an arm it cannot see, so the angle
    is a number, just not one to average in."""
    elbow = pullup_angles_per_frame(_arms(30.0, 150.0, 0.9, 0.3))[0]["elbow"]
    assert elbow == pytest.approx(30.0, abs=0.5)


def test_with_neither_arm_confident_the_better_seen_one_is_used():
    """Both arms under the gate but both readable: the better seen is used."""
    elbow = pullup_angles_per_frame(_arms(30.0, 90.0, 0.5, 0.4))[0]["elbow"]
    assert elbow == pytest.approx(30.0, abs=0.5)


def test_which_side_was_picked_no_longer_changes_the_rep_count():
    """On sequence 1173 the choice of arm decided whether the gate kept 2 reps
    or 1. The side argument must not change the result now."""
    scale = 0.25
    travel = np.array([0.0, 0.30, 0.0, 0.30, 0.0])
    reps = [(0, 1, 2), (2, 3, 4)]
    angles = [{"elbow": 170.0, "trunk": 2.0},
              {"elbow": 40.0, "trunk": 2.0},
              {"elbow": 170.0, "trunk": 2.0},
              {"elbow": 45.0, "trunk": 2.0},
              {"elbow": 170.0, "trunk": 2.0}]
    left = pullup.keep_real_reps(reps, travel, scale,
                                 angles_per_frame=angles, side="left")
    right = pullup.keep_real_reps(reps, travel, scale,
                                  angles_per_frame=angles, side="right")
    assert left == right == reps
    assert (pullup.frame_valid(angles[1], "left")
            == pullup.frame_valid(angles[1], "right"))


def test_a_degenerate_arm_never_wins_on_visibility_alone():
    """Sequence 1173: neither arm cleared the gate, the better-seen arm read 0
    and the other read 4. Ranking on visibility alone lost 46 frames."""
    elbow = pullup_angles_per_frame(_arms(4.0, 0.0, 0.28, 0.35))[0]["elbow"]
    assert elbow == pytest.approx(4.0, abs=0.5)


def test_a_confident_but_degenerate_arm_is_not_averaged_in():
    """170 averaged with a degenerate 0 would be a plausible 85 that passes every
    gate."""
    elbow = pullup_angles_per_frame(_arms(170.0, 0.0, 0.9, 0.9))[0]["elbow"]
    assert elbow == pytest.approx(170.0, abs=0.5)

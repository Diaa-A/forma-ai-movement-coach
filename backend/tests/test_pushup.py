"""Push-up analyser: angles, validity, cues, and the shared machinery under it.

Two things these are written to protect. First the push-up's own maths, which is
new. Second the property the shared machinery exists for: that adding a second
exercise did not require forking it, so a fix to one benefits both.
"""
import numpy as np
import pytest

from backend.exercises import mechanics, pushup, squat
from backend.exercises.pushup_cues import PUSHUP_CUES, PUSHUP_PROFILE, evaluate_pushup
from backend.exercises.registry import EXERCISES, exercise_ids
from backend.pipeline.angles import _signed_body_line, pushup_angles_per_frame


# ---------------------------------------------------------------------------
# body line: the measurement the whole push-up assessment turns on
# ---------------------------------------------------------------------------

def test_body_line_is_zero_when_straight():
    assert abs(_signed_body_line((0, 0), (1, 0), (2, 0))) < 0.01


def test_sagging_hips_read_positive_and_piked_negative():
    """Sag and pike bend the hip by the same amount and need opposite fixes, so
    the sign is the entire point. Image y increases downward, which is easy to get
    backwards -- hence a test rather than a comment."""
    sag = _signed_body_line((0, 0), (1, 0.3), (2, 0))    # hip lower than the line
    pike = _signed_body_line((0, 0), (1, -0.3), (2, 0))  # hip higher than the line
    assert sag > 0
    assert pike < 0
    assert abs(sag + pike) < 0.01, "the two should be mirror images"


def test_body_line_grows_with_deviation():
    small = _signed_body_line((0, 0), (1, 0.1), (2, 0))
    large = _signed_body_line((0, 0), (1, 0.4), (2, 0))
    assert large > small


# ---------------------------------------------------------------------------
# validity gate
# ---------------------------------------------------------------------------

def _frame(elbow=90.0, body=5.0):
    return {"elbow_left": elbow, "elbow_right": elbow,
            "body_left": body, "body_right": body}


def test_impossible_body_line_is_rejected():
    """Two of the six reference clips swung the full +/-180 because the subject was
    too small in frame for the hip to be placed at all. That is the pose estimate
    coming apart, not a person with terrible form, and scoring it would produce a
    confident number from nonsense."""
    assert pushup.frame_valid(_frame(body=10.0), "left")
    assert not pushup.frame_valid(_frame(body=175.0), "left")
    assert not pushup.frame_valid(_frame(body=-175.0), "left")


def test_missing_elbow_is_rejected():
    assert not pushup.frame_valid({"body_left": 5.0}, "left")
    assert not pushup.frame_valid(_frame(elbow=float("nan")), "left")


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def test_going_deeper_than_target_is_not_penalised():
    """Same rule as the squat: depth only counts against you when you stop short."""
    deep, _ = pushup._score_frame(_frame(elbow=70.0, body=0.0), "left")
    target, _ = pushup._score_frame(_frame(elbow=pushup.ELBOW_TARGET_DEPTH, body=0.0), "left")
    assert deep == target == 0.0


def test_stopping_short_is_penalised():
    short, breakdown = pushup._score_frame(_frame(elbow=140.0, body=0.0), "left")
    assert short > 0
    assert breakdown["depth"] > 0


def test_sag_and_pike_are_penalised_equally():
    """Both are failures to hold the plank, even though the corrections differ."""
    sag, _ = pushup._score_frame(_frame(elbow=90.0, body=45.0), "left")
    pike, _ = pushup._score_frame(_frame(elbow=90.0, body=-45.0), "left")
    assert sag == pike > 0


# ---------------------------------------------------------------------------
# cue firing
# ---------------------------------------------------------------------------

def _set_of_reps(n_reps, elbow, body, frames_per_rep=60, fps=30.0):
    """Build a synthetic set: each rep dips to `elbow` at its midpoint."""
    angles, reps = [], []
    for r in range(n_reps):
        base = r * frames_per_rep
        for i in range(frames_per_rep):
            deep = (i == frames_per_rep // 2)
            angles.append(_frame(elbow=elbow if deep else 170.0,
                                 body=body if deep else 3.0))
        reps.append((base, base + frames_per_rep // 2, base + frames_per_rep - 1))
    return angles, reps


def test_shallow_set_fires_shallow_depth():
    angles, reps = _set_of_reps(3, elbow=140.0, body=3.0)
    ev = evaluate_pushup(angles, reps, "left", 30.0)
    assert "shallow_depth" in [c.flag for c in ev.cues_fired]


def test_deep_set_earns_the_depth_positive_and_fires_nothing():
    angles, reps = _set_of_reps(3, elbow=80.0, body=3.0)
    ev = evaluate_pushup(angles, reps, "left", 30.0)
    assert "shallow_depth" not in [c.flag for c in ev.cues_fired]
    assert any("full depth" in p for p in ev.positives)


def test_sagging_and_piking_fire_different_cues():
    sag_angles, reps = _set_of_reps(3, elbow=85.0, body=40.0)
    piked_angles, _ = _set_of_reps(3, elbow=85.0, body=-40.0)

    sag_flags = [c.flag for c in evaluate_pushup(sag_angles, reps, "left", 30.0).cues_fired]
    pike_flags = [c.flag for c in evaluate_pushup(piked_angles, reps, "left", 30.0).cues_fired]

    assert "hips_sagging" in sag_flags and "hips_piked" not in sag_flags
    assert "hips_piked" in pike_flags and "hips_sagging" not in pike_flags


def test_straight_body_positive_is_withheld_when_it_sagged():
    angles, reps = _set_of_reps(3, elbow=85.0, body=40.0)
    ev = evaluate_pushup(angles, reps, "left", 30.0)
    assert not any("straight line" in p for p in ev.positives)


def test_symmetry_is_withheld_without_landmarks_to_check_it():
    """Cross-side comparison needs the far arm to actually be visible. With no
    landmark array there is nothing to gate on, so the cue must stay silent rather
    than claim a symmetry it could not measure -- the squat's confidence gate,
    applied to a new exercise without new machinery."""
    angles, reps = _set_of_reps(3, elbow=85.0, body=3.0)
    ev = evaluate_pushup(angles, reps, "left", 30.0, landmarks=None)
    assert "elbow_asymmetry" not in [c.flag for c in ev.cues_fired]
    assert not any("both arms" in p for p in ev.positives)


def test_uneven_arms_fire_elbow_asymmetry_once_the_far_arm_is_seen():
    """The other half of the gate above: with every landmark confident, arms 40
    degrees apart at the bottom fire the cue and even arms earn the positive."""
    landmarks = np.zeros((180, 33, 4))
    landmarks[:, :, 3] = 0.95
    even, reps = _set_of_reps(3, elbow=85.0, body=3.0)
    uneven = [dict(f) for f in even]
    for _, bottom, _ in reps:
        uneven[bottom]["elbow_right"] = 125.0

    ev_uneven = evaluate_pushup(uneven, reps, "left", 30.0, landmarks=landmarks)
    ev_even = evaluate_pushup(even, reps, "left", 30.0, landmarks=landmarks)
    assert "elbow_asymmetry" in [c.flag for c in ev_uneven.cues_fired]
    assert "elbow_asymmetry" not in [c.flag for c in ev_even.cues_fired]
    assert any("both arms" in p for p in ev_even.positives)


def test_reps_that_bottom_out_at_different_depths_fire_rep_inconsistency():
    even, reps = _set_of_reps(3, elbow=85.0, body=3.0)
    uneven = [dict(f) for f in even]
    uneven[reps[2][1]] = _frame(elbow=115.0, body=3.0)   # the last rep stops 30 degrees higher

    assert "rep_inconsistency" in [c.flag for c in evaluate_pushup(uneven, reps, "left", 30.0).cues_fired]
    assert "rep_inconsistency" not in [c.flag for c in evaluate_pushup(even, reps, "left", 30.0).cues_fired]


def test_dropping_into_the_bottom_fires_fast_descent():
    """A third of a second on the way down fires; a full second does not."""
    quick, quick_reps = _set_of_reps(3, elbow=85.0, body=3.0, frames_per_rep=20)
    slow, slow_reps = _set_of_reps(3, elbow=85.0, body=3.0)

    assert "fast_descent" in [c.flag for c in evaluate_pushup(quick, quick_reps, "left", 30.0).cues_fired]
    assert "fast_descent" not in [c.flag for c in evaluate_pushup(slow, slow_reps, "left", 30.0).cues_fired]


def test_parked_cue_never_fires():
    """head_dropped is defined but its detector is not trustworthy -- the nose is
    the only head landmark tracked and it moves with rotation as well as with the
    neck. Same honesty as the squat's parked knee-valgus cue."""
    assert PUSHUP_CUES["head_dropped"]["available"] is False


def test_every_cue_has_a_fault_and_a_separate_fix():
    """Layer 2 is only allowed to rephrase these, and keeping them apart is what
    makes it clear which half is being rephrased."""
    for flag, cue in PUSHUP_CUES.items():
        assert cue["fault"] and cue["fix"], flag
        assert cue["fault"] != cue["fix"], flag
        assert cue["severity"] in ("primary", "secondary"), flag


# ---------------------------------------------------------------------------
# the abstraction itself
# ---------------------------------------------------------------------------

def test_both_exercises_share_one_rep_filter():
    """The whole point of sharing it. If these had been forked, a fix to one would
    silently not reach the other -- which is exactly how the earlier defects
    would have come back."""
    assert squat.keep_real_reps.__module__ == pushup.keep_real_reps.__module__ \
        or mechanics.keep_real_reps is not None
    # both delegate to the same function object
    import inspect
    assert 'mechanics.keep_real_reps' in inspect.getsource(squat.keep_real_reps)
    assert 'mechanics.keep_real_reps' in inspect.getsource(pushup.keep_real_reps)


def test_body_scale_uses_true_length_for_a_horizontal_body():
    """A push-up body is horizontal, so the squat's vertical measure collapses to
    near zero and every rep would look impossibly large against it. This is the
    one place the original abstraction genuinely did not generalise."""
    assert squat.SQUAT.scale_metric == "vertical"
    assert pushup.PUSHUP.scale_metric == "euclidean"

    # a flat body: shoulder and ankle at the same height, one unit apart
    lm = np.zeros((5, 33, 4))
    from backend.pipeline.pose import LM
    lm[:, LM["left_shoulder"]] = [0.2, 0.5, 0, 1.0]
    lm[:, LM["left_ankle"]] = [0.8, 0.5, 0, 1.0]
    lm[:, LM["right_shoulder"]] = [0.2, 0.5, 0, 1.0]
    lm[:, LM["right_ankle"]] = [0.8, 0.5, 0, 1.0]

    assert mechanics.body_scale(pushup.PUSHUP, lm) == pytest.approx(0.6, abs=0.01)


def test_registry_wires_every_exercise_completely():
    """A half-registered exercise would fail deep inside the pipeline rather than
    here, so check the entries are whole."""
    # Updated when the pull-up landed. The list is asserted rather than
    # counted so that adding an exercise has to be a deliberate edit here.
    assert exercise_ids() == ["pullup", "pushup", "squat"]
    for name, spec in EXERCISES.items():
        assert spec.profile.name == name
        assert callable(spec.angles) and callable(spec.evaluate)
        assert callable(spec.flag_frames) and callable(spec.caption)
        assert spec.movement.name == name


def test_pushup_profile_declares_both_planes():
    assert PUSHUP_PROFILE.assessments("sagittal")
    assert "left/right arm symmetry" in PUSHUP_PROFILE.assessments("frontal")
    assert PUSHUP_PROFILE.label == "Push-up"


def test_angles_cover_the_joints_the_cues_need():
    lm = np.random.rand(3, 33, 4)
    frames = pushup_angles_per_frame(lm)
    assert len(frames) == 3
    for key in ("elbow_left", "elbow_right", "body_left", "body_right",
                "shoulder_left", "shoulder_right", "neck", "spine"):
        assert key in frames[0], key


def test_caption_names_pushup_joints_not_squat_ones():
    """Found on a real clip: the renderer built the readout itself, so a push-up
    frame was captioned 'knee L:-- R:-- shin --' with every field empty. The
    caption belongs to the exercise."""
    lines = pushup.frame_caption(_frame(elbow=74.0, body=29.0), "left")
    joined = " ".join(lines)
    assert "elbow" in joined and "body" in joined
    assert "knee" not in joined and "shin" not in joined


# ---------------------------------------------------------------------------
# Rep gating: why push-up does not use body travel
# ---------------------------------------------------------------------------

def test_pushup_ignores_body_travel_when_deciding_reps():
    """From a real upload: fourteen push-ups each moved the hips about 0.10 of
    body length, while lowering into position at the start moved them 0.59. Under
    a travel gate that 0.59 became the largest travel in the clip and the relative
    floor exceeded every genuine rep -- one survived out of fourteen.

    Retuning could not fix it: measured across seven clips, hip travel for a real
    push-up rep ranged 0.10 to 0.55 of body length, because the body pivots at the
    toes and how much of that reaches the camera depends on the camera. The joint
    that flexes is the reliable signal.
    """
    assert pushup.PUSHUP.use_travel_gate is False
    assert squat.SQUAT.use_travel_gate is True, "the squat's hips ARE the movement"

    # travel that would fail every squat floor; the reps must survive anyway
    angles = (_set_of_reps(4, elbow=95.0, body=4.0)[0])
    reps = [(i * 60, i * 60 + 30, i * 60 + 59) for i in range(4)]
    flat = np.zeros(240)          # no body travel at all

    kept = mechanics.keep_real_reps(pushup.PUSHUP, reps, flat, scale=0.5,
                                    angles_per_frame=angles, side="left", fps=30.0)
    assert len(kept) == 4


def test_pushup_still_rejects_a_dip_where_the_elbow_did_not_bend():
    """Dropping the travel gate must not mean accepting anything. Getting into
    position is rejected because the elbow stays near straight, which is what
    caught it on the real clip -- flexion 15 degrees against 66-88 for the reps."""
    def dip(n, at, elbow):
        return [_frame(elbow=elbow if i == at else 172.0, body=4.0) for i in range(n)]

    angles = (dip(60, 30, 165.0)      # settling into position: elbow barely bends
              + dip(60, 30, 95.0)
              + dip(60, 30, 92.0))
    reps = [(0, 30, 59), (60, 90, 119), (120, 150, 179)]
    flat = np.zeros(180)

    kept = mechanics.keep_real_reps(pushup.PUSHUP, reps, flat, scale=0.5,
                                    angles_per_frame=angles, side="left", fps=30.0)
    assert (0, 30, 59) not in kept
    assert len(kept) == 2


# ---------------------------------------------------------------------------
# a segment made of frames the validity gate already rejects is not a rep
# ---------------------------------------------------------------------------
# Reported by a participant in the first testing round: their last "rep" was
# them getting up and walking back to the phone. The pose estimate came apart,
# and the single frame in that window that squeaked under the plausibility
# ceiling was picked as the deepest and reported as the worst form of the set.
# Nine clean reps scored zero; the only thing the report pointed at was them
# reaching for the camera.
#
# Measured on that clip, and on this project's own push-up reference, the two
# populations do not overlap: real reps are 100% valid frames, artefacts run
# 36-54%. The floor sits at 60% in a 46-point gap rather than against either
# edge.

def _window(valid, invalid):
    """A rep window: `valid` scoreable frames then `invalid` implausible ones."""
    good = {"elbow_left": 90.0, "body_left": 4.0}
    bad = {"elbow_left": 170.0, "body_left": 175.0}   # body line past the ceiling
    return [dict(good)] * valid + [dict(bad)] * invalid


def test_a_window_of_mostly_invalid_frames_is_not_a_rep():
    from backend.exercises import mechanics
    from backend.exercises.pushup import PUSHUP

    angles = _window(33, 42)            # the participant's rep 11, 44% valid
    kept = mechanics.keep_real_reps(PUSHUP, [(0, 20, 74)], travel_y=None, scale=1.0,
                                    angles_per_frame=angles, side="left", fps=30.0)
    assert kept == []


def test_a_clean_rep_is_kept():
    from backend.exercises import mechanics
    from backend.exercises.pushup import PUSHUP

    angles = _window(50, 0)
    kept = mechanics.keep_real_reps(PUSHUP, [(0, 25, 49)], travel_y=None, scale=1.0,
                                    angles_per_frame=angles, side="left", fps=30.0)
    assert len(kept) == 1


def test_a_rep_with_a_few_bad_frames_survives():
    """Tracking drops a handful of frames on almost every real clip. The gate has
    to tolerate that, or it becomes a second detection-rate test."""
    from backend.exercises import mechanics
    from backend.exercises.pushup import PUSHUP

    angles = _window(40, 10)            # 80% valid
    kept = mechanics.keep_real_reps(PUSHUP, [(0, 25, 49)], travel_y=None, scale=1.0,
                                    angles_per_frame=angles, side="left", fps=30.0)
    assert len(kept) == 1


def test_a_rep_at_the_end_is_judged_with_the_frames_after_it():
    """The reference push-up clip's fourteenth "rep" is the dip on the way to standing
    up: 100% valid inside its own window and 44% once getting up is counted with it. The
    window a rep is measured in stops at the rhythm of the set; this gate still looks to
    the end of the clip, which is the only thing telling the two apart."""
    from backend.exercises import mechanics
    from backend.exercises.pushup import PUSHUP

    angles = _window(80, 0) + _window(0, 90)
    kept = mechanics.keep_real_reps(PUSHUP, [(0, 20, 39), (40, 60, 79)], travel_y=None,
                                    scale=1.0, angles_per_frame=angles, side="left",
                                    fps=30.0)

    assert kept == [(0, 20, 39)]


def test_parked_cues_are_declared_and_not_claimed_as_covered():
    """Holds the coverage contract for the push-up the way the squat and pull-up
    suites already do."""
    from backend.exercises.base import SAGITTAL, FRONTAL
    from backend.exercises.pushup_cues import PUSHUP_CUES
    covered = " ".join(PUSHUP_PROFILE.assessments(SAGITTAL)
                       + PUSHUP_PROFILE.assessments(FRONTAL)).lower()
    declared = " ".join(PUSHUP_PROFILE.not_yet_assessed).lower()
    keywords = {"elbow_flare": "elbow flare", "head_dropped": "head and neck"}
    for flag, cue in PUSHUP_CUES.items():
        if not cue.get("available", True):
            assert keywords[flag] in declared, f"{flag} is parked and undeclared"
            assert keywords[flag] not in covered, \
                f"{flag} is parked but claimed as covered"


# ---------------------------------------------------------------------------
# one push-up split in two
# ---------------------------------------------------------------------------

def _two_bottoms(elbow_between):
    """Bottoms of 85 and 86 degrees at frames 15 and 45, with the elbow at
    `elbow_between` on the other frames from 8 to 52."""
    angles = [_frame(elbow=170.0) for _ in range(61)]
    for i in range(8, 53):
        angles[i] = _frame(elbow=elbow_between)
    angles[15] = _frame(elbow=85.0)
    angles[45] = _frame(elbow=86.0)
    return angles, [(0, 15, 30), (30, 45, 60)]


def test_a_pushup_whose_elbow_never_reopens_counts_once():
    """The elbow opens 4 degrees between the two bottoms, so they are one rep, kept at
    the deeper one."""
    angles, reps = _two_bottoms(elbow_between=90.0)
    assert mechanics.merge_reps_without_reopening(pushup.PUSHUP, reps, angles, "left", 30.0) \
        == [(0, 15, 60)]


def test_pushups_with_the_arms_straightening_between_them_stay_two():
    angles, reps = _two_bottoms(elbow_between=160.0)
    assert mechanics.merge_reps_without_reopening(pushup.PUSHUP, reps, angles, "left", 30.0) \
        == reps


def test_only_the_pushup_merges_reps_on_the_elbow():
    """Guard: the limit was measured on push-up clips only."""
    from backend.exercises.pullup import PULLUP
    assert pushup.PUSHUP.min_reopen > 0
    assert squat.SQUAT.min_reopen == 0
    assert PULLUP.min_reopen == 0


def test_flag_frames_marks_the_frame_the_rep_was_judged_at():
    """Same as the squat -- the red goes round the frame the score picked, so the
    worst key frame can't come out green next to a red window"""
    angles = ([_frame(elbow=170.0)] * 20 + [_frame(elbow=121.0)]
              + [_frame(elbow=170.0)] * 19 + [_frame(elbow=125.0)]
              + [_frame(elbow=170.0)] * 20)
    reps = [(0, 40, 60)]

    flagged = pushup.flag_frames(angles, reps, "left", fps=30.0)
    worst = mechanics.worst_frame(pushup.PUSHUP, angles, reps, "left", 30.0)

    assert worst == 40
    assert "left_elbow" in flagged[worst]

"""Push-up analyser: angles, validity, cues, and the shared machinery under it.

Two things these are written to protect. First the push-up's own maths, which is
new. Second the property WP-04 exists to test: that adding a second exercise did
not require forking the rep machinery, so a fix to one benefits both.
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
    """The point of WP-04. If these had been forked, a fix to one would silently
    not reach the other -- which is exactly how the earlier defects would have
    come back."""
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
    # Updated when the pull-up landed in WP-08. The list is asserted rather than
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

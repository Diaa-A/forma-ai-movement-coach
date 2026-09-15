"""Layer 1 cue evaluator: thresholds, visibility gating, view guidance."""
import numpy as np

from backend.exercises.squat_cues import evaluate_squat, SQUAT_CUES, SQUAT_PROFILE
from backend.pipeline.pose import LM

FPS = 30.0
REPS = [(0, 15, 30)]   # 15-frame descent at 30fps = 0.5s, not "fast"


def _frame(knee_l=90.0, knee_r=None, spine=35.0, shin=30.0):
    if knee_r is None:
        knee_r = knee_l
    return {
        "knee_left": knee_l, "knee_right": knee_r,
        "hip_left": 90.0, "hip_right": 90.0,
        "ankle_left": 80.0, "ankle_right": 80.0,
        "shin_left": shin, "shin_right": shin,
        "spine": spine,
    }


def _clip(n=31, **kw):
    return [_frame(**kw) for _ in range(n)]


def _flags(ev):
    return {c.flag for c in ev.cues_fired}


def _landmarks(n=31, far_leg_vis=0.95):
    """Visibility-only landmark stack: everything confident except the far leg."""
    lms = np.zeros((n, 33, 4))
    lms[:, :, 3] = 0.95
    for name in ("right_hip", "right_knee", "right_ankle"):
        lms[:, LM[name], 3] = far_leg_vis
    return lms


def test_shallow_depth_fires():
    ev = evaluate_squat(_clip(knee_l=125.0), REPS, "left", FPS)
    assert "shallow_depth" in _flags(ev)
    assert not any("full depth" in p for p in ev.positives)


def test_deep_squat_earns_the_depth_positive():
    ev = evaluate_squat(_clip(knee_l=85.0), REPS, "left", FPS)
    assert "shallow_depth" not in _flags(ev)
    assert any("full depth" in p for p in ev.positives)


def test_forward_lean_fires_on_trunk_shin_excess():
    ev = evaluate_squat(_clip(spine=60.0, shin=35.0), REPS, "left", FPS)   # +25
    assert "excessive_forward_lean" in _flags(ev)


def test_balanced_lean_does_not_fire():
    ev = evaluate_squat(_clip(spine=40.0, shin=35.0), REPS, "left", FPS)   # +5
    assert "excessive_forward_lean" not in _flags(ev)
    assert any("parallel to your shins" in p for p in ev.positives)


def test_asymmetry_fires_when_far_leg_is_visible():
    ev = evaluate_squat(_clip(knee_l=90.0, knee_r=130.0), REPS, "left", FPS,
                        landmarks=_landmarks(far_leg_vis=0.95))
    assert "knee_left_right_asymmetry" in _flags(ev)
    # The frontal plane was observable, so there must be no "film from the front"
    # instruction. There may still be guidance: detectors that are parked stay
    # unassessed whichever way the clip was filmed, and the system now says so
    # rather than letting that silence read as a clean result.
    assert "film a set from the front" not in (ev.view_guidance or "")


def test_asymmetry_suppressed_when_far_leg_is_occluded():
    ev = evaluate_squat(_clip(knee_l=90.0, knee_r=130.0), REPS, "left", FPS,
                        landmarks=_landmarks(far_leg_vis=0.45))
    assert "knee_left_right_asymmetry" not in _flags(ev)
    # neither the fault nor the matching positive may be claimed
    assert not any("side-to-side" in p for p in ev.positives)
    assert ev.view_guidance is not None
    assert "front" in ev.view_guidance


def test_symmetry_positive_when_visible_and_even():
    ev = evaluate_squat(_clip(knee_l=90.0, knee_r=92.0), REPS, "left", FPS,
                        landmarks=_landmarks(far_leg_vis=0.95))
    assert any("side-to-side" in p for p in ev.positives)


def test_rep_inconsistency_across_reps():
    frames = _clip(31, knee_l=88.0) + _clip(31, knee_l=112.0)   # 24 degree spread
    reps = [(0, 15, 30), (31, 46, 61)]
    ev = evaluate_squat(frames, reps, "left", FPS)
    assert "rep_inconsistency" in _flags(ev)


def test_fast_descent_fires_on_snappy_reps():
    ev = evaluate_squat(_clip(), [(0, 8, 30)], "left", FPS)   # 0.27s down
    assert "fast_descent" in _flags(ev)


def test_controlled_tempo_positive():
    ev = evaluate_squat(_clip(), REPS, "left", FPS)
    assert "fast_descent" not in _flags(ev)
    assert any("controlled" in p for p in ev.positives)


def test_hips_rising_before_the_knees_fires_hip_rise_first():
    """Halfway up the knee is nearly straight while the trunk still has 25 degrees
    to come back. When the knee is still opening, it does not fire."""
    phases = ["ascent"] * 31
    hips_first = _clip()
    hips_first[22] = _frame(knee_l=150.0, spine=50.0)   # (15 + 30) // 2, the ascent midpoint
    hips_first[30] = _frame(knee_l=160.0, spine=25.0)
    together = _clip()
    together[22] = _frame(knee_l=120.0, spine=35.0)
    together[30] = _frame(knee_l=165.0, spine=25.0)

    assert "hip_rise_first" in _flags(
        evaluate_squat(hips_first, REPS, "left", FPS, phase_per_frame=phases))
    assert "hip_rise_first" not in _flags(
        evaluate_squat(together, REPS, "left", FPS, phase_per_frame=phases))


def test_no_reps_yields_nothing():
    ev = evaluate_squat(_clip(), [], "left", FPS)
    assert ev.rep_count == 0
    assert ev.cues_fired == []
    assert ev.view_guidance is None


# ---------------------------------------------------------------------------
# Coverage as a contract (handoff section 15)
# ---------------------------------------------------------------------------

def test_every_cue_is_declared_somewhere_in_the_profile():
    """A cue that appears in no plane list and no not-yet-assessed list is
    invisible: it never fires, nothing tells the user it was not checked, and the
    report reads as though it was checked and found fine.

    That is exactly what happened with the push-up's elbow flare -- a user asked
    about their arms and got silence, because flare was declared nowhere. This
    asserts the squat cannot drift into the same state.
    """
    from backend.exercises.base import SAGITTAL, FRONTAL
    declared = set(SQUAT_PROFILE.assessments(SAGITTAL)) \
        | set(SQUAT_PROFILE.assessments(FRONTAL)) \
        | set(SQUAT_PROFILE.not_yet_assessed)
    # matched loosely on keywords, since the declared strings are user-facing
    # prose rather than cue keys
    expected = {
        "shallow_depth": "depth",
        "excessive_forward_lean": "lean",
        "hip_rise_first": "hip drive",
        "knee_left_right_asymmetry": "symmetry",
        "rep_inconsistency": "consistency",
        "fast_descent": "tempo",
        "knee_valgus": "knee tracking",
        "heel_lift": "heel",
    }
    assert set(expected) == set(SQUAT_CUES), "a cue was added without declaring it"
    blob = " ".join(declared).lower()
    for flag, keyword in expected.items():
        assert keyword in blob, f"{flag} is not declared in the profile"


def test_parked_cues_are_not_claimed_as_covered():
    """A view must not say it 'covers' something no working detector reports on.
    Claiming coverage is a different lie from staying silent, and a worse one --
    the user reads it as a clean bill of health."""
    from backend.exercises.base import SAGITTAL, FRONTAL
    covered = " ".join(SQUAT_PROFILE.assessments(SAGITTAL)
                       + SQUAT_PROFILE.assessments(FRONTAL)).lower()
    for flag, cue in SQUAT_CUES.items():
        if not cue.get("available", False):
            keyword = {"knee_valgus": "knee tracking", "heel_lift": "heel"}[flag]
            assert keyword not in covered, f"{flag} is parked but claimed as covered"

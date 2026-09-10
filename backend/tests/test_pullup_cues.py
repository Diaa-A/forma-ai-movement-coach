"""Layer 1 for the pull-up.

Four of the five cues are parked, so most of what is worth testing here is that
they stay parked and that the profile says so out loud. The coverage contract is
the whole point of the exercise: a user must be able to tell "checked and fine"
from "not looked at", and with this many parked detectors the second case is the
common one.
"""
import numpy as np

from backend.exercises.base import FRONTAL, SAGITTAL
from backend.exercises.pullup_cues import (PULLUP_CUES, PULLUP_PROFILE,
                                           _RANGE_INCONSISTENCY_DEG,
                                           evaluate_pullup)


def _set(tops):
    """A clip whose reps reach the given elbow angles at the top.

    Three frames per rep so deepest_frame has something to pick from, with the
    top in the middle where the rep's bottom index points.
    """
    angles, reps = [], []
    for n, top in enumerate(tops):
        base = n * 3
        angles += [{"elbow_left": 170.0, "trunk": 2.0},
                   {"elbow_left": top, "trunk": 2.0},
                   {"elbow_left": 170.0, "trunk": 2.0}]
        reps.append((base, base + 1, base + 2))
    return angles, reps


def test_reps_that_agree_are_reported_as_consistent():
    angles, reps = _set([30.0, 34.0, 28.0])
    ev = evaluate_pullup(angles, reps, "left", 30.0)

    assert ev.rep_count == 3
    assert not ev.cues_fired
    assert any("same height" in p for p in ev.positives)


def test_reps_that_differ_by_more_than_the_instrument_fire_the_cue():
    """The threshold is three times the measured elbow error, so a spread this
    wide is the person changing rather than the measurement wobbling."""
    angles, reps = _set([25.0, 30.0, 70.0])
    ev = evaluate_pullup(angles, reps, "left", 30.0)

    flags = [c.flag for c in ev.cues_fired]
    assert flags == ["inconsistent_range"]
    assert not ev.positives


def test_a_spread_just_under_the_tolerance_says_nothing():
    """Pins the boundary. Just inside is silence, not a quiet fault."""
    angles, reps = _set([30.0, 30.0 + _RANGE_INCONSISTENCY_DEG - 1.0])
    ev = evaluate_pullup(angles, reps, "left", 30.0)
    assert not ev.cues_fired


def test_a_single_rep_is_not_judged_for_consistency():
    """There is nothing to be consistent with. Firing here would be asserting a
    fault from one observation."""
    angles, reps = _set([65.0])
    ev = evaluate_pullup(angles, reps, "left", 30.0)

    assert ev.rep_count == 1
    assert not ev.cues_fired
    assert not ev.positives


def test_the_parked_cues_never_fire():
    """They have no calibrated threshold behind them. _fire refuses them, and
    that refusal is what keeps an uncalibrated claim off the user's screen."""
    parked = [f for f, c in PULLUP_CUES.items() if not c.get("available", False)]
    assert set(parked) == {"partial_range", "incomplete_extension", "kipping",
                           "grip_too_wide"}

    # a set that would trip every one of them if they were live
    angles, reps = _set([95.0, 20.0, 88.0])
    ev = evaluate_pullup(angles, reps, "left", 30.0)
    assert all(c.flag not in parked for c in ev.cues_fired)


def test_parked_cues_are_declared_rather_than_silently_dropped():
    """§15's rule. Every parked detector has to be named in not_yet_assessed, or
    the user has no way to tell it was never looked at."""
    declared = " ".join(PULLUP_PROFILE.not_yet_assessed).lower()
    keywords = {"partial_range": "how high you pull",
                "incomplete_extension": "fully straighten",
                "kipping": "kipping",
                "grip_too_wide": "grip width"}
    for flag, cue in PULLUP_CUES.items():
        if not cue.get("available", False):
            assert keywords[flag] in declared, f"{flag} is parked and undeclared"


def test_the_profile_claims_no_sagittal_coverage():
    """Body swing is the only sagittal thing a pull-up has and it is parked, so
    listing anything here would read as a clean bill of health on it."""
    assert PULLUP_PROFILE.assessments(SAGITTAL) == []
    assert PULLUP_PROFILE.assessments(FRONTAL)
    assert PULLUP_PROFILE.view_label == "front-on"
    assert PULLUP_PROFILE.label == "Pull-up"


def test_parked_cues_are_not_claimed_as_covered():
    """The other half of the same rule: declared-but-parked must not also appear
    in the coverage list, which is what the user reads as 'checked'."""
    covered = " ".join(PULLUP_PROFILE.assessments(SAGITTAL)
                       + PULLUP_PROFILE.assessments(FRONTAL)).lower()
    for word in ("chin", "straighten", "kipping", "grip"):
        assert word not in covered, f"'{word}' is parked but claimed as covered"


def test_the_report_says_what_was_not_assessed():
    angles, reps = _set([30.0, 32.0])
    ev = evaluate_pullup(angles, reps, "left", 30.0)

    assert ev.view_guidance, "a pull-up report with no coverage line hides four parked cues"
    assert "not assessed yet" in ev.view_guidance.lower()
    assert any(ev.view_guidance == n for n in ev.notes)


def test_no_reps_means_no_coverage_claim_either():
    """With nothing detected there is no clip to describe, and a coverage line
    would imply one was analysed."""
    ev = evaluate_pullup([], [], "left", 30.0)
    assert ev.rep_count == 0
    assert ev.view_guidance is None


def test_an_invalid_top_frame_is_not_counted_as_a_rep_height():
    """A degenerate elbow reading must not become a rep's measured height, or the
    spread is computed against a landmark error."""
    angles = [{"elbow_left": 170.0, "trunk": 2.0},
              {"elbow_left": 0.0, "trunk": 2.0},      # degenerate
              {"elbow_left": 170.0, "trunk": 2.0}]
    ev = evaluate_pullup(angles, [(0, 1, 2)], "left", 30.0)
    assert not any("spread" in n for n in ev.notes)

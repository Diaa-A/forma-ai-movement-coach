"""Layer 1 for the pull-up.

Four of the five cues are parked, so most of this checks that they stay parked
and that the profile declares them: a user has to be able to tell "checked" from
"not looked at".
"""
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
        angles += [{"elbow": 170.0, "trunk": 2.0},
                   {"elbow": top, "trunk": 2.0},
                   {"elbow": 170.0, "trunk": 2.0}]
        reps.append((base, base + 1, base + 2))
    return angles, reps


def test_reps_that_agree_are_reported_as_consistent():
    angles, reps = _set([30.0, 34.0, 28.0])
    ev = evaluate_pullup(angles, reps, "left", 30.0)

    assert ev.rep_count == 3
    assert not ev.cues_fired
    assert any("same height" in p for p in ev.positives)


def test_reps_that_differ_by_more_than_the_instrument_fire_the_cue():
    """20 degrees is about three times the elbow error, so a spread this wide is
    the reps differing, not noise."""
    angles, reps = _set([25.0, 30.0, 70.0])
    ev = evaluate_pullup(angles, reps, "left", 30.0)

    flags = [c.flag for c in ev.cues_fired]
    assert flags == ["inconsistent_range"]
    assert not ev.positives


def test_a_spread_just_under_the_tolerance_says_nothing():
    """Pins the boundary: just inside the tolerance says nothing."""
    angles, reps = _set([30.0, 30.0 + _RANGE_INCONSISTENCY_DEG - 1.0])
    ev = evaluate_pullup(angles, reps, "left", 30.0)
    assert not ev.cues_fired


def test_a_frame_outside_the_evaluation_window_is_not_read_as_a_rep_height():
    """A real set read 47 degrees of spread against 18 because the last rep's span
    ran to the end of the clip, where letting go of the bar reads as a bent elbow."""
    angles, reps = [], []
    for n, top in enumerate([70.0, 72.0, 68.0]):
        base = n * 60
        angles += [{"elbow": 170.0, "trunk": 2.0} for _ in range(60)]
        angles[base + 30] = {"elbow": top, "trunk": 2.0}
        reps.append((base, base + 30, base + 59))
    angles[reps[-1][2]] = {"elbow": 20.0, "trunk": 2.0}

    ev = evaluate_pullup(angles, reps, "left", 30.0)

    assert not ev.cues_fired
    assert any("70" in n or "average elbow" in n for n in ev.notes)


def test_a_set_that_fades_says_so():
    """Every rep lower than the last is what running out of strength looks like, and a
    real set read 68, 76, 80, 85, 98 at the elbow. The spread alone cannot say that."""
    angles, reps = _set([68.0, 76.0, 80.0, 85.0, 98.0])
    ev = evaluate_pullup(angles, reps, "left", 30.0)

    assert [c.flag for c in ev.cues_fired] == ["inconsistent_range"]
    assert any("each rep finished lower than the one before" in n for n in ev.notes)


def test_one_low_first_rep_is_named_as_one_rep():
    """Two stock clips fire the cue on a low first rep and four that match within a
    degree. That is a warm-up rep, and the report should say which rep it was."""
    angles, reps = _set([66.0, 37.0, 37.0, 36.0])
    ev = evaluate_pullup(angles, reps, "left", 30.0)

    assert [c.flag for c in ev.cues_fired] == ["inconsistent_range"]
    assert any("only the first rep came up lower" in n for n in ev.notes)


def test_a_set_with_no_pattern_is_not_given_one():
    """Reps that wander have no shape worth describing, and inventing one would be the
    report telling a user something the measurement does not support."""
    angles, reps = _set([34.0, 27.0, 49.0, 36.0, 31.0])
    ev = evaluate_pullup(angles, reps, "left", 30.0)

    assert [c.flag for c in ev.cues_fired] == ["inconsistent_range"]
    assert not any("each rep" in n or "only the" in n for n in ev.notes)


def test_a_single_rep_is_not_judged_for_consistency():
    """One rep has nothing to be consistent with."""
    angles, reps = _set([65.0])
    ev = evaluate_pullup(angles, reps, "left", 30.0)

    assert ev.rep_count == 1
    assert not ev.cues_fired
    assert not ev.positives


def test_the_parked_cues_never_fire():
    """No calibrated threshold behind them, so _fire refuses them."""
    parked = [f for f, c in PULLUP_CUES.items() if not c.get("available", False)]
    assert set(parked) == {"partial_range", "incomplete_extension", "kipping",
                           "grip_too_wide"}

    # a set that would trip every one of them if they were live
    angles, reps = _set([95.0, 20.0, 88.0])
    ev = evaluate_pullup(angles, reps, "left", 30.0)
    assert all(c.flag not in parked for c in ev.cues_fired)


def test_parked_cues_are_declared_rather_than_silently_dropped():
    """Every parked cue has to be named in not_yet_assessed, or the user cannot
    tell it was never looked at."""
    declared = " ".join(PULLUP_PROFILE.not_yet_assessed).lower()
    keywords = {"partial_range": "chin clears the bar",
                "incomplete_extension": "straighten fully",
                "kipping": "kipping",
                "grip_too_wide": "grip width"}
    for flag, cue in PULLUP_CUES.items():
        if not cue.get("available", False):
            assert keywords[flag] in declared, f"{flag} is parked and undeclared"


def test_the_profile_claims_no_sagittal_coverage():
    """Body swing is the only sagittal assessment and it is parked, so nothing is
    listed there."""
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

    assert ev.view_guidance, \
        "a pull-up report with no coverage line hides four parked cues"
    assert "not assessed yet" in ev.view_guidance.lower()
    assert any(ev.view_guidance == n for n in ev.notes)


def test_no_reps_means_no_coverage_claim_either():
    """With nothing detected there is no clip to describe, and a coverage line
    would imply one was analysed."""
    ev = evaluate_pullup([], [], "left", 30.0)
    assert ev.rep_count == 0
    assert ev.view_guidance is None


def test_a_frame_the_validity_gate_rejects_is_not_used_as_a_rep_height():
    """The second rep's lowest reading comes with a trunk lean no hanging body
    makes. Its height has to come from the 36-degree frame beside it: counting the
    0 would put the spread at 30 and fire the cue."""
    angles = [{"elbow": 170.0, "trunk": 2.0},
              {"elbow": 30.0, "trunk": 2.0},
              {"elbow": 170.0, "trunk": 2.0},
              {"elbow": 170.0, "trunk": 2.0},
              {"elbow": 0.0, "trunk": 75.0},     # tracker lost the torso
              {"elbow": 36.0, "trunk": 2.0},
              {"elbow": 170.0, "trunk": 2.0}]
    ev = evaluate_pullup(angles, [(0, 1, 2), (3, 4, 6)], "left", 30.0)

    assert not ev.cues_fired
    assert any("33°" in n for n in ev.notes)


def test_the_not_assessed_line_reads_as_a_list():
    """coverage_guidance joins items with commas and a final "and", so an item
    carrying either of its own makes the sentence ambiguous. The first version
    rendered "body swing and kipping and grip width"."""
    for item in PULLUP_PROFILE.not_yet_assessed:
        assert "," not in item, item
        assert " and " not in item, item


def test_tempo_and_arm_evenness_are_declared():
    """Squat and push-up both time the lowering, so a pull-up user will ask about
    tempo, and front-on is the view that shows both arms at once."""
    declared = " ".join(PULLUP_PROFILE.not_yet_assessed).lower()
    assert "how fast you lower" in declared
    assert "evenly" in declared

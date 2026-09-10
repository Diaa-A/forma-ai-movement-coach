"""Layer 1 for the pull-up: the cue database and the evaluator that fires it.

Same contract as `squat_cues.py` and `pushup_cues.py` — `fault` and `fix` kept
apart so it stays obvious which side Layer 2 is rephrasing, and cues whose
detector is not trustworthy carrying `available: False`.

**Four of the five cues here are parked, which is the outcome of calibration
rather than an oversight.** WP-08 step 2 measured all 25 Penn Action sequences
and found the pipeline reads a pull-up well — elbow error is 5.5 degrees at deep
flexion against the push-up benchmark's 8.1 — but Penn Action labels what an
action is, not whether it was done well. There is no incomplete repetition in the
set to calibrate a boundary against, and across 26 repetitions the elbow at the
top ran 2.8 to 73.8 degrees in one unbroken spread. A threshold placed in that
range would separate a population from itself.

The one cue that does fire needs no such boundary, which is why it survived. It
compares a repetition against the others in the same set, so the person is their
own reference, and the only number it needs is how much difference exceeds the
instrument. That number is measured.

The camera view is the other thing that differs. Pull-ups are filmed from the
front — median shoulder separation across the subset was 0.34 of torso length,
where a genuinely side-on view collapses the shoulders on top of each other —
because the bar is overhead and the photographer stands in front of it. So the
frontal plane is the productive one here, the reverse of both other exercises.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .base import ExerciseProfile, FRONTAL, SAGITTAL, coverage_guidance
from .mechanics import CueHit, Evaluation
from .pullup import _elbow_key, frame_valid
from . import pullup


PULLUP_CUES = {
    "inconsistent_range": {
        "severity": "secondary",
        "fault": ("your reps did not all come up to the same height — some "
                  "finished noticeably lower than others."),
        "fix": ("pick a height you can reach on every rep and end the set when "
                "you can no longer reach it, rather than shortening the last "
                "few."),
        "joints": ["left_elbow", "right_elbow"],
        "phase": "set",
        "plane": FRONTAL,
        "available": True,
    },
    "partial_range": {
        "severity": "primary",
        "fault": ("you stopped short of the bar — the elbows never closed far "
                  "enough to bring the chin above it."),
        "fix": ("pull until your chin clears the bar, or use a band so you can. "
                "Fewer full reps beat more partial ones."),
        "joints": ["left_elbow", "right_elbow"],
        "phase": "bottom",
        "plane": FRONTAL,
        # Parked, and the reason is not the usual one. This is measurable: elbow
        # error at deep flexion is 5.5 degrees, the best band of the six. What is
        # missing is any labelled partial repetition to set the boundary from.
        # Across 26 scored reps the elbow at the top ran 2.8 to 73.8 with no
        # division anywhere in it, so a threshold would be invented rather than
        # calibrated. Unparks when clips with known partial reps exist.
        "available": False,
    },
    "incomplete_extension": {
        "severity": "secondary",
        "fault": ("you started the next rep before the arms had straightened, so "
                  "the bottom of each rep was cut short."),
        "fix": ("let the arms come to a full hang between reps. The stretched "
                "position is part of the movement, not a rest."),
        "joints": ["left_elbow", "right_elbow"],
        "phase": "bottom",
        "plane": FRONTAL,
        # Same reason as partial_range. Elbow at the hang is well measured — 6.2
        # degrees of error, median 173.9 across the set — and nothing in the set
        # is marked as a partial hang to calibrate against.
        "available": False,
    },
    "kipping": {
        "severity": "secondary",
        "fault": ("your body swung to generate momentum rather than the arms "
                  "doing the work."),
        "fix": ("squeeze the glutes and hold the legs still, and drop the rep "
                "count if that makes the set impossible. A strict rep trains "
                "more than a swung one."),
        "joints": ["left_hip", "right_hip", "left_shoulder", "right_shoulder"],
        "phase": "set",
        "plane": SAGITTAL,
        # Swing is toward and away from a front-on camera, which is the one
        # direction it cannot see. Trunk lean across the subset ran 0.6 to 43.9
        # degrees with a median of 2.9, and the single clip at the top of that
        # range is visibly a kipping rep -- so the signal exists, from the wrong
        # view, on one example. Both problems have to be solved before it fires.
        "available": False,
    },
    "grip_too_wide": {
        "severity": "secondary",
        "fault": ("your hands were set wider than the shoulders, which shortens "
                  "the pull and loads the shoulder joint."),
        "fix": "start with the hands about shoulder-width and adjust from there.",
        "joints": ["left_wrist", "right_wrist"],
        "phase": "bottom",
        "plane": FRONTAL,
        # Wrist separation over shoulder separation ran 0.50 to 2.32, which looks
        # usable until you notice shoulder spread itself varied 0.25 to 0.43 of
        # torso across the same clips. The ratio moves with camera angle as much
        # as with grip, so it is confounded rather than measured.
        "available": False,
    },
}


PULLUP_POSITIVES = {
    "consistent_range": {
        "text": ("every rep came up to about the same height, which is the part "
                 "most people lose first as a set gets hard.")},
}


PULLUP_PROFILE = ExerciseProfile(
    name="pullup",
    display_name="Pull-up",
    view_label="front-on",
    filming_guide=("Film from in front of the bar, far enough back that your "
                   "whole body stays in frame at the top and the bottom. 1080p "
                   "is plenty. Filming from the side hides one arm behind the "
                   "other and there is nothing this system can do about that."),
    plane_assessments={
        FRONTAL: ["repetition count", "how consistent your range of motion is"],
        # Deliberately empty. Nothing is claimed for the sagittal plane, because
        # the only thing that lives there for a pull-up is body swing and that is
        # parked. Listing it would read as coverage.
        SAGITTAL: [],
    },
    not_yet_assessed=["how high you pull, and whether the chin clears the bar",
                      "whether the arms fully straighten between reps",
                      "body swing and kipping",
                      "grip width"],
)


# Spread of the top-of-rep elbow angle across a set, past which the reps really
# did differ. Set from the instrument rather than from labelled form, which is
# what makes it defensible when the other thresholds were not: elbow error is
# 5.5 to 6.7 degrees over the angles this reads, so a spread of 20 is roughly
# three times the noise and cannot be the measurement wobbling.
#
# It matches the push-up's depth-inconsistency figure, which is a coincidence of
# the same reasoning rather than a value copied across.
_RANGE_INCONSISTENCY_DEG = 20.0


def _fire(fired, flag, rep_index):
    """Record a cue hit, or extend the one already recorded for that cue."""
    spec = PULLUP_CUES[flag]
    if not spec.get("available", True):
        return
    hit = fired.get(flag)
    if hit is None:
        hit = CueHit(flag=flag, severity=spec["severity"], fault=spec["fault"],
                     fix=spec["fix"], joints=list(spec["joints"]))
        fired[flag] = hit
    if rep_index is not None and rep_index not in hit.rep_indices:
        hit.rep_indices.append(rep_index)


def evaluate_pullup(angles_per_frame, reps, side, fps, phase_per_frame=None,
                    landmarks=None):
    """Fire the cue database against one analysed set.

    Shorter than its two siblings because four of the five cues are parked. What
    is left reads the elbow at the top of each rep and asks whether the reps
    agreed with each other.

    `phase_per_frame` and `landmarks` are accepted and unused. Every cue that
    would need them is parked, and the signature is shared with the other two
    evaluators so the registry can call any of them the same way.
    """
    fired = {}
    notes: List[str] = []
    positives: List[str] = []

    key = _elbow_key(side)
    tops = []

    for (s, b, e) in reps:
        f = pullup.deepest_frame(angles_per_frame, max(s, 0),
                                 min(e, len(angles_per_frame) - 1), side)
        if f is None:
            continue
        angle = angles_per_frame[f].get(key)
        if angle is not None and np.isfinite(angle) and frame_valid(angles_per_frame[f], side):
            tops.append(float(angle))

    spread = max(tops) - min(tops) if len(tops) >= 2 else None

    if spread is not None and spread > _RANGE_INCONSISTENCY_DEG:
        _fire(fired, "inconsistent_range", None)
    elif spread is not None:
        positives.append(PULLUP_POSITIVES["consistent_range"]["text"])

    if tops:
        notes.append("average elbow angle at the top of the rep: {:.0f}°"
                     .format(float(np.mean(tops))))
    if spread is not None:
        notes.append("spread between the highest and lowest rep: {:.0f}° "
                     "(tolerance {:.0f}°)".format(spread, _RANGE_INCONSISTENCY_DEG))

    # Always True: a pull-up is filmed front-on, so the frontal plane is the one
    # the clip is expected to reach. This leaves coverage_guidance to say what is
    # not assessed yet, which for this exercise is most of it.
    guidance = coverage_guidance(PULLUP_PROFILE, frontal_observed=True)
    view_guidance = guidance if reps else None
    if view_guidance:
        notes.append(view_guidance)

    return Evaluation(
        exercise="pullup",
        side=side,
        rep_count=len(reps),
        cues_fired=list(fired.values()),
        positives=positives,
        notes=notes,
        view_guidance=view_guidance,
    )

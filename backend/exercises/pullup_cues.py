"""Layer 1 for the pull-up: the cue database and the evaluator that fires it.

Same contract as `squat_cues.py` and `pushup_cues.py`: `fault` and `fix` are kept
apart so it is clear what Layer 2 rephrases, and a cue with no trustworthy
detector carries `available: False` and never fires.

Four of the five cues are parked. The pull-up is measured well, but calibration
found nothing to set a height or extension threshold from (see `pullup.py`). The
cue that does fire compares the reps in a set with each other, so all it needs to
know is how much difference is bigger than the measurement error.

The profile is front-on, the reverse of the squat and the push-up, so the
assessments sit in the frontal plane.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .base import ExerciseProfile, FRONTAL, SAGITTAL, coverage_guidance
from .mechanics import CueHit, Evaluation
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
        # Parked. There is no labelled partial rep to set a boundary from: across 23
        # reps the elbow at the top ran 0 to 84 with no gap in it.
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
        # Parked for the same reason: nothing is labelled as a partial hang.
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
        # Parked. Swing moves toward and away from a front-on camera, and nothing in
        # Penn Action is labelled as kipping to set a threshold from.
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
        # Parked. Wrist separation over shoulder separation ran 0.50 to 2.32, but
        # shoulder spread itself ran 0.25 to 0.43 of torso over the same clips, so
        # the ratio follows camera angle as much as grip.
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
        # Empty on purpose: body swing is the only sagittal assessment and it is
        # parked, so listing it here would read as covered.
        SAGITTAL: [],
    },
    # coverage_guidance joins these with commas and a final "and", so no item can
    # contain either.
    not_yet_assessed=["whether your chin clears the bar",
                      "whether your arms straighten fully between reps",
                      "body swing or kipping",
                      "grip width",
                      "how evenly your two arms pull",
                      "how fast you lower"],
    # "lower" on its own is left out: the one live cue is about reps that finish
    # lower than others.
    not_assessed_words=["chin", "bar", "straighten", "extend", "extension", "lock out",
                        "lockout", "full hang", "dead hang", "swing", "swinging", "kip",
                        "kipping", "momentum", "grip", "hands", "shoulder-width",
                        "evenly", "both arms", "one arm", "lowering", "descent",
                        "tempo", "slowly", "control"],
)


# Spread of the top-of-rep elbow angle across a set beyond which the reps really
# differed. Set from measurement error rather than labelled form: per-arm elbow
# error is 5.5 to 6.7 degrees over the angles this reads, so 20 is about three
# times the noise. The two-arm average this now reads has not been benchmarked
# on its own.
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

    Reads the elbow at the top of each rep, both arms combined, and checks the
    reps against each other. `phase_per_frame` and `landmarks` are unused -- the
    cues that would need them are parked -- but stay in the signature the
    registry calls every evaluator with.
    """
    fired = {}
    notes: List[str] = []
    positives: List[str] = []

    key = pullup.PULLUP.primary_angle(side)
    tops = []

    for (s, b, e) in reps:
        f = pullup.deepest_frame(angles_per_frame, max(s, 0),
                                 min(e, len(angles_per_frame) - 1), side)
        if f is None:
            continue
        angle = angles_per_frame[f].get(key)
        if angle is not None and np.isfinite(angle):
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

    # A pull-up is filmed front-on, so the frontal plane is always the one in view
    # and coverage_guidance only has to list what is not assessed yet.
    guidance = coverage_guidance(PULLUP_PROFILE, frontal_observed=True)
    view_guidance = guidance if reps else None
    if view_guidance:
        notes.append(view_guidance)

    # "both" rather than the runner's pick: the elbow measured is both arms, and
    # Layer 2 should not be handed one side to talk about. The results screen's
    # side chip still shows the runner's pick.
    return Evaluation(
        exercise="pullup",
        side="both",
        rep_count=len(reps),
        cues_fired=list(fired.values()),
        positives=positives,
        notes=notes,
        view_guidance=view_guidance,
    )

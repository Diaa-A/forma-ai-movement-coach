"""Layer 1 for the push-up: the cue database and the evaluator that fires it.

Same contract as `squat_cues.py`, deliberately. Each cue carries a `fault` (what
went wrong, in the user's terms) and a `fix` (what to do about it) as separate
strings, because Layer 2 is only ever allowed to rephrase them and keeping the two
apart is what makes it obvious which side is being rephrased. Cues whose detector
is not trustworthy yet carry `available: False` and never fire — the squat set
does the same for knee valgus, and parking a detector honestly is better than
shipping one that is wrong some of the time.

Thresholds live in `pushup.py` next to the scoring that uses them, so the overlay
and the cue layer cannot disagree about what counts as shallow.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .base import ExerciseProfile, SAGITTAL, FRONTAL, coverage_guidance
from .mechanics import CueHit, Evaluation, joints_visible
from .pushup import (
    BODY_LINE_LIMIT, DEPTH_FLAG_ELBOW_ANGLE, ELBOW_ASYMMETRY_DEG,
    ELBOW_TARGET_DEPTH, MIN_CUE_VISIBILITY, NECK_NEUTRAL_MIN,
    frame_valid, _elbow_key, _body_key,
)
from . import pushup


PUSHUP_CUES = {
    "shallow_depth": {
        "severity": "primary",
        "fault": ("you stopped short of the bottom — the elbows never bent far "
                  "enough for the chest to come down to the floor."),
        "fix": ("lower until your chest is a fist's height off the floor, or as "
                "close as you can hold the body line. Fewer full reps beat more "
                "partial ones."),
        "joints": ["left_elbow", "right_elbow"],
        "phase": "bottom",
        "plane": SAGITTAL,
        "available": True,
    },
    "hips_sagging": {
        "severity": "primary",
        "fault": ("your hips dropped below the line from shoulders to heels, so "
                  "the lower back took load the arms and trunk should have held."),
        "fix": ("squeeze your glutes and brace your stomach as if about to be "
                "pushed, and think of the body as one plank from head to heels."),
        "joints": ["left_hip", "right_hip"],
        "phase": "bottom",
        "plane": SAGITTAL,
        "available": True,
    },
    "hips_piked": {
        "severity": "secondary",
        "fault": ("your hips rode high, which shortens the movement and shifts "
                  "the work away from the chest and arms."),
        "fix": ("drop the hips until shoulders, hips and heels line up, then hold "
                "that line as you lower."),
        "joints": ["left_hip", "right_hip"],
        "phase": "bottom",
        "plane": SAGITTAL,
        "available": True,
    },
    "rep_inconsistency": {
        "severity": "secondary",
        "fault": ("your depth varied noticeably between reps — some went "
                  "considerably lower than others."),
        "fix": ("pick a depth you can repeat and stop the set when you can no "
                "longer reach it, rather than shortening the last few."),
        "joints": ["left_elbow", "right_elbow"],
        "phase": "set",
        "plane": SAGITTAL,
        "available": True,
    },
    "fast_descent": {
        "severity": "secondary",
        "fault": "you dropped into the bottom quickly rather than lowering under control.",
        "fix": ("take about two seconds on the way down. The lowering half is "
                "where most of the strength is built."),
        "joints": ["left_elbow", "right_elbow"],
        "phase": "descent",
        "plane": SAGITTAL,
        "available": True,
    },
    "elbow_asymmetry": {
        "severity": "secondary",
        "fault": ("one arm bent noticeably more than the other, so the work was "
                  "not shared evenly."),
        "fix": ("film a set from the front and watch for one shoulder dipping "
                "first. Slow the tempo until both sides move together."),
        "joints": ["left_elbow", "right_elbow"],
        "phase": "bottom",
        "plane": FRONTAL,
        "available": True,
    },
    "elbow_flare": {
        "severity": "primary",
        "fault": ("your upper arms travelled out wide from your body, closer to a "
                  "T shape than an arrow, which loads the front of the shoulder."),
        "fix": ("tuck the elbows to roughly 45 degrees from your sides, so the "
                "arms form an arrow rather than a T. Point your fingers forward "
                "and screw your hands into the floor to help hold it."),
        "joints": ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder"],
        "phase": "bottom",
        "plane": FRONTAL,
        # Declared but PARKED, and the distinction matters. Flare is lateral
        # abduction: filmed side-on the arm moves almost perpendicular to the
        # camera, so a flared arm and a tucked one both project to nearly the same
        # shoulder angle -- measured at 0-11 degrees across every rep of a real
        # side-on clip, whether or not the arms were flared. It is genuinely
        # unmeasurable from that view.
        #
        # It is listed anyway because declaring it is what lets the system tell a
        # user that their question needs a front-on clip, instead of silently
        # saying nothing about arms. The cue stays unavailable until there is
        # front-on footage to calibrate a threshold against; shipping a guessed
        # number would be the thing this project keeps refusing to do.
        "available": False,
    },
    "head_dropped": {
        "severity": "secondary",
        "fault": "your head dropped toward the floor instead of staying in line with the spine.",
        "fix": "look at a point slightly ahead of your hands and keep the chin tucked.",
        "joints": ["nose"],
        "phase": "bottom",
        "plane": SAGITTAL,
        # Parked. The nose is the only head landmark this project tracks, and it
        # moves with head ROTATION as well as with the neck flexing, so a dropped
        # head and a head turned to the side read alike. Measuring it needs the
        # ear landmarks; until then, firing this would be guessing.
        "available": False,
    },
}


PUSHUP_POSITIVES = {
    "good_depth": {"text": "you reached full depth — the chest came down on every rep."},
    "straight_body": {"text": ("you held a straight line from shoulders to heels "
                               "throughout, which is the hard part of a push-up.")},
    "consistent_reps": {"text": "your depth was consistent from the first rep to the last."},
    "controlled_tempo": {"text": "you lowered under control rather than dropping into the bottom."},
    "elbows_symmetric": {"text": "both arms bent evenly, so the work was shared."},
}


PUSHUP_PROFILE = ExerciseProfile(
    name="pushup",
    display_name="Push-up",
    view_label="side-on",
    # Same measured resolution note as the squat profile — see the comment there
    # for the numbers behind it.
    filming_guide=("Film side-on from about a metre away, low down so the camera "
                   "is roughly level with your shoulders, with your whole body "
                   "from hands to feet in frame. 1080p is plenty — filming in 4K "
                   "makes the upload slower and doesn't improve the analysis. For "
                   "an even-arms check, film a second set from in front of your "
                   "head."),
    plane_assessments={
        SAGITTAL: ["push-up depth", "hip position", "descent tempo", "rep consistency"],
        # Both of these need the camera in front of the head. Elbow flare is
        # lateral, so a side-on clip cannot see it at all -- listing it here is
        # what turns "the app ignored my question about my arms" into "film from
        # the front and it will answer it".
        FRONTAL: ["left/right arm symmetry"],
    },
    not_yet_assessed=["elbow flare (how wide the arms travel from the body)",
                      "head and neck position"],
)


_DEPTH_INCONSISTENCY_DEG = 20.0   # spread of bottom elbow angle across reps
_FAST_DESCENT_SEC = 0.5


def _elbow_keys(side):
    if side == "left":
        return "elbow_left", "elbow_right"
    return "elbow_right", "elbow_left"


def evaluate_pushup(angles_per_frame, reps, side, fps, phase_per_frame=None,
                    landmarks=None):
    """Fire the cue database against one analysed set.

    Mirrors `evaluate_squat` in shape and in the rules it obeys: a cue only fires
    from frames that pass the validity gate, cross-side comparisons are withheld
    unless the far side is genuinely visible, and a positive is only claimed when
    the matching fault did not fire and the measurement was actually possible.
    """
    fired = {}
    notes: List[str] = []
    positives: List[str] = []

    other_side = "right" if side == "left" else "left"
    elbow_primary, elbow_other = _elbow_keys(side)
    body_key = _body_key(side)

    bottom_elbows = []
    bottom_bodies = []
    descent_durations = []
    asym_assessable = False
    asym_reps = []

    for rep_i, (s, b, e) in enumerate(reps):
        f = pushup.deepest_frame(angles_per_frame, max(s, 0), min(e, len(angles_per_frame) - 1), side)
        if f is None:
            continue
        ang = angles_per_frame[f]

        elbow = ang.get(elbow_primary)
        body = ang.get(body_key)

        if elbow is not None and np.isfinite(elbow):
            bottom_elbows.append(elbow)
            if elbow > DEPTH_FLAG_ELBOW_ANGLE:
                _fire(fired, "shallow_depth", rep_i)

        # sag and pike are the same measurement read in opposite directions, but
        # they are separate cues because the corrections are opposite and telling
        # someone to squeeze their glutes when their hips are already too high
        # would make it worse
        if body is not None and np.isfinite(body):
            bottom_bodies.append(body)
            if body > BODY_LINE_LIMIT:
                _fire(fired, "hips_sagging", rep_i)
            elif body < -BODY_LINE_LIMIT:
                _fire(fired, "hips_piked", rep_i)

        other = ang.get(elbow_other)
        if (elbow is not None and other is not None
                and np.isfinite(elbow) and np.isfinite(other)
                and joints_visible(landmarks, f,
                                   [f"{other_side}_shoulder", f"{other_side}_elbow",
                                    f"{other_side}_wrist"], MIN_CUE_VISIBILITY)):
            asym_assessable = True
            if abs(elbow - other) > ELBOW_ASYMMETRY_DEG:
                asym_reps.append(rep_i)
                _fire(fired, "elbow_asymmetry", rep_i)

        if fps and fps > 0:
            descent = max(1, b - s) / fps
            descent_durations.append(descent)
            if descent < _FAST_DESCENT_SEC:
                _fire(fired, "fast_descent", rep_i)

    if len(bottom_elbows) >= 2:
        spread = max(bottom_elbows) - min(bottom_elbows)
        if spread > _DEPTH_INCONSISTENCY_DEG:
            _fire(fired, "rep_inconsistency", None)

    # positives, only where the opposite cue did not fire AND the thing was
    # actually measurable
    if bottom_elbows and max(bottom_elbows) <= DEPTH_FLAG_ELBOW_ANGLE:
        positives.append(PUSHUP_POSITIVES["good_depth"]["text"])
    if bottom_bodies and max(abs(v) for v in bottom_bodies) <= BODY_LINE_LIMIT:
        positives.append(PUSHUP_POSITIVES["straight_body"]["text"])
    if (len(bottom_elbows) >= 2
            and (max(bottom_elbows) - min(bottom_elbows)) <= _DEPTH_INCONSISTENCY_DEG
            and "rep_inconsistency" not in fired):
        positives.append(PUSHUP_POSITIVES["consistent_reps"]["text"])
    if descent_durations and min(descent_durations) >= _FAST_DESCENT_SEC:
        positives.append(PUSHUP_POSITIVES["controlled_tempo"]["text"])
    if asym_assessable and not asym_reps and "elbow_asymmetry" not in fired:
        positives.append(PUSHUP_POSITIVES["elbows_symmetric"]["text"])

    if bottom_elbows:
        notes.append("average elbow angle at the bottom: {:.0f}° (target ~{:.0f}° or lower)"
                     .format(float(np.mean(bottom_elbows)), ELBOW_TARGET_DEPTH))
    if bottom_bodies:
        mean_body = float(np.mean(bottom_bodies))
        direction = "sagging" if mean_body > 0 else "piked"
        notes.append("average hip position at the bottom: {:.0f}° off the "
                     "shoulder-to-heel line ({}), tolerance {:.0f}°"
                     .format(abs(mean_body), direction, BODY_LINE_LIMIT))
    if descent_durations:
        notes.append("average descent duration: {:.2f}s".format(float(np.mean(descent_durations))))

    guidance = coverage_guidance(PUSHUP_PROFILE, frontal_observed=asym_assessable)
    view_guidance = guidance if reps else None
    if view_guidance:
        notes.append(view_guidance)

    return Evaluation(
        exercise="pushup",
        side=side,
        rep_count=len(reps),
        cues_fired=list(fired.values()),
        positives=positives,
        notes=notes,
        view_guidance=view_guidance,
    )


def _fire(fired, flag, rep_i):
    """Record a cue hit, or extend one that already fired. `rep_i` of None means
    the cue is about the set as a whole rather than one rep."""
    cue = PUSHUP_CUES.get(flag)
    if cue is None or not cue.get("available", False):
        return
    if flag in fired:
        if rep_i is not None:
            fired[flag].rep_indices.append(rep_i)
        return
    fired[flag] = CueHit(
        flag=flag,
        severity=cue["severity"],
        fault=cue["fault"],
        fix=cue["fix"],
        joints=list(cue["joints"]),
        rep_indices=[] if rep_i is None else [rep_i],
    )

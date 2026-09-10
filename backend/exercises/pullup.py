"""Pull-up mechanics: thresholds, validity, scoring, fault colouring.

Third exercise on the shared machinery in `mechanics.py`, and the first one that
did not fit it. Two things differ from both the squat and the push-up, and the
first is the reason this file exists at all:

  - **The effort happens at the top.** A squat and a push-up both put peak joint
    flexion at the lowest point of the body's travel, which is exactly what
    `detect_bottoms` looks for. A pull-up puts it at the highest. Measured across
    six Penn Action sequences, peak elbow flexion landed at normalised travel
    position 0.00-0.16, where 0 is the body at its highest -- six out of six, no
    ambiguity. Left alone, the detector would have returned the dead hang between
    reps and every downstream measurement would have read the wrong frame while
    reporting a rep count that looked entirely reasonable. `Movement.effort_at_top`
    is the flag; the runner negates the travel signal when it is set.

  - **The camera goes in front, not to the side.** Median shoulder separation
    across eight sequences was 0.34 of torso length (range 0.25-0.43), and a
    genuinely side-on view collapses the shoulders nearly on top of each other.
    People film pull-ups from the front because the bar is overhead. That inverts
    the plane assignment relative to the other two exercises and is handled in
    `pullup_cues.py` rather than here.

**Thresholds below are provisional.** They come from an eight-sequence probe of
the Penn Action pull-up subset, which was enough to establish direction and rule
out the impossible, and is not enough to set a number anyone should trust. WP-08
step 2 calibrates them against the full 25 sequences. Where a value is a guess it
says so, rather than being stated with the confidence of the push-up numbers next
door.
"""
from __future__ import annotations

import numpy as np

from ..pipeline.pose import LM, VISIBILITY_THRESHOLD
from . import mechanics
from .mechanics import Movement


# Elbow angle at the top of a complete pull-up. The clips that tracked cleanly
# reached 17-32 degrees at their highest point, so a full rep closes the elbow
# well past a right angle. PROVISIONAL: set at 60 so a rep that stops appreciably
# short is caught, pending step 2.
ELBOW_TARGET_TOP = 60.0

# Above this at the highest point, the rep did not bring the chin near the bar.
# PROVISIONAL -- no partial-rep clips have been measured yet, so this is reasoned
# from the target rather than from a measured gap between good and bad reps.
HEIGHT_FLAG_ELBOW_ANGLE = 90.0

# Below this at the lowest point, the arms never straightened and the rep started
# from a partial hang. Measured at the hang across eight sequences: median 172,
# range 151-179. The one clip at 151 is the outlier and may itself be a partial.
# PROVISIONAL floor at 150.
HANG_EXTENSION_MIN = 150.0

# An elbow cannot fold flat. Two of the eight sequences produced minima of 0 and
# 1 degree, which is the pose estimate coming apart rather than a person pulling
# very hard, and the same class of artefact the push-up guards against with its
# body-line ceiling. Frames below this are excluded from scoring rather than
# scored and reported with a straight face.
MIN_PLAUSIBLE_ELBOW = 15.0

# Torso lean from vertical. A hanging body reads near zero; swing moves it. Only
# a proxy for kipping and not calibrated -- `pullup_cues.py` parks the kipping cue
# and this constant is here for the validity gate, not for a cue.
MAX_PLAUSIBLE_TRUNK_LEAN = 60.0

MIN_CUE_VISIBILITY = VISIBILITY_THRESHOLD

W_HEIGHT = 1.0

# Rep gating. Body scale for a pull-up is the TORSO, not shoulder-to-ankle: most
# people hang with the knees bent or crossed, which shortens a shoulder-ankle
# measure by an amount that has nothing to do with body size. Torso is stable
# whatever the legs do.
#
# That choice makes travel large relative to scale. Across the probe the body
# moved roughly 0.9-1.3 times its own torso length per clip, where a squat moves
# about 0.45 of leg length, so the shared default ceiling of 1.0 would have
# discarded genuine reps as implausible. Raised to 2.5, which still catches a
# subject walking out of frame.
MAX_PLAUSIBLE_TRAVEL = 2.5
TRAVEL_ABS_FLOOR = 0.30
TRAVEL_REL_FLOOR = 0.35
MIN_REP_ELBOW_FLEXION = 30.0
MIN_REP_FLEXION_RATIO = 0.4


def _elbow_key(side):
    return "elbow_left" if side == "left" else "elbow_right"


def frame_valid(angle_dict, side):
    """Is this frame usable for scoring?

    Two ways a pull-up frame goes wrong. The elbow can come back impossibly
    closed, which is a landmark scramble. Or the trunk can swing past anything a
    hanging body does, which is the tracker having lost the torso rather than a
    violent kip.
    """
    elbow = angle_dict.get(_elbow_key(side))
    if elbow is None or not np.isfinite(elbow):
        return False
    if elbow < MIN_PLAUSIBLE_ELBOW:
        return False
    trunk = angle_dict.get("trunk")
    if trunk is not None and np.isfinite(trunk) and abs(trunk) > MAX_PLAUSIBLE_TRUNK_LEAN:
        return False
    return True


def _score_frame(angle_dict, side):
    """Form deviation at a single frame. Higher is worse.

    Only height for now. Pulling higher than the target is not a fault, so the
    penalty is one-sided, the same way the push-up treats depth. Grip width and
    kipping are the other two things a pull-up is judged on and neither has a
    trustworthy measurement yet -- see the coverage audit -- so scoring on them
    would be inventing a number to fill the shape of a function.
    """
    elbow = angle_dict.get(_elbow_key(side))
    if elbow is None or not np.isfinite(elbow):
        return float("-inf"), {"height": None}

    height_pen = max(0.0, elbow - ELBOW_TARGET_TOP)
    return W_HEIGHT * height_pen, {"height": round(height_pen, 1)}


def flag_frames(angles_per_frame, reps, side, top_window=8):
    """Which joints to draw in fault colour, per frame, for the overlay.

    Same rule as the other two: colour follows the measurement so the video shows
    when form broke down, not just that it did. A rep that did not get high enough
    marks the elbow and shoulder around its top, and only around its top, because
    the hang below it was not the problem.
    """
    n = len(angles_per_frame)
    out = [set() for _ in range(n)]
    elbow_key = _elbow_key(side)

    for (s, b, e) in reps:
        highest, elbow_min = None, None
        for i in range(max(0, s), min(n, e + 1)):
            if not frame_valid(angles_per_frame[i], side):
                continue
            v = angles_per_frame[i].get(elbow_key)
            if v is None or not np.isfinite(v):
                continue
            if elbow_min is None or v < elbow_min:
                highest, elbow_min = i, v
        if highest is None or elbow_min <= HEIGHT_FLAG_ELBOW_ANGLE:
            continue
        lo = max(0, highest - top_window)
        hi = min(n - 1, highest + top_window)
        for i in range(lo, hi + 1):
            if frame_valid(angles_per_frame[i], side):
                out[i] |= {f"{side}_elbow", f"{side}_shoulder"}

    return out


# ---------------------------------------------------------------------------
# The pull-up as the generic machinery sees it
# ---------------------------------------------------------------------------

PULLUP = Movement(
    name="pullup",
    primary_angle=_elbow_key,
    side_joints=lambda side: [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"],
    # torso only, for the reason in the rep-gating note above
    scale_pairs=[("left_shoulder", "left_hip"), ("right_shoulder", "right_hip")],
    scale_metric="euclidean",
    is_valid=frame_valid,
    score_frame=_score_frame,
    # hanging from a bar, so a correctly decoded frame shows an upright body --
    # the same expectation as the squat, and the opposite of the push-up
    body_axis="vertical",
    # the whole body translates, unlike the push-up where it pivots at the toes
    use_travel_gate=True,
    max_travel=MAX_PLAUSIBLE_TRAVEL,
    travel_abs_floor=TRAVEL_ABS_FLOOR,
    travel_rel_floor=TRAVEL_REL_FLOOR,
    min_flexion=MIN_REP_ELBOW_FLEXION,
    flexion_rel_floor=MIN_REP_FLEXION_RATIO,
    effort_at_top=True,
)


def pick_side(landmarks, reps):
    return mechanics.pick_side(PULLUP, landmarks, reps)


def body_scale(landmarks):
    return mechanics.body_scale(PULLUP, landmarks)


def keep_real_reps(reps, travel_y, scale, angles_per_frame=None, side="left", fps=30.0):
    return mechanics.keep_real_reps(PULLUP, reps, travel_y, scale,
                                    angles_per_frame=angles_per_frame,
                                    side=side, fps=fps)


def deepest_frame(angles_per_frame, start, end, side):
    return mechanics.deepest_frame(PULLUP, angles_per_frame, start, end, side)


def score_reps(angles_per_frame, reps, side, fps=30.0):
    return mechanics.score_reps(PULLUP, angles_per_frame, reps, side, fps)


def worst_frame(angles_per_frame, reps, side, fps=30.0):
    return mechanics.worst_frame(PULLUP, angles_per_frame, reps, side, fps)


def best_frame(angles_per_frame, reps, side, fps=30.0):
    return mechanics.best_frame(PULLUP, angles_per_frame, reps, side, fps)


def frame_caption(angle_dict, side):
    """Measurement readout for the overlay and key frames.

    Elbow and trunk lean. Trunk is shown even though nothing cues on it yet,
    because it is the number a reader would want when the swing is what they can
    see happening in the clip.
    """
    trunk = angle_dict.get("trunk")
    lines = [
        "elbow L:{} R:{}".format(_fmt(angle_dict.get("elbow_left")),
                                 _fmt(angle_dict.get("elbow_right"))),
    ]
    if trunk is not None and np.isfinite(trunk):
        lines.append("trunk {:.0f} from vertical".format(trunk))
    else:
        lines.append("trunk --")
    return lines


def _fmt(v):
    if v is None or not np.isfinite(v):
        return "--"
    return "{:.0f}".format(v)

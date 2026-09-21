"""Push-up mechanics: thresholds, validity, scoring, fault colouring.

Structurally the same as `squat.py` and for the same reason — the shared rep
machinery in `mechanics.py` does the work, and this file supplies only what is
specific to the movement. Nothing here is copied from the squat module; where the
two need the same behaviour they call the same function.

Two things genuinely differ from the squat and are worth knowing before reading
the numbers:

  - Body scale is measured as true length, not vertical drop. A push-up body
    is horizontal, so the shoulder and ankle sit at nearly the same height and the
    squat's |dy| measure collapses to almost nothing. Shoulder-to-ankle Euclidean
    distance is the right reference.
  - Hip travel is smaller. The body pivots at the toes, so the hips move a
    fraction of what they do in a squat. Measured across the reference clips it
    runs about 0.2-0.55 of body length, against roughly 0.45 of leg length for a
    squat, which is why the gating floors are per-movement rather than shared.

Thresholds below were calibrated on six CC0 clips (`data/test_videos/pexels/
push_up_exercise/`). Two of those six could not be tracked at all — the subject
was small in frame and the body-line angle swung through the full +/-180 range,
which is the tracking flipping rather than a person moving. They are useful as
negative cases and the validity gate is set to exclude exactly that.
"""
from __future__ import annotations

import numpy as np

from ..pipeline.pose import LM, VISIBILITY_THRESHOLD
from . import mechanics
from .mechanics import Movement


# Elbow angle at the bottom of a good push-up. Chest-to-floor work reaches about
# 90 degrees or below; the deep reference clips bottom out at 56-93. Set at 100
# so a rep that stops appreciably short is flagged without punishing someone who
# simply is not going lower than parallel.
ELBOW_TARGET_DEPTH = 100.0

# Above this at the deepest point, the rep did not go down far enough to count as
# a full repetition. The shallow reference clips sit at 106-134.
DEPTH_FLAG_ELBOW_ANGLE = 120.0

# How far the hips may sit off the shoulder-to-ankle line before it stops being a
# plank. Signed: positive is sagging, negative is piked (see angles.py). Sag and
# pike get separate cues because the corrections are opposite, but the tolerance
# is the same either way. The clips that tracked cleanly stayed inside +/-40, and
# a visible sag reads well past that.
BODY_LINE_LIMIT = 25.0

# Beyond this the body-line reading is not a person doing a push-up badly, it is
# the pose estimate coming apart. Two of the six reference clips swung the full
# +/-180 because the subject was too small in frame for the hip to be located at
# all. Frames like that are excluded rather than scored.
MAX_PLAUSIBLE_BODY_DEVIATION = 60.0

# Neck angle (nose-shoulder-hip). Measured on the clips that tracked, a neutral
# head sits around 140-170 and a dropped head falls well below that. Parked as
# unavailable in the cue set for now — see pushup_cues.py for why.
NECK_NEUTRAL_MIN = 120.0

# Left/right elbow difference at the bottom that reads as asymmetric. Only
# assessable from the front, and gated on visibility like the squat's knee check.
ELBOW_ASYMMETRY_DEG = 12.0

MIN_CUE_VISIBILITY = VISIBILITY_THRESHOLD

# Score weights. Depth and body line are both real faults and neither obviously
# dominates, so they are weighted equally until there is evidence to do otherwise.
W_DEPTH = 1.0
W_BODY = 1.0

# Rep gating. Travel numbers are multiples of body length and come from the
# measured range above; the flexion pair matches the squat because "did the joint
# bend at all" is the same question whichever joint it is.
MAX_PLAUSIBLE_HIP_TRAVEL = 1.0
TRAVEL_ABS_FLOOR = 0.10
# Lower than the squat's 0.35. Push-up rep depth varies more within a set — one
# reference clip ran 0.17 to 0.56 across five reps — and 0.35 of the largest would
# have discarded a rep that was simply shallower, which the shallow-depth cue is
# there to report rather than hide.
TRAVEL_REL_FLOOR = 0.30
MIN_REP_ELBOW_FLEXION = 30.0
MIN_REP_FLEXION_RATIO = 0.4
# How far the elbow has to open again between two reps for them to be two. On four
# stock clips one push-up was counted twice, and the elbow opened 0 to 11.8 degrees
# between the halves; on the clips that analyse, separate reps opened 25.5 or more.
MIN_ELBOW_REOPEN = 15.0


def _elbow_key(side):
    return "elbow_left" if side == "left" else "elbow_right"


def _body_key(side):
    return "body_left" if side == "left" else "body_right"


def frame_valid(angle_dict, side):
    """Is this frame usable for scoring?

    Same shape as the squat's gate and the same intent: exclude frames where the
    pose estimate has produced something anatomically impossible, rather than
    scoring it and reporting the result with a straight face. Here the tell is the
    body line — a hip that appears more than 60 degrees off the shoulder-ankle
    line is not a sagging push-up, it is a landmark in the wrong place.
    """
    elbow = angle_dict.get(_elbow_key(side))
    if elbow is None or not np.isfinite(elbow):
        return False
    body = angle_dict.get(_body_key(side))
    if body is None or not np.isfinite(body):
        return False
    if abs(body) > MAX_PLAUSIBLE_BODY_DEVIATION:
        return False
    return True


def _score_frame(angle_dict, side):
    """Form deviation at a single frame. Higher is worse.

    Depth only penalises a rep that stopped short — going deeper than the target
    is not a fault. Body line penalises deviation in either direction, since sag
    and pike are both failures to hold the plank even though they need opposite
    fixes.
    """
    elbow = angle_dict.get(_elbow_key(side))
    body = angle_dict.get(_body_key(side))
    if (elbow is None or body is None
            or not np.isfinite(elbow) or not np.isfinite(body)):
        return float("-inf"), {"depth": None, "body": None}

    depth_pen = max(0.0, elbow - ELBOW_TARGET_DEPTH)
    body_pen = max(0.0, abs(body) - BODY_LINE_LIMIT)
    total = W_DEPTH * depth_pen + W_BODY * body_pen
    return total, {"depth": round(depth_pen, 1), "body": round(body_pen, 1)}


def flag_frames(angles_per_frame, reps, side, fps=30.0, landmarks=None,
                depth_window=8):
    """Which joints to draw in fault colour, per frame, for the overlay.

    Follows the squat's rule: colour tracks the measurement rather than being
    applied once to a single representative frame, so the video shows *when* form
    breaks down. Body-line faults mark the hip and shoulder because that is the
    segment that has bent; depth marks the elbow around the bottom of a rep that
    did not get low enough, and only around the bottom, because the descent itself
    was not the problem.

    Depth is marked round the frame the rep was judged at, same as the squat, so
    the worst key frame can't fall outside its own red window. `landmarks` is
    accepted and unused, to keep one signature across the three exercises
    """
    n = len(angles_per_frame)
    out = [set() for _ in range(n)]
    elbow_key = _elbow_key(side)
    body_key = _body_key(side)

    for i, ang in enumerate(angles_per_frame):
        if not frame_valid(ang, side):
            continue
        body = ang.get(body_key)
        if body is not None and np.isfinite(body) and abs(body) > BODY_LINE_LIMIT:
            out[i] |= {"left_hip", "right_hip", "left_shoulder", "right_shoulder"}

    for (s, b, e) in reps:
        lo, hi = mechanics.eval_window(PUSHUP, s, b, e, fps)
        f = mechanics.deepest_frame(PUSHUP, angles_per_frame, lo, hi, side)
        if f is None:
            continue
        v = angles_per_frame[f].get(elbow_key)
        if v is None or not np.isfinite(v) or v <= DEPTH_FLAG_ELBOW_ANGLE:
            continue
        for i in range(max(0, f - depth_window), min(n - 1, f + depth_window) + 1):
            if frame_valid(angles_per_frame[i], side):
                out[i] |= {f"{side}_elbow", f"{side}_shoulder"}

    return out


# ---------------------------------------------------------------------------
# The push-up as the generic machinery sees it
# ---------------------------------------------------------------------------

PUSHUP = Movement(
    name="pushup",
    primary_angle=_elbow_key,
    side_joints=lambda side: [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"],
    # shoulder to ankle, measured as true distance: the body is horizontal, so a
    # vertical measure would read close to zero
    scale_pairs=[("left_shoulder", "left_ankle"), ("right_shoulder", "right_ankle")],
    scale_metric="euclidean",
    is_valid=frame_valid,
    score_frame=_score_frame,
    # done on the floor, so a correctly decoded frame shows the body lying down.
    # The rotation guard reads this the other way round from the squat: a
    # push-up whose axis comes back near-vertical is a sideways-decoded file.
    body_axis="horizontal",
    use_travel_gate=False,
    min_flexion=MIN_REP_ELBOW_FLEXION,
    flexion_rel_floor=MIN_REP_FLEXION_RATIO,
    min_reopen=MIN_ELBOW_REOPEN,
)


def pick_side(landmarks, reps):
    return mechanics.pick_side(PUSHUP, landmarks, reps)


def body_scale(landmarks):
    return mechanics.body_scale(PUSHUP, landmarks)


def keep_real_reps(reps, hip_y, scale, angles_per_frame=None, side="left", fps=30.0):
    return mechanics.keep_real_reps(PUSHUP, reps, hip_y, scale,
                                    angles_per_frame=angles_per_frame,
                                    side=side, fps=fps)


def deepest_frame(angles_per_frame, start, end, side):
    return mechanics.deepest_frame(PUSHUP, angles_per_frame, start, end, side)


def score_reps(angles_per_frame, reps, side, fps=30.0):
    return mechanics.score_reps(PUSHUP, angles_per_frame, reps, side, fps)


def worst_frame(angles_per_frame, reps, side, fps=30.0):
    return mechanics.worst_frame(PUSHUP, angles_per_frame, reps, side, fps)


def best_frame(angles_per_frame, reps, side, fps=30.0):
    return mechanics.best_frame(PUSHUP, angles_per_frame, reps, side, fps)


def frame_caption(angle_dict, side):
    """Measurement readout for the overlay and key frames — elbow and body line,
    which are the two things a push-up is judged on."""
    body = angle_dict.get(_body_key(side))
    lines = [
        "elbow L:{} R:{}".format(_fmt(angle_dict.get("elbow_left")),
                                 _fmt(angle_dict.get("elbow_right"))),
    ]
    if body is not None and np.isfinite(body):
        direction = "sag" if body > 0 else "pike"
        lines.append("body {:+.0f} ({}, cap {:.0f})".format(body, direction, BODY_LINE_LIMIT))
    else:
        lines.append("body --")
    return lines


def _fmt(v):
    if v is None or not np.isfinite(v):
        return "--"
    return "{:.0f}".format(v)

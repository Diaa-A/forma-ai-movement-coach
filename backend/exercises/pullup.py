"""Pull-up mechanics: thresholds, validity, scoring, fault colouring.

Built on `mechanics.py` like the squat and the push-up, with two differences that
shape the rest of the file:

  - **The effort is at the top.** A squat and a push-up flex hardest at the lowest
    point of the body's travel, which is what `detect_bottoms` looks for. A
    pull-up flexes hardest at the highest point: on 6 of 6 Penn Action sequences
    peak elbow flexion fell at normalised travel 0.00-0.16, with 0 the top.
    Without `Movement.effort_at_top` the detector returns the dead hang between
    reps and the rep count still looks right.

  - **It is filmed from the front.** Median shoulder separation across eight
    sequences was 0.34 of torso length; side-on, one shoulder sits almost on top
    of the other. The camera goes in front because the bar is overhead.

Calibrated on all 25 Penn Action sequences, tracked the way the app tracks a
video. The rep gate's floor and ceiling fit the measured travel, but height and
extension thresholds could not be set: Penn Action labels the action, not how
well it was done, so there is no incomplete rep to place a boundary against.
`pullup_cues.py` parks those cues and declares them.
"""
from __future__ import annotations

import numpy as np

from . import mechanics
from .mechanics import Movement


# Orders the reps in a clip so worst_frame and best_frame can pick one. Not a form
# threshold, and no cue reads it. 40 is about the 75th percentile of the elbow at
# the top across the calibration set (23 reps, p75 40.9), so most reps score zero
# and the ones that came up short sort to the end.
ELBOW_TARGET_TOP = 40.0

# Past this the tracker has lost the torso; a hanging body sits near zero. Not a
# kipping threshold -- that cue is parked.
MAX_PLAUSIBLE_TRUNK_LEAN = 60.0

# Rep gating. Scale is the torso, not shoulder-to-ankle, because most people hang
# with bent or crossed knees. Against the torso a rep travels 0.4 to 1.3 (median
# 0.8), so the shared ceiling of 1.0 would drop real reps; 2.5 still catches
# someone walking out of frame.
#
# The floors are low because a set that does not come all the way back down between
# reps travels little, and those are still reps: on a set filmed to order whose chin
# cleared the bar every time but which stopped short on the way down, floors of
# 0.30 and 0.35 counted one rep of five. Across 25 clips counted by eye they cost
# four more counts and gained none. What keeps a hang or a reach for the bar out is
# the flexion floor below, not travel.
MAX_PLAUSIBLE_TRAVEL = 2.5
TRAVEL_ABS_FLOOR = 0.15
TRAVEL_REL_FLOOR = 0.20
MIN_REP_ELBOW_FLEXION = 30.0
MIN_REP_FLEXION_RATIO = 0.4

# What every scoring function here reads: both arms combined, see _both_elbows in
# angles.py. The side argument they take is part of the shared signature and does
# not change the result. From the front the arms are about equally visible, so
# choosing one by visibility is close to arbitrary.
ELBOW_KEY = "elbow"


def frame_valid(angle_dict, side):
    """Is this frame usable for scoring?

    Not when there is no elbow reading, or when the trunk leans further than a
    hanging body does, which means the tracker has lost the torso. A reading near
    0 counts: at the top of a pull the forearm folds onto the upper arm in the
    image.
    """
    elbow = angle_dict.get(ELBOW_KEY)
    if elbow is None or not np.isfinite(elbow):
        return False
    trunk = angle_dict.get("trunk")
    if (trunk is not None and np.isfinite(trunk)
            and abs(trunk) > MAX_PLAUSIBLE_TRUNK_LEAN):
        return False
    return True


def _score_frame(angle_dict, side):
    """Form deviation at a single frame. Higher is worse.

    Only used to rank the reps for the key frames. One-sided, like the push-up's
    depth penalty: pulling higher than the reference costs nothing.
    """
    elbow = angle_dict.get(ELBOW_KEY)
    if elbow is None or not np.isfinite(elbow):
        return float("-inf"), {"height": None}

    height_pen = max(0.0, elbow - ELBOW_TARGET_TOP)
    return height_pen, {"height": round(height_pen, 1)}


def flag_frames(angles_per_frame, reps, side):
    """Which joints to draw in fault colour, per frame. None, for now.

    A red joint says something went wrong there, and with no calibrated height or
    extension threshold there is nothing to back that up. Same return shape as the
    squat and push-up versions, so the renderer needs no special case.
    """
    return [set() for _ in angles_per_frame]


# ---------------------------------------------------------------------------
# The pull-up as the generic machinery sees it
# ---------------------------------------------------------------------------

PULLUP = Movement(
    name="pullup",
    primary_angle=lambda side: ELBOW_KEY,
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
    # a second rep needs the body to come back down at least as far as one rep has
    # to travel, or a hold at the top counts twice
    min_return=TRAVEL_ABS_FLOOR,
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
    """Measurement readout for the overlay and key frames: both elbows and the
    trunk lean. Nothing cues on the trunk, but it is the number to look at when
    the clip shows the body swinging."""
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

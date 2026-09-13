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

**Calibrated against all 25 Penn Action sequences on 10 September**, 26 scored
repetitions. The rep-gating numbers survived. The form thresholds did not, and
the reason is worth stating precisely: it is not that the pipeline cannot measure
a pull-up. Overall elbow error is 9.1 degrees median, comparable to the push-up's
8.1, and in the deep-flexion band where a height cue would actually read it drops
to 5.5. The problem is that Penn Action labels *what the action is*, not *whether
it was done well*, so there are no incomplete repetitions to calibrate a boundary
against. A threshold set here would separate nothing that has been shown to need
separating.

So no cue fires on form. The exercise counts repetitions, declares what it cannot
assess, and says so — see `pullup_cues.py` and the audit in the engineering notes.
"""
from __future__ import annotations

import numpy as np

from ..pipeline.angles import DEGENERATE_ELBOW_DEG
from ..pipeline.pose import LM, VISIBILITY_THRESHOLD
from . import mechanics
from .mechanics import Movement


# Reference point for ORDERING repetitions within a clip, not a form threshold.
# The distinction matters and is the whole outcome of step 2.
#
# A score has to rank reps so worst_frame and best_frame can pick one, and
# ranking only needs the ordering to be right. A cue asserts that something was
# wrong, and that needs a boundary somebody can defend. The first is available
# here; the second is not.
#
# 40 is the 75th percentile of what the calibration set actually reached at the
# top on the two-arm elbow: n=26, min 2.6, p25 13.6, median 24.7, p75 39.8, max
# 86.3. Most reps therefore score zero and the ones that came up short of the
# field sort to the end, which is all the ordering needs.
ELBOW_TARGET_TOP = 40.0

# Deliberately absent: a height-flag threshold and a hang-extension threshold.
#
# Both are measurable. Elbow error against Penn Action ground truth is 5.5 deg in
# the deep-flexion band and 6.2 at full extension -- the two bands a height cue and
# an extension cue would read. Overall the pull-up's elbow median is 9.1 deg,
# against the push-up's 8.1, so the two exercises are measured about as well as
# each other and the bands that matter here happen to be the accurate ones. What
# is missing is any labelled example of the fault.
# Penn Action says an action is a pull-up; it does not say whether it
# was a good one. Across 26 scored reps the elbow at the top ran 2.6 to 86.3 with
# no marked boundary anywhere in it, and a number chosen from that range would
# separate a population from itself.
#
# So the height and extension cues are declared and parked, on the same grounds as
# push-up elbow flare and lockout: measurable in principle, not yet calibrated
# against anything that would show the threshold was right.

# Only catches a degenerate reading, and deliberately nothing more.
#
# This started at 15 degrees on the reasoning that an elbow cannot fold flat, so
# the 0 and 1 degree minima seen in the probe had to be the pose estimate coming
# apart. That reasoning was wrong and the number was doing real damage: on
# sequence 1172 the top of the pull reads 8 degrees, the floor rejected that
# frame, and with the effort frame gone the flexion gate saw a straight arm and
# threw the entire repetition away. Three of 25 clips scored nothing for this
# reason.
#
# Checked against Penn Action ground truth rather than argued about. Elbow error
# by true angle, 2709 measurements:
#
#     true 0-30      n=206   median error  -2.1    abs 5.5
#     true 30-60     n=475                 +4.5    abs 6.7
#     true 90-120    n=442                +12.1    abs 15.0
#     true 150-181   n=625                 +3.1    abs 6.2
#
# Deep flexion is the most accurate band of the six, not the least, and ground
# truth itself holds 206 readings below 30 degrees. These are projected 2D
# angles: at the top of a pull-up filmed from the front the upper arm and the
# forearm overlap in projection, so a small angle there is a correct measurement
# of what the camera can see rather than a failure to track.
MIN_PLAUSIBLE_ELBOW = DEGENERATE_ELBOW_DEG

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


# The key every scoring function here reads: both arms, averaged where both are
# tracked (see _both_elbows in angles.py). The side argument these functions
# still take is kept for the shared signature and deliberately changes nothing.
# Under a front-on camera the two arms are about equally visible, and on
# sequence 1173 a four-point visibility gap between them was the difference
# between telling the user they did 2 reps and telling them they did 1.
ELBOW_KEY = "elbow"


def frame_valid(angle_dict, side):
    """Is this frame usable for scoring?

    Two ways a pull-up frame goes wrong. The elbow can come back degenerate,
    which is coincident landmarks rather than a hard pull -- MIN_PLAUSIBLE_ELBOW
    explains why that floor sits so low. Or the trunk can swing past anything a
    hanging body does, which is the tracker having lost the torso rather than a
    violent kip.
    """
    elbow = angle_dict.get(ELBOW_KEY)
    if elbow is None or not np.isfinite(elbow):
        return False
    if elbow < MIN_PLAUSIBLE_ELBOW:
        return False
    trunk = angle_dict.get("trunk")
    if (trunk is not None and np.isfinite(trunk)
            and abs(trunk) > MAX_PLAUSIBLE_TRUNK_LEAN):
        return False
    return True


def _score_frame(angle_dict, side):
    """Form deviation at a single frame. Higher is worse.

    Used to ORDER the reps in a clip so a key frame can be chosen, and for
    nothing else -- no cue reads it. Pulling higher than the reference is not a
    fault, so the penalty is one-sided, the same way the push-up treats depth.
    """
    elbow = angle_dict.get(ELBOW_KEY)
    if elbow is None or not np.isfinite(elbow):
        return float("-inf"), {"height": None}

    height_pen = max(0.0, elbow - ELBOW_TARGET_TOP)
    return W_HEIGHT * height_pen, {"height": round(height_pen, 1)}


def flag_frames(angles_per_frame, reps, side, top_window=8):
    """Which joints to draw in fault colour. For now, none of them.

    A red joint on the overlay is an assertion that something was wrong there,
    and §15.7 already established that an unfounded claim is worse than silence:
    silence is ambiguous, but a fault marking reads as a diagnosis. With no
    calibrated threshold there is nothing to assert, so the skeleton draws in the
    neutral colour throughout and the report says what was not assessed instead.

    The signature and the per-frame shape are kept so the renderer needs no
    special case, and so this becomes a real implementation the moment a cue has
    a threshold behind it.
    """
    return [set() for _ in range(len(angles_per_frame))]


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

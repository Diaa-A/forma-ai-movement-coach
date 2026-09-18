"""Velocity zero-crossing phase detection.

For a squat (and push-up) the bottom of the rep is where vertical velocity flips
from descending to ascending. This is exercise-agnostic: works for any user
height/mobility/camera placement because we never assume a specific angle value.

IMPORTANT y-axis convention: in MediaPipe image-normalised coordinates, y INCREASES
going down. So when the hip is descending (body sinking), hip_y is INCREASING and
velocity is POSITIVE. We negate before differentiating so the rest of the code
reads naturally — descending = negative velocity, ascending = positive — matching
the snippet in the technical spec.
"""
import numpy as np


def detect_bottoms(hip_y_smoothed, min_separation=5):
    """Return indices of rep bottoms (descent->ascent transitions).

    `min_separation` filters out tiny back-to-back zero crossings caused by
    residual jitter — bottoms closer than this many frames are merged.
    """
    y = np.asarray(hip_y_smoothed, dtype=np.float64)

    # negate so descent (y increasing in image space) = negative velocity
    velocity = np.gradient(-y)

    # locate sign changes in velocity
    sign = np.sign(velocity)
    # treat zeros as positive to avoid spurious crossings
    sign[sign == 0] = 1.0
    sign_changes = np.where(np.diff(sign) != 0)[0]

    # we want neg -> pos transitions = the bottom of the movement
    bottoms = [int(i) for i in sign_changes if velocity[i] < 0]

    # merge anything closer than min_separation
    merged = []
    for b in bottoms:
        if merged and (b - merged[-1]) < min_separation:
            continue
        merged.append(b)
    return merged


def segment_reps(bottoms, n_frames):
    """Wrap each bottom in a (descent_start, bottom, ascent_end) tuple.

    Boundaries are half-way to the neighbouring bottom on each side. The first and
    last rep have a neighbour on one side only, and running them out to the edges of
    the video puts the approach and the walk back to the phone inside a rep: a real
    pull-up set lost its fifth rep that way, because dropping off the bar afterwards
    measured 4.2 body lengths of travel and failed the plausibility ceiling. With one
    neighbour, the gap to it is mirrored, so the end rep gets the same width as the
    rhythm around it. With no neighbour at all there is nothing to mirror and the
    clip's edges are all there is.
    """
    reps = []
    for i, b in enumerate(bottoms):
        prev_b = bottoms[i - 1] if i > 0 else None
        next_b = bottoms[i + 1] if i + 1 < len(bottoms) else None
        if prev_b is None and next_b is not None:
            prev_b = b - (next_b - b)
        if next_b is None and prev_b is not None:
            next_b = b + (b - prev_b)
        start = (prev_b + b) // 2 if prev_b is not None else 0
        end = (b + next_b) // 2 if next_b is not None else n_frames - 1
        reps.append((int(max(0, start)), int(b), int(min(n_frames - 1, end))))
    return reps


# phase tags used in the JSON + colour-coding
PHASE_STANDING = "standing"
PHASE_DESCENT  = "descent"
PHASE_BOTTOM   = "bottom"
PHASE_ASCENT   = "ascent"

# how wide the "bottom" plateau is around the velocity zero-crossing (in frames)
_BOTTOM_HALF_WINDOW_FRAMES = 4


def label_phases(reps, n_frames, bottom_window=_BOTTOM_HALF_WINDOW_FRAMES):
    """Assign a phase tag to every frame in the video.

    Rules:
        - descent  = inside a rep, before the bottom plateau
        - bottom   = within +/- `bottom_window` frames of the rep's bottom index
        - ascent   = inside a rep, after the bottom plateau
        - standing = outside any rep
    """
    phases = [PHASE_STANDING] * n_frames
    for (s, b, e) in reps:
        bot_lo = max(s, b - bottom_window)
        bot_hi = min(e, b + bottom_window)
        for i in range(s, bot_lo):
            phases[i] = PHASE_DESCENT
        for i in range(bot_lo, bot_hi + 1):
            phases[i] = PHASE_BOTTOM
        for i in range(bot_hi + 1, e + 1):
            phases[i] = PHASE_ASCENT
    return phases

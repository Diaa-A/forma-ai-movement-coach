"""Squat-specific analysis — phase boundaries, rep validity, form scoring.

Phase B implementation . Form scoring follows these biomechanical standards:
    - Knee at bottom should reach ~90° (parallel). Deeper is fine if mobility
      allows — only "not deep enough" is a fault.
    - Spine should stay >45° from horizontal, i.e. forward lean from vertical
      should stay UNDER 45°. More lean = more spinal load.

Phase C will swap the score-then-rank approach below for the deterministic
safety layer (a structured cue database mapping deviations to expert-vetted
coaching cues). For now this gives Phase B usable worst/best frame selection.
"""
import numpy as np
from ..pipeline.pose import LM, VISIBILITY_THRESHOLD
from . import mechanics
from .mechanics import Movement


# --- biomechanical standards---------------------------------------------------

# parallel (knee 90°) is the target depth. Going deeper is acceptable, only
# being SHALLOWER than this counts against the score.
KNEE_TARGET_DEPTH = 100.0       # degrees of knee flexion at the bottom

# Forward lean is judged RELATIVEL Y, not against a absolute angle. In a balanced
# squat the trunk stays roughly parallel to the shin; a back-dominant ("good
# morning") fault is the trunk leaning notably MORE than the shin. We flag when
# trunk-lean exceeds shin-lean by more than this margin. This is far more robust
# than an absolute "trunk < 45° from vertical" rule, which wrongly penalised deep
# squats (where large but balanced forward lean is normal) and was camera-angle
# sensitive; the relative measure foreshortens with the camera on both segments.
# Margin calibrated on test clips: balanced/upright squats showed <=6° excess,
# clearly back-dominant ones 15-28°.
LEAN_EXCESS_LIMIT = 15.0        # degrees the trunk may lead the shin before flagged

# A cue that depends on a joint below this visibility is SUPPRESSED rather than
# fired on noisy coordinates — e.g. the occluded far-side leg in a side-on view,
# which would otherwise make the left/right comparison meaningless. Single source
# of truth lives in pose.py (data-driven: occluded far legs 0.47-0.55, visible
# 0.95+, so the cut sits in the gap).
MIN_CUE_VISIBILITY = VISIBILITY_THRESHOLD

# Score component weights — kept simple and tweakable. Phase C cue database will
# absorb this anyway.
W_DEPTH = 1.0
W_SPINE = 1.0

# Per-frame validity gate. MediaPipe occasionally scrambles the torso landmarks
# (left/right hip or shoulder swap during fast motion / occlusion), which makes
# the spine angle read far past anything anatomically possible. A controlled
# squat never leans past ~70-75° from vertical, so anything beyond this ceiling
# is a tracking glitch — exclude such frames rather than scoring garbage.
MAX_PLAUSIBLE_LEAN = 85.0

# Ceiling on how far the hips may travel in a single rep, as a multiple of leg
# length. A deep squat moves the hips roughly 0.4-0.6 of a leg length; anything
# approaching a whole leg length is not a squat. Set well clear of real movement
# at 1.0 so it only ever catches the tracking falling apart — a subject walking
# out of or back into frame, or a landmark snapping across the image.
MAX_PLAUSIBLE_HIP_TRAVEL = 1.0

# Knee angle at the deepest point above which a rep counts as shallow. Lives here
# rather than in squat_cues so the overlay can use it without importing the cue
# database (squat_cues already imports from this module, so the arrow only points
# one way).
DEPTH_FLAG_KNEE_ANGLE = 110.0

# Left and right knee more than this apart at the bottom reads as a weight shift
# to one side. Here for the same reason as the depth line -- the overlay colours
# it too, and can't import squat_cues
KNEE_ASYMMETRY_DEG = 10.0

# The knee has to bend for a dip to count as a rep at all. Distinct from
# DEPTH_FLAG_KNEE_ANGLE, and the distinction matters: 110 is "was the squat deep
# enough", this is "did a squat happen".
#
# Absolute floor: 30 degrees of bend off straight. Loose on purpose — it only
# catches someone standing still.
MIN_REP_KNEE_FLEXION = 30.0
# Relative floor: a rep must bend at least this fraction of the deepest bend in
# the same clip. Same self-calibrating idea as the hip-travel test, and it is the
# check that does the real work. On a 4K upload the first detected "rep" was the
# subject settling into position — knee reached 137.8 degrees, about 42 of bend,
# while the four genuine reps reached 34-48 degrees, about 132-146 of bend. An
# absolute cut sitting between those is a number fitted to one clip; a ratio is
# not, and it still lets a set of uniformly shallow squats through to be counted
# and then flagged shallow, which is the whole reason the rep test is by movement
# rather than by depth.
MIN_REP_FLEXION_RATIO = 0.4


def frame_valid(angle_dict, side):
    """True if a frame is usable for scoring: the chosen-side knee is present and
    the spine angle is anatomically plausible (not a landmark-scramble glitch) ."""
    key = _knee_key(side) if side in ("left", "right") else "knee_left"
    k = angle_dict.get(key)
    sp = angle_dict.get("spine")
    if k is None or not np.isfinite(k):
        return False
    if sp is None or not np.isfinite(sp) or sp > MAX_PLAUSIBLE_LEAN:
        return False
    return True


def _knee_key(side):
    return "knee_left" if side == "left" else "knee_right"


def joints_visible(landmarks, frame, names, thresh=MIN_CUE_VISIBILITY):
    """True if every named landmark at `frame` has visibility >= thresh. When no
    landmark array is supplied, returns True (visibility gating disabled). Used to
    suppress cues that would otherwise fire on unreliable (e.g. occluded) joints —
    the confidence-weighting principle applied to the deterministic cue layer."""
    if landmarks is None:
        return True
    try:
        return all(landmarks[frame, LM[n], 3] >= thresh for n in names)
    except (IndexError, KeyError):
        return True


def flag_frames(angles_per_frame, reps, side, fps=30.0, landmarks=None,
                depth_window=8):
    """Which joints to draw in fault colour, PER FRAME, for the overlay video.

    Previously only the single worst frame was flagged, which meant one frame in
    six hundred was ever red — a 30th of a second, invisible at playback speed,
    so the overlay looked uniformly green no matter how the set went. The colour
    is supposed to show *when* form breaks down, so it has to track the
    measurement frame by frame.

    Three faults, and only where they are actually measurable:
      - forward lean, per frame: the torso reddens exactly while the trunk leads
        the shin by more than LEAN_EXCESS_LIMIT.
      - shallow depth, per rep: knee and hip, around the frame the rep was judged
        at. Not the whole rep -- the descent itself is fine.
      - left/right knee asymmetry, per rep: both knees, same frame, and only when
        the far leg is visible enough to compare. Same gate as the cue

    "The frame the rep was judged at" is the one the score and the report both
    use, the deepest valid frame in the window round the bottom. Searching the
    whole rep let the red drift off it: on a front-on set with some sway the worst
    key frame landed two frames outside its own red window and came out green,
    while the report named two faults on that same rep

    Frames that fail the validity gate are left unflagged rather than guessed at,
    and the renderer independently fades anything below the visibility threshold,
    so a joint MediaPipe is unsure about can never be shown as confidently bad."""
    n = len(angles_per_frame)
    out = [set() for _ in range(n)]
    shin_key = "shin_left" if side == "left" else "shin_right"
    knee_key = _knee_key(side)
    other = "right" if side == "left" else "left"
    other_knee = _knee_key(other)

    for i, ang in enumerate(angles_per_frame):
        if not frame_valid(ang, side):
            continue
        spine = ang.get("spine")
        shin = ang.get(shin_key)
        if (spine is not None and shin is not None
                and np.isfinite(spine) and np.isfinite(shin)
                and (spine - shin) > LEAN_EXCESS_LIMIT):
            out[i] |= {"left_shoulder", "right_shoulder", "left_hip", "right_hip"}

    for (s, b, e) in reps:
        lo, hi = mechanics.eval_window(SQUAT, s, b, e, fps)
        f = mechanics.deepest_frame(SQUAT, angles_per_frame, lo, hi, side)
        if f is None:
            continue
        knee = angles_per_frame[f].get(knee_key)
        if knee is None or not np.isfinite(knee):
            continue

        joints = set()
        if knee > DEPTH_FLAG_KNEE_ANGLE:
            joints |= {f"{side}_knee", f"{side}_hip"}
        k_other = angles_per_frame[f].get(other_knee)
        if (k_other is not None and np.isfinite(k_other)
                and abs(knee - k_other) > KNEE_ASYMMETRY_DEG
                and joints_visible(landmarks, f, [f"{other}_hip", f"{other}_knee",
                                                  f"{other}_ankle"])):
            joints |= {"left_knee", "right_knee"}
        if not joints:
            continue

        for i in range(max(0, f - depth_window), min(n - 1, f + depth_window) + 1):
            if frame_valid(angles_per_frame[i], side):
                out[i] |= joints

    return out


def _shin_key(side):
    return "shin_left" if side == "left" else "shin_right"


def _score_frame(angle_dict, side):
    """Combined form-deviation score at a single frame. Higher = worse.

    Components:
        depth_penalty: only fires when the knee is SHALLOWER than the target
                       (deep squats are not penalised).
        lean_penalty:  fires when the trunk leans more than the shin by over
                       LEAN_EXCESS_LIMIT (the trunk/shin parallelism check).

    Returns (total_score, breakdown_dict). NaN inputs propagate as -inf so this
    frame won't be picked as worst or best."""
    knee = angle_dict.get(_knee_key(side))
    spine = angle_dict.get("spine")
    shin = angle_dict.get(_shin_key(side))
    if (knee is None or spine is None or shin is None or
            not np.isfinite(knee) or not np.isfinite(spine) or not np.isfinite(shin)):
        return float("-inf"), {"depth": None, "lean": None}

    depth_pen = max(0.0, knee - KNEE_TARGET_DEPTH)          # too shallow only
    lean_excess = spine - shin                            # trunk over shin
    lean_pen = max(0.0, lean_excess - LEAN_EXCESS_LIMIT)  # back-dominant only
    total = W_DEPTH * depth_pen + W_SPINE * lean_pen
    return total, {"depth": round(depth_pen, 1), "lean": round(lean_pen, 1)}


# ---------------------------------------------------------------------------
# The squat as the generic machinery sees it
# ---------------------------------------------------------------------------
# Everything above is squat knowledge: which angles matter, where the thresholds
# sit, which joints to colour when a fault fires. Everything below hands that to
# mechanics.py, which does the rep-finding and scoring without knowing what a
# squat is.

SQUAT = Movement(
    name="squat",
    primary_angle=_knee_key,
    side_joints=lambda side: [f"{side}_hip", f"{side}_knee", f"{side}_ankle"],
    # hip to ankle, measured vertically: for a standing body that is leg length,
    # and squat depth scales with it
    scale_pairs=[("left_hip", "left_ankle"), ("right_hip", "right_ankle")],
    scale_metric="vertical",
    is_valid=frame_valid,
    score_frame=_score_frame,
    max_travel=MAX_PLAUSIBLE_HIP_TRAVEL,
    travel_abs_floor=0.10,
    travel_rel_floor=0.35,
    min_flexion=MIN_REP_KNEE_FLEXION,
    flexion_rel_floor=MIN_REP_FLEXION_RATIO,
)


# Thin pass-throughs so callers and tests keep the names they already use. Worth
# the four lines: the alternative is churning every call site to prove a
# refactor, which is how refactors acquire bugs that have nothing to do with the
# refactor.

def pick_side(landmarks, reps):
    return mechanics.pick_side(SQUAT, landmarks, reps)


def body_scale(landmarks):
    return mechanics.body_scale(SQUAT, landmarks)


def keep_real_reps(reps, hip_y, scale, angles_per_frame=None, side="left", fps=30.0):
    return mechanics.keep_real_reps(SQUAT, reps, hip_y, scale,
                                    angles_per_frame=angles_per_frame,
                                    side=side, fps=fps)


def deepest_frame(angles_per_frame, start, end, side):
    return mechanics.deepest_frame(SQUAT, angles_per_frame, start, end, side)


def score_reps(angles_per_frame, reps, side, fps=30.0):
    return mechanics.score_reps(SQUAT, angles_per_frame, reps, side, fps)


def worst_frame(angles_per_frame, reps, side, fps=30.0):
    return mechanics.worst_frame(SQUAT, angles_per_frame, reps, side, fps)


def best_frame(angles_per_frame, reps, side, fps=30.0):
    return mechanics.best_frame(SQUAT, angles_per_frame, reps, side, fps)


def frame_caption(angle_dict, side):
    """The measurement readout burned into the overlay and the key frames.

    Lives with the exercise because it names the exercise's joints. The renderer
    used to build this string itself, which meant a push-up frame was captioned
    "knee L:-- R:-- shin --" — every field empty, because none of them applied.
    """
    trunk = angle_dict.get("spine")
    shin = angle_dict.get(_shin_key(side))
    lines = [
        "knee L:{} R:{}".format(_fmt(angle_dict.get("knee_left")),
                                _fmt(angle_dict.get("knee_right"))),
        "trunk {}  shin {}".format(_fmt(trunk), _fmt(shin)),
    ]
    if trunk is not None and shin is not None and np.isfinite(trunk) and np.isfinite(shin):
        lines.append("lean {:+.0f} (cap +{:.0f})".format(trunk - shin, LEAN_EXCESS_LIMIT))
    return lines


def _fmt(v):
    if v is None or not np.isfinite(v):
        return "--"
    return "{:.0f}".format(v)

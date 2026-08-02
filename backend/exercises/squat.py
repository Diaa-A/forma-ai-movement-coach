"""Squat-specific analysis — phase boundaries, rep validity, form scoring.

Phase B implementation . Form scoring follows the biomechanical standards in the
spec (§5 Phase B):
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


def pick_side(landmarks, reps):
    """Decide whether to evaluate the left or right side based on mean visibility
    of the lower-body landmarks across the analysed reps. Returns 'left' or 'right'."""
    if not reps:
        # fall back to whole-video mean
        spans = [(0, landmarks.shape[0])]
    else:
        spans = [(s, e) for (s, _, e) in reps]

    def vis_of(joint_names, span):
        s, e = span
        idxs = [LM[n] for n in joint_names]
        sub = landmarks[s:e, idxs, 3]   # visibility channel
        return float(np.nanmean(sub)) if sub.size else 0.0

    left_score = np.mean([vis_of(["left_hip", "left_knee", "left_ankle"], sp) for sp in spans])
    right_score = np.mean([vis_of(["right_hip", "right_knee", "right_ankle"], sp) for sp in spans])
    return "left" if left_score >= right_score else "right"


def _knee_key(side):
    return "knee_left" if side == "left" else "knee_right"


def body_scale(landmarks):
    """Body-size reference in normalised units: median standing hip->ankle length
    (the more-visible leg). Squat depth scales with leg length, so hip travel is
    judged relative to this — making the rep test invariant to how large the
    person appears in frame (camera distance, zoom, resolution)."""
    def leg_len(hip_name, ank_name):
        d = np.abs(landmarks[:, LM[ank_name], 1] - landmarks[:, LM[hip_name], 1])
        d = d[np.isfinite(d)]
        return float(np.median(d)) if d.size else 0.0
    return max(leg_len("left_hip", "left_ankle"),
              leg_len("right_hip", "right_ankle"))


def keep_real_reps(reps, hip_y, scale, abs_floor=0.10, rel_floor=0.35,
                   max_travel=MAX_PLAUSIBLE_HIP_TRAVEL,
                   angles_per_frame=None, side="left", fps=30.0):
    """Decide which detected bottoms are genuine reps by HIP TRAVEL, not knee
    angle. Three checks:
      - ceiling: travel <= max_travel * leg length — anything above that is not a
        squat, it is the tracking coming apart (see below).
      - absolute: travel >= abs_floor * leg length — rejects clips that are all
        jitter (no real movement at all).
      - relative: travel >= rel_floor * the largest PLAUSIBLE travel in THIS clip
        — self-calibrates per clip, so real reps survive and noise zero-crossings
        drop out without any per-upload tuning.
    Keeping the travel test (rather than a knee-angle cutoff) means genuinely
    SHALLOW squats survive and can fire the shallow-depth cue, instead of being
    silently discarded as 'not a rep'.

    The ceiling has to be applied BEFORE the relative floor calibrates, and that
    ordering is the whole point of it. On a real upload the subject walked back
    toward the camera at the end of the clip; the hips travelled 2.9x leg length
    in that segment, which is not a movement a human makes. That one glitch
    became `biggest`, every genuine rep measured about 0.15 against it, and all
    six real reps were binned as noise while the artefact was kept and reported
    as the only rep. Self-calibration is only safe once obvious nonsense is out
    of the sample it calibrates against."""
    travels = []
    for (s, b, e) in reps:
        w = hip_y[s:e + 1]
        w = w[np.isfinite(w)]
        travels.append(float(w.max() - w.min()) if w.size else 0.0)
    if not travels:
        return []

    def implausible(t):
        return scale > 0 and (t / scale) > max_travel

    plausible = [t for t in travels if not implausible(t)]
    # if everything looks impossible the clip is a mess anyway — fall back to the
    # old behaviour rather than dividing by nothing, and let the abs_floor and
    # the detection-rate status machinery have the final word
    biggest = max(plausible) if plausible else max(travels)

    passed_travel = []
    for (s, b, e), t in zip(reps, travels):
        if implausible(t):
            continue
        norm = t / scale   if scale   > 0 else 0.0
        rel  = t / biggest if biggest > 0 else 0.0
        if norm >= abs_floor and rel >= rel_floor:
            passed_travel.append((s, b, e))

    return _keep_reps_that_bent(passed_travel, angles_per_frame, side, fps)


def _keep_reps_that_bent(reps, angles_per_frame, side, fps):
    """Drop dips where the knee never really bent.

    Hip travel says something moved, not what. On a real upload the subject
    settled into position before starting: the hips dropped enough to clear the
    travel floor while the knee only reached 137.8 degrees — standing, not
    squatting. It counted as a rep, scored least-bad among the set, and so became
    the 'worst form' key frame. The headline image was a photo of someone standing
    still before their first rep.

    Judged the same way hip travel is: an absolute floor to catch no-movement, and
    a ratio against the deepest bend in this clip to catch a movement that is real
    but nothing like the others. Skipped when no angles are supplied, so the
    travel-only behaviour stays available on its own.
    """
    if angles_per_frame is None or not reps:
        return list(reps)

    flexions = []
    for (s, b, e) in reps:
        lo, hi = _eval_window(s, b, e, fps)
        f = deepest_frame(angles_per_frame, lo, hi, side)
        knee = None if f is None else angles_per_frame[f].get(_knee_key(side))
        if knee is None or not np.isfinite(knee):
            flexions.append(0.0)
        else:
            flexions.append(max(0.0, 180.0 - float(knee)))

    deepest = max(flexions) if flexions else 0.0
    floor = max(MIN_REP_KNEE_FLEXION, deepest * MIN_REP_FLEXION_RATIO)
    return [rep for rep, flex in zip(reps, flexions) if flex >= floor]


def flag_frames(angles_per_frame, reps, side, depth_window=8):
    """Which joints to draw in fault colour, PER FRAME, for the overlay video.

    Previously only the single worst frame was flagged, which meant one frame in
    six hundred was ever red — a 30th of a second, invisible at playback speed,
    so the overlay looked uniformly green no matter how the set went. The colour
    is supposed to show *when* form breaks down, so it has to track the
    measurement frame by frame.

    Two faults are marked, and only where they are actually measurable:
      - forward lean: any valid frame where the trunk leads the shin by more than
        LEAN_EXCESS_LIMIT. Genuinely per-frame, so the torso reddens exactly
        while the lean is happening.
      - shallow depth: a per-REP property, not a per-frame one, so it marks the
        knee and hip around the deepest point of any rep that never got low
        enough. Marking the whole rep would be wrong — the descent itself is fine.

    Frames that fail the validity gate are left unflagged rather than guessed at,
    and the renderer independently fades anything below the visibility threshold,
    so a joint MediaPipe is unsure about can never be shown as confidently bad."""
    n = len(angles_per_frame)
    out = [set() for _ in range(n)]
    shin_key = "shin_left" if side == "left" else "shin_right"
    knee_key = _knee_key(side)

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
        deepest, knee_min = None, None
        for i in range(max(0, s), min(n, e + 1)):
            if not frame_valid(angles_per_frame[i], side):
                continue
            k = angles_per_frame[i].get(knee_key)
            if k is None or not np.isfinite(k):
                continue
            if knee_min is None or k < knee_min:
                deepest, knee_min = i, k
        if deepest is None or knee_min <= DEPTH_FLAG_KNEE_ANGLE:
            continue
        lo = max(0, deepest - depth_window)
        hi = min(n - 1, deepest + depth_window)
        for i in range(lo, hi + 1):
            if frame_valid(angles_per_frame[i], side):
                out[i] |= {f"{side}_knee", f"{side}_hip"}

    return out


def deepest_frame(angles_per_frame, start, end, side):
    """Frame of maximum knee flexion (minimum knee angle) within a rep window,
    considering only VALID frames . The hip-lowest frame (velocity bottom) and the
    deepest knee bend don't always coincide, so depth is read here — but a naive
    argmin would grab a single-frame tracking glitch, so glitched frames are
    excluded via frame_valid(). Returns None if the whole window is unusable."""
    key = _knee_key(side)
    best_i, best_k = None, float("inf")
    for i in range(start, end + 1):
        if not frame_valid(angles_per_frame[i], side):
            continue
        k = angles_per_frame[i][key]
        if k < best_k:
            best_k, best_i = k, i
    return best_i


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


def _eval_window(start, b, end, fps):
    """Search window for peak flexion: ~0.4s either side of the kinematic bottom,
    clamped to the rep. The deepest knee bend is physically near the hip's lowest
    point, so we don't scan the whole rep (which lets the eval frame wander into
    adjacent movement on long or messy clips)."""
    hw = max(5, int(round(0.4 * (fps or 30.0))))
    return max(start, b - hw), min(end, b + hw)


def score_reps(angles_per_frame, reps, side, fps=30.0):
    """Score each rep at its deepest VALID frame near the bottom. Returns a list
    of (eval_frame, score, breakdown) in rep order. A rep whose window is entirely
    glitched yields (None, NaN, ...) and is excluded from worst/best selection
    rather than scored on garbage."""
    out = []
    for (s, b, e) in reps:
        lo, hi = _eval_window(s, b, e, fps)
        f = deepest_frame(angles_per_frame, lo, hi, side)
        if f is None:
            out.append((None, float("nan"), {"depth": None, "spine": None}))
        else:
            score, breakdown = _score_frame(angles_per_frame[f], side)
            out.append((f, score, breakdown))
    return out


def worst_frame(angles_per_frame, reps, side, fps=30.0):
    """Eval frame of the worst-scoring rep (highest deviation)."""
    scored = [(f, s) for (f, s, _) in score_reps(angles_per_frame, reps, side, fps)
              if np.isfinite(s)]
    if not scored:
        return None
    return max(scored, key=lambda x: x[1])[0]


def best_frame(angles_per_frame, reps, side, fps=30.0):
    """Eval frame of the best-scoring rep (lowest deviation)."""
    scored = [(f, s) for (f, s, _) in score_reps(angles_per_frame, reps, side, fps)
              if np.isfinite(s)]
    if not scored:
        return None
    return min(scored, key=lambda x: x[1])[0]

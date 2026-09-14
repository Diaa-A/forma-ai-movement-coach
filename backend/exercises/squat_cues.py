"""Squat cue database — Layer 1 of the two-layer coaching architecture.

Every entry is a pre-vetted coaching statement. The LLM in Layer 2 may rephrase
these for tone but MUST NOT invent new biomechanical claims, weights, or rep
recommendations.

Rationale for the "fault wording" + "fix wording" split: the fault tells the
user WHAT was wrong (their model of what happened); the fix tells them what to
do next. Splitting them keeps the LLM honest about which side it's rephrasing.

Detection lives in `evaluate_squat()` below. Each cue declares the *category*
of detection it relies on so we can mark cues as future-work when their detector
isn't reliable yet (e.g. knee valgus needs 3D or a front-camera and is parked).
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

from .squat import (frame_valid, LEAN_EXCESS_LIMIT, MIN_CUE_VISIBILITY,
                    DEPTH_FLAG_KNEE_ANGLE as _DEPTH_FLAG_KNEE_ANGLE)
from .base import ExerciseProfile, SAGITTAL, FRONTAL, coverage_guidance
# CueHit and Evaluation live in mechanics now (push-up needs them too);
# re-exported here so existing imports of squat_cues keep working.
from .mechanics import CueHit, Evaluation  # noqa: F401
from ..pipeline.pose import LM


# ----------------------------------------------------------------------------
# Cue database
# ----------------------------------------------------------------------------
# Each cue: severity controls the report slot (primary -> "Primary issue",
# secondary -> "Secondary issues"). `joints` drives the skeleton overlay colour
# on the worst frame. `available` lets us park cues that need detection we
# can't yet do reliably (e.g. knee valgus in 2D from a side angle ).

SQUAT_CUES: Dict[str, dict] = {
    "shallow_depth": {
        "severity": "primary",
        "fault": "knees stayed above parallel at the bottom — hips never dropped level with the knees.",
        "fix": "Aim to reach hip-crease level with the knee at the bottom. If your range is limited, work on ankle and hip mobility separately rather than forcing depth.",
        "joints": ["knee"],
        "phase": "bottom",
        "plane": SAGITTAL,
        "available": True,
    },
    "excessive_forward_lean": {
        "severity": "primary",
        "fault": "at the bottom the torso leaned noticeably more than the shins — the chest dropped toward the floor while the lower legs stayed more upright, shifting load onto the lower back.",
        "fix": "Drive your chest up as you descend so the torso stays roughly parallel to your shins. Think 'proud chest' — the sternum should point forward, not at the floor.",
        "joints": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        "phase": "bottom",
        "plane": SAGITTAL,
        "available": True,
    },
    "hip_rise_first": {
        "severity": "primary",
        "fault": "on the way up, the hips rose faster than the chest — the spine angle steepened before the knees finished extending. This is the 'good morning' fault.",
        "fix": "Drive the chest up and the hips up together. Cue: imagine pushing the floor away with your feet rather than lifting your hips.",
        "joints": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        "phase": "ascent",
        "plane": SAGITTAL,
        "available": True,
    },
    "knee_left_right_asymmetry": {
        "severity": "secondary",
        "fault": "knee angle at the bottom differed by more than 10° between left and right, suggesting a side-to-side weight shift.",
        "fix": "Film yourself from the front to check whether one hip is dropping. Even out by emphasising the weak side.",
        "joints": ["left_knee", "right_knee"],
        "phase": "bottom",
        "plane": FRONTAL,
        "available": True,
    },
    "rep_inconsistency": {
        "severity": "secondary",
        "fault": "depth varied by more than 15° across reps — some reps were noticeably shallower or deeper than others.",
        "fix": "Use a consistent depth target. If you can't hit the same depth every rep, you may be fatiguing — reduce the load or rep count.",
        "joints": [],
        "phase": None,
        "plane": SAGITTAL,
        "available": True,
    },
    "fast_descent": {
        "severity": "secondary",
        "fault": "descent was completed in under half a second — gravity is doing most of the work.",
        "fix": "Lower yourself in 2-3 seconds under control. Pause briefly at the bottom before driving up.",
        "joints": [],
        "phase": "descent",
        "plane": SAGITTAL,
        "available": True,
    },

    # --- parked: detection not yet reliable in our 2D / side-view scope ---
    "knee_valgus": {
        "severity": "primary",
        "fault": "knees travelled inward toward each other during the descent or ascent.",
        "fix": "Push the knees outward in line with the toes throughout the movement. Cue: 'spread the floor'.",
        "joints": ["left_knee", "right_knee"],
        "phase": "descent",
        "plane": FRONTAL,
        "available": False,   # needs front-on camera or 3D — parked
    },
    "heel_lift": {
        "severity": "secondary",
        "fault": "the heels rose off the ground at the bottom — usually a sign of limited ankle dorsiflexion.",
        "fix": "Work on calf and ankle mobility, or try squatting with 5-10mm heel elevation (a thin plate under each heel).",
        "joints": ["left_heel", "right_heel"],
        "phase": "bottom",
        "plane": SAGITTAL,
        "available": False,   # needs ground-plane reference, parked
    },
}


# Camera-view profile for the squat — the per-exercise scalability layer. The
# squat is filmed side-on (sagittal plane), which exposes depth and forward lean
# but NOT left/right symmetry (that needs a front view). push-up / pull-up will
# each define their own profile the same way.
SQUAT_PROFILE = ExerciseProfile(
    name="squat",
    view_label="side-on",
    # The resolution line is measured, not a guess. The same clip analysed at
    # 576x1024, 1080p and 2160p produced the same 7 reps, the same cues, and knee
    # angles within 0.4 degrees -- well inside the ~9.5 degree error the Penn
    # Action benchmark already reports for this joint. MediaPipe resizes its input
    # internally, so the extra pixels are discarded before they reach the model.
    # They cost the user upload time and cost the server roughly double the memory
    # (238 MB at 576x1024 against 774 MB at 2160p), and buy nothing.
    filming_guide=("Film side-on at about hip height with your whole body in "
                   "frame. 1080p is plenty — filming in 4K makes the upload "
                   "slower and doesn't improve the analysis. For a left/right "
                   "symmetry check, film a second set from the front."),
    # Audited against the cue set on 4 Aug 2026 and found short in three places:
    # hip_rise_first fires but was declared nowhere, and both parked detectors
    # (knee_valgus, heel_lift) were invisible -- so a user asking "do my knees
    # cave in?" got silence with no way to tell whether that meant "no" or "not
    # assessed". Same fourth-state failure as the push-up's elbow flare. Every cue
    # in SQUAT_CUES now appears here, parked or not.
    plane_assessments={
        SAGITTAL: ["squat depth", "forward lean", "descent tempo", "rep consistency",
                   "hip drive out of the bottom"],
        FRONTAL:  ["left/right symmetry"],
    },
    # Cues that exist but are parked (available: False). Declared so the silence
    # is legible, kept out of the covered lists so no view claims to assess them.
    not_yet_assessed=["knee tracking (whether the knees cave inward)",
                      "whether the heels stay down"],
)

# Which anatomical plane each cue lives in — a cue is only meaningful from a view
# that exposes its plane. The confidence gate enforces this at runtime; this tag
# is the a-priori design statement and is what push-up / pull-up cue sets reuse.
CUE_PLANE: Dict[str, str] = {
    "shallow_depth":            SAGITTAL,
    "excessive_forward_lean":   SAGITTAL,
    "hip_rise_first":           SAGITTAL,
    "knee_left_right_asymmetry": FRONTAL,
    "rep_inconsistency":        SAGITTAL,
    "fast_descent":             SAGITTAL,
    "knee_valgus":              FRONTAL,
    "heel_lift":                SAGITTAL,
}


SQUAT_POSITIVES: Dict[str, dict] = {
    "good_depth": {
        "text": "you reached full depth — hips clearly below parallel at the bottom of every rep.",
    },
    "upright_torso": {
        "text": "your torso stayed roughly parallel to your shins, keeping the load on the legs rather than the lower back.",
    },
    "consistent_reps": {
        "text": "rep-to-rep consistency was good — depth varied by less than 10° across the set.",
    },
    "controlled_tempo": {
        "text": "descent tempo was controlled — no rushing or bouncing out of the bottom.",
    },
    "knees_track_symmetrically": {
        "text": "left and right knees reached the same depth at the bottom — no obvious side-to-side shift.",
    },
}


# ----------------------------------------------------------------------------
# Layer 1 evaluator
# ----------------------------------------------------------------------------

# Detection thresholds. Kept here (next to the cue text they fire) rather than
# scattered. Phase D / E may move some of these to per-user calibration.
# knee angle at bottom > _DEPTH_FLAG_KNEE_ANGLE => shallow. Imported from
# squat.py above, because the overlay needs the same number to colour a shallow
# rep and cannot import this module (squat_cues imports squat, not the reverse).
# forward lean is relative: trunk leading the shin by > LEAN_EXCESS_LIMIT
# (imported from squat.py) flags as excessive — see that module for the rationale
_KNEE_ASYMMETRY_DEG      =  10.0   # |knee_L - knee_R| at bottom > this => asym
_DEPTH_INCONSISTENCY_DEG =  15.0   # max-min knee at bottom across reps > this
_FAST_DESCENT_SEC        =   0.5   # descent shorter than this => uncontrolled


def _knee_keys(side):
    if side == "left":
        return "knee_left", "knee_right"
    return "knee_right", "knee_left"


def _joints_visible(landmarks, frame, names, thresh=MIN_CUE_VISIBILITY):
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


def _deepest(angles_per_frame, start, end, side):
    """Deepest VALID knee frame in [start, end] — skips tracking-glitch frames
    (see squat.frame_valid). Returns None if the window is entirely unusable."""
    key = "knee_left" if side == "left" else "knee_right"
    best_i, best_k = None, float("inf")
    for i in range(start, end + 1):
        if not frame_valid(angles_per_frame[i], side):
            continue
        k = angles_per_frame[i][key]
        if k < best_k:
            best_k, best_i = k, i
    return best_i


def evaluate_squat(angles_per_frame, reps, side, fps, phase_per_frame=None,
                   landmarks=None):
    """Run Layer 1 over the pipeline output. Returns an Evaluation.

    `angles_per_frame` comes from squat_angles_per_frame, `reps` from
    segment_reps as (start, bottom, end) tuples, and `side` picks which knee
    counts as primary. `fps` is only read by the tempo cues.

    The two optional arguments each switch something off when missing. Without
    `phase_per_frame` (from label_phases) the tempo cues are skipped. Without
    `landmarks`, an (n_frames, 33, 4) array, there is no confidence weighting,
    so cues fire on joints the model may be guessing at — the one that matters
    is the left/right comparison, which means nothing when the far leg is
    occluded.
    """
    knee_primary, knee_secondary = _knee_keys(side)
    shin_key = "shin_left" if side == "left" else "shin_right"
    other_side = "right" if side == "left" else "left"
    notes = []
    fired: Dict[str, CueHit] = {}
    bottom_knees = []
    bottom_trunks = []        # trunk lean from vertical
    lean_excesses = []        # trunk lean minus shin lean (the relative measure)
    bottom_asym_reps = []
    asym_assessable = False   # did any rep have the far leg visible enough to compare?
    descent_durations = []

    hw = max(5, int(round(0.4 * (fps or 30.0))))
    for rep_i, (s, b, e) in enumerate(reps):
        # read depth/spine/asymmetry at the DEEPEST VALID frame near the bottom,
        # not the hip-velocity bottom itself — the two don't always coincide
        # (see squat.deepest_frame). If the window is glitched, skip the rep.
        lo, hi = max(s, b - hw), min(e, b + hw)
        f = _deepest(angles_per_frame, lo, hi, side)
        if f is None:
            continue
        ang_b = angles_per_frame[f]
        k_main = ang_b.get(knee_primary)
        k_other = ang_b.get(knee_secondary)
        spine = ang_b.get("spine")
        shin = ang_b.get(shin_key)

        if k_main is not None and np.isfinite(k_main):
            bottom_knees.append(k_main)
            # shallow depth — only the chosen side
            if k_main > _DEPTH_FLAG_KNEE_ANGLE:
                _fire(fired, "shallow_depth", rep_i)

        # forward lean is judged relative to the shin (trunk/shin parallelism),
        # not against an absolute angle — robust to deep squats and camera tilt
        if (spine is not None and shin is not None
                and np.isfinite(spine) and np.isfinite(shin)):
            bottom_trunks.append(spine)
            excess = spine - shin
            lean_excesses.append(excess)
            if excess > LEAN_EXCESS_LIMIT:
                _fire(fired, "excessive_forward_lean", rep_i)

        # left/right knee asymmetry — a cross-side comparison, so only trust it
        # when the far-side leg is actually visible. In a side-on view the far leg
        # is occluded (low confidence) and the comparison is meaningless, so we
        # suppress the cue rather than fire it on unreliable coordinates.
        if (k_main is not None and k_other is not None
                and np.isfinite(k_main) and np.isfinite(k_other)
                and _joints_visible(landmarks, f,
                                    [other_side + "_hip", other_side + "_knee",
                                     other_side + "_ankle"])):
            asym_assessable = True
            if abs(k_main - k_other) > _KNEE_ASYMMETRY_DEG:
                bottom_asym_reps.append(rep_i)
                _fire(fired, "knee_left_right_asymmetry", rep_i)

        # descent duration (frames between start and bottom -> seconds)
        if fps and fps > 0:
            descent_frames = max(1, b - s)
            descent_sec = descent_frames / fps
            descent_durations.append(descent_sec)
            if descent_sec < _FAST_DESCENT_SEC:
                _fire(fired, "fast_descent", rep_i)

        # hip-rise-first on ascent: does the spine angle steepen (lean increase)
        # before the knee finishes extending? Compare midpoints of ascent.
        if (phase_per_frame is not None and e > b + 4
                and frame_valid(angles_per_frame[(b + e) // 2], side)
                and frame_valid(angles_per_frame[e], side)):
            mid = (b + e) // 2
            ang_mid = angles_per_frame[mid]
            ang_end = angles_per_frame[e]
            sp_mid = ang_mid.get("spine")
            sp_end = ang_end.get("spine")
            k_mid = ang_mid.get(knee_primary)
            k_end = ang_end.get(knee_primary)
            if all(v is not None and np.isfinite(v) for v in (sp_mid, sp_end, k_mid, k_end)):
                # at ascent midpoint we're not done extending, but if the spine
                # has already returned ~closer to upright than the knee has
                # extended, hips rose first
                spine_progress = max(0.0, (sp_mid - sp_end))  # decreasing toward upright
                knee_progress = max(0.0, (k_end - k_mid))    # increasing toward extension
                # crude proxy: if spine swung back more aggressively than knee opened
                if spine_progress > 8.0 and knee_progress < 15.0:
                    _fire(fired, "hip_rise_first", rep_i)

    # whole-set: depth inconsistency across reps
    if len(bottom_knees) >= 2:
        spread = max(bottom_knees) - min(bottom_knees)
        if spread > _DEPTH_INCONSISTENCY_DEG:
            # every rep contributed — leave rep_indices empty to signal "across reps"
            fired.setdefault("rep_inconsistency", CueHit(
                flag="rep_inconsistency",
                severity=SQUAT_CUES["rep_inconsistency"]["severity"],
                fault=SQUAT_CUES["rep_inconsistency"]["fault"],
                fix=SQUAT_CUES["rep_inconsistency"]["fix"],
                joints=list(SQUAT_CUES["rep_inconsistency"]["joints"]),
            ))

    # positives — only fire when the opposite cue did NOT fire
    positives: List[str] = []
    if bottom_knees and max(bottom_knees) <= _DEPTH_FLAG_KNEE_ANGLE:
        positives.append(SQUAT_POSITIVES["good_depth"]["text"])
    if lean_excesses and max(lean_excesses) <= LEAN_EXCESS_LIMIT:
        positives.append(SQUAT_POSITIVES["upright_torso"]["text"])
    if (len(bottom_knees) >= 2
            and (max(bottom_knees) - min(bottom_knees)) <= 10.0
            and "rep_inconsistency" not in fired):
        positives.append(SQUAT_POSITIVES["consistent_reps"]["text"])
    if descent_durations and min(descent_durations) >= _FAST_DESCENT_SEC:
        positives.append(SQUAT_POSITIVES["controlled_tempo"]["text"])
    if (asym_assessable and not bottom_asym_reps
            and "knee_left_right_asymmetry" not in fired):
        positives.append(SQUAT_POSITIVES["knees_track_symmetrically"]["text"])

    # informational stats so the LLM (or the report) can quote specifics
    if bottom_knees:
        notes.append(
            "average knee angle at the bottom: {:.0f}° (target ~90°, deeper acceptable)"
            .format(float(np.mean(bottom_knees)))
        )
    if lean_excesses:
        notes.append(
            "average trunk lean at the bottom: {:.0f}° from vertical, {:.0f}° more "
            "than the shins (trunk and shins should stay roughly parallel)"
            .format(float(np.mean(bottom_trunks)), float(np.mean(lean_excesses)))
        )
    if descent_durations:
        notes.append(
            "average descent duration: {:.2f}s".format(float(np.mean(descent_durations)))
        )
    # profile-driven view guidance: turn an un-assessable plane into an actionable
    # instruction ("film from the front for symmetry") instead of a silent gap.
    # Kept as its own field so it is surfaced deterministically, not left to the LLM.
    guidance = coverage_guidance(SQUAT_PROFILE, frontal_observed=asym_assessable)
    view_guidance = guidance if reps else None
    if view_guidance:
        notes.append(view_guidance)   # also give the LLM the context

    return Evaluation(
        exercise="squat",
        side=side,
        rep_count=len(reps),
        cues_fired=list(fired.values()),
        positives=positives,
        notes=notes,
        view_guidance=view_guidance,
    )


def _fire(fired, flag, rep_i):
    """Helper: record (or extend) a cue hit."""
    if flag not in SQUAT_CUES:
        return
    if not SQUAT_CUES[flag].get("available", False):
        return
    if flag in fired:
        fired[flag].rep_indices.append(rep_i)
    else:
        fired[flag] = CueHit(
            flag=flag,
            severity=SQUAT_CUES[flag]["severity"],
            fault=SQUAT_CUES[flag]["fault"],
            fix=SQUAT_CUES[flag]["fix"],
            joints=list(SQUAT_CUES[flag]["joints"]),
            rep_indices=[rep_i],
        )

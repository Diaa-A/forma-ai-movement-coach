"""Rep-level machinery that turned out not to be about squats.

When the squat analyser was the only one, all of this lived in `squat.py` with
the knee and the hip written directly into it. Adding push-up made it obvious how
much of it was general: picking which side of the body to trust, deciding which
detected bottoms are real reps, finding the deepest frame of a rep, scoring reps
and choosing the worst and best. None of that cares which joint bends.

What differs between exercises is small and specific, so it is collected in a
`Movement` and passed in: which angle is the one that flexes, which landmarks
decide side visibility, what sets the body-size reference, and how a single frame
is scored. Everything below is written against that and nothing else.

The alternative was to copy `squat.py` and rename the joints, which is how you end
up fixing the same bug twice. Extracting instead also made a genuine question
answerable — whether the original design generalised — and the honest answer is
mostly, with two exceptions worth recording:

  - `body_scale` measured VERTICAL distance. Fine for a standing body, useless for
    a push-up where the shoulder and ankle sit at nearly the same height, so the
    metric had to become part of the movement rather than an assumption.
  - The rep-gating floors are per-movement numbers. A squat drops the hips about
    half a leg length; a push-up pivots at the toes and moves them far less. One
    set of constants could never have covered both.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..pipeline.pose import LM, VISIBILITY_THRESHOLD


@dataclass(frozen=True)
class Movement:
    """Everything the generic machinery needs to know about one exercise.

    Deliberately a bag of small functions rather than a class hierarchy: there are
    two exercises and the differences are a handful of joint names and thresholds,
    so inheritance would be more scaffolding than substance.
    """

    name: str

    # side -> the angle key that flexes during the movement ("knee_left", "elbow_right")
    primary_angle: Callable[[str], str]
    # side -> landmark names whose visibility decides which side to analyse
    side_joints: Callable[[str], List[str]]
    # landmark pairs whose length gives the body-size reference; the longest wins
    scale_pairs: Sequence[Tuple[str, str]]
    # "vertical" measures |dy| (a standing body); "euclidean" measures true length
    scale_metric: str
    # is this frame usable at all, or is it a tracking glitch
    is_valid: Callable[[dict, str], bool]
    # single-frame deviation score, higher is worse -> (score, breakdown)
    score_frame: Callable[[dict, str], Tuple[float, dict]]

    # Which way the shoulder->ankle line runs in a correctly decoded frame:
    # "vertical" for a standing movement, "horizontal" for one done on the floor.
    # Read by the rotation guard, which refuses a clip whose measured axis is on
    # the wrong side of the cut for this movement -- see body_axis_tilt below.
    body_axis: str = "vertical"

    # Does the hard part of the rep happen at the top of the body's travel?
    #
    # detect_bottoms looks for the frame where the body stops going down and
    # starts back up, which is where a squat and a push-up flex most. A pull-up
    # flexes most at the top, so the runner negates the travel signal when this is
    # set (see travel_for_phase). Phase labels stay positional: for a pull-up
    # "descent" is the pull and "ascent" the lowering, so a tempo cue added later
    # has to time "ascent".
    effort_at_top: bool = False

    # --- rep gating, all as multiples of the body scale or degrees of bend ---
    #
    # Whether body travel is a usable signal at all. It is for a squat: the hips
    # ARE the movement, and travel separates a rep from a noise zero-crossing
    # cleanly. It is not for a push-up. The body pivots at the toes, so the hips
    # move a fraction of what the arms do, and how much of that fraction reaches
    # the camera depends on where the camera is. Measured across seven clips, hip
    # travel for a genuine push-up rep ranged from 0.10 to 0.55 of body length --
    # a five-fold spread for the same exercise, which no pair of thresholds can
    # straddle. The joint that flexes is the honest test there.
    use_travel_gate: bool = True
    max_travel: float = 1.0        # above this the tracking has come apart
    travel_abs_floor: float = 0.10
    travel_rel_floor: float = 0.35
    min_flexion: float = 30.0      # degrees off straight before a dip counts
    flexion_rel_floor: float = 0.4
    # How much of a rep's window has to survive the per-frame validity gate for
    # the rep to be scoreable at all. See _windows_mostly_valid.
    min_valid_fraction: float = 0.6

    # how far either side of the kinematic bottom to look for peak flexion
    eval_window_sec: float = 0.4


# ---------------------------------------------------------------------------
# what a Layer 1 evaluation produces
# ---------------------------------------------------------------------------
# Shared verbatim by every exercise. The cue DATABASE is per-exercise -- the
# thresholds and the wording are the domain knowledge -- but the shape of a fired
# cue and of a finished evaluation is not, and `coaching.py` consumes this shape
# without caring which exercise produced it.

@dataclass
class CueHit:
    flag: str                  # which cue (matches SQUAT_CUES key)
    severity: str
    fault: str
    fix: str
    joints: List[str] = field(default_factory=list)
    rep_indices: List[int] = field(default_factory=list)  # which reps fired it


@dataclass
class Evaluation:
    exercise: str
    side: str
    rep_count: int
    cues_fired: List[CueHit]
    positives: List[str]       # plain-text positive sentences
    notes: List[str] = field(default_factory=list)   # diagnostic / informational
    view_guidance: Optional[str] = None   # actionable camera-view tip, if any

    def primary(self) -> Optional[CueHit]:
        # The first primary cue that fired, ordered by how many reps it hit.
        primaries = [c for c in self.cues_fired if c.severity == "primary"]
        if not primaries:
            return None
        return max(primaries, key=lambda c: len(c.rep_indices))

    def secondaries(self, limit=2) -> List[CueHit]:
        sec = [c for c in self.cues_fired if c.severity == "secondary"]
        primary = self.primary()
        # if no primary fired, the most-fired secondary becomes the primary slot
        # (handled by the report builder, not here)
        return sec[:limit] if sec else []


# ---------------------------------------------------------------------------
# side selection and body size
# ---------------------------------------------------------------------------

def pick_side(movement: Movement, landmarks, reps) -> str:
    """Left or right, by mean landmark visibility over the analysed reps.

    Which joints matter is the movement's business — legs for a squat, arms for a
    push-up — but the comparison itself is the same either way.
    """
    if not reps:
        spans = [(0, landmarks.shape[0])]      # nothing segmented yet, use it all
    else:
        spans = [(s, e) for (s, _, e) in reps]

    def visibility(joint_names, span):
        s, e = span
        idxs = [LM[n] for n in joint_names]
        sub = landmarks[s:e, idxs, 3]          # visibility channel
        return float(np.nanmean(sub)) if sub.size else 0.0

    left = np.mean([visibility(movement.side_joints("left"), sp) for sp in spans])
    right = np.mean([visibility(movement.side_joints("right"), sp) for sp in spans])
    return "left" if left >= right else "right"


def body_scale(movement: Movement, landmarks) -> float:
    """Body-size reference in normalised units, so rep travel can be judged
    relative to the person rather than to the frame. Makes the rep test invariant
    to camera distance, zoom and resolution.

    The metric is the movement's, not a constant. Measuring vertical distance is
    right for a standing body and meaningless for a horizontal one — in a push-up
    the shoulder and ankle are at almost the same height, so |dy| collapses to
    near zero and every rep would look impossibly large against it.
    """
    lengths = []
    for a_name, b_name in movement.scale_pairs:
        a = landmarks[:, LM[a_name], :2]
        b = landmarks[:, LM[b_name], :2]
        if movement.scale_metric == "euclidean":
            d = np.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1])
        else:
            d = np.abs(a[:, 1] - b[:, 1])
        d = d[np.isfinite(d)]
        lengths.append(float(np.median(d)) if d.size else 0.0)
    return max(lengths) if lengths else 0.0


# Refusal cuts for body_axis_tilt, one per expected axis, with a wide dead band
# between them so a diagonal-ish framing is never refused on this evidence.
# Calibrated on the fixture set (per-clip medians, full model):
#
#     correct squats              8.7 - 10.5     (4 clips)
#     correct push-ups           54.5 - 72.6     (8 clips, incl. one whose
#                                                 rotation flag IS honoured)
#     squat with flag stripped          81.3
#     push-up stored sideways           18.6
#
# So every correctly decoded clip clears its cut by at least 24 degrees, and both
# genuinely rotated ones land past it by more than 21.
TILT_ROTATED_ABOVE = 60.0    # a "vertical" movement reading this horizontal is rotated
TILT_ROTATED_BELOW = 30.0    # a "horizontal" movement reading this vertical is rotated

# Below either floor the median is not evidence, so no verdict is given. The
# clips that fall here are the ones the subject is barely tracked in, and those
# already come back as no_reps or low_detection.
_TILT_MIN_FRAMES = 10
_TILT_MIN_FRACTION = 0.2


def body_axis_tilt(landmarks):
    """Median angle of the shoulder->ankle line from image vertical, in degrees:
    0 is an upright body, 90 a horizontal one.

    Exists for one failure. A file whose pixels are stored rotated and whose
    rotation flag has been stripped -- messaging-app re-encodes do this --
    decodes sideways, and the pipeline then measures a tipped-over person
    without noticing: the reference clip went from 7 reps to 1 with the same
    frame reported as best and as a forward-lean fault, status still ok.
    Metadata cannot catch it because there is no metadata. The pose can,
    because a body has an axis and the movement knows which way it should run.

    Per frame the better-seen side is used, gated at the shared visibility
    threshold -- side-on footage rarely clears the gate on both sides at once.
    Returns None when too few frames qualify to support a verdict either way.
    """
    n = landmarks.shape[0]
    tilts = []
    for f in range(n):
        best = None
        for side in ("left", "right"):
            sh = landmarks[f, LM[f"{side}_shoulder"]]
            an = landmarks[f, LM[f"{side}_ankle"]]
            vis = min(sh[3], an[3])
            if not np.isfinite(vis) or vis < VISIBILITY_THRESHOLD:
                continue
            if not np.all(np.isfinite([sh[0], sh[1], an[0], an[1]])):
                continue
            if best is None or vis > best[0]:
                best = (vis, sh, an)
        if best is None:
            continue
        _, sh, an = best
        dx = an[0] - sh[0]
        dy = an[1] - sh[1]
        if dx == 0 and dy == 0:
            continue
        tilts.append(float(np.degrees(np.arctan2(abs(dx), abs(dy)))))
    if len(tilts) < _TILT_MIN_FRAMES or len(tilts) < _TILT_MIN_FRACTION * n:
        return None
    return float(np.median(tilts))


def axis_looks_rotated(movement: Movement, tilt) -> bool:
    """Is the measured body axis on the wrong side of this movement's cut?

    None -- no measurement -- is never refused. A clip we cannot see a body in
    is a detection problem, and the detection statuses already cover it.
    """
    if tilt is None:
        return False
    if movement.body_axis == "horizontal":
        return tilt < TILT_ROTATED_BELOW
    return tilt > TILT_ROTATED_ABOVE


# ---------------------------------------------------------------------------
# which detected bottoms are actually reps
# ---------------------------------------------------------------------------


def _windows_mostly_valid(movement: Movement, reps, angles_per_frame, side):
    """Drop segments that are mostly frames the validity gate already rejects.

    Found by a participant in the first testing round, on a push-up clip. Their
    last "rep" was the moment they got up and walked back to the phone: the pose
    estimate came apart, the body line swung to 170 degrees and back, and the one
    frame in that window that squeaked under the plausibility ceiling at 40
    degrees was picked as the deepest and reported to them as the worst form of
    the set, with hips and shoulders flagged. Nine genuine reps scored zero and
    the only thing the report pointed at was them reaching for the camera.

    The per-frame gate was doing its job -- 42 of that window's 75 frames were
    already being discarded. Nothing was asking whether a rep built out of mostly
    discarded frames should exist. Measured on that clip: the nine real reps were
    100% valid, the tenth 73%, and the false one 44%.

    Applied before the travel and flexion gates rather than after, for the reason
    the travel ceiling exists (Decision 25): both of those calibrate a ratio
    against the clip's own maximum, and a segment of tracking noise in the sample
    distorts what "normal" means for every real rep beside it.
    """
    if angles_per_frame is None:
        return list(reps)

    kept = []
    for (s, b, e) in reps:
        window = range(s, min(e, len(angles_per_frame) - 1) + 1)
        total = sum(1 for _ in window)
        if total == 0:
            continue
        valid = sum(1 for f in window if movement.is_valid(angles_per_frame[f], side))
        if valid / total >= movement.min_valid_fraction:
            kept.append((s, b, e))
    return kept


def keep_real_reps(movement: Movement, reps, travel_y, scale,
                   angles_per_frame=None, side="left", fps=30.0):
    """Filter detected bottoms down to genuine repetitions.

    Two independent questions, asked in order, because they fail differently:

    1. Did the body move far enough, and not absurdly far? Travel is measured
       against body scale. The ceiling matters as much as the floor and has to be
       applied FIRST: on a real upload the subject walked back toward the camera
       and the hips "travelled" 2.9x leg length, which is not a movement a human
       makes. Because the relative floor calibrates against the largest travel in
       the clip, that one artefact became the yardstick, every genuine rep
       measured about 0.15 against it, and all six were discarded as noise while
       the artefact was kept and reported as the only rep. Self-calibration is
       only safe once obvious nonsense is out of the sample it calibrates against.

    2. Did the joint that is supposed to bend actually bend? Travel establishes
       that something moved, not what. Someone settling into position before their
       first rep cleared the travel floor with the knee at 137.8 degrees, and
       because that false rep contained almost no deviation it scored least-bad
       and became the key frame representing the set.

    Both use an absolute floor plus a ratio against the clip's own maximum. The
    ratio is what avoids per-upload tuning, and it preserves the property the
    travel test existed for: a set of uniformly shallow reps still counts, and is
    then flagged as shallow rather than silently dropped.
    """
    if not reps:
        return []

    # Plausibility first, on the same principle as the travel ceiling below: a
    # segment that is mostly frames the validity gate rejects is not a rep, and
    # leaving it in the sample would let it set the scale the ratios calibrate
    # against.
    reps = _windows_mostly_valid(movement, list(reps), angles_per_frame, side)
    if not reps:
        return []

    if not movement.use_travel_gate:
        # Travel says nothing useful for this movement, so let the flexion gate
        # decide on its own. It is the stronger test anyway: it asks whether the
        # joint under assessment actually moved, rather than whether the body did.
        return _keep_reps_that_flexed(movement, list(reps), angles_per_frame,
                                      side, fps)

    travels = []
    for (s, b, e) in reps:
        window = travel_y[s:e + 1]
        window = window[np.isfinite(window)]
        travels.append(float(window.max() - window.min()) if window.size else 0.0)
    if not travels:
        return []

    def implausible(t):
        return scale > 0 and (t / scale) > movement.max_travel

    plausible = [t for t in travels if not implausible(t)]
    # if the whole clip looks impossible, fall back rather than divide by nothing
    # and let the absolute floor and the detection-rate status have the last word
    biggest = max(plausible) if plausible else max(travels)

    moved_enough = []
    for (s, b, e), t in zip(reps, travels):
        if implausible(t):
            continue
        absolute = t / scale if scale > 0 else 0.0
        relative = t / biggest if biggest > 0 else 0.0
        if absolute >= movement.travel_abs_floor and relative >= movement.travel_rel_floor:
            moved_enough.append((s, b, e))

    return _keep_reps_that_flexed(movement, moved_enough, angles_per_frame, side, fps)


def _keep_reps_that_flexed(movement, reps, angles_per_frame, side, fps):
    """Drop dips where the primary joint never really bent. Skipped entirely when
    no angles are supplied, so travel-only filtering stays available on its own."""
    if angles_per_frame is None or not reps:
        return list(reps)

    key = movement.primary_angle(side)
    flexions = []
    for (s, b, e) in reps:
        lo, hi = eval_window(movement, s, b, e, fps)
        f = deepest_frame(movement, angles_per_frame, lo, hi, side)
        angle = None if f is None else angles_per_frame[f].get(key)
        if angle is None or not np.isfinite(angle):
            flexions.append(0.0)
        else:
            flexions.append(max(0.0, 180.0 - float(angle)))

    deepest = max(flexions) if flexions else 0.0
    floor = max(movement.min_flexion, deepest * movement.flexion_rel_floor)
    return [rep for rep, flex in zip(reps, flexions) if flex >= floor]


# ---------------------------------------------------------------------------
# locating and scoring the moment that represents a rep
# ---------------------------------------------------------------------------

def eval_window(movement: Movement, start, bottom, end, fps):
    """Where to look for peak flexion: a short window either side of the kinematic
    bottom, clamped to the rep. The lowest point of the body and the deepest joint
    bend do not always land on the same frame, but they are close — scanning the
    whole rep lets the chosen frame wander into adjacent movement on long clips."""
    half = max(5, int(round(movement.eval_window_sec * (fps or 30.0))))
    return max(start, bottom - half), min(end, bottom + half)


def deepest_frame(movement: Movement, angles_per_frame, start, end, side):
    """Frame of maximum flexion (minimum angle) in a window, considering only
    valid frames. A naive argmin would happily pick a single-frame tracking
    glitch, which is what the validity gate is there to exclude. None if the whole
    window is unusable."""
    key = movement.primary_angle(side)
    best_i, best_angle = None, float("inf")
    for i in range(start, end + 1):
        if not movement.is_valid(angles_per_frame[i], side):
            continue
        angle = angles_per_frame[i].get(key)
        if angle is None or not np.isfinite(angle):
            continue
        if angle < best_angle:
            best_angle, best_i = angle, i
    return best_i


def score_reps(movement: Movement, angles_per_frame, reps, side, fps=30.0):
    """Score each rep at its deepest valid frame. Returns (eval_frame, score,
    breakdown) per rep, in order. A rep whose window is entirely glitched yields
    a NaN score and drops out of worst/best selection rather than being scored on
    garbage."""
    out = []
    for (s, b, e) in reps:
        lo, hi = eval_window(movement, s, b, e, fps)
        f = deepest_frame(movement, angles_per_frame, lo, hi, side)
        if f is None:
            out.append((None, float("nan"), {}))
        else:
            score, breakdown = movement.score_frame(angles_per_frame[f], side)
            out.append((f, score, breakdown))
    return out


def worst_frame(movement: Movement, angles_per_frame, reps, side, fps=30.0):
    """Eval frame of the worst-scoring rep. Note this is the worst of what was
    filmed — on a clean set it is simply the rep that came closest to a limit, and
    callers are expected to label it accordingly rather than implying a fault."""
    return _pick(movement, angles_per_frame, reps, side, fps, max)


def best_frame(movement: Movement, angles_per_frame, reps, side, fps=30.0):
    return _pick(movement, angles_per_frame, reps, side, fps, min)


def _pick(movement, angles_per_frame, reps, side, fps, chooser):
    scored = [(f, s) for (f, s, _) in
              score_reps(movement, angles_per_frame, reps, side, fps)
              if f is not None and np.isfinite(s)]
    if not scored:
        return None
    return chooser(scored, key=lambda pair: pair[1])[0]


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------

def joints_visible(landmarks, frame, names, threshold) -> bool:
    """Are all of these landmarks confidently tracked on this frame?

    Used to suppress cues that depend on a joint the model is guessing at, rather
    than firing them on unreliable coordinates (Decision 19).
    """
    if landmarks is None:
        return False
    for name in names:
        idx = LM.get(name)
        if idx is None:
            return False
        v = landmarks[frame, idx, 3]
        if not np.isfinite(v) or v < threshold:
            return False
    return True


def travel_for_phase(movement: Movement, travel_y):
    """The travel signal as phase detection should see it.

    Negated for a movement whose effort is at the top, so `detect_bottoms` finds
    that frame instead of the hang. Kept out of the runner so it can be tested:
    the wrong sign still gives a believable rep count, built on the wrong frames.
    """
    if not movement.effort_at_top:
        return travel_y
    return -np.asarray(travel_y, dtype=np.float64)


def travel_series(landmarks, left_name, right_name):
    """Mean vertical position of a symmetric landmark pair, which is the signal
    phase detection runs on. Averaging the two sides is more stable than either
    alone when one is partly occluded."""
    left = landmarks[:, LM[left_name], 1]
    right = landmarks[:, LM[right_name], 1]
    return np.nanmean(np.stack([left, right], axis=0), axis=0)

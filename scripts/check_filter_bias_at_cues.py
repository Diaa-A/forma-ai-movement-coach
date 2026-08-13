"""How much does the One Euro filter bias the value a cue actually reads?

    python scripts/check_filter_bias_at_cues.py

The Penn Action ablation reports cue-joint angle error averaged over every
evaluated frame. The cue layer does not read every frame. It reads one: the
deepest valid frame of each rep. This measures the difference there.

It matters because a causal filter's lag is proportional to velocity, and at a
rep bottom the velocity is zero - that is what a local minimum is. So the lag
term largely vanishes exactly where the cue reads, and what is left is peak
attenuation, which is smaller. The averaged figure therefore overstates the error
in the quantity the thresholds are compared against.

Reps are located from the filtered series in every column, because that is what
the pipeline does and this is not an ablation of phase detection. Only the value
read at the located frame changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.exercises import mechanics, pushup
from backend.exercises.registry import get_spec
from backend.pipeline.filter import smooth_series
from backend.pipeline.phase_detection import detect_bottoms, segment_reps
from backend.pipeline.pose import extract_landmarks
from backend.pipeline.runner import _smooth_landmarks

MIN_CUTOFF, BETA = 1.0, 0.007       # RunOptions defaults
WINDOW = 2                          # +/- frames for the median read

CLIPS = [
    ("push-up-shallow-rot.mp4", "pushup", pushup.DEPTH_FLAG_ELBOW_ANGLE,
     "shallow_depth fires above"),
    ("squat.mp4", "squat", None, None),
]


def cue_reads(clip: str, exercise: str):
    spec = get_spec(exercise)
    pose = extract_landmarks(ROOT / "data" / "test_videos" / clip, model="full")
    ts, fps = pose["timestamps"], pose["fps"]
    lm_s = _smooth_landmarks(pose["landmarks"], ts, MIN_CUTOFF, BETA)

    hip = mechanics.travel_series(lm_s, *spec.travel_landmarks)
    hip_s = smooth_series(hip, ts, min_cutoff=0.5, beta=0.001)
    reps_all = segment_reps(
        detect_bottoms(hip_s, min_separation=max(3, int(fps * 0.4))),
        pose["frame_count"])
    side = mechanics.pick_side(spec.movement, lm_s, reps_all)
    scale = mechanics.body_scale(spec.movement, lm_s)

    ang_filt = spec.angles(lm_s)
    ang_raw = spec.angles(pose["landmarks"])
    reps = mechanics.keep_real_reps(spec.movement, reps_all, hip_s, scale,
                                    angles_per_frame=ang_filt, side=side, fps=fps)
    key = spec.movement.primary_angle(side)

    rows = []
    for (s, _b, e) in reps:
        lo, hi = max(0, s), min(len(ang_filt) - 1, e)
        f = mechanics.deepest_frame(spec.movement, ang_filt, lo, hi, side)
        if f is None:
            continue
        filt, raw = ang_filt[f].get(key), ang_raw[f].get(key)
        win = [ang_raw[i].get(key)
               for i in range(max(0, f - WINDOW), min(len(ang_raw), f + WINDOW + 1))]
        win = [v for v in win if v is not None and np.isfinite(v)]
        rows.append((f, filt, raw, float(np.median(win)) if win else None))
    return side, key, rows


def main():
    for clip, exercise, threshold, label in CLIPS:
        side, key, rows = cue_reads(clip, exercise)
        print(f"\n=== {clip}  ({exercise}, side {side}, {key}) ===")
        print(f"{'frame':>7}{'filtered':>10}{'raw':>8}"
              f"{'median+-' + str(WINDOW):>11}{'filt-raw':>10}")
        for f, filt, raw, med in rows:
            print(f"{f:>7}{filt:>10.1f}{raw:>8.1f}{med:>11.1f}{filt - raw:>+10.1f}")

        diffs = [filt - raw for _, filt, raw, _ in rows]
        print(f"  bias at the cue read: {min(diffs):+.1f} to {max(diffs):+.1f} deg, "
              f"mean {np.mean(diffs):+.1f}")

        if threshold is None:
            continue
        fired = {name: sum(1 for r in rows if r[i] > threshold)
                 for name, i in (("filtered", 1), ("median", 3))}
        print(f"  {label} {threshold:.0f} deg — "
              f"filtered {fired['filtered']}/{len(rows)} reps, "
              f"median {fired['median']}/{len(rows)}")
        if fired["filtered"] != fired["median"]:
            print("  >>> the cue changes behaviour on this clip")


if __name__ == "__main__":
    main()

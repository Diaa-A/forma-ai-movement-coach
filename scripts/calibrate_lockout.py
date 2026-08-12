"""Can a lockout threshold be set from the push-up fixture set?

    python scripts/calibrate_lockout.py

Arm B of the lockout experiment wants a cue backed by elbow angle at the top of
the rep, with the threshold calibrated on clips and written down the way
DEPTH_FLAG_ELBOW_ANGLE was. This measures the clips first, because the threshold
is only worth setting if the clips separate.

Three candidate measures per rep, over frames the existing validity gate already
accepts, then the median across the reps of a clip:

    max     most extended elbow anywhere in the rep
    edges   elbow at the rep boundary frames, where the top actually is
    p90     90th percentile, which blunts a single-frame spike

Writes report/chapter5/lockout_calibration.json and .txt. The clips live under
data/test_videos/, which is gitignored, so the Pexels ids are recorded here -
they are the reproduction path for anyone who wants to re-measure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.exercises import mechanics, pushup
from backend.exercises.registry import get_spec
from backend.pipeline.filter import smooth_series
from backend.pipeline.phase_detection import detect_bottoms, segment_reps
from backend.pipeline.pose import extract_landmarks
from backend.pipeline.runner import _smooth_landmarks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "chapter5"
CLIP_DIRS = [ROOT / "data/test_videos/pexels/push_up",
             ROOT / "data/test_videos/pexels/push_up_exercise"]

# Mean elbow-angle error from the Penn Action push-up benchmark. Any gap in the
# numbers below that is smaller than this is not a class boundary the pipeline
# can actually resolve.
ELBOW_ERROR_DEG = 8.1

SPEC = get_spec("pushup")


def clips():
    seen, out = set(), []
    for d in CLIP_DIRS:
        for p in sorted(d.glob("*.mp4")):
            if p.name not in seen:
                seen.add(p.name)
                out.append(p)
    return out


def rep_tops(path):
    """Per-rep top-of-movement elbow angle, by each of the three measures."""
    pose = extract_landmarks(path, model="full")
    if pose["frame_count"] == 0:
        return None
    fps = pose["fps"]
    lm = _smooth_landmarks(pose["landmarks"], pose["timestamps"], 1.0, 0.007)
    angles = SPEC.angles(lm)
    hip = mechanics.travel_series(lm, *SPEC.travel_landmarks)
    hip_s = smooth_series(hip, pose["timestamps"], min_cutoff=0.5, beta=0.001)
    bottoms = detect_bottoms(hip_s, min_separation=max(3, int(fps * 0.4)))
    reps_all = segment_reps(bottoms, pose["frame_count"])
    side = mechanics.pick_side(SPEC.movement, lm, reps_all)
    scale = mechanics.body_scale(SPEC.movement, lm)
    reps = mechanics.keep_real_reps(SPEC.movement, reps_all, hip_s, scale,
                                    angles_per_frame=angles, side=side, fps=fps)
    if not reps:
        return {"side": side, "reps": 0, "measures": {}, "bottoms": []}

    k = pushup._elbow_key(side)
    per = {"max": [], "edges": [], "p90": []}
    bottom_vals = []
    for (s, b, e) in reps:
        lo, hi = max(0, s), min(len(angles) - 1, e)
        vals = [angles[f].get(k) for f in range(lo, hi + 1)
                if pushup.frame_valid(angles[f], side)]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if not vals:
            continue
        per["max"].append(round(max(vals), 1))
        per["p90"].append(round(float(np.percentile(vals, 90)), 1))
        bottom_vals.append(round(min(vals), 1))
        edge = [angles[f].get(k) for f in (lo, hi)
                if pushup.frame_valid(angles[f], side)
                and angles[f].get(k) is not None and np.isfinite(angles[f].get(k))]
        if edge:
            per["edges"].append(round(max(edge), 1))
    return {"side": side, "reps": len(per["max"]), "measures": per,
            "bottoms": bottom_vals}


def largest_gap(values):
    """Biggest step between adjacent clips, which is where a threshold would go."""
    v = sorted(values)
    if len(v) < 2:
        return {"gap": 0.0, "between": None}
    gaps = [(round(b - a, 1), [a, b]) for a, b in zip(v, v[1:])]
    gap, between = max(gaps)
    return {"gap": gap, "between": between}


def build():
    per_clip = {}
    for path in clips():
        got = rep_tops(path)
        if got is None:
            per_clip[path.stem] = {"error": "no frames decoded"}
            continue
        per_clip[path.stem] = got

    usable = {n: d for n, d in per_clip.items() if d.get("reps")}
    medians = {}
    for measure in ("max", "edges", "p90"):
        vals = {n: round(float(np.median(d["measures"][measure])), 1)
                for n, d in usable.items() if d["measures"].get(measure)}
        medians[measure] = {"per_clip": vals, **largest_gap(vals.values())}

    return {
        "clips_examined": len(per_clip),
        "clips_with_reps": len(usable),
        "reps_measured": sum(d["reps"] for d in usable.values()),
        "elbow_angle_error_deg": ELBOW_ERROR_DEG,
        "measures": medians,
        "per_clip": per_clip,
        "sources": "data/test_videos/pexels/*/SOURCE.txt (Pexels ids, CC0)",
    }


def table(d) -> str:
    L = ["Lockout threshold calibration - push-up fixture set", ""]
    L.append(f"{d['clips_with_reps']} clips with usable reps out of {d['clips_examined']}, "
             f"{d['reps_measured']} reps.")
    L.append("Median top-of-rep elbow angle per clip, three ways of measuring it.")
    L.append("")
    hdr = f"{'clip':<14}{'reps':>5}{'max':>9}{'edges':>9}{'p90':>9}"
    L.append(hdr)
    L.append("-" * len(hdr))
    order = sorted(d["measures"]["max"]["per_clip"].items(), key=lambda kv: kv[1])
    for name, _ in order:
        row = [d["measures"][m]["per_clip"].get(name) for m in ("max", "edges", "p90")]
        reps = d["per_clip"][name]["reps"]
        cells = "".join(f"{('-' if v is None else f'{v:.1f}'):>9}" for v in row)
        L.append(f"{name:<14}{reps:>5}{cells}")
    L.append("-" * len(hdr))
    L.append("")
    for m in ("max", "edges", "p90"):
        g = d["measures"][m]
        span = sorted(g["per_clip"].values())
        L.append(f"{m:<6} {span[0]:.1f} to {span[-1]:.1f}, largest step between adjacent "
                 f"clips {g['gap']:.1f} deg at {g['between']}")
    L.append("")
    L.append(f"Penn Action puts mean elbow-angle error at {d['elbow_angle_error_deg']} "
             f"degrees for push-up.")
    return "\n".join(L)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = build()
    (OUT / "lockout_calibration.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    t = table(d)
    (OUT / "lockout_calibration.txt").write_text(t + "\n", encoding="utf-8")
    print(t)
    print(f"\nwritten to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()

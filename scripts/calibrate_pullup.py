"""Can the pull-up thresholds be set from the Penn Action subset?

    python scripts/calibrate_pullup.py

WP-08 step 1 shipped `pullup.py` with every cue threshold marked PROVISIONAL,
measured on an eight-sequence probe that was enough to establish direction and
not enough to set a number. This measures all 25 and reports what they support.

Runs the real pipeline path per sequence -- smooth, angles, travel, the
effort-at-top flip, bottom detection, rep segmentation, the rep gate -- so the
numbers describe reps the system would actually score, not an idealised pass over
the frames.

Two things are measured in the same pass because both need MediaPipe over the
same footage:

    resolution  elbow-angle error against Penn Action ground truth. A gap in the
                distributions narrower than this is not a class boundary the
                pipeline can resolve, so it is the floor on any threshold
    spread      per-rep elbow at the top and at the hang, travel against body
                scale, and trunk lean

Writes report/chapter5/pullup_calibration.json and .txt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.evaluation.metrics import angle_errors                      # noqa: E402
from backend.evaluation.penn_action import (PENN_JOINTS, PENN_TO_MP,     # noqa: E402
                                            find_sequences_by_action,
                                            load_sequence)
from backend.exercises import mechanics, pullup                          # noqa: E402
from backend.pipeline.angles import pullup_angles_per_frame              # noqa: E402
from backend.pipeline.filter import smooth_series                        # noqa: E402
from backend.pipeline.phase_detection import detect_bottoms, segment_reps  # noqa: E402
from backend.pipeline.pose import extract_landmarks_from_frames          # noqa: E402
from backend.pipeline.runner import _smooth_landmarks                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "chapter5"
DATA = ROOT / "data" / "penn_action"

# Penn Action does not record frame rate. 30 is the usual capture rate for the
# source footage and only affects the smoothing time constants and the tempo
# figures, not any angle.
NOMINAL_FPS = 30.0


def mp_to_pixels(mp_frame, w, h):
    """MediaPipe normalised coords to pixels, in Penn Action's joint order.

    Same mapping as the benchmark harness does. Duplicated rather than imported
    because scripts/ is not a package and making it one to share eight lines
    would change how every other script is run.
    """
    out = np.full((len(PENN_JOINTS), 2), np.nan)
    for j, name in enumerate(PENN_JOINTS):
        x, y = mp_frame[PENN_TO_MP[name], 0], mp_frame[PENN_TO_MP[name], 1]
        if np.isfinite(x) and np.isfinite(y):
            out[j] = [x * w, y * h]
    return out


def measure(seq):
    """Per-rep measurements for one sequence, or None if it cannot be scored."""
    paths = seq.frame_paths()[: seq.nframes]
    if len(paths) < 8:
        return None

    pose = extract_landmarks_from_frames(paths, model="full")
    lm = pose["landmarks"]
    if np.all(np.isfinite(lm[:, :, 0]), axis=1).sum() < 8:
        return None

    ts = np.arange(len(paths), dtype=np.float64) / NOMINAL_FPS
    lm_s = _smooth_landmarks(lm, ts, 1.0, 0.007)
    angles = pullup_angles_per_frame(lm_s)

    travel = mechanics.travel_series(lm_s, "left_hip", "right_hip")
    travel = mechanics.travel_for_phase(pullup.PULLUP, travel)
    travel_s = smooth_series(travel, ts, min_cutoff=0.5, beta=0.001)

    bottoms = detect_bottoms(travel_s, min_separation=max(3, int(NOMINAL_FPS * 0.4)))
    reps_all = segment_reps(bottoms, len(paths))
    side = mechanics.pick_side(pullup.PULLUP, lm_s, reps_all)
    scale = mechanics.body_scale(pullup.PULLUP, lm_s)
    reps = pullup.keep_real_reps(reps_all, travel_s, scale,
                                 angles_per_frame=angles, side=side,
                                 fps=NOMINAL_FPS)

    key = pullup.PULLUP.primary_angle(side)
    per_rep = []
    for (s, b, e) in reps:
        window = [angles[i] for i in range(s, min(e + 1, len(angles)))
                  if pullup.frame_valid(angles[i], side)]
        vals = [a[key] for a in window
                if a.get(key) is not None and np.isfinite(a[key])]
        if not vals:
            continue
        trunks = [abs(a["trunk"]) for a in window
                  if a.get("trunk") is not None and np.isfinite(a["trunk"])]
        span = travel_s[s:e + 1]
        span = span[np.isfinite(span)]
        per_rep.append({
            "elbow_at_top": float(min(vals)),
            "elbow_at_hang": float(max(vals)),
            "travel_over_scale": (float(span.max() - span.min()) / scale
                                  if scale and span.size else None),
            "trunk_max": float(max(trunks)) if trunks else None,
            "frames": int(e - s + 1),
        })

    # elbow-angle error against ground truth, over the same frames
    errs = []
    for i in range(len(paths)):
        w, h = pose["sizes"][i]
        if w == 0:
            continue
        try:
            pred = mp_to_pixels(lm[i], w, h)
        except Exception:
            continue
        d = angle_errors(pred, seq.gt_xy(i), seq.gt_visibility(i))
        errs.extend(v for k, v in d.items() if k.startswith("elbow"))

    return {
        "sequence": seq.seq_id,
        "frames": len(paths),
        "reps_detected": len(reps_all),
        "reps_kept": len(reps),
        "side": side,
        "scale": float(scale) if scale else None,
        "per_rep": per_rep,
        "elbow_error_deg": errs,
    }


def stats(values):
    v = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not v.size:
        return None
    return {"n": int(v.size), "min": round(float(v.min()), 1),
            "p25": round(float(np.percentile(v, 25)), 1),
            "median": round(float(np.median(v)), 1),
            "p75": round(float(np.percentile(v, 75)), 1),
            "max": round(float(v.max()), 1),
            "mean": round(float(v.mean()), 1)}


def main():
    ids = find_sequences_by_action(DATA, "pullup")
    print(f"{len(ids)} pull-up sequences\n")

    rows, skipped = [], []
    for n, sid in enumerate(ids, 1):
        seq = load_sequence(sid, DATA)
        r = measure(seq)
        if r is None:
            skipped.append(sid)
            print(f"  [{n:2d}/{len(ids)}] {sid}  skipped, not enough tracked frames")
            continue
        rows.append(r)
        tops = [x["elbow_at_top"] for x in r["per_rep"]]
        detail = f"top {min(tops):.0f} deg" if tops else "no scorable rep"
        print(f"  [{n:2d}/{len(ids)}] {sid}  {r['frames']:3d}f  "
              f"{r['reps_detected']}->{r['reps_kept']} reps  {detail}")

    all_rep = [x for r in rows for x in r["per_rep"]]
    all_err = [e for r in rows for e in r["elbow_error_deg"]]

    summary = {
        "sequences_total": len(ids),
        "sequences_measured": len(rows),
        "sequences_skipped": skipped,
        "reps_scored": len(all_rep),
        "clips_with_multiple_reps": sum(1 for r in rows if len(r["per_rep"]) > 1),
        "elbow_angle_error_deg": stats(all_err),
        "elbow_at_top": stats([x["elbow_at_top"] for x in all_rep]),
        "elbow_at_hang": stats([x["elbow_at_hang"] for x in all_rep]),
        "travel_over_scale": stats([x["travel_over_scale"] for x in all_rep]),
        "trunk_max_deg": stats([x["trunk_max"] for x in all_rep]),
        "frames_per_rep": stats([x["frames"] for x in all_rep]),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pullup_calibration.json").write_text(
        json.dumps({"summary": summary, "per_sequence": rows}, indent=2),
        encoding="utf-8")

    lines = ["Pull-up threshold calibration - Penn Action subset", ""]
    lines.append(f"sequences: {len(rows)} measured of {len(ids)}, "
                 f"{len(skipped)} skipped")
    lines.append(f"reps scored: {len(all_rep)}  "
                 f"(clips with more than one scorable rep: "
                 f"{summary['clips_with_multiple_reps']})")
    lines.append("")
    hdr = (f"{'measure':24s} {'n':>4s} {'min':>7s} {'p25':>7s} "
           f"{'med':>7s} {'p75':>7s} {'max':>7s}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for name in ("elbow_angle_error_deg", "elbow_at_top", "elbow_at_hang",
                 "travel_over_scale", "trunk_max_deg", "frames_per_rep"):
        s = summary[name]
        if not s:
            lines.append(f"{name:24s}  no data")
            continue
        lines.append(f"{name:24s} {s['n']:4d} {s['min']:7.1f} {s['p25']:7.1f} "
                     f"{s['median']:7.1f} {s['p75']:7.1f} {s['max']:7.1f}")
    text = "\n".join(lines)
    (OUT / "pullup_calibration.txt").write_text(text + "\n", encoding="utf-8")

    print()
    print(text)


if __name__ == "__main__":
    main()

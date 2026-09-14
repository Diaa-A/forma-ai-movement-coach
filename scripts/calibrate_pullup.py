"""Can the pull-up thresholds be set from the Penn Action subset?

    python scripts/calibrate_pullup.py

Each sequence's frames are written to a near-lossless video and tracked with the
app's own VIDEO-mode extraction, then taken through the runner's steps --
smoothing, angles, travel with the effort-at-top flip, bottom detection, bottoms
merged without a return, rep segmentation, the rep gate. The Penn Action benchmark
detects pose per image, which is right for pose accuracy but counted differently
from the app on the same clips.

One pass gives:

    resolution  the app's elbow angle against Penn Action ground truth, by
                labelled angle. A gap in the distributions narrower than this is
                not a boundary the pipeline can resolve
    spread      per-rep elbow at the top and at the hang, travel against body
                scale, and trunk lean
    arm gap     left against right elbow, labelled and as tracked, which sets
                ARM_DISAGREEMENT_DEG

Writes report/chapter5/pullup_calibration.json and .txt.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.evaluation.metrics import elbow_angle_from_joints           # noqa: E402
from backend.evaluation.penn_action import (PENN_JOINTS,                 # noqa: E402
                                            find_sequences_by_action,
                                            load_sequence)
from backend.exercises import mechanics, pullup                          # noqa: E402
from backend.pipeline.angles import (ARM_DISAGREEMENT_DEG,               # noqa: E402
                                     _arm_visibility,
                                     pullup_angles_per_frame)
from backend.pipeline.filter import smooth_series                        # noqa: E402
from backend.pipeline.phase_detection import detect_bottoms, segment_reps  # noqa: E402
from backend.pipeline.pose import VISIBILITY_THRESHOLD, extract_landmarks  # noqa: E402
from backend.pipeline.runner import _smooth_landmarks                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "chapter5"
DATA = ROOT / "data" / "penn_action"

# Penn Action does not record frame rate. 30 is the usual capture rate for the
# source footage and only affects the smoothing time constants and the tempo
# figures, not any angle.
NOMINAL_FPS = 30.0

# MediaPipe names the subject's own arms; Penn Action names them as seen from the
# camera, so its "right" elbow is MediaPipe's left.
PENN_SIDE = {"left": "right", "right": "left"}

BANDS = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 181)]


def tracked(paths):
    """Landmarks the way the app gets them, from the frames written to a
    near-lossless video, plus the frame size."""
    h, w = cv2.imread(str(paths[0])).shape[:2]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        video = Path(tmp) / "sequence.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), NOMINAL_FPS, (w, h))
        writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
        for p in paths:
            frame = cv2.imread(str(p))
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
        writer.release()
        return extract_landmarks(video, model="full"), w, h


def measure(seq):
    """Per-rep measurements for one sequence, or None if it cannot be scored."""
    paths = seq.frame_paths()[: seq.nframes]
    if len(paths) < 8:
        return None

    pose, w, h = tracked(paths)
    lm = pose["landmarks"]
    if len(lm) != len(paths) or np.all(np.isfinite(lm[:, :, 0]), axis=1).sum() < 8:
        return None

    ts = pose["timestamps"]
    lm_s = _smooth_landmarks(lm, ts, 1.0, 0.007)
    angles = pullup_angles_per_frame(lm_s)

    travel = mechanics.travel_series(lm_s, "left_hip", "right_hip")
    travel = mechanics.travel_for_phase(pullup.PULLUP, travel)
    travel_s = smooth_series(travel, ts, min_cutoff=0.5, beta=0.001)

    bottoms = detect_bottoms(travel_s, min_separation=max(3, int(NOMINAL_FPS * 0.4)))
    scale = mechanics.body_scale(pullup.PULLUP, lm_s)
    reps_all = segment_reps(
        mechanics.merge_bottoms_without_return(pullup.PULLUP, bottoms, travel_s, scale),
        len(paths))
    side = mechanics.pick_side(pullup.PULLUP, lm_s, reps_all)
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

    # the app's elbow angle against ground truth, and left against right
    errors, labels, gap_labelled, gap_tracked = [], [], [], []
    for i in range(len(paths)):
        vis = seq.gt_visibility(i)
        gt_xy = seq.gt_xy(i)
        labelled = {}
        for arm in ("left", "right"):
            penn = PENN_SIDE[arm]
            if any(vis[PENN_JOINTS.index(f"{penn}_{j}")] < 0.5
                   for j in ("shoulder", "elbow", "wrist")):
                continue
            label = float(elbow_angle_from_joints(gt_xy, penn))
            reading = angles[i][f"elbow_{arm}"]
            if np.isfinite(label):
                labelled[arm] = label
                if np.isfinite(reading):
                    errors.append(float(reading) - label)
                    labels.append(label)
        if len(labelled) == 2:
            gap_labelled.append(abs(labelled["left"] - labelled["right"]))
        a = angles[i]
        if (np.isfinite(a["elbow_left"]) and np.isfinite(a["elbow_right"])
                and _arm_visibility(lm_s[i], "left") >= VISIBILITY_THRESHOLD
                and _arm_visibility(lm_s[i], "right") >= VISIBILITY_THRESHOLD):
            gap_tracked.append(abs(a["elbow_left"] - a["elbow_right"]))

    return {
        "sequence": seq.seq_id,
        "frames": len(paths),
        "frame_size": [w, h],
        "reps_detected": len(reps_all),
        "reps_kept": len(reps),
        "side": side,
        "scale": float(scale) if scale else None,
        "per_rep": per_rep,
        "elbow_error_deg": errors,
        "elbow_label_deg": labels,
        "elbow_gap_labelled_deg": gap_labelled,
        "elbow_gap_tracked_deg": gap_tracked,
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
    errors = [e for r in rows for e in r["elbow_error_deg"]]
    labels = [t for r in rows for t in r["elbow_label_deg"]]
    gap_labelled = [g for r in rows for g in r["elbow_gap_labelled_deg"]]
    gap_tracked = [g for r in rows for g in r["elbow_gap_tracked_deg"]]

    bands = []
    for lo, hi in BANDS:
        signed = np.array([e for e, t in zip(errors, labels) if lo <= t < hi])
        bands.append({"labelled": f"{lo}-{hi}", "n": int(signed.size),
                      "median_signed": round(float(np.median(signed)), 1) if signed.size else None,
                      "median_abs": round(float(np.median(np.abs(signed))), 1) if signed.size else None})

    summary = {
        "sequences_total": len(ids),
        "sequences_measured": len(rows),
        "sequences_skipped": skipped,
        "reps_scored": len(all_rep),
        "clips_with_multiple_reps": sum(1 for r in rows if len(r["per_rep"]) > 1),
        "elbow_angle_error_deg": stats([abs(e) for e in errors]),
        "elbow_error_by_labelled_angle": bands,
        "elbow_at_top": stats([x["elbow_at_top"] for x in all_rep]),
        "elbow_at_hang": stats([x["elbow_at_hang"] for x in all_rep]),
        "travel_over_scale": stats([x["travel_over_scale"] for x in all_rep]),
        "trunk_max_deg": stats([x["trunk_max"] for x in all_rep]),
        "frames_per_rep": stats([x["frames"] for x in all_rep]),
        "elbow_gap_labelled_deg": stats(gap_labelled),
        "elbow_gap_tracked_deg": stats(gap_tracked),
        "tracked_frames_past_arm_guard": sum(1 for g in gap_tracked if g > ARM_DISAGREEMENT_DEG),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pullup_calibration.json").write_text(
        json.dumps({"summary": summary, "per_sequence": rows}, indent=2),
        encoding="utf-8")

    lines = ["Pull-up threshold calibration - Penn Action subset, tracked as the app does", ""]
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
                 "travel_over_scale", "trunk_max_deg", "frames_per_rep",
                 "elbow_gap_labelled_deg", "elbow_gap_tracked_deg"):
        s = summary[name]
        if not s:
            lines.append(f"{name:24s}  no data")
            continue
        lines.append(f"{name:24s} {s['n']:4d} {s['min']:7.1f} {s['p25']:7.1f} "
                     f"{s['median']:7.1f} {s['p75']:7.1f} {s['max']:7.1f}")
    lines.append("")
    lines.append(f"tracked frames with both arms confident and more than "
                 f"{ARM_DISAGREEMENT_DEG:.0f} degrees apart: "
                 f"{summary['tracked_frames_past_arm_guard']} of {len(gap_tracked)}")
    lines.append("")
    lines.append("elbow error by labelled angle, app minus label")
    lines.append(f"{'labelled':10s} {'n':>5s} {'median':>8s} {'abs':>6s}")
    for b in bands:
        if b["n"]:
            lines.append(f"{b['labelled']:10s} {b['n']:5d} {b['median_signed']:+8.1f} "
                         f"{b['median_abs']:6.1f}")
    text = "\n".join(lines)
    (OUT / "pullup_calibration.txt").write_text(text + "\n", encoding="utf-8")

    print()
    print(text)


if __name__ == "__main__":
    main()

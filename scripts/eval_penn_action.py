"""Penn Action 2D evaluation harness.

Validates MediaPipe pose accuracy against Penn Action ground-truth annotations
for squat clips. Produces the quantitative numbers Chapter 4 cites: per-joint
pixel error, PCK@0.2, and (most importantly for coaching) knee/hip angle error.

Typical use :
    # one-shot: extract squats from the downloaded tar, then evaluate
    python scripts/eval_penn_action.py --tar data/Penn_Action.tar.gz \\
        --limit 25 --sanity

    # if already extracted to data/penn_action/, skip the tar
    python scripts/eval_penn_action.py --data-root data/penn_action --limit 25

Outputs to data/outputs/penn_eval/:
    metrics.json        aggregate + per-sequence numbers
    per_joint_error.png bar chart
    angle_error.png     knee/hip angle-error histogram
    sanity_<seq>.jpg    GT (red) vs MediaPipe (green) overlay for visual check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.pipeline.pose import extract_landmarks_from_frames
from backend.evaluation.penn_action import (
    PENN_JOINTS, PENN_TO_MP, load_sequence, find_sequences_by_action,
    extract_squats_from_tar,
)
from backend.evaluation.metrics import Accumulator, per_joint_pixel_error


def mp_to_penn_pixels(mp_frame, w, h):
    """Map a MediaPipe (33,4) frame to Penn Action (13,2) pixel coords."""
    out = np.full((len(PENN_JOINTS), 2), np.nan)
    for j, name in enumerate(PENN_JOINTS):
        idx = PENN_TO_MP[name]
        x, y = mp_frame[idx, 0], mp_frame[idx, 1]
        if np.isfinite(x) and np.isfinite(y):
            out[j] = [x * w, y * h]
    return out


def _swap_lr(xy):
    """Swap left/right joints — used to detect a L/R convention mismatch."""
    swapped = xy.copy()
    pairs = [("left_shoulder", "right_shoulder"), ("left_elbow", "right_elbow"),
             ("left_wrist", "right_wrist"), ("left_hip", "right_hip"),
             ("left_knee", "right_knee"), ("left_ankle", "right_ankle")]
    idx = {n: i for i, n in enumerate(PENN_JOINTS)}
    for a, b in pairs:
        swapped[[idx[a], idx[b]]] = swapped[[idx[b], idx[a]]]
    return swapped


def evaluate_sequence(seq, model, frame_stride, mp_cache=None):
    """Run MediaPipe over a sequence's frames, return (Accumulator, raw_records)."""
    frame_paths = seq.frame_paths()
    n = min(len(frame_paths), seq.nframes)
    idxs = list(range(0, n, frame_stride))
    chosen = [frame_paths[i] for i in idxs]

    pose = extract_landmarks_from_frames(chosen, model=model)

    acc = Accumulator()
    acc_swapped = Accumulator()
    for k, i in enumerate(idxs):
        w, h = pose["sizes"][k]
        if w == 0:
            continue
        mp_frame = pose["landmarks"][k]
        detected = bool(np.all(np.isfinite(mp_frame[:, 0])))
        pred = mp_to_penn_pixels(mp_frame, w, h)
        gt = seq.gt_xy(i)
        vis = seq.gt_visibility(i)
        ref = seq.torso_diag(i)
        acc.add_frame(pred, gt, vis, ref, detected=detected)
        acc_swapped.add_frame(_swap_lr(pred), gt, vis, ref, detected=detected)
    return acc, acc_swapped, pose, idxs


def save_sanity_overlay(seq, pose, idxs, out_path):
    """Overlay GT (red) and MediaPipe-mapped (green) joints on a mid-sequence
    frame so the joint mapping can be visually confirmed."""
    mid = len(idxs) // 2
    frame_i = idxs[mid]
    img = cv2.imread(str(seq.frame_paths()[frame_i]))
    if img is None:
        return None
    w, h = pose["sizes"][mid]
    pred = mp_to_penn_pixels(pose["landmarks"][mid], w, h)
    gt = seq.gt_xy(frame_i)
    vis = seq.gt_visibility(frame_i)
    for j in range(len(PENN_JOINTS)):
        if vis[j] >= 0.5 and np.all(np.isfinite(gt[j])):
            cv2.circle(img, (int(gt[j][0]), int(gt[j][1])), 6, (0, 0, 220), -1)
        if np.all(np.isfinite(pred[j])):
            cv2.circle(img, (int(pred[j][0]), int(pred[j][1])), 4, (0, 220, 0), -1)
    cv2.putText(img, "GT=red  MediaPipe=green", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)
    return out_path


def _num(value, spec=".1f", missing="n/a"):
    """Format a metric that may legitimately be absent.

    A sequence where MediaPipe detected nothing has no MPJPE — not zero, not an
    error, simply nothing to average. Every squat sequence produced detections so
    this never came up; the first push-up run hit it immediately and took the
    whole evaluation down at the print statement, after all the expensive work.
    Absent is a real outcome here and the harness has to be able to say so.
    """
    if value is None:
        return missing
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return missing


def _default_output(action):
    return Path("data/outputs/penn_eval" if action == "squat"
                else f"data/outputs/penn_eval_{action}")


# which joint angle actually matters for each exercise -- the squat is judged on
# the knee, the push-up on the elbow. Anything unlisted falls back to all of them.
ANGLES_OF_INTEREST = {
    "squat": ("knee", "hip"),
    "pushup": ("elbow",),
}


def plot_per_joint(summary, out_path, action="squat"):
    joints = [j for j in PENN_JOINTS if summary["per_joint_pixel_error"].get(j) is not None]
    vals = [summary["per_joint_pixel_error"][j] for j in joints]
    plt.figure(figsize=(10, 4))
    plt.bar(joints, vals, color="#3b7dd8")
    plt.ylabel("mean pixel error")
    plt.title(f"MediaPipe vs Penn Action — per-joint 2D error ({action})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_angle_hist(all_angle_errs, out_path, action="squat"):
    if not all_angle_errs:
        return
    plt.figure(figsize=(7, 4))
    plt.hist(all_angle_errs, bins=30, color="#46a35e", edgecolor="white")
    plt.xlabel("absolute angle error (degrees)")
    plt.ylabel("frame count")
    names = " + ".join(n.capitalize() for n in ANGLES_OF_INTEREST.get(action, ("joint",)))
    plt.title(f"{names} angle error: MediaPipe vs Penn Action ({action})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Penn Action 2D evaluation.")
    ap.add_argument("--tar", default="", help="path to Penn_Action.tar.gz "
                    "(extracts squat subset if --data-root not yet populated)")
    ap.add_argument("--data-root", default="data/penn_action",
                    help="extracted Penn_Action root (frames/ + labels/)")
    ap.add_argument("--action", default="squat")
    ap.add_argument("--limit", type=int, default=25, help="max sequences")
    ap.add_argument("--frame-stride", type=int, default=2,
                    help="evaluate every Nth frame (speed)")
    ap.add_argument("--model", default="full", choices=["lite", "full", "heavy"])
    # Empty means "derive from the action". The squat keeps the original path
    # because the report figures already reference it; anything else gets its own
    # directory. Running push-up with the squat's default would have overwritten
    # metrics.json and all three figures that Chapter 4 cites.
    ap.add_argument("--output", default="",
                    help="results directory (default: data/outputs/penn_eval "
                         "for squat, penn_eval_<action> otherwise)")
    ap.add_argument("--sanity", action="store_true",
                    help="save GT-vs-prediction overlay images")
    ap.add_argument("--delete-tar", action="store_true",
                    help="delete the tar after extracting the subset")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.output) if args.output else _default_output(args.action)
    out_dir.mkdir(parents=True, exist_ok=True)

    # refuse to write one action's results over another's
    existing = out_dir / "metrics.json"
    if existing.is_file():
        try:
            prev = json.loads(existing.read_text()).get("action")
        except (json.JSONDecodeError, OSError):
            prev = None
        if prev and prev != args.action:
            print(f"error: {out_dir} holds '{prev}' results; refusing to "
                  f"overwrite them with '{args.action}'. Pass --output.",
                  file=sys.stderr)
            return 1

    # Extract when this ACTION is missing, not merely when labels/ is empty. The
    # old check looked for any .mat at all, so once the squat subset was present
    # a --tar run for a different action silently skipped extraction and then
    # reported "no sequences found".
    if args.tar and not find_sequences_by_action(data_root, args.action):
        print(f"[+] extracting '{args.action}' clips from {args.tar} ...")
        ids = extract_squats_from_tar(args.tar, data_root,
                                      action_substr=args.action,
                                      max_sequences=args.limit)
        print(f"    extracted {len(ids)} sequence(s): {ids}")
        if args.delete_tar:
            Path(args.tar).unlink(missing_ok=True)
            print(f"    deleted {args.tar}")

    seq_ids = find_sequences_by_action(data_root, args.action)[:args.limit]
    if not seq_ids:
        print(f"error: no '{args.action}' sequences found under {data_root}",
              file=sys.stderr)
        return 1
    print(f"[+] evaluating {len(seq_ids)} sequence(s) (model={args.model}, "
          f"stride={args.frame_stride})")

    overall = Accumulator()
    overall_swapped = Accumulator()
    per_seq = {}
    all_angle_errs = []

    for sid in seq_ids:
        seq = load_sequence(sid, data_root)
        acc, acc_sw, pose, idxs = evaluate_sequence(
            seq, args.model, args.frame_stride)
        s = acc.summary()
        per_seq[sid] = s
        # fold into overall
        for n, lst in acc.joint_errors.items():
            overall.joint_errors[n].extend(lst)
        overall.pck_correct += acc.pck_correct
        overall.pck_total += acc.pck_total
        overall.frames_evaluated += acc.frames_evaluated
        overall.frames_detected += acc.frames_detected
        wanted = ANGLES_OF_INTEREST.get(args.action)
        for k, v in acc.angle_errors.items():
            overall.angle_errors.setdefault(k, []).extend(v)
            # the histogram is the headline figure, so it shows only the joint
            # this exercise is actually judged on; the full per-joint numbers stay
            # in metrics.json either way
            if wanted is None or k.rsplit("_", 1)[0] in wanted:
                all_angle_errs.extend(v)
        for n, lst in acc_sw.joint_errors.items():
            overall_swapped.joint_errors[n].extend(lst)

        print(f"    {sid}: MPJPE={_num(s['mpjpe_2d_pixels'])}px  "
              f"PCK@0.2={_num(s['pck@0.2'], '.2f')}  "
              f"detect={_num(s['detection_rate'], '.2f')}  "
              f"frames={s['frames_evaluated']}")

        if args.sanity:
            save_sanity_overlay(seq, pose, idxs, out_dir / f"sanity_{sid}.jpg")

    dead = [sid for sid, s in per_seq.items() if s["frames_detected"] == 0]
    if dead:
        print(f"\n[!] {len(dead)} sequence(s) produced no detections at all: "
              f"{', '.join(dead)}")
        print("    These still count against the detection rate, which is the "
              "honest treatment -- they are clips the system could not use.")

    summary = overall.summary()

    # L/R convention check
    sw_all = [e for v in overall_swapped.joint_errors.values() for e in v]
    direct_all = [e for v in overall.joint_errors.values() for e in v]
    if sw_all and direct_all:
        sw_mean = float(np.mean(sw_all))
        summary["lr_swapped_mpjpe_pixels"] = sw_mean
        if sw_mean < 0.8 * summary["mpjpe_2d_pixels"]:
            summary["WARNING"] = ("L/R swap gives much lower error — the joint "
                                  "left/right convention may be mismatched. "
                                  "Check a sanity overlay.")

    with open(out_dir / "metrics.json", "w") as fh:
        json.dump({
            # recorded so a later run for a different exercise can refuse to
            # overwrite these results, and so a figure can be traced back to
            # what produced it
            "action": args.action,
            "sequences": len(seq_ids),
            "model": args.model,
            "frame_stride": args.frame_stride,
            "aggregate": summary,
            "per_sequence": per_seq,
        }, fh, indent=2)

    plot_per_joint(summary, out_dir / "per_joint_error.png", args.action)
    plot_angle_hist(all_angle_errs, out_dir / "angle_error.png", args.action)

    # ---- console summary
    print("\n=== aggregate ===")
    print(f"  sequences:        {len(seq_ids)}")
    print(f"  frames evaluated: {summary['frames_evaluated']}")
    print(f"  detection rate:   {_num(summary['detection_rate'], '.1%')}")
    print(f"  MPJPE (2D):       {_num(summary['mpjpe_2d_pixels'])} px "
          f"(median {_num(summary['mpjpe_2d_median'])})")
    print(f"  PCK@0.2:          {_num(summary['pck@0.2'], '.3f')}")
    print("  angle error (deg):")
    for k, v in summary["angle_error_degrees"].items():
        print(f"     {k:12s} mean={_num(v['mean'])}  "
              f"median={_num(v['median'])}  n={v['n']}")
    if "WARNING" in summary:
        print(f"  [!] {summary['WARNING']}")
    print(f"\noutputs: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

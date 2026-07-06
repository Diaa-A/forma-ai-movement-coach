"""Evaluation metrics: MediaPipe predictions vs Penn Action ground truth.

Three families:
    1. Per-joint pixel error (Euclidean distance, GT-visible joints only)
    2. PCK@alpha — Percentage of Correct Keypoints within alpha * reference
       length. The standard Penn Action metric. alpha=0.2 is conventional.
    3. Joint-angle error — knee and hip angles derived from both predicted and
       GT joints, compared in degrees. This is the metric that actually matters
       for coaching, since the cue layer keys off angles, not raw pixels.

All functions take per-frame arrays and ignore joints flagged invisible in the
ground truth.
"""
from __future__ import annotations

import numpy as np

from ..pipeline.angles import joint_angle
from .penn_action import PENN_JOINTS


_J = {n: i for i, n in enumerate(PENN_JOINTS)}


def per_joint_pixel_error(pred_xy, gt_xy, gt_vis):
    """Euclidean pixel error per joint for one frame.

    pred_xy, gt_xy: (13, 2). gt_vis: (13,). Returns (13,) with NaN where the
    joint is invisible in GT or the prediction is missing.
    """
    err = np.full(len(PENN_JOINTS), np.nan)
    for j in range(len(PENN_JOINTS)):
        if gt_vis[j] < 0.5:
            continue
        p = pred_xy[j]
        g = gt_xy[j]
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(g))):
            continue
        err[j] = float(np.hypot(p[0] - g[0], p[1] - g[1]))
    return err


def pck_frame(pred_xy, gt_xy, gt_vis, ref_len, alpha=0.2):
    """Per-frame PCK: fraction of visible joints within alpha*ref_len.

    Returns (n_correct, n_evaluated). Aggregate across frames by summing.
    """
    if ref_len <= 1e-6:
        return 0, 0
    thresh = alpha * ref_len
    correct = 0
    total = 0
    errs = per_joint_pixel_error(pred_xy, gt_xy, gt_vis)
    for e in errs:
        if np.isnan(e):
            continue
        total += 1
        if e <= thresh:
            correct += 1
    return correct, total


def knee_angle_from_joints(xy, side):
    """Knee angle (hip-knee-ankle) from a (13,2) joint array. side: 'left'/'right'."""
    hip = xy[_J[f"{side}_hip"]]
    knee = xy[_J[f"{side}_knee"]]
    ankle = xy[_J[f"{side}_ankle"]]
    if not all(np.all(np.isfinite(p)) for p in (hip, knee, ankle)):
        return float("nan")
    return joint_angle(hip, knee, ankle)


def hip_angle_from_joints(xy, side):
    """Hip angle (shoulder-hip-knee) from a (13,2) joint array."""
    sh = xy[_J[f"{side}_shoulder"]]
    hip = xy[_J[f"{side}_hip"]]
    knee = xy[_J[f"{side}_knee"]]
    if not all(np.all(np.isfinite(p)) for p in (sh, hip, knee)):
        return float("nan")
    return joint_angle(sh, hip, knee)


def angle_errors(pred_xy, gt_xy, gt_vis):
    """Knee + hip angle error (degrees) for one frame, both sides.

    Only computed when all three joints of the triplet are GT-visible AND
    predicted. Returns a dict of {name: abs_error_deg} (missing -> absent).
    """
    out = {}
    for side in ("left", "right"):
        for name, fn in (("knee", knee_angle_from_joints),
                         ("hip",  hip_angle_from_joints)):
            triplet = _triplet_joints(name, side)
            if any(gt_vis[_J[j]] < 0.5 for j in triplet):
                continue
            a_pred = fn(pred_xy, side)
            a_gt = fn(gt_xy, side)
            if np.isfinite(a_pred) and np.isfinite(a_gt):
                out[f"{name}_{side}"] = abs(a_pred - a_gt)
    return out


def _triplet_joints(name, side):
    if name == "knee":
        return [f"{side}_hip", f"{side}_knee", f"{side}_ankle"]
    return [f"{side}_shoulder", f"{side}_hip", f"{side}_knee"]


class Accumulator:
    """Running aggregation across frames/sequences."""

    def __init__(self):
        self.joint_errors = {n: [] for n in PENN_JOINTS}
        self.angle_errors = {}
        self.pck_correct = 0
        self.pck_total = 0
        self.frames_evaluated = 0
        self.frames_detected = 0

    def add_frame(self, pred_xy, gt_xy, gt_vis, ref_len, alpha=0.2,
                  detected=True):
        self.frames_evaluated += 1
        if detected:
            self.frames_detected += 1
        errs = per_joint_pixel_error(pred_xy, gt_xy, gt_vis)
        for j, n in enumerate(PENN_JOINTS):
            if not np.isnan(errs[j]):
                self.joint_errors[n].append(errs[j])
        c, t = pck_frame(pred_xy, gt_xy, gt_vis, ref_len, alpha)
        self.pck_correct += c
        self.pck_total += t
        for k, v in angle_errors(pred_xy, gt_xy, gt_vis).items():
            self.angle_errors.setdefault(k, []).append(v)

    def summary(self):
        joint_means = {
            n: (float(np.mean(v)) if v else None)
            for n, v in self.joint_errors.items()
        }
        all_errs = [e for v in self.joint_errors.values() for e in v]
        angle_means = {
            k: {"mean": float(np.mean(v)), "median": float(np.median(v)),
                "n": len(v)}
            for k, v in self.angle_errors.items()
        }
        return {
            "frames_evaluated": self.frames_evaluated,
            "frames_detected": self.frames_detected,
            "detection_rate": (self.frames_detected / self.frames_evaluated
                               if self.frames_evaluated else 0.0),
            "mpjpe_2d_pixels": float(np.mean(all_errs)) if all_errs else None,
            "mpjpe_2d_median": float(np.median(all_errs)) if all_errs else None,
            "per_joint_pixel_error": joint_means,
            "pck@0.2": (self.pck_correct / self.pck_total
                        if self.pck_total else None),
            "pck_evaluated_joints": self.pck_total,
            "angle_error_degrees": angle_means,
        }

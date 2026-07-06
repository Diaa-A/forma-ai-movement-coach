"""Joint angle calculations from 2D landmark coordinates.

Angles use the standard 3-point formula:
    angle at B = arccos( (BA · BC) / (|BA| * |BC|) )

For the squat we expose knee, hip, spine, and ankle angles for both sides — the
exercise module picks which side to use based on visibility.
"""
import numpy as np
from .pose import LM


def joint_angle(a, b, c):
    """Angle in degrees at vertex b, formed by points a and c. 2D (x, y)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    ba = a - b
    bc = c - b
    na = np.linalg.norm(ba)
    nb = np.linalg.norm(bc)
    if na == 0 or nb == 0 or not np.isfinite(na) or not np.isfinite(nb):
        return float("nan")
    cosine = np.dot(ba, bc) / (na * nb)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def angle_from_vertical(p_top, p_bottom):
    """Angle (degrees) of the segment p_top->p_bottom from the image vertical.
    0 = vertical, increasing = more tilt. Lets a limb (e.g. the shin) be compared
    against the trunk on the same scale for the trunk/shin parallelism check."""
    v = np.asarray(p_bottom, dtype=np.float64) - np.asarray(p_top, dtype=np.float64)
    n = np.linalg.norm(v)
    if n == 0 or not np.all(np.isfinite(v)):
        return float("nan")
    cosine = float(np.clip(np.dot(v, [0.0, 1.0]) / n, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _xy(landmarks_frame, idx):
    """Pull (x, y) from a single-frame landmark array. Returns NaNs if missing."""
    row = landmarks_frame[idx]
    return row[0], row[1]


def squat_angles_per_frame(landmarks):
    """Compute per-frame squat angle dict.

    `landmarks` has shape (n_frames, 33, 4). Returns a list of dicts, one per frame,
    keyed by: knee_left, knee_right, hip_left, hip_right, spine, ankle_left,
    ankle_right. NaN where the underlying landmarks are missing.
    """
    out = []
    for f in range(len(landmarks)):
        fr = landmarks[f]

        # knee = hip-knee-ankle
        knee_l = joint_angle(_xy(fr, LM["left_hip"]),  _xy(fr, LM["left_knee"]),  _xy(fr, LM["left_ankle"]))
        knee_r = joint_angle(_xy(fr, LM["right_hip"]), _xy(fr, LM["right_knee"]), _xy(fr, LM["right_ankle"]))

        # hip = shoulder-hip-knee
        hip_l = joint_angle(_xy(fr, LM["left_shoulder"]),  _xy(fr, LM["left_hip"]),  _xy(fr, LM["left_knee"]))
        hip_r = joint_angle(_xy(fr, LM["right_shoulder"]), _xy(fr, LM["right_hip"]), _xy(fr, LM["right_knee"]))

        # ankle = knee-ankle-foot_index (dorsiflexion proxy)
        ank_l = joint_angle(_xy(fr, LM["left_knee"]),  _xy(fr, LM["left_ankle"]),  _xy(fr, LM["left_foot_index"]))
        ank_r = joint_angle(_xy(fr, LM["right_knee"]), _xy(fr, LM["right_ankle"]), _xy(fr, LM["right_foot_index"]))

        # spine angle relative to vertical, using mid-shoulder -> mid-hip
        ls = np.array(_xy(fr, LM["left_shoulder"]))
        rs = np.array(_xy(fr, LM["right_shoulder"]))
        lh = np.array(_xy(fr, LM["left_hip"]))
        rh = np.array(_xy(fr, LM["right_hip"]))
        mid_sh = (ls + rs) / 2.0
        mid_hp = (lh + rh) / 2.0
        spine_vec = mid_sh - mid_hp
        # angle from vertical (image y axis points DOWN, so "up" is -y)
        # we measure the lean from upright: 0 = perfectly upright, larger = more lean
        if np.linalg.norm(spine_vec) == 0 or not np.all(np.isfinite(spine_vec)):
            spine = float("nan")
        else:
            up = np.array([0.0, -1.0])
            cosine = np.dot(spine_vec, up) / (np.linalg.norm(spine_vec) * 1.0)
            cosine = float(np.clip(cosine, -1.0, 1.0))
            spine = float(np.degrees(np.arccos(cosine)))

        # shin lean from vertical (knee->ankle), per side — compared against the
        # trunk lean for the parallelism-based forward-lean cue
        shin_l = angle_from_vertical(_xy(fr, LM["left_knee"]),  _xy(fr, LM["left_ankle"]))
        shin_r = angle_from_vertical(_xy(fr, LM["right_knee"]), _xy(fr, LM["right_ankle"]))

        out.append({
            "knee_left":  knee_l, "knee_right":  knee_r,
            "hip_left":   hip_l,  "hip_right":   hip_r,
            "ankle_left": ank_l,  "ankle_right": ank_r,
            "shin_left":  shin_l, "shin_right":  shin_r,
            "spine":      spine,
        })
    return out

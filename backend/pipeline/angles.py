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


def _signed_body_line(shoulder, hip, ankle):
    """How far the hip sits off the straight shoulder-to-ankle line, and which way.

    A push-up is judged on whether the body holds a plank. The interior angle at
    the hip alone cannot say what went wrong: sagging and piking both bend it the
    same amount, and they need opposite corrections. So the perpendicular offset
    of the hip from the shoulder-ankle line is measured instead, and signed by the
    cross product.

    Returns degrees of deviation from straight, **positive when the hips are LOW
    (sagging) and negative when they are HIGH (piked)**. Sag is the far more
    common fault, so it gets the positive direction. Expressed as an angle rather
    than a distance so it does not change meaning with camera distance.

    Image coordinates have y increasing downward, which is easy to get backwards
    here — the direction is pinned by a test rather than by reading the sign off
    the cross product and hoping.
    """
    s = np.asarray(shoulder, dtype=np.float64)
    h = np.asarray(hip, dtype=np.float64)
    a = np.asarray(ankle, dtype=np.float64)
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(h)) and np.all(np.isfinite(a))):
        return float("nan")

    interior = joint_angle(s, h, a)
    if not np.isfinite(interior):
        return float("nan")
    deviation = 180.0 - interior          # 0 when perfectly straight

    # cross product of shoulder->ankle with shoulder->hip. y points down, so a
    # negative z means the hip lies above the line: piked.
    line = a - s
    to_hip = h - s
    cross = line[0] * to_hip[1] - line[1] * to_hip[0]
    if not np.isfinite(cross) or cross == 0:
        return deviation
    return -deviation if cross < 0 else deviation


def pushup_angles_per_frame(landmarks):
    """Per-frame push-up angle dict.

    Same shape and conventions as `squat_angles_per_frame`, and built from the
    same primitives — only the joints of interest differ. Keys:

        elbow_left / elbow_right   shoulder-elbow-wrist, the angle that flexes
        body_left / body_right     signed hip deviation from the shoulder-ankle
                                   line: positive sagging, negative piked
        shoulder_left / shoulder_right
                                   elbow-shoulder-hip, how far the upper arm is
                                   from the torso (flared elbows read wide)
        neck                       head deviation from the shoulder-hip line,
                                   for the dropped- or craned-head cue
        spine                      trunk angle from vertical, carried over so the
                                   shared validity gate has something to check
    """
    out = []
    for f in range(len(landmarks)):
        fr = landmarks[f]

        elbow_l = joint_angle(_xy(fr, LM["left_shoulder"]),  _xy(fr, LM["left_elbow"]),  _xy(fr, LM["left_wrist"]))
        elbow_r = joint_angle(_xy(fr, LM["right_shoulder"]), _xy(fr, LM["right_elbow"]), _xy(fr, LM["right_wrist"]))

        body_l = _signed_body_line(_xy(fr, LM["left_shoulder"]),  _xy(fr, LM["left_hip"]),  _xy(fr, LM["left_ankle"]))
        body_r = _signed_body_line(_xy(fr, LM["right_shoulder"]), _xy(fr, LM["right_hip"]), _xy(fr, LM["right_ankle"]))

        sh_l = joint_angle(_xy(fr, LM["left_elbow"]),  _xy(fr, LM["left_shoulder"]),  _xy(fr, LM["left_hip"]))
        sh_r = joint_angle(_xy(fr, LM["right_elbow"]), _xy(fr, LM["right_shoulder"]), _xy(fr, LM["right_hip"]))

        # neck: nose-shoulder-hip. The ear would be the textbook landmark for head
        # carriage, but this project's LM map is a curated subset that stops at the
        # nose, and adding two landmarks to a shared module for one cue is not a
        # trade worth making. The nose sits further forward than the ear so the
        # absolute value differs; the cue is calibrated on what this measures, not
        # on a number borrowed from elsewhere.
        ls = np.array(_xy(fr, LM["left_shoulder"]))
        rs = np.array(_xy(fr, LM["right_shoulder"]))
        lh = np.array(_xy(fr, LM["left_hip"]))
        rh = np.array(_xy(fr, LM["right_hip"]))
        mid_sh = (ls + rs) / 2.0
        mid_hp = (lh + rh) / 2.0
        neck = joint_angle(_xy(fr, LM["nose"]), mid_sh, mid_hp)

        spine_vec = mid_sh - mid_hp
        if np.linalg.norm(spine_vec) == 0 or not np.all(np.isfinite(spine_vec)):
            spine = float("nan")
        else:
            up = np.array([0.0, -1.0])
            cosine = float(np.clip(np.dot(spine_vec, up) / np.linalg.norm(spine_vec), -1.0, 1.0))
            spine = float(np.degrees(np.arccos(cosine)))

        out.append({
            "elbow_left":    elbow_l, "elbow_right":    elbow_r,
            "body_left":     body_l,  "body_right":     body_r,
            "shoulder_left": sh_l,    "shoulder_right": sh_r,
            "neck":          neck,
            "spine":         spine,
        })
    return out


def pullup_angles_per_frame(landmarks):
    """Per-frame pull-up angle dict.

    Same primitives again, and a shorter list than the push-up's because a
    hanging body offers less that is worth measuring. Keys:

        elbow_left / elbow_right   shoulder-elbow-wrist, the angle that flexes.
                                   Unlike the push-up this reaches its MINIMUM at
                                   the top of the movement, with the chin at the
                                   bar
        shoulder_left / shoulder_right
                                   elbow-shoulder-hip, how far the upper arm sits
                                   from the torso. Wide on a wide grip, narrow on
                                   a chin-up
        trunk                      torso angle from image vertical. A hanging
                                   body reads near zero; swing and kipping move
                                   it, which is the only handle on either that a
                                   single frame gives

    No body-line key on purpose. The push-up's shoulder-hip-ankle measure asks
    whether the body held a plank, and a pull-up with the knees tucked -- which is
    how most people hang -- would read as a severe fault while being perfectly
    correct.
    """
    out = []
    for f in range(len(landmarks)):
        fr = landmarks[f]

        elbow_l = joint_angle(_xy(fr, LM["left_shoulder"]),  _xy(fr, LM["left_elbow"]),  _xy(fr, LM["left_wrist"]))
        elbow_r = joint_angle(_xy(fr, LM["right_shoulder"]), _xy(fr, LM["right_elbow"]), _xy(fr, LM["right_wrist"]))

        sh_l = joint_angle(_xy(fr, LM["left_elbow"]),  _xy(fr, LM["left_shoulder"]),  _xy(fr, LM["left_hip"]))
        sh_r = joint_angle(_xy(fr, LM["right_elbow"]), _xy(fr, LM["right_shoulder"]), _xy(fr, LM["right_hip"]))

        ls = np.array(_xy(fr, LM["left_shoulder"]))
        rs = np.array(_xy(fr, LM["right_shoulder"]))
        lh = np.array(_xy(fr, LM["left_hip"]))
        rh = np.array(_xy(fr, LM["right_hip"]))
        mid_sh = (ls + rs) / 2.0
        mid_hp = (lh + rh) / 2.0

        trunk = angle_from_vertical(mid_sh, mid_hp)

        out.append({
            "elbow_left":    elbow_l, "elbow_right":    elbow_r,
            "shoulder_left": sh_l,    "shoulder_right": sh_r,
            "trunk":         trunk,
        })
    return out

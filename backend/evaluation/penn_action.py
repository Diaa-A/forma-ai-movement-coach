"""Penn Action dataset loader + MediaPipe joint correspondence.

Penn Action ships per-sequence:
    frames/<seq>/NNNNNN.jpg        extracted video frames
    labels/<seq>.mat              ground-truth 2D joint annotations + metadata

Each label .mat (MATLAB struct) holds:
    action       e.g. 'squat'
    pose         camera view string
    x, y         (nframes x 13) joint pixel coordinates
    visibility   (nframes x 13) 1 = labelled/visible
    bbox         (nframes x 4)  [x1, y1, x2, y2]
    dimensions   [H, W, nframes]
    nframes, train

We use this for 2D landmark accuracy validation: run MediaPipe on the frames,
compare against these annotations. See `metrics.py` for the error calculations.

Joint-order note: the 13 Penn Action joints are validated empirically by
overlaying ground truth on a frame (scripts/eval_penn_action.py --sanity) before
any metric is trusted — the left/right convention (subject vs image) is the
classic gotcha and we verify rather than assume.
"""
from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.io import loadmat


# Canonical 13-joint order for Penn Action labels.
PENN_JOINTS = [
    "head",
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_wrist",    "right_wrist",
    "left_hip",      "right_hip",
    "left_knee",     "right_knee",
    "left_ankle",    "right_ankle",
]

# Map each Penn Action joint to the MediaPipe landmark index that represents it.
# 'head' has no exact MP equivalent — nose (0) is the standard proxy and is
# excluded from angle calculations (only used for completeness in pixel error).
#
# LEFT/RIGHT CONVENTION (important): Penn Action labels joints by the mirrored
# (image-observer) convention, whereas MediaPipe labels them anatomically (the
# subject's own left/right). They are therefore opposite. This was verified
# empirically over the squat subset: a left/right swap lowered per-joint error
# on 23 of 25 clips (the 2 exceptions were near-symmetric side-views where the
# left/right joints overlap in image space, so the swap is immaterial). We align
# the conventions ONCE, globally, below — Penn's "left_*" maps to MediaPipe's
# "right_*" index and vice versa. This is a fixed convention alignment applied
# uniformly to every clip, not per-clip tuning against ground truth.
from ..pipeline.pose import LM as MP_LM

PENN_TO_MP = {
    "head":           MP_LM["nose"],
    "left_shoulder":  MP_LM["right_shoulder"],
    "right_shoulder": MP_LM["left_shoulder"],
    "left_elbow":     MP_LM["right_elbow"],
    "right_elbow":    MP_LM["left_elbow"],
    "left_wrist":     MP_LM["right_wrist"],
    "right_wrist":    MP_LM["left_wrist"],
    "left_hip":       MP_LM["right_hip"],
    "right_hip":      MP_LM["left_hip"],
    "left_knee":      MP_LM["right_knee"],
    "right_knee":     MP_LM["left_knee"],
    "left_ankle":     MP_LM["right_ankle"],
    "right_ankle":    MP_LM["left_ankle"],
}


class PennSequence:
    """One Penn Action clip: frame paths + ground-truth joint arrays."""

    def __init__(self, seq_id, frames_dir, action, x, y, visibility, bbox, dims):
        self.seq_id = seq_id
        self.frames_dir = Path(frames_dir)
        self.action = action
        self.x = x                  # (nframes, 13)
        self.y = y                  # (nframes, 13)
        self.visibility = visibility
        self.bbox = bbox            # (nframes, 4)
        self.dims = dims            # [H, W, nframes]

    @property
    def nframes(self):
        return self.x.shape[0]

    def frame_paths(self) -> List[Path]:
        # frames are 000001.jpg ... numbered 1-based, contiguous
        paths = sorted(self.frames_dir.glob("*.jpg"))
        return paths

    def gt_xy(self, frame_idx) -> np.ndarray:
        """Ground-truth (13, 2) pixel coords for one frame."""
        return np.stack([self.x[frame_idx], self.y[frame_idx]], axis=1)

    def gt_visibility(self, frame_idx) -> np.ndarray:
        return self.visibility[frame_idx]

    def torso_diag(self, frame_idx) -> float:
        """Reference length for PCK normalisation — shoulder-to-opposite-hip
        diagonal. Falls back to bbox diagonal if those joints are unlabelled."""
        js = {n: i for i, n in enumerate(PENN_JOINTS)}
        ls = np.array([self.x[frame_idx, js["left_shoulder"]],
                       self.y[frame_idx, js["left_shoulder"]]])
        rh = np.array([self.x[frame_idx, js["right_hip"]],
                       self.y[frame_idx, js["right_hip"]]])
        d = float(np.linalg.norm(ls - rh))
        if d > 1e-3:
            return d
        x1, y1, x2, y2 = self.bbox[frame_idx]
        return float(np.hypot(x2 - x1, y2 - y1))


def _as_2d(arr):
    """Penn Action .mat arrays sometimes load as (n,) of objects or odd shapes.
    Coerce to a clean float (nframes, 13) array."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


def load_label(mat_path) -> dict:
    """Read one Penn Action label .mat into plain Python/NumPy."""
    m = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    action = str(m.get("action", "")).strip()
    pose = str(m.get("pose", "")).strip()
    x = _as_2d(m["x"])
    y = _as_2d(m["y"])
    vis = _as_2d(m.get("visibility", np.ones_like(x)))
    bbox = np.asarray(m["bbox"], dtype=np.float64)
    if bbox.ndim == 1:
        bbox = bbox.reshape(-1, 4)
    dims = np.asarray(m.get("dimensions", [0, 0, x.shape[0]])).ravel()
    return {
        "action": action, "pose": pose,
        "x": x, "y": y, "visibility": vis, "bbox": bbox, "dimensions": dims,
    }


def load_sequence(seq_id, root) -> PennSequence:
    """Load a sequence by id from an extracted Penn_Action root directory."""
    root = Path(root)
    label = load_label(root / "labels" / f"{seq_id}.mat")
    return PennSequence(
        seq_id=seq_id,
        frames_dir=root / "frames" / seq_id,
        action=label["action"],
        x=label["x"], y=label["y"], visibility=label["visibility"],
        bbox=label["bbox"], dims=label["dimensions"],
    )


def find_sequences_by_action(root, action_substr="squat") -> List[str]:
    """Return sequence ids whose action label contains `action_substr`.

    Case-insensitive substring match handles 'squat' vs 'squats' and any
    capitalisation differences in the dataset.
    """
    root = Path(root)
    labels_dir = root / "labels"
    hits = []
    for mat in sorted(labels_dir.glob("*.mat")):
        try:
            lbl = load_label(mat)
        except Exception:
            continue
        if action_substr.lower() in lbl["action"].lower():
            hits.append(mat.stem)
    return hits


# ---------------------------------------------------------------------------
# tar subset extraction — pull just the squat clips out of the monolithic
# Penn_Action.tar.gz so we never need the whole ~10GB extracted.
# ---------------------------------------------------------------------------

def extract_squats_from_tar(tar_path, dest_root, action_substr="squat",
                            max_sequences=None, prefix="Penn_Action"):
    """Two-pass extraction from the gzip tar.

    Pass 1: stream the labels (small) into memory, find sequences matching the
            action. Labels live at the END of the archive (frames come first),
            so this pass reads to the end regardless.
    Pass 2: random-access extract frames + labels for the matched sequences.

    Both passes operate on a LOCAL tar file (seekable), so this is cheap once
    the file is downloaded. Returns the list of extracted sequence ids.
    """
    dest_root = Path(dest_root)
    (dest_root / "frames").mkdir(parents=True, exist_ok=True)
    (dest_root / "labels").mkdir(parents=True, exist_ok=True)

    # --- pass 1: find matching sequence ids by reading label .mat members ---
    squat_ids = []
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            name = member.name
            if "/labels/" in name and name.endswith(".mat"):
                seq_id = Path(name).stem
                f = tf.extractfile(member)
                if f is None:
                    continue
                import io
                buf = io.BytesIO(f.read())
                try:
                    m = loadmat(buf, squeeze_me=True, struct_as_record=False)
                    action = str(m.get("action", "")).strip().lower()
                except Exception:
                    continue
                if action_substr.lower() in action:
                    squat_ids.append(seq_id)

    squat_ids = sorted(squat_ids)
    if max_sequences:
        squat_ids = squat_ids[:max_sequences]
    want = set(squat_ids)

    # --- pass 2: extract frames + labels for the chosen sequences ---
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            name = member.name
            # normalise: strip the top-level prefix dir
            rel = name[len(prefix) + 1:] if name.startswith(prefix + "/") else name
            parts = rel.split("/")
            if len(parts) < 2:
                continue
            kind = parts[0]   # 'frames' or 'labels'
            if kind == "labels" and rel.endswith(".mat"):
                if Path(rel).stem in want:
                    tf.extract(member, dest_root, set_attrs=False)
                    _flatten(dest_root, name, rel)
            elif kind == "frames" and len(parts) >= 3:
                seq_id = parts[1]
                if seq_id in want and member.isfile():
                    tf.extract(member, dest_root, set_attrs=False)
                    _flatten(dest_root, name, rel)

    return squat_ids


def _flatten(dest_root, member_name, rel):
    """tar.extract recreates the full member path (incl. the Penn_Action/ prefix).
    Move the file to dest_root/<rel> so callers see a clean frames/ + labels/ tree."""
    extracted = Path(dest_root) / member_name
    target = Path(dest_root) / rel
    if extracted == target:
        return
    if extracted.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        if extracted.is_file():
            extracted.replace(target)

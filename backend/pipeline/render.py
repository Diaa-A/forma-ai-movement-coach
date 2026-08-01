"""Skeleton overlay rendering. We draw a small explicit edge list instead of using
MediaPipe's drawing helper so per-joint colour control is straightforward."""
import cv2
import numpy as np
from .pose import LM, VISIBILITY_THRESHOLD
from .encoder import VideoEncoder


# Edge list — only the joints we actually analyse. Keeps the overlay readable.
_EDGES = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("left_ankle",     "left_foot_index"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
    ("right_ankle",    "right_foot_index"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
]

GOOD = (60, 200, 60)      # green-ish
BAD  = (50, 50, 220)      # red-ish (BGR)
NEUTRAL = (200, 200, 200) # light grey
UNCERTAIN = (140, 140, 140)  # muted grey for low-confidence joints

# Joints below VISIBILITY_THRESHOLD (shared with the cue layer, defined in pose.py)
# are drawn faded — a joint MediaPipe is guessing (e.g. an occluded far-side leg
# that collapses toward the near side) must not be presented as a confident one.


def _pt(landmark, w, h):
    """Normalised landmark to pixel coords. Returns None for NaN."""
    x, y = landmark[0], landmark[1]
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return (int(round(x * w)), int(round(y * h)))


def _vis(landmark):
    v = landmark[3]
    return float(v) if np.isfinite(v) else 0.0


def draw_overlay(frame_bgr, landmarks_frame, flagged_joints=None, info_text=None):
    """Draw skeleton + (optionally) angle text on a single BGR frame.

    Joints/edges MediaPipe is not confident about (visibility < VISIBILITY_THRESHOLD) are
    drawn faded and thin, so an occluded, guessed limb is not presented as a
    confidently tracked one."""
    flagged = flagged_joints or set()
    h, w = frame_bgr.shape[:2]

    # edges
    for a_name, b_name in _EDGES:
        a = _pt(landmarks_frame[LM[a_name]], w, h)
        b = _pt(landmarks_frame[LM[b_name]], w, h)
        if a is None or b is None:
            continue
        confident = (_vis(landmarks_frame[LM[a_name]]) >= VISIBILITY_THRESHOLD
                     and _vis(landmarks_frame[LM[b_name]]) >= VISIBILITY_THRESHOLD)
        if confident:
            colour = BAD if (a_name in flagged or b_name in flagged) else GOOD
            cv2.line(frame_bgr, a, b, colour, 3, cv2.LINE_AA)
        else:
            # uncertain limb (e.g. occluded far leg) — faint and thin
            cv2.line(frame_bgr, a, b, UNCERTAIN, 1, cv2.LINE_AA)

    # joint dots
    for name, idx in LM.items():
        p = _pt(landmarks_frame[idx], w, h)
        if p is None:
            continue
        if _vis(landmarks_frame[idx]) >= VISIBILITY_THRESHOLD:
            col = BAD if name in flagged else NEUTRAL
            cv2.circle(frame_bgr, p, 5, col, -1, cv2.LINE_AA)
        else:
            # low-confidence joint — small hollow marker, not a solid confident dot
            cv2.circle(frame_bgr, p, 4, UNCERTAIN, 1, cv2.LINE_AA)

    if info_text:
        y0 = 30
        for line in info_text.split("\n"):
            cv2.putText(frame_bgr, line, (12, y0), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame_bgr, line, (12, y0), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (240, 240, 240), 1, cv2.LINE_AA)
            y0 += 26

    return frame_bgr


def render_video(input_path, landmarks_all, angles_all, output_path,
                 flagged_per_frame=None):
    """Re-encode the input video with skeleton overlays per frame.

    The encoding itself lives in `encoder.py` — see the docstring there for why
    we don't just hand this to cv2.VideoWriter. Short version: the file it
    produces doesn't play in a browser, and this output is the main thing the
    PWA shows the user."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n = len(landmarks_all)
    flagged_per_frame = flagged_per_frame or [set()] * n

    with VideoEncoder(output_path, fps, w, h) as writer:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or i >= n:
                break
            ang = angles_all[i]
            info = "knee L:{} R:{}  spine:{}".format(
                _fmt(ang.get("knee_left")), _fmt(ang.get("knee_right")),
                _fmt(ang.get("spine")),
            )
            draw_overlay(frame, landmarks_all[i], flagged_per_frame[i], info_text=info)
            writer.write(frame)
            i += 1

    cap.release()
    return output_path


def read_frames_exact(input_path, indices):
    """Decode forward to each wanted frame and return {index: frame}.

    Deliberately NOT cv2's CAP_PROP_POS_FRAMES seek. That seek is approximate on
    plenty of real files — on a 621-frame .mov straight off a phone, asking for
    frame 44 returned frame 40, 200 returned 196, 575 returned 574. Small, but the
    landmarks drawn on a key frame come from the index we asked for, so a seek
    that lands four frames early puts the skeleton somewhere the body no longer
    is, and the reported timestamp is wrong too.

    Sequential decode is exact by construction. It costs one pass, stopping at the
    last index we need — about 2 ms a frame, so well under a second for a normal
    clip, against a ~12 s pipeline.
    """
    wanted = sorted({int(i) for i in indices if i is not None and i >= 0})
    if not wanted:
        return {}

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open input video: {input_path}")

    out, last, i = {}, wanted[-1], 0
    remaining = set(wanted)
    while i <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if i in remaining:
            out[i] = frame.copy()
            remaining.discard(i)
        i += 1
    cap.release()
    return out


def save_key_frame(input_path, frame_idx, landmarks_frame, angle_frame,
                   output_path, flagged_joints=None, label="", side=None,
                   frame=None):
    """Save one frame with overlay + label.

    Pass `frame` if you already decoded it (see read_frames_exact) — otherwise
    this reads it, exactly, itself.

    When `side` is given, the trunk-vs-shin relationship (the basis of the forward
    lean cue) is shown explicitly, so the figure explains why a frame was flagged."""
    if frame is None:
        got = read_frames_exact(input_path, [frame_idx])
        frame = got.get(frame_idx)
    if frame is None:
        raise RuntimeError(f"could not read frame {frame_idx} from {input_path}")

    trunk = angle_frame.get("spine")
    shin = angle_frame.get("shin_left" if side == "left" else "shin_right")
    info_lines = [
        label.upper() if label else "",
        "frame {}".format(frame_idx),
        "knee L:{} R:{}".format(_fmt(angle_frame.get("knee_left")),
                                _fmt(angle_frame.get("knee_right"))),
        "trunk {}  shin {}".format(_fmt(trunk), _fmt(shin)),
    ]
    if (trunk is not None and shin is not None
            and np.isfinite(trunk) and np.isfinite(shin)):
        info_lines.append("lean {:+.0f} (cap +15)".format(trunk - shin))
    info_text = "\n".join(s for s in info_lines if s)
    draw_overlay(frame, landmarks_frame, flagged_joints, info_text=info_text)
    cv2.imwrite(str(output_path), frame)
    return output_path


def _fmt(v):
    if v is None or not np.isfinite(v):
        return "--"
    return "{:.0f}".format(v)

"""Skeleton overlay rendering. We draw a small explicit edge list instead of using
MediaPipe's drawing helper so per-joint colour control is straightforward."""
import cv2
import numpy as np
from .pose import LM, VISIBILITY_THRESHOLD, fit_dims, fit_within
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


def _scale(w, h):
    """Overlay geometry as a fraction of the frame, not a pixel count.

    Everything here used to be fixed — 3px limbs, 5px joints, 0.7 font. That is
    fine on the 576x1024 development clips and close to invisible on the 2160x3840
    a modern phone actually records: a 3px line across a 4K frame is a hairline,
    and the angle readout ends up unreadable. Phones keep getting more pixels, so
    scaling off the frame is the only version that keeps working.

    Keyed to the SHORT edge, so a portrait clip and the same clip landscape get
    the same weight of line. Floors keep it visible on small inputs; the numbers
    are tuned to reproduce roughly the old look at 576 wide, which is what the
    existing report figures were made at."""
    # Fractions calibrated against the render people said looked right: a 3px limb
    # and 5px joint on a 360x480 clip, i.e. about 0.85% and 1.2% of the short
    # edge. Anything meaningfully finer reads as a hairline once the frame gets
    # big, which is what a 4K upload showed.
    ref = min(w, h)
    return {
        "limb": max(3, round(ref * 0.0085)),
        "joint": max(5, round(ref * 0.012)),
        "thin": max(1, round(ref * 0.003)),
        "font": max(0.6, ref * 0.0017),
        "font_thick": max(1, round(ref * 0.003)),
        "outline": max(3, round(ref * 0.008)),
        "line_step": max(24, round(ref * 0.058)),
        "margin": max(10, round(ref * 0.025)),
    }


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
    s = _scale(w, h)

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
            cv2.line(frame_bgr, a, b, colour, s["limb"], cv2.LINE_AA)
        else:
            # uncertain limb (e.g. occluded far leg) — faint and thin
            cv2.line(frame_bgr, a, b, UNCERTAIN, s["thin"], cv2.LINE_AA)

    # joint dots
    for name, idx in LM.items():
        p = _pt(landmarks_frame[idx], w, h)
        if p is None:
            continue
        if _vis(landmarks_frame[idx]) >= VISIBILITY_THRESHOLD:
            col = BAD if name in flagged else NEUTRAL
            cv2.circle(frame_bgr, p, s["joint"], col, -1, cv2.LINE_AA)
        else:
            # low-confidence joint — small hollow marker, not a solid confident dot
            cv2.circle(frame_bgr, p, max(3, s["joint"] - 1), UNCERTAIN,
                       s["thin"], cv2.LINE_AA)

    if info_text:
        y0 = s["line_step"]
        for line in info_text.split("\n"):
            # dark outline under light text so it stays readable over a bright
            # gym floor or a window
            cv2.putText(frame_bgr, line, (s["margin"], y0), cv2.FONT_HERSHEY_SIMPLEX,
                        s["font"], (0, 0, 0), s["outline"], cv2.LINE_AA)
            cv2.putText(frame_bgr, line, (s["margin"], y0), cv2.FONT_HERSHEY_SIMPLEX,
                        s["font"], (240, 240, 240), s["font_thick"], cv2.LINE_AA)
            y0 += s["line_step"]

    return frame_bgr


def render_video(input_path, landmarks_all, angles_all, output_path,
                 flagged_per_frame=None, caption=None, side=None):
    """Re-encode the input video with skeleton overlays per frame.

    The encoding itself lives in `encoder.py` — see the docstring there for why
    we don't just hand this to cv2.VideoWriter. Short version: the file it
    produces doesn't play in a browser, and this output is the main thing the
    PWA shows the user."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Encode at the analysed size, not the source size. A 4K input used to produce
    # a 4K annotated video, which is expensive to encode and then has to travel
    # down a phone's mobile connection before anyone can watch it.
    w, h = fit_dims(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    n = len(landmarks_all)
    flagged_per_frame = flagged_per_frame or [set()] * n

    with VideoEncoder(output_path, fps, w, h) as writer:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or i >= n:
                break
            frame = fit_within(frame)
            ang = angles_all[i]
            # the caption names joints, so the exercise supplies it -- see
            # squat.frame_caption / pushup.frame_caption
            info = "\n".join(caption(ang, side)) if caption else None
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
            # capped like everything else, so a key frame matches the video it
            # was taken from
            out[i] = fit_within(frame).copy()
            remaining.discard(i)
        i += 1
    cap.release()
    return out


def save_key_frame(input_path, frame_idx, landmarks_frame, angle_frame,
                   output_path, flagged_joints=None, label="", side=None,
                   frame=None, caption=None):
    """Save one frame with overlay + label.

    Pass `frame` if you already decoded it (see read_frames_exact) — otherwise
    this reads it, exactly, itself.

    `caption` comes from the exercise and produces the measurement lines, so the
    figure explains why the frame was flagged in that exercise's own terms."""
    if frame is None:
        got = read_frames_exact(input_path, [frame_idx])
        frame = got.get(frame_idx)
    if frame is None:
        raise RuntimeError(f"could not read frame {frame_idx} from {input_path}")

    info_lines = [
        label.upper() if label else "",
        "frame {}".format(frame_idx),
    ]
    if caption is not None:
        info_lines.extend(caption(angle_frame, side))
    info_text = "\n".join(s for s in info_lines if s)
    draw_overlay(frame, landmarks_frame, flagged_joints, info_text=info_text)
    cv2.imwrite(str(output_path), frame)
    return output_path


def _fmt(v):
    if v is None or not np.isfinite(v):
        return "--"
    return "{:.0f}".format(v)

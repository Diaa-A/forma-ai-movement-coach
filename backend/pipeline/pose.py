"""MediaPipe Pose wrapper (new Tasks API).

Old `mp.solutions.pose` is gone in mediapipe 0.10.x — we use `PoseLandmarker`
with the downloadable `pose_landmarker_lite.task` model file.

Takes a video path, returns per-frame landmarks. We expose both image-normalised
coords (for 2D angles and drawing) and world coords in metres (kept for the
future 3D investigation against Fit3D). Missing detections become NaN so the
filter can mask them.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

N_LANDMARKS = 33

# Shared landmark-confidence threshold. A landmark below this is treated as
# unreliable (MediaPipe is guessing it — typically an occluded far-side limb).
# Data-driven, not MediaPipe's 0.5 default: across the test clips an occluded
# far-side leg scored 0.47-0.55 while a clearly visible one scored 0.95+, so the
# cut sits in that gap with margin. Used by both the cue layer (suppress cues that
# depend on such a joint) and the overlay (draw such joints faded, not confident).
VISIBILITY_THRESHOLD = 0.65

# Available MediaPipe pose models, by size/quality. Full is the sensible default
# for analysis work — lite drops frames on fast motion; heavy is overkill for
# single-person batch processing.
MODELS = {
    "lite":  "pose_landmarker_lite.task",
    "full":  "pose_landmarker_full.task",
    "heavy": "pose_landmarker_heavy.task",
}
_MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"


def _model_path(name="full"):
    fname = MODELS.get(name, name)
    # env override stays as last word so deployments can swap without code edits
    p = os.environ.get("POSE_MODEL_PATH", str(_MODEL_DIR / fname))
    if not Path(p).exists():
        raise FileNotFoundError(
            f"pose model not found at {p}. Download with:\n"
            f"  curl -L -o {p} \\\n"
            f"    https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            f"pose_landmarker_{name}/float16/1/{fname}"
        )
    return p


# Longest edge we will analyse or draw on. 1920 is 1080p, which is what both
# filming guides already tell people to shoot, so the code and the advice finally
# agree.
#
# This is a memory fix before it is a speed one. The container idles at 447 MB of
# 1000 and a 4K analysis wants roughly 690 MB on top of that, so an iPhone filming
# at its default setting was a guaranteed kill -- it died three seconds in, before
# the overlay render was even reached.
#
# It costs nothing in accuracy. MediaPipe resizes its input internally, so the
# extra pixels are discarded before the model ever sees them, and the same clip at
# three resolutions gave identical rep counts with knee angles inside 0.4 degrees
# across a 14x pixel range (handoff 19.2). Landmarks are normalised, so an
# aspect-preserving resize leaves the angle maths untouched.
#
# Below the cap this is a no-op and returns the frame it was given, which is why
# no published number moves: every fixture in the repo is already smaller.
MAX_ANALYSIS_EDGE = 1920


def fit_dims(w, h, max_edge=MAX_ANALYSIS_EDGE):
    """Target size for a frame, capped on its long edge, aspect preserved."""
    longest = max(w, h)
    if longest <= max_edge or longest == 0:
        return w, h
    s = max_edge / longest
    return max(1, int(round(w * s))), max(1, int(round(h * s)))


def fit_within(frame, max_edge=MAX_ANALYSIS_EDGE):
    """The frame, shrunk to fit the cap. Returned untouched when it already does."""
    h, w = frame.shape[:2]
    tw, th = fit_dims(w, h, max_edge)
    if (tw, th) == (w, h):
        return frame
    # INTER_AREA is the right one for shrinking; the others alias badly and this
    # image is about to have a pose estimated off it.
    return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)


def extract_landmarks(video_path, model="full"):
    """Run MediaPipe PoseLandmarker on every frame.

    `model` selects lite / full / heavy. Returns:
        fps, width, height, frame_count, timestamps,
        landmarks       (n_frames, 33, 4)  x, y (normalised 0-1), z, visibility
        world_landmarks (n_frames, 33, 3)  metric x, y, z
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=_model_path(model)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    lm_buf = []
    wl_buf = []
    timestamps = []

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_bgr = fit_within(frame_bgr)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(round((frame_idx / fps) * 1000))
            result = landmarker.detect_for_video(mp_image, ts_ms)

            if not result.pose_landmarks:
                lm_buf.append(np.full((N_LANDMARKS, 4), np.nan, dtype=np.float64))
                wl_buf.append(np.full((N_LANDMARKS, 3), np.nan, dtype=np.float64))
            else:
                pose_lms = result.pose_landmarks[0]   # one person
                lm = np.array(
                    [[p.x, p.y, p.z, p.visibility] for p in pose_lms],
                    dtype=np.float64,
                )
                lm_buf.append(lm)

                if result.pose_world_landmarks:
                    world = result.pose_world_landmarks[0]
                    wl = np.array([[p.x, p.y, p.z] for p in world], dtype=np.float64)
                    wl_buf.append(wl)
                else:
                    wl_buf.append(np.full((N_LANDMARKS, 3), np.nan, dtype=np.float64))

            timestamps.append(frame_idx / fps)
            frame_idx += 1

    cap.release()

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_idx,
        "timestamps": np.array(timestamps, dtype=np.float64),
        "landmarks": np.stack(lm_buf) if lm_buf else np.zeros((0, N_LANDMARKS, 4)),
        "world_landmarks": np.stack(wl_buf) if wl_buf else np.zeros((0, N_LANDMARKS, 3)),
    }


def extract_landmarks_from_frames(frame_paths, model="full"):
    """Run PoseLandmarker in IMAGE mode over a list of still images.

    Used by the Penn Action evaluation — the dataset ships individual frames,
    and benchmarking pose accuracy is fairest with each frame detected
    independently (no temporal smoothing confound). Returns the same shape as
    `extract_landmarks` minus the video-only fields.
    """
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=_model_path(model)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )

    lm_buf, wl_buf, sizes = [], [], []
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        for path in frame_paths:
            bgr = cv2.imread(str(path))
            if bgr is None:
                lm_buf.append(np.full((N_LANDMARKS, 4), np.nan))
                wl_buf.append(np.full((N_LANDMARKS, 3), np.nan))
                sizes.append((0, 0))
                continue
            h, w = bgr.shape[:2]
            sizes.append((w, h))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect(mp_image)
            if not res.pose_landmarks:
                lm_buf.append(np.full((N_LANDMARKS, 4), np.nan))
                wl_buf.append(np.full((N_LANDMARKS, 3), np.nan))
            else:
                pl = res.pose_landmarks[0]
                lm_buf.append(np.array([[p.x, p.y, p.z, p.visibility] for p in pl]))
                if res.pose_world_landmarks:
                    wl = res.pose_world_landmarks[0]
                    wl_buf.append(np.array([[p.x, p.y, p.z] for p in wl]))
                else:
                    wl_buf.append(np.full((N_LANDMARKS, 3), np.nan))

    return {
        "frame_count": len(frame_paths),
        "sizes": sizes,   # (w, h) per frame — frames may differ in Penn Action
        "landmarks": np.stack(lm_buf) if lm_buf else np.zeros((0, N_LANDMARKS, 4)),
        "world_landmarks": np.stack(wl_buf) if wl_buf else np.zeros((0, N_LANDMARKS, 3)),
    }


# MediaPipe pose landmark indices — names mirror the official enum for searchability.
LM = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13,    "right_elbow": 14,
    "left_wrist": 15,    "right_wrist": 16,
    "left_hip": 23,      "right_hip": 24,
    "left_knee": 25,     "right_knee": 26,
    "left_ankle": 27,    "right_ankle": 28,
    "left_heel": 29,     "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}

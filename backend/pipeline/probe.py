"""Read a clip's length without decoding it.

There is a byte cap on uploads but no length check, so a ten-minute clip is
accepted and then costs the pipeline ten minutes of pose inference before anyone
finds out. This answers "how long is it" cheaply enough to ask before starting.

The reading comes from container metadata rather than from counting frames, which
makes it approximate. That is worth being explicit about, because this project has
already been caught by an approximate OpenCV reading once: `CAP_PROP_POS_FRAMES`
seeks landed up to four frames early, which mattered a lot when the requirement
was "give me frame 44 exactly" (see render.read_frames_exact). It does not matter
here. Being a few frames out on a 45-second limit changes nothing, and the
question is a band rather than a value.

When the metadata is missing or nonsense the answer is None, and the caller is
expected to let the clip through rather than refuse on a number it does not have.
Refusing on an unreadable probe would reject valid files for a reason the user
could do nothing about.
"""
from __future__ import annotations

import cv2


def duration_seconds(path) -> float | None:
    """Clip length in seconds, or None if the container will not say.

    Opens the file and reads two properties; no frames are decoded, so this costs
    microseconds against a pipeline that costs seconds.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        cap.release()

    # A container that reports zero fps or a negative frame count is telling us it
    # does not know, not that the clip is empty.
    if not fps or fps <= 0 or not frames or frames <= 0:
        return None
    return float(frames) / float(fps)

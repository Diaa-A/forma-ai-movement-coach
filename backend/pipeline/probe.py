"""Look at an uploaded file before handing it to the pipeline.

Two questions, both answered from container metadata, both worth asking before
spending twelve seconds a clip-minute on pose inference:

  - Is there a picture in there at all? Not a rhetorical question. An audio-only
    .mp4 is what you get by picking a voice memo out of a phone gallery by
    mistake, and it used to travel all the way into `extract_landmarks`, fail
    with "no frames decoded", and come back to the user as a 500. That is a bad
    upload, not a server fault.
  - How long is it? There is a byte cap but there was no length check, so a
    ten-minute clip was accepted and then analysed in full.

Measured on the fixtures, which is what the checks below are built from:

    squat.mp4 (real)          fps 30   frames 608   576x1024
    notefortest.mp4 (audio)   fps 1    frames  -1   0x0
    requirements.txt as .mp4  does not open as video       (also .mov/.webm/.m4v)
    requirements.txt as .txt  fps 25   frames   5   640x400
    README.md                 does not open at all

So a zero frame size is the test. The last two rows are the same bytes: OpenCV
picks its backend partly from the extension, and only the unrecognised one falls
through to something that reports a nonsense 640x400. Uploads are always staged
under one of the allowed video extensions, so that row is not a path a request
can take — it is recorded because it is why the check reads frame size rather
than trusting the file to be what it is named.

Nothing decodes here, on purpose. An earlier version called `cap.read()` to
confirm a frame actually came out, which is the stronger test, and on
requirements.txt that call **hangs indefinitely** rather than failing. A probe
that can hang is worse than the problem it was added for.

Since the pipeline decodes with that same call, a file that got past this check
while not really being video could still hang a request. Nothing in the fixtures
manages it, but the check is metadata and metadata can lie. The bound for that is
a request timeout at the serving layer, which belongs to deployment (WP-03).

`seconds` is None when the container will not say. The caller lets those through
rather than refusing on a number it does not have; refusing would reject valid
files for a reason the user cannot act on, and the byte cap bounds the bad case.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class ClipProbe:
    readable: bool          # there is a picture in it, whatever else is true
    seconds: float | None   # length from metadata, None if unknown


def probe_clip(path) -> ClipProbe:
    """Open the file once and read what it says about itself. Decodes nothing."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return ClipProbe(readable=False, seconds=None)

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    finally:
        cap.release()

    if not (width > 0 and height > 0 and frames > 0):
        return ClipProbe(readable=False, seconds=None)

    # fps of zero is the container declining to say, not a still image
    if not fps or fps <= 0:
        return ClipProbe(readable=True, seconds=None)
    return ClipProbe(readable=True, seconds=float(frames) / float(fps))

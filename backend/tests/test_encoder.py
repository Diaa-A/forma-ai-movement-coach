"""Tests for the annotated-video encoder.

The thing being guarded here is specific: OpenCV will happily hand you a writer
that reports isOpened() == True and then produces a file no browser can play (or
in the H.264 case, no file at all). So these tests check the *bytes that came
out*, not whether the call returned without complaining.
"""
import cv2
import numpy as np
import pytest

from backend.pipeline.encoder import VideoEncoder, _ffmpeg_exe


def _frames(n, w=64, h=48):
    """A few frames with something moving in them, so x264 has real work to do
    and can't collapse the whole thing into one tiny keyframe."""
    out = []
    for i in range(n):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        f[:, (i * 3) % w:((i * 3) % w) + 6] = 255
        out.append(f)
    return out


def _decodable_frames(path):
    """Read the file back and count the frames that actually come out.

    Stronger than checking the file size: OpenCV's broken H.264 path wrote a
    plausible-looking 1.3 KB header with nothing in it, and a size threshold
    only catches that by luck. Decoding it catches it properly.
    """
    cap = cv2.VideoCapture(str(path))
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def _codec_tag(path):
    """Pull the video sample-description fourcc out of the container.

    Reading the box structure properly would be overkill — the tags we care
    about telling apart ('avc1' vs 'mp4v') are distinctive enough to just look
    for in the raw bytes.
    """
    data = path.read_bytes()
    for tag in (b"avc1", b"mp4v", b"hev1", b"hvc1"):
        if tag in data:
            return tag.decode()
    return None


needs_ffmpeg = pytest.mark.skipif(
    _ffmpeg_exe() is None,
    reason="imageio-ffmpeg not installed — H.264 path unavailable",
)


@needs_ffmpeg
def test_writes_h264_not_mpeg4_part2(tmp_path):
    """The whole point of the module: browsers play avc1, not mp4v."""
    out = tmp_path / "clip.mp4"
    with VideoEncoder(out, fps=30, width=64, height=48) as enc:
        assert enc.codec == "h264"
        for f in _frames(20):
            enc.write(f)

    assert out.exists()
    assert _codec_tag(out) == "avc1"
    # a header-only file is the exact failure mode we're guarding against —
    # OpenCV's broken avc1 path wrote 1.3 KB and reported success
    assert _decodable_frames(out) == 20


@needs_ffmpeg
def test_faststart_moov_is_near_the_front(tmp_path):
    """-movflags +faststart, so a phone can start playing before the whole file
    has downloaded. Without it moov lands at the end and playback waits."""
    out = tmp_path / "clip.mp4"
    with VideoEncoder(out, fps=30, width=64, height=48) as enc:
        for f in _frames(30):
            enc.write(f)

    data = out.read_bytes()
    moov = data.find(b"moov")
    mdat = data.find(b"mdat")
    assert moov != -1 and mdat != -1
    assert moov < mdat, "moov after mdat — faststart didn't apply"


@needs_ffmpeg
def test_odd_dimensions_are_made_even(tmp_path):
    """yuv420p can't do odd width/height and ffmpeg errors rather than rounding.
    Rotated / cropped phone clips do turn up odd-sized."""
    out = tmp_path / "odd.mp4"
    enc = VideoEncoder(out, fps=30, width=65, height=49)
    assert (enc.width, enc.height) == (64, 48)
    for f in _frames(10, w=65, h=49):
        enc.write(f)
    enc.close()

    assert _codec_tag(out) == "avc1"
    assert _decodable_frames(out) == 10


def test_falls_back_to_opencv_when_ffmpeg_missing(tmp_path, monkeypatch):
    """No ffmpeg shouldn't mean no analysis — we degrade to the old writer and
    say so in `codec` rather than raising."""
    monkeypatch.setattr("backend.pipeline.encoder._ffmpeg_exe", lambda: None)

    out = tmp_path / "fallback.mp4"
    with VideoEncoder(out, fps=30, width=64, height=48) as enc:
        assert enc.codec == "mp4v"
        for f in _frames(10):
            enc.write(f)

    assert out.exists()


def test_zero_fps_does_not_explode(tmp_path):
    """cv2 returns 0.0 fps on some containers. Don't pass that to the encoder."""
    enc = VideoEncoder(tmp_path / "z.mp4", fps=0, width=64, height=48)
    assert enc.fps == 30.0
    enc.close()

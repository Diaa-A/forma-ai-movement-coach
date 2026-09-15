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


# ---------------------------------------------------------------------------
# Exact frame reads (render.read_frames_exact)
# ---------------------------------------------------------------------------

@needs_ffmpeg
def test_read_frames_exact_returns_the_frame_actually_asked_for(tmp_path):
    """cv2's CAP_PROP_POS_FRAMES seek is approximate on plenty of real files. On a
    621-frame .mov off a phone it returned frame 40 for 44, 196 for 200, 574 for
    575 — small, but the landmarks drawn onto a key frame come from the index we
    asked for, so an off-by-four puts the skeleton where the body isn't.

    Each frame carries its index as the POSITION of a white stripe. Position
    survives the encode; brightness does not — yuv420p is limited-range, so a
    grey of 28 comes back as 24 and an absolute pixel value would fail for
    reasons that have nothing to do with which frame was read.
    """
    from backend.pipeline.render import read_frames_exact

    out = tmp_path / "counted.mp4"
    with VideoEncoder(out, fps=30, width=64, height=48) as enc:
        for i in range(60):
            f = np.zeros((48, 64, 3), dtype=np.uint8)
            f[:, i:i + 2] = 255            # stripe sits at column == frame index
            enc.write(f)

    got = read_frames_exact(out, [7, 23, 55])
    assert sorted(got) == [7, 23, 55]
    for idx in (7, 23, 55):
        stripe = int(got[idx][:, :, 0].mean(axis=0).argmax())
        assert abs(stripe - idx) <= 1, f"asked for frame {idx}, got the one at {stripe}"


def test_read_frames_exact_handles_nothing_to_read(tmp_path):
    from backend.pipeline.render import read_frames_exact
    assert read_frames_exact(tmp_path / "nope.mp4", []) == {}
    assert read_frames_exact(tmp_path / "nope.mp4", [None]) == {}


def test_saving_a_key_frame_leaves_the_decoded_frame_untouched(tmp_path):
    """A one-rep set has the same worst and best frame, and the runner decodes it
    once for both saves. Drawing in place meant best.jpg carried both labels, one
    printed over the other."""
    from backend.pipeline.render import save_key_frame

    frame = np.full((120, 160, 3), 40, dtype=np.uint8)
    before = frame.copy()
    landmarks = np.zeros((33, 4))
    save_key_frame(None, 5, landmarks, {}, tmp_path / "worst.jpg",
                   label="closest to the limit", frame=frame)
    assert np.array_equal(frame, before)
    save_key_frame(None, 5, landmarks, {}, tmp_path / "best.jpg", label="best", frame=frame)
    assert np.array_equal(frame, before)
    assert cv2.imread(str(tmp_path / "best.jpg")) is not None


# ---------------------------------------------------------------------------
# Overlay geometry scales with the frame (render._scale)
# ---------------------------------------------------------------------------

def test_overlay_geometry_scales_with_resolution():
    """Fixed 3px limbs and 5px joints look right on the 576x1024 development
    clips and are a hairline on the 2160x3840 a phone actually records. Reported
    from a real device as 'the skeleton is thin and does not look better'."""
    from backend.pipeline.render import _scale

    small = _scale(576, 1024)
    big = _scale(2160, 3840)

    assert big["limb"] > small["limb"] * 3, "limbs did not scale up for 4K"
    assert big["joint"] > small["joint"] * 3
    assert big["font"] > small["font"] * 2

    # and the old look is roughly preserved where the report figures were made
    assert 4 <= small["limb"] <= 6
    assert 5 <= small["joint"] <= 9


def test_overlay_stays_visible_on_a_tiny_frame():
    """Floors matter too — a thumbnail-sized clip must not end up with 0px lines."""
    from backend.pipeline.render import _scale
    tiny = _scale(120, 160)
    assert tiny["limb"] >= 3 and tiny["joint"] >= 4 and tiny["thin"] >= 1


def test_scale_is_orientation_independent():
    """Keyed to the short edge, so the same clip portrait and landscape gets the
    same weight of line."""
    from backend.pipeline.render import _scale
    assert _scale(1080, 1920) == _scale(1920, 1080)

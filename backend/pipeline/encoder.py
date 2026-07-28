"""Writing the annotated video out in something a browser will actually play.

Background, because the obvious code is the broken code here. `cv2.VideoWriter`
with the "mp4v" fourcc gives you MPEG-4 Part 2, which VLC is happy with and no
browser is — it isn't in the codec set `<video>` accepts inside an MP4. And you
can't just ask OpenCV for H.264 instead: on this build `avc1`/`H264`/`VP80` all
return isOpened() == True and then quietly write a header with no frames in it,
because libopenh264 fails to load. Silent, and it produces a file that looks
plausible until you try to play it.

So we pipe raw frames to ffmpeg instead. imageio-ffmpeg ships a static binary for
Windows/macOS/Linux as a normal pip install, so there's no "first install ffmpeg"
step in the setup instructions — which matters, a marker who can't run the thing
marks what they can see.

If ffmpeg somehow isn't there we fall back to the old mp4v writer rather than
blowing up the whole pipeline: a clip that plays in VLC but not Safari is worse
than one that plays everywhere, and much better than no analysis at all. The
fallback records itself in `codec` so callers can tell.
"""
from __future__ import annotations

import subprocess
import tempfile

import cv2


# x264 defaults. veryfast because a 20 s clip is ~600 frames and we're already
# spending ~9.5 s in pose estimation — no sense adding to that for a preview
# render nobody is archiving. crf 23 is x264's own default and looks fine over
# a skeleton overlay.
X264_PRESET = "veryfast"
X264_CRF = "23"


def _ffmpeg_exe():
    """Path to the bundled ffmpeg, or None if the package isn't installed."""
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        # get_ffmpeg_exe raises if the binary didn't download / got removed
        return None


class VideoEncoder:
    """Frame sink. Feed it BGR frames, it writes an MP4.

    Prefers H.264 via ffmpeg; falls back to OpenCV's mp4v if it has to. Use it
    as a context manager, or call close() yourself.
    """

    def __init__(self, path, fps, width, height):
        self.path = str(path)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        # yuv420p needs even dimensions. Odd sizes turn up more than you'd think
        # once phones start rotating and cropping things, and ffmpeg just refuses
        # rather than rounding for you.
        self.width = int(width) - (int(width) % 2)
        self.height = int(height) - (int(height) % 2)
        self._crop = (self.width != int(width)) or (self.height != int(height))

        self._proc = None
        self._errlog = None
        self._cv_writer = None
        self.codec = None

        exe = _ffmpeg_exe()
        if exe:
            self._start_ffmpeg(exe)
        if self._proc is None:
            self._start_opencv()

    # -- setup ---------------------------------------------------------------

    def _start_ffmpeg(self, exe):
        cmd = [
            exe, "-y",
            "-loglevel", "error",
            # input: raw frames on stdin, exactly as OpenCV hands them to us
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", f"{self.fps:.6f}",
            "-i", "pipe:0",
            "-an",                          # no audio track, we never had one
            "-c:v", "libx264",
            "-preset", X264_PRESET,
            "-crf", X264_CRF,
            # yuv420p is the compatibility pixel format — Safari and older
            # Android decoders won't touch yuv444p even though x264 will emit it
            "-pix_fmt", "yuv420p",
            # puts the moov atom at the front so a browser can start playing
            # before the whole file has arrived. Easy to forget, very noticable
            # on a phone over mobile data.
            "-movflags", "+faststart",
            self.path,
        ]
        # stderr goes to a real temp file, not a pipe — nobody is draining a pipe
        # while we're busy writing frames into stdin, and a full stderr buffer
        # deadlocks the whole thing. We only read it if something goes wrong.
        self._errlog = tempfile.TemporaryFile()
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=self._errlog,
            )
            self.codec = "h264"
        except OSError:
            self._proc = None
            self._errlog.close()
            self._errlog = None

    def _start_opencv(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._cv_writer = cv2.VideoWriter(self.path, fourcc, self.fps,
                                          (self.width, self.height))
        self.codec = "mp4v"

    # -- use -----------------------------------------------------------------

    def write(self, frame_bgr):
        if self._crop:
            frame_bgr = frame_bgr[:self.height, :self.width]
        if self._proc is not None:
            try:
                self._proc.stdin.write(frame_bgr.tobytes())
            except (BrokenPipeError, ValueError):
                # ffmpeg died mid-render. Salvage what we can: switch to the
                # OpenCV writer for the remaining frames so the caller still
                # gets a file. The already-written frames are lost, which is
                # ugly, but this path should basically never happen.
                self._drain_ffmpeg()
                self._start_opencv()
                self._cv_writer.write(frame_bgr)
        else:
            self._cv_writer.write(frame_bgr)

    def close(self):
        if self._proc is not None:
            self._drain_ffmpeg()
        if self._cv_writer is not None:
            self._cv_writer.release()
            self._cv_writer = None

    def _drain_ffmpeg(self):
        proc, self._proc = self._proc, None
        try:
            proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        proc.wait(timeout=120)
        if proc.returncode not in (0, None) and self._errlog is not None:
            self._errlog.seek(0)
            detail = self._errlog.read().decode("utf-8", "replace").strip()
            # not raising — a bad annotated video shouldn't lose the analysis
            print(f"[encoder] ffmpeg exited {proc.returncode}: {detail[:500]}")
        if self._errlog is not None:
            self._errlog.close()
            self._errlog = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

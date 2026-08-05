"""/analyze endpoint.

Multipart upload (video, optional voice note, exercise_type) -> run the same
pipeline the CLI uses -> JSON response with URLs to artefacts the static mount
in `backend.main` serves.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..pipeline.probe import probe_clip
from ..pipeline.runner import run_pipeline, new_job_id, RunOptions, RunResult
from ..exercises.registry import PROFILES, exercise_ids
from .schemas import (AnalyzeResponse, KeyFrame, RepStat, CoachingReportOut,
                      ExerciseOut, ExercisesResponse, LimitsOut)


router = APIRouter()

log = logging.getLogger("coach.api")


# Upload + output roots, kept sibling so the /results static mount in backend.main
# serves both the uploaded original and the rendered artefacts.
#
# The default is absolute and anchored to the repo rather than to the working
# directory. It used to be Path("data"), which quietly means "data relative to
# wherever this process was started" — fine when that is always the repo root, and
# a deployment where artefacts land somewhere unexpected and /results serves an
# empty directory as soon as it is not.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT       = Path(os.environ.get("DATA_ROOT") or (_REPO_ROOT / "data"))
UPLOAD_ROOT     = DATA_ROOT / "uploads"
OUTPUT_ROOT     = DATA_ROOT / "outputs"


MAX_VIDEO_BYTES = 100 * 1024 * 1024     # 100MB — generous but bounded
MAX_AUDIO_BYTES = 10 * 1024 * 1024
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".m4v"}
# .mp4 is here because Safari's MediaRecorder produced MP4/AAC only until 18.4, so
# an iOS voice note arrives as audio/mp4. The frontend has been renaming those to
# .m4a to get past this list — same container, but the workaround only exists
# because the list was wrong.
ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".mp4", ".webm", ".ogg"}
# derived from the profile registry, not written out again here — one list means
# the allowlist and the exercise picker can't drift apart. push-up / pull-up
# arrive by registering their profile (Phase G).
ALLOWED_EXERCISES = set(PROFILES)

# Hard band the server refuses outside of. The spec asks for 5-30 s, which is the
# ideal pair below; the hard band is looser so a slightly long clip is analysed
# rather than thrown away. Under 3 s cannot hold a full rep, and over 45 s is
# minutes of pose inference for a set nobody needs analysed in full.
MIN_CLIP_SECONDS = 3.0
MAX_CLIP_SECONDS = 45.0
IDEAL_MIN_SECONDS = 5.0
IDEAL_MAX_SECONDS = 30.0


# How many analyses may run at once. Everything else -- /health, /exercises, the
# static files -- is unaffected.
#
# This is a memory bound, and it is set from a measured number rather than a
# guess. One analysis peaks at 239 MB on the 576x1024 reference clip and 353 MB
# on a 1080p phone clip; a 4K upload is higher again. The container has far less
# headroom than that times six.
#
# Six is what it was, expressed as uvicorn's --limit-concurrency, and that was
# wrong twice over. The number was invented rather than measured, and the flag
# caps ALL connections rather than analyses -- so lowering it far enough to bound
# memory would eventually block the platform's healthcheck and put the service in
# a restart loop. A semaphore around the expensive part is the right instrument.
#
# What actually happened without it: a phone upload was cancelled and retried.
# Cancelling closes the connection but does not stop the handler, so each retry
# ADDED an analysis instead of replacing one, and the third one took the
# container past its limit. The log said "Killed".
MAX_CONCURRENT_ANALYSES = int(os.environ.get("MAX_CONCURRENT_ANALYSES", "2"))
_analysis_slots = threading.Semaphore(MAX_CONCURRENT_ANALYSES)


def _mb(n_bytes: int) -> str:
    """Byte counts in the units a person reads. The 413 used to say 'upload
    exceeds 104857600 bytes', on a phone."""
    return "{:.0f} MB".format(n_bytes / (1024 * 1024))


def _save_upload(up: UploadFile, dest: Path, max_bytes: int, what: str):
    """Stream an upload to disk, stopping if it goes over the cap.

    Streamed rather than read whole so a 100 MB clip is never held in memory.
    `what` names the file in the error, because "upload exceeds the limit" does
    not tell someone whether it was their video or their voice note.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with dest.open("wb") as fh:
        while True:
            chunk = up.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    413, f"That {what} is over the {_mb(max_bytes)} limit.")
            fh.write(chunk)
    return dest


def _discard(upload_dir: Path):
    """Remove a staged upload we have decided not to analyse.

    Rejections used to leave the video on disk: the audio suffix is checked after
    the video has been written, so a bad voice-note extension cost a full upload
    and left it there. Nothing has been analysed at these points, so there is no
    reason to keep the file — and until WP-07 ships there is nothing else that
    would ever remove it.
    """
    shutil.rmtree(upload_dir, ignore_errors=True)


@router.get("/exercises", response_model=ExercisesResponse)
def exercises():
    """What the app can analyse, and how to film each one.

    The PWA calls this on load to build the exercise picker and, more usefully,
    to show the filming guidance *before* the user records anything — bad camera
    placement is the single biggest cause of a clip we can't assess properly
    (Decision 23), and it's much cheaper to prevent than to detect.
    """
    return ExercisesResponse(
        exercises=[
            ExerciseOut(
                id=ex_id,
                name=PROFILES[ex_id].label,
                view_label=PROFILES[ex_id].view_label,
                filming_guide=PROFILES[ex_id].filming_guide,
                assesses=PROFILES[ex_id].plane_assessments,
            )
            for ex_id in exercise_ids()
        ],
        limits=current_limits(),
    )


def current_limits() -> LimitsOut:
    """The upload constraints, read straight off the constants the route enforces.

    Built here rather than written out again so the served numbers cannot say one
    thing while /analyze does another.
    """
    return LimitsOut(
        max_video_bytes=MAX_VIDEO_BYTES,
        max_audio_bytes=MAX_AUDIO_BYTES,
        video_suffixes=sorted(ALLOWED_VIDEO_SUFFIXES),
        audio_suffixes=sorted(ALLOWED_AUDIO_SUFFIXES),
        min_seconds=MIN_CLIP_SECONDS,
        max_seconds=MAX_CLIP_SECONDS,
        ideal_min_seconds=IDEAL_MIN_SECONDS,
        ideal_max_seconds=IDEAL_MAX_SECONDS,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    video: UploadFile = File(..., description="exercise clip"),
    exercise_type: str = Form("squat"),
    voice_note: Optional[UploadFile] = File(default=None),
    voice_note_text: str = Form(""),
    pose_model: str = Form("full"),
    dry_run_coach: bool = Form(False),
):
    # ---- validate everything cheap before writing a byte
    if exercise_type not in ALLOWED_EXERCISES:
        raise HTTPException(400, f"unsupported exercise_type '{exercise_type}'. "
                                 f"Supported: {sorted(ALLOWED_EXERCISES)}")

    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(400, f"unsupported video format '{suffix}'. "
                                 f"Supported: {sorted(ALLOWED_VIDEO_SUFFIXES)}")

    # The audio suffix is checked here rather than after the video is saved. It
    # used to be checked later, which meant a mistyped voice-note extension was
    # only discovered once the whole clip had been uploaded and written to disk.
    a_suffix = ""
    if voice_note is not None and voice_note.filename:
        a_suffix = Path(voice_note.filename).suffix.lower()
        if a_suffix not in ALLOWED_AUDIO_SUFFIXES:
            raise HTTPException(400, f"unsupported audio format '{a_suffix}'. "
                                     f"Supported: {sorted(ALLOWED_AUDIO_SUFFIXES)}")

    # ---- stage uploads
    job_id = new_job_id(exercise_type)
    upload_dir = UPLOAD_ROOT / job_id
    video_path = _save_upload(video, upload_dir / f"input{suffix}",
                              MAX_VIDEO_BYTES, "clip")

    # Neither of these is knowable until the file is here, and both are cheaper
    # than finding out inside the pipeline.
    clip = probe_clip(video_path)
    if not clip.readable:
        # The extension said .mp4 and there is no video behind it. This used to
        # reach extract_landmarks, fail with "no frames decoded", and come back as
        # a 500 — a bad upload reported as our fault.
        _discard(upload_dir)
        raise HTTPException(
            400, "We couldn't read any video in that file. If you picked it from "
                 "your gallery, check it's the clip itself and not a photo or a "
                 "voice memo.")

    seconds = clip.seconds
    if seconds is not None and not (MIN_CLIP_SECONDS <= seconds <= MAX_CLIP_SECONDS):
        _discard(upload_dir)
        if seconds < MIN_CLIP_SECONDS:
            raise HTTPException(
                400, f"That clip is {seconds:.1f}s, which is too short to hold a "
                     f"full rep. Aim for {IDEAL_MIN_SECONDS:.0f}-"
                     f"{IDEAL_MAX_SECONDS:.0f} seconds.")
        raise HTTPException(
            400, f"That clip is {seconds:.0f}s and the limit is "
                 f"{MAX_CLIP_SECONDS:.0f}s. Trim it to a few good reps.")

    voice_path = ""
    if a_suffix:
        voice_path = str(_save_upload(voice_note,
                                      upload_dir / f"voice{a_suffix}",
                                      MAX_AUDIO_BYTES, "voice note"))

    # ---- run pipeline (same code as the CLI)
    opts = RunOptions(
        pose_model=pose_model,
        coach=True,
        force_dry_run_coach=dry_run_coach,
        voice_transcript=voice_note_text,
        voice_audio_path=voice_path,
    )
    # Refuse rather than pile up. A queued request would hold its upload in
    # memory while it waited and the phone would sit on a spinner it cannot
    # interpret; a 503 is something the PWA already renders as a retryable
    # server error, and it arrives immediately.
    if not _analysis_slots.acquire(blocking=False):
        _discard(upload_dir)
        log.warning("refused an analysis: all %d slots busy", MAX_CONCURRENT_ANALYSES)
        raise HTTPException(
            503,
            "The server is already analysing as many clips as it can handle at "
            "once. Give it a minute and try again — nothing was lost.")

    try:
        # exercise_type has to reach the pipeline, not just the job name. It
        # previously did not: this called the squat-shaped alias, so a push-up
        # upload was validated as a push-up, named pushup_<ts>, echoed back as a
        # push-up, and then analysed as a squat. Everything downstream looked
        # right except the measurements.
        result = run_pipeline(video_path, OUTPUT_ROOT, exercise_type,
                              options=opts, job_id=job_id)
    except Exception:
        # FileNotFoundError used to be caught separately and returned as a 400
        # carrying str(e). Neither case that raises it is the caller's fault: the
        # input video is missing only if our own staging failed, and the other
        # source is a missing pose model, whose message is a server path plus the
        # curl command to fix it. Both are ours, so both go through here.
        # Whatever went wrong here is ours, and the detail is ours too — it used
        # to be interpolated straight into the response, so a stack of internal
        # paths and module names went to whoever sent the request. The reference
        # is the compromise: the user gets something short they can quote, and it
        # ties their report to the traceback in the log without either of us
        # guessing which run they meant. It rides in a header so the rule that 5xx
        # bodies are never shown to users stays intact on the client.
        ref = secrets.token_hex(4)
        log.exception("analyze failed (ref %s, job %s, exercise %s)",
                      ref, job_id, exercise_type)
        raise HTTPException(
            500,
            "Something went wrong while analysing that clip.",
            headers={"X-Error-Reference": ref},
        )
    finally:
        _analysis_slots.release()

    return _build_response(result, exercise_type)


def _build_response(result: RunResult, exercise_type: str) -> AnalyzeResponse:
    # the static mount in main.py serves /results/<job_id>/<filename>
    base = f"/results/{result.job_id}"

    # we already have the data in angles.json — read it back rather than
    # re-deriving so the API response is exactly what the file contains
    with open(result.angles_path) as fh:
        angles_payload = json.load(fh)

    key_frames = []
    if result.worst_frame_path:
        idx = angles_payload.get("worst_frame")
        key_frames.append(KeyFrame(
            url=f"{base}/{result.worst_frame_path.name}",
            label="worst",
            timestamp=(idx / angles_payload["fps"]) if idx is not None else 0.0,
            highlighted=bool(angles_payload.get("worst_frame_flagged")),
        ))
    if result.best_frame_path:
        idx = angles_payload.get("best_frame")
        key_frames.append(KeyFrame(
            url=f"{base}/{result.best_frame_path.name}",
            label="best",
            timestamp=(idx / angles_payload["fps"]) if idx is not None else 0.0,
        ))

    reps_out = [RepStat(**r) for r in result.summary.get("reps", [])]

    report_out = None
    transcript = None
    if result.coaching_path:
        with open(result.coaching_path) as fh:
            cpayload = json.load(fh)
        report_out = CoachingReportOut(**cpayload["report"])
        transcript = cpayload.get("voice_transcript") or None

    return AnalyzeResponse(
        job_id=result.job_id,
        exercise_type=exercise_type,
        status=angles_payload.get("status", "ok"),
        fps=angles_payload["fps"],
        frame_count=angles_payload["frame_count"],
        side=angles_payload["side"],
        annotated_video_url=f"{base}/{result.annotated_video_path.name}",
        angles_url=f"{base}/angles.json",
        key_frames=key_frames,
        reps=reps_out,
        coaching_report=report_out,
        voice_transcript=transcript,
        warnings=result.summary.get("warnings", []),
    )

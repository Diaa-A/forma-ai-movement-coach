"""/analyze endpoint.

Multipart upload (video, optional voice note, exercise_type) -> run the same
pipeline the CLI uses -> JSON response with URLs to artefacts the static mount
in `backend.main` serves.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..pipeline.runner import run_pipeline, RunOptions, RunResult
from ..exercises.registry import PROFILES, exercise_ids
from .schemas import (AnalyzeResponse, KeyFrame, RepStat, CoachingReportOut,
                      ExerciseOut, ExercisesResponse)


router = APIRouter()


# upload + output roots. Keep them sibling so the /results static mount in
# backend.main serves both the uploaded original and the rendered artefacts.
DATA_ROOT       = Path("data")
UPLOAD_ROOT     = DATA_ROOT / "uploads"
OUTPUT_ROOT     = DATA_ROOT / "outputs"


MAX_VIDEO_BYTES = 100 * 1024 * 1024     # 100MB — generous but bounded
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".m4v"}
ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".webm", ".ogg"}
# derived from the profile registry, not written out again here — one list means
# the allowlist and the exercise picker can't drift apart. push-up / pull-up
# arrive by registering their profile (Phase G).
ALLOWED_EXERCISES = set(PROFILES)


def _save_upload(up: UploadFile, dest: Path, max_bytes: int):
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
                raise HTTPException(413, f"upload exceeds {max_bytes} bytes")
            fh.write(chunk)
    return dest


@router.get("/exercises", response_model=ExercisesResponse)
def exercises():
    """What the app can analyse, and how to film each one.

    The PWA calls this on load to build the exercise picker and, more usefully,
    to show the filming guidance *before* the user records anything — bad camera
    placement is the single biggest cause of a clip we can't assess properly
    (Decision 23), and it's much cheaper to prevent than to detect.
    """
    return ExercisesResponse(exercises=[
        ExerciseOut(
            id=ex_id,
            name=PROFILES[ex_id].label,
            view_label=PROFILES[ex_id].view_label,
            filming_guide=PROFILES[ex_id].filming_guide,
            assesses=PROFILES[ex_id].plane_assessments,
        )
        for ex_id in exercise_ids()
    ])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    video: UploadFile = File(..., description="exercise clip"),
    exercise_type: str = Form("squat"),
    voice_note: Optional[UploadFile] = File(default=None),
    voice_note_text: str = Form(""),
    pose_model: str = Form("full"),
    dry_run_coach: bool = Form(False),
):
    # ---- validate
    if exercise_type not in ALLOWED_EXERCISES:
        raise HTTPException(400, f"unsupported exercise_type '{exercise_type}'. "
                                 f"Supported: {sorted(ALLOWED_EXERCISES)}")

    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(400, f"unsupported video format '{suffix}'. "
                                 f"Supported: {sorted(ALLOWED_VIDEO_SUFFIXES)}")

    # ---- stage uploads
    job_id = f"{exercise_type}_{time.strftime('%Y%m%d_%H%M%S')}"
    upload_dir = UPLOAD_ROOT / job_id
    video_path = _save_upload(video, upload_dir / f"input{suffix}", MAX_VIDEO_BYTES)

    voice_path = ""
    if voice_note is not None and voice_note.filename:
        a_suffix = Path(voice_note.filename).suffix.lower()
        if a_suffix not in ALLOWED_AUDIO_SUFFIXES:
            raise HTTPException(400, f"unsupported audio format '{a_suffix}'")
        voice_path = str(_save_upload(voice_note,
                                      upload_dir / f"voice{a_suffix}",
                                      10 * 1024 * 1024))

    # ---- run pipeline (same code as the CLI)
    opts = RunOptions(
        pose_model=pose_model,
        coach=True,
        force_dry_run_coach=dry_run_coach,
        voice_transcript=voice_note_text,
        voice_audio_path=voice_path,
    )
    try:
        # exercise_type has to reach the pipeline, not just the job name. It
        # previously did not: this called the squat-shaped alias, so a push-up
        # upload was validated as a push-up, named pushup_<ts>, echoed back as a
        # push-up, and then analysed as a squat. Everything downstream looked
        # right except the measurements.
        result = run_pipeline(video_path, OUTPUT_ROOT, exercise_type,
                              options=opts, job_id=job_id)
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        # don't leak internals — log server-side, return generic
        # in dev we keep the message; in prod swap for an opaque error
        raise HTTPException(500, f"pipeline error: {e}")

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

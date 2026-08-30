"""Pipeline orchestration shared by the CLI and the API, for every exercise.

Single source of truth for the analysis flow so the API cannot drift from the
CLI. Both call `run_pipeline()` and get back the same artefacts.

Nothing here knows what a squat or a push-up is. Which angles to compute, how to
decide a rep happened, which cues to fire and which joints to colour all arrive
through the registry entry for the requested exercise. Adding an exercise is a
registration, not an edit to this file -- which was the point of WP-04, and is
easy to check: there is no exercise name below the imports.
"""
from __future__ import annotations

import json
import logging
import math
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .pose import extract_landmarks, N_LANDMARKS
from .filter import smooth_series
from .phase_detection import detect_bottoms, segment_reps, label_phases
from .render import render_video, save_key_frame, read_frames_exact
from .coaching import generate_coaching_report, not_analyzed_report
from . import whisper_wrapper
from ..exercises import mechanics
from ..exercises.registry import get_spec


log = logging.getLogger("coach.pipeline")


def new_job_id(exercise: str) -> str:
    """A job id has to be unique, not merely descriptive.

    The timestamp on its own collides for two uploads inside the same second, and
    that is not hypothetical: /analyze is a plain `def`, so FastAPI runs it in a
    threadpool and two people sharing a link genuinely run at once. Both would
    resolve to the same job directory, the second would overwrite the first, and
    the first person's /results URL would then serve the second person's video and
    key frames. Six hex characters make that collision not worth thinking about
    while keeping the timestamp readable, which is what the id is for.
    """
    return "{}_{}_{}".format(exercise, time.strftime("%Y%m%d_%H%M%S"),
                             secrets.token_hex(3))


@dataclass
class RunOptions:
    pose_model: str = "full"
    min_cutoff: float = 1.0
    beta: float = 0.007
    coach: bool = True
    force_dry_run_coach: bool = False
    voice_transcript: str = ""        # pl ain text (used directly if non-empty)
    voice_audio_path: str = ""        # optional — transcribed if provided
    whisper_model: Optional[str] = None
    # None lets coaching.active_model() decide at call time. A dataclass default
    # is evaluated at import, which is what made GROQ_MODEL look configurable
    # while doing nothing.
    llm_model: Optional[str] = None


@dataclass
class RunResult:
    job_id: str
    output_dir: Path
    angles_path: Path
    annotated_video_path: Path
    worst_frame_path: Optional[Path]
    best_frame_path: Optional[Path]
    coaching_path: Optional[Path]
    summary: dict = field(default_factory=dict)


def _smooth_landmarks(landmarks, timestamps, min_cutoff, beta):
    n_frames = landmarks.shape[0]
    out = landmarks.copy()
    for j in range(N_LANDMARKS):
        for c in range(3):   # x, y, z; visibility untouched
            out[:, j, c] = smooth_series(landmarks[:, j, c], timestamps,
                                         min_cutoff=min_cutoff, beta=beta)
    return out


def _angles_jsonable(angles_per_frame):
    out = []
    for d in angles_per_frame:
        cleaned = {}
        for k, v in d.items():
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                cleaned[k] = None
            else:
                cleaned[k] = round(float(v), 2)
        out.append(cleaned)
    return out


def _world_jsonable(world):
    out = []
    for f in range(len(world)):
        frame = []
        for j in range(world.shape[1]):
            x, y, z = world[f, j]
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                frame.append(None)
            else:
                frame.append([round(float(x), 4), round(float(y), 4), round(float(z), 4)])
        out.append(frame)
    return out


def run_pipeline(input_path, output_root, exercise: str = "squat",
                 options: RunOptions = None,
                 job_id: Optional[str] = None) -> RunResult:
    """One orchestrator, shared by the CLI and the API, for every exercise.

    Everything exercise-specific arrives through the registry: which angles to
    compute, how to gate reps, which cues to fire, which joints to colour. There
    is deliberately no exercise name anywhere below this line -- if one appeared,
    the abstraction would not be doing its job.
    """
    options = options or RunOptions()
    spec = get_spec(exercise)
    if spec is None:
        raise ValueError(f"unknown exercise '{exercise}'")
    movement = spec.movement
    in_path = Path(input_path).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"input video not found: {in_path}")

    if job_id is None:
        job_id = new_job_id(exercise)
    out_dir = Path(output_root).resolve() / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- pose -> filter -> angles
    pose_data = extract_landmarks(in_path, model=options.pose_model)
    n_frames = pose_data["frame_count"]
    fps = pose_data["fps"]
    if n_frames == 0:
        raise RuntimeError("no frames decoded from input video")

    lm_smooth = _smooth_landmarks(pose_data["landmarks"], pose_data["timestamps"],
                                  options.min_cutoff, options.beta)
    angles = spec.angles(lm_smooth)

    # -- phase detection
    hip_y = mechanics.travel_series(lm_smooth, *spec.travel_landmarks)
    hip_y_smooth = smooth_series(hip_y, pose_data["timestamps"],
                                 min_cutoff=0.5, beta=0.001)
    bottoms = detect_bottoms(hip_y_smooth, min_separation=max(3, int(fps * 0.4)))

    # -- side / rep validity (by body travel) / phases / scoring
    reps_all = segment_reps(bottoms, n_frames)
    side = mechanics.pick_side(movement, lm_smooth, reps_all)
    scale = mechanics.body_scale(movement, lm_smooth)
    reps = mechanics.keep_real_reps(movement, reps_all, hip_y_smooth, scale,
                                    angles_per_frame=angles, side=side, fps=fps)
    phases = label_phases(reps, n_frames)
    scores = mechanics.score_reps(movement, angles, reps, side, fps)
    worst_idx = mechanics.worst_frame(movement, angles, reps, side, fps)
    best_idx  = mechanics.best_frame(movement, angles, reps, side, fps)

    # -- analysis status: be honest when the clip can't be analysed.
    # "valid" = pose present AND anatomically plausible (excludes tracking
    # glitches), so the rate reflects usable frames, not just any detection.
    valid = sum(1 for a in angles if movement.is_valid(a, side))
    detection_rate = valid / n_frames if n_frames else 0.0
    # The rotation check comes first because it explains the others: a sideways-
    # decoded file usually still tracks well, produces a rep count and a status
    # of ok, and is wrong about all of it -- the reference clip with its rotation
    # flag stripped came back 1 rep instead of 7, with its best frame flagged as
    # a forward-lean fault. Declared rather than analysed, per the coverage rule.
    tilt = mechanics.body_axis_tilt(lm_smooth)
    if mechanics.axis_looks_rotated(movement, tilt):
        status = "rotated"
    elif not reps:
        status = "no_reps"
    elif detection_rate < 0.5:
        status = "low_detection"
    else:
        status = "ok"

    # Fault colouring for the overlay, evaluated per frame rather than only on the
    # single worst one — one red frame in six hundred is a 30th of a second and
    # reads as "nothing was ever wrong". See the exercise module.
    flagged = spec.flag_frames(angles, reps, side)

    # -- write angles.json (with world landmarks for the Fit3D path)
    angles_path = out_dir / "angles.json"
    angles_payload = {
        "input": str(in_path),
        "model": options.pose_model,
        "fps": fps,
        "frame_count": n_frames,
        "side": side,
        "status": status,
        "detection_rate": round(detection_rate, 3),
        "body_axis_tilt": round(tilt, 1) if tilt is not None else None,
        "bottoms_raw": bottoms,
        "bottoms": [b for (_, b, _) in reps],
        "reps": [
            {
                "start": s, "bottom": b, "end": e, "eval_frame": f,
                "score": round(sc, 2) if np.isfinite(sc) else None,
                "breakdown": bd,
            } for (s, b, e), (f, sc, bd) in zip(reps, scores)
        ],
        "phase_per_frame": phases,
        "worst_frame": worst_idx,
        "best_frame":  best_idx,
        # Did anything actually get marked as a fault on the worst frame? On a
        # clean set nothing does, and then calling the image "worst form" sends
        # the user hunting for a red limb that was never there. The label and the
        # caption both key off this.
        "worst_frame_flagged": bool(worst_idx is not None and flagged[worst_idx]),
        "worst_frame_joints": sorted(flagged[worst_idx]) if worst_idx is not None else [],
        "angles_per_frame": _angles_jsonable(angles),
        "world_landmarks": _world_jsonable(pose_data["world_landmarks"]),
    }
    with open(angles_path, "w") as fh:
        json.dump(angles_payload, fh, indent=2)

    # -- annotated video + key frames
    annotated_path = out_dir / "annotated.mp4"
    render_video(in_path, lm_smooth, angles, annotated_path,
                 flagged_per_frame=flagged, caption=spec.caption, side=side)

    # one exact decode pass for both key frames — see read_frames_exact for why
    # this isn't a seek
    key_frames = read_frames_exact(in_path, [worst_idx, best_idx])

    worst_path = None
    best_path  = None
    if worst_idx is not None and worst_idx in key_frames:
        worst_path = out_dir / "worst.jpg"
        # only call it the worst if something is actually marked on it
        worst_label = "worst form" if flagged[worst_idx] else "closest to the limit"
        save_key_frame(in_path, worst_idx, lm_smooth[worst_idx], angles[worst_idx],
                       worst_path, flagged[worst_idx], label=worst_label, side=side,
                       frame=key_frames[worst_idx], caption=spec.caption)
    if best_idx is not None and best_idx in key_frames:
        best_path = out_dir / "best.jpg"
        save_key_frame(in_path, best_idx, lm_smooth[best_idx], angles[best_idx],
                       best_path, set(), label="best", side=side,
                       frame=key_frames[best_idx], caption=spec.caption)

    summary = _build_summary(angles, reps, side, scores, movement)
    summary["status"] = status
    summary["detection_rate"] = round(detection_rate, 3)

    # -- voice transcription (optional) + Phase C coaching
    coaching_path = None
    if options.coach:
        if status != "ok":
            # nothing to coach — emit an honest report and skip the LLM entirely.
            # The profile goes in so the message names the exercise the user
            # actually uploaded and re-uses that exercise's own filming guidance:
            # this text used to be squat-shaped for everything, so a push-up that
            # failed to track was told we could not find a complete squat rep.
            report = not_analyzed_report(status, spec.profile)
            payload = {"status": status, "evaluation": None,
                       "voice_transcript": "", "report": report.to_dict()}
        else:
            voice_text = options.voice_transcript or ""
            if options.voice_audio_path:
                try:
                    voice_text = whisper_wrapper.transcribe(
                        options.voice_audio_path, model_size=options.whisper_model,
                    )
                except RuntimeError:
                    # The exception text here is the provider's own error body,
                    # which the API returns in warnings[] and the PWA renders
                    # verbatim in a banner. An invalid key put "Groq transcription
                    # error 401: {...}" on the user's screen. Keep the failure
                    # visible — dropping it silently would be worse — but say it in
                    # the user's terms and leave the detail in the log.
                    log.warning("voice transcription failed for job %s", job_id,
                                exc_info=True)
                    summary.setdefault("warnings", []).append(
                        "Your voice note couldn't be transcribed, so the coaching "
                        "below doesn't take it into account. Everything else was "
                        "analysed normally."
                    )

            evaluation = spec.evaluate(angles, reps , side, fps,
                                       phase_per_frame=phases, landmarks=lm_smooth)
            report = generate_coaching_report(
                evaluation, voice_transcript=voice_text,
                model=options.llm_model,
                force_dry_run=options.force_dry_run_coach,
            )
            payload = {
                "status": status,
                "evaluation": {
                    "exercise": evaluation.exercise,
                    "side": evaluation.side,
                    "rep_count": evaluation.rep_count,
                    "cues_fired": [
                        {
                            "flag": c.flag, "severity": c.severity,
                            "fault": c.fault, "fix": c.fix,
                            "rep_indices": c.rep_indices,
                        } for c in evaluation.cues_fired
                    ],
                    "positives": evaluation.positives,
                    "notes": evaluation.notes,
                },
                "voice_transcript": voice_text,
                "report": report.to_dict(),
            }

        coaching_path = out_dir / "coaching.json"
        with open(coaching_path, "w") as fh:
            json.dump(payload, fh, indent=2)

    return RunResult(
        job_id=job_id,
        output_dir=out_dir,
        angles_path=angles_path,
        annotated_video_path=annotated_path,
        worst_frame_path=worst_path,
        best_frame_path=best_path,
        coaching_path=coaching_path,
        summary=summary,
    )


def _build_summary(angles, reps, side, scores, movement=None):
    # the angle that flexes, whichever joint that is for this exercise
    key = movement.primary_angle(side) if movement else ("knee_left" if side == "left" else "knee_right")
    series = np.array([a.get(key) for a in angles], dtype=np.float64)
    finite = series[np.isfinite(series)]
    # "knee_*" names are kept because the API schema and the report figures
    # already use them; for a push-up they carry the elbow. Renaming would ripple
    # into the response model and the chapter 4 numbers for no gain.
    out = {
        "side": side,
        "rep_count": len(reps),
        "primary_angle": key,
        "knee_min": float(finite.min()) if finite.size else None,
        "knee_max": float(finite.max()) if finite.size else None,
        "knee_mean": float(finite.mean()) if finite.size else None,
        "reps": [],
    }
    for (s, b, e), (f, sc, bd) in zip(reps, scores):
        rep_series = series[s:e + 1]
        rep_series = rep_series[np.isfinite(rep_series)]
        out["reps"].append({
            "start": s, "bottom": b, "end": e, "eval_frame": f,
            "knee_min": float(rep_series.min()) if rep_series.size else None,
            "knee_max": float(rep_series.max()) if rep_series.size else None,
            "score": sc if (sc is not None and np.isfinite(sc)) else None,
            "breakdown": bd,
        })
    return out


def run_squat_pipeline(input_path , output_root, options: RunOptions = None,
                       job_id: Optional[str] = None) -> RunResult:
    """Squat-shaped alias kept so existing callers and tests do not have to change.
    It is the same single orchestrator, not a second one."""
    return run_pipeline(input_path, output_root, "squat", options=options,
                        job_id=job_id)

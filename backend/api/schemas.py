"""Pydantic models for the /analyze and /exercises responses.

The request is multipart/form-data and is unpacked in the route handler.
"""
from typing import List, Optional, Any
from pydantic import BaseModel, Field


class ExerciseOut(BaseModel):
    """One entry in GET /exercises.

    `filming_guide` is shown before recording, so guidance and coaching come from
    the same string. `assesses` maps plane -> assessments, which is how the UI
    says what a camera view can cover without knowing any anatomy.
    """
    id: str
    name: str
    view_label: str
    filming_guide: str
    assesses: dict


class LimitsOut(BaseModel):
    """What /analyze will accept, served rather than duplicated.

    Same argument as filming_guide. The frontend used to keep its own copy of
    these numbers and it had already drifted -- it claimed a 3-45 second gate
    that didn't exist anywhere in the server.

    min/max is what gets refused outright, ideal_* only earns a warning.
    """
    max_video_bytes: int
    max_audio_bytes: int
    video_suffixes: List[str]
    audio_suffixes: List[str]
    min_seconds: float
    max_seconds: float
    ideal_min_seconds: float
    ideal_max_seconds: float
    # The consent copy states this period, so it comes from the sweep that
    # enforces it. retention_note is the wording, built from the number.
    retention_hours: float
    retention_note: str


class ExercisesResponse(BaseModel):
    exercises: List[ExerciseOut]
    limits: LimitsOut
    # only set while a testing round is running. The {job_id} placeholder gets
    # filled in by the client -- that join is what turns "participant 3 rated
    # clarity 4" into something you can actually use
    feedback_form_url: Optional[str] = None


class KeyFrame(BaseModel):
    url: str
    label: str
    timestamp: float
    # True when joints were marked at fault on this frame. On a clean set there
    # is no red limb, and captioning it "where form drifted most" sends the user
    # hunting for one that isn't there.
    highlighted: bool = False


class RepStat(BaseModel):
    start: int
    bottom: int
    end: int
    knee_min: Optional[float] = None
    knee_max: Optional[float] = None
    score: Optional[float] = None
    breakdown: Optional[dict] = None


class CoachingReportOut(BaseModel):
    # Declared, or pydantic drops it and the field silently never arrives.
    answer_to_question: Optional[str] = None
    what_went_well: List[str]
    primary_issue: str
    secondary_issues: List[str]
    corrective_cues: List[str]
    next_session_focus: str
    source: str
    model: Optional[str] = None
    filming_tip: Optional[str] = None


class AnalyzeResponse(BaseModel):
    job_id: str
    exercise_type: str
    status: str               # "ok" | "no_reps" | "low_detection" | "rotated"
    fps: float
    frame_count: int
    side: str                 # "left" | "right", or "both" for the pull-up
    annotated_video_url: str
    angles_url: str
    key_frames: List[KeyFrame]
    reps: List[RepStat]
    coaching_report: Optional[CoachingReportOut] = None
    voice_transcript: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

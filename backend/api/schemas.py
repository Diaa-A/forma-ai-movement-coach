"""Pydantic models for /analyze response.

The request itself is multipart/form-data so we don't model it here — FastAPI
unpacks the parts directly in the route handler. See `routes.py`.
"""
from typing import List, Optional, Any
from pydantic import BaseModel, Field


class KeyFrame(BaseModel):
    url: str
    label: str
    timestamp: float


class RepStat(BaseModel):
    start: int
    bottom: int
    end: int
    knee_min: Optional[float] = None
    knee_max: Optional[float] = None
    score: Optional[float] = None
    breakdown: Optional[dict] = None


class CoachingReportOut(BaseModel):
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
    status: str               # "ok" | "no_reps" | "low_detection"
    fps: float
    frame_count: int
    side: str
    annotated_video_url: str
    angles_url: str
    key_frames: List[KeyFrame]
    reps: List[RepStat]
    coaching_report: Optional[CoachingReportOut] = None
    voice_transcript: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

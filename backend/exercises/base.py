"""Per-exercise camera-view profiles — the scalability layer.

A single camera cannot measure a joint it cannot see, so the robust, industry
approach is to constrain the input: each exercise declares the camera VIEW that
makes its key joints visible, and which assessments live in which anatomical
plane. Metrics in the sagittal plane (depth, forward lean) need a side-on view;
metrics in the frontal plane (left/right symmetry, knee valgus) need a front-on
view. The confidence gate then enforces this automatically at runtime — but the
profile lets the system explain it *a priori* and tell the user which view to use
for what, instead of silently going quiet.

push-up and pull-up reuse this structure: each gets an ExerciseProfile plus a
cue database whose cues are tagged with the plane they belong to.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Anatomical planes -> the camera view that exposes them.
SAGITTAL = "sagittal"   # side-on: flexion/extension, depth, forward lean
FRONTAL  = "frontal"    # front-on: left/right symmetry, knee valgus / tracking


@dataclass
class ExerciseProfile:
    name: str
    view_label: str                         # short, e.g. "side-on"
    filming_guide: str                      # one-line instruction shown pre-upload
    plane_assessments: Dict[str, List[str]] # plane -> what that view lets you assess

    def assessments(self, plane: str) -> List[str]:
        return self.plane_assessments.get(plane, [])


def _join(items: List[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def coverage_guidance(profile: ExerciseProfile, frontal_observed: bool) -> Optional[str]:
    """Actionable guidance about what THIS clip's view could and couldn't assess.

    Returns None when nothing extra needs saying (the frontal plane was visible,
    so both planes are covered). Otherwise it names the frontal-plane assessments
    that this side-on clip cannot reach and tells the user to film a front view —
    turning a silent gap into a clear instruction."""
    if frontal_observed:
        return None
    frontal_items = profile.assessments(FRONTAL)
    if not frontal_items:
        return None
    sagittal_items = profile.assessments(SAGITTAL)
    covered = _join(sagittal_items)
    missing = _join(frontal_items)
    lead = f"This clip is {profile.view_label}, which covers {covered}. " if covered \
        else f"This clip is {profile.view_label}. "
    return lead + f"To check {missing}, film a set from the front."

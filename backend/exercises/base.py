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
    # What the UI puts on the button. Optional because `name` is a fine default
    # for the ones that are a single lowercase word; it exists for the ones that
    # aren't ("pushup" should read "Push-up", not "Pushup").
    display_name: Optional[str] = None
    # Declared, but no working detector behind it yet -- a cue carrying
    # `available: False`, like the squat's knee valgus or the push-up's elbow
    # flare. These must NOT appear in plane_assessments: saying a view "covers"
    # something the system never actually reports on is a different lie from
    # staying silent about it, and arguably a worse one, because the user reads
    # coverage as a clean bill of health. They get named separately instead.
    not_yet_assessed: List[str] = field(default_factory=list)

    def assessments(self, plane: str) -> List[str]:
        return self.plane_assessments.get(plane, [])

    @property
    def label(self) -> str:
        return self.display_name or self.name.capitalize()


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
    parts = []

    frontal_items = profile.assessments(FRONTAL)
    if not frontal_observed and frontal_items:
        covered = _join(profile.assessments(SAGITTAL))
        lead = f"This clip is {profile.view_label}, which covers {covered}. " if covered \
            else f"This clip is {profile.view_label}. "
        parts.append(lead + f"To check {_join(frontal_items)}, film a set from the front.")

    # Named whichever way the clip was filmed, because a detector that does not
    # work yet is not fixed by changing the camera angle. Saying so is what keeps
    # "nothing was reported" from reading as "nothing was wrong".
    if profile.not_yet_assessed:
        parts.append(f"Not assessed yet by this system: "
                     f"{_join(profile.not_yet_assessed)}.")

    return " ".join(parts) if parts else None

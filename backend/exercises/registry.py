"""One place that knows which exercises exist.

`routes.py` used to carry its own `ALLOWED_EXERCISES = {"squat"}` literal, which
is fine right up until someone adds an exercise and updates one of the two lists.
Everything now derives from `PROFILES` below, so the API's allowlist and the
frontend's exercise picker cannot disagree with each other — adding push-up means
registering a profile here and nothing else.

The frontend reads this through `GET /exercises` rather than hard-coding the
filming guidance, which matters because that text also feeds the coaching output.
Two copies of it would drift the first time a threshold moved.
"""
from typing import Dict, List, Optional

from .base import ExerciseProfile
from .squat_cues import SQUAT_PROFILE


# id -> profile. The id is what the client sends as `exercise_type`, so keep it
# lowercase and url-safe.
PROFILES: Dict[str, ExerciseProfile] = {
    "squat": SQUAT_PROFILE,
}


def exercise_ids() -> List[str]:
    """Sorted so the picker order is stable rather than dict-insertion order."""
    return sorted(PROFILES)


def get_profile(exercise_id: str) -> Optional[ExerciseProfile]:
    return PROFILES.get(exercise_id)


def is_supported(exercise_id: str) -> bool:
    return exercise_id in PROFILES

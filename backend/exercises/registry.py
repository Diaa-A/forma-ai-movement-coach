"""One place that knows which exercises exist and how to run each one.

`routes.py` used to carry its own `ALLOWED_EXERCISES = {"squat"}` literal, which
is fine right up until someone adds an exercise and updates one of the two lists.
Everything now derives from `EXERCISES` below.

Adding an exercise means writing its three modules and adding one entry here. The
pipeline, the API allowlist, the frontend picker and the filming guidance all
follow from that — none of them needs editing, and none of them contains a name
of an exercise.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from ..pipeline.angles import (pullup_angles_per_frame, pushup_angles_per_frame,
                               squat_angles_per_frame)
from .base import ExerciseProfile
from .mechanics import Movement
from . import pullup, pushup, squat
from .pullup_cues import PULLUP_PROFILE, evaluate_pullup
from .pushup_cues import PUSHUP_PROFILE, evaluate_pushup
from .squat_cues import SQUAT_PROFILE, evaluate_squat


@dataclass(frozen=True)
class ExerciseSpec:
    """Everything the pipeline needs to analyse one exercise.

    The runner reads this and nothing else, which is what keeps exercise names out
    of `pipeline/`. That is the whole point of the split and it is the part worth
    checking: if a name had leaked, this indirection would be pointless.
    """
    profile: ExerciseProfile
    movement: Movement
    angles: Callable          # landmarks -> list of per-frame angle dicts
    evaluate: Callable        # Layer 1 cue evaluation
    flag_frames: Callable     # per-frame joints to draw in fault colour
    caption: Callable         # angle dict + side -> overlay readout lines
    # symmetric landmark pair whose mean vertical position drives phase detection.
    # Both current exercises use the hips, which is a finding rather than a
    # coincidence -- see the note in the work-package documentation.
    travel_landmarks: tuple = ("left_hip", "right_hip")
    # scored on both arms together rather than on the better-seen side, so the API
    # reports "both" instead of the side the runner picked for drawing
    scores_both_sides: bool = False


EXERCISES: Dict[str, ExerciseSpec] = {
    "squat": ExerciseSpec(
        profile=SQUAT_PROFILE,
        movement=squat.SQUAT,
        angles=squat_angles_per_frame,
        evaluate=evaluate_squat,
        flag_frames=squat.flag_frames,
        caption=squat.frame_caption,
    ),
    "pushup": ExerciseSpec(
        profile=PUSHUP_PROFILE,
        movement=pushup.PUSHUP,
        angles=pushup_angles_per_frame,
        evaluate=evaluate_pushup,
        flag_frames=pushup.flag_frames,
        caption=pushup.frame_caption,
    ),
    "pullup": ExerciseSpec(
        profile=PULLUP_PROFILE,
        movement=pullup.PULLUP,
        angles=pullup_angles_per_frame,
        evaluate=evaluate_pullup,
        flag_frames=pullup.flag_frames,
        caption=pullup.frame_caption,
        scores_both_sides=True,
    ),
}


# kept for the API and the /exercises endpoint, which only want the profiles
PROFILES: Dict[str, ExerciseProfile] = {k: v.profile for k, v in EXERCISES.items()}


def exercise_ids() -> List[str]:
    """Sorted so the picker order is stable rather than dict-insertion order."""
    return sorted(EXERCISES)


def get_spec(exercise_id: str) -> Optional[ExerciseSpec]:
    return EXERCISES.get(exercise_id)


def get_profile(exercise_id: str) -> Optional[ExerciseProfile]:
    spec = EXERCISES.get(exercise_id)
    return spec.profile if spec else None


def is_supported(exercise_id: str) -> bool:
    return exercise_id in EXERCISES

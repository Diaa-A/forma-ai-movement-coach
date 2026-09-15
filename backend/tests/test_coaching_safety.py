"""What the coaching layer is allowed to say when something has gone wrong.

Two failure paths, and both used to say too much or say the wrong thing. Neither
is about the quality of the coaching — they are about the report the user reads
when the system could not do its job, which is where wording matters most because
there is no analysis for it to be checked against.
"""
import json

import pytest

from backend.exercises.mechanics import Evaluation
from backend.exercises.pullup_cues import PULLUP_PROFILE
from backend.exercises.pushup_cues import PUSHUP_PROFILE
from backend.exercises.squat_cues import SQUAT_PROFILE
from backend.pipeline import coaching


def an_evaluation(exercise="squat"):
    return Evaluation(
        exercise=exercise,
        side="left",
        rep_count=4,
        cues_fired=[],
        positives=["you completed the set"],
        notes=["average knee angle at the bottom: 88°"],
    )


# ---------------------------------------------------------------------------
# advice about something the system did not check
# ---------------------------------------------------------------------------

def the_model_replies(monkeypatch, next_session_focus):
    monkeypatch.setenv("GROQ_API_KEY", "not-a-real-key")
    reply = {
        "what_went_well": ["every rep came up to about the same height"],
        "primary_issue": "No primary fault was detected in this set.",
        "secondary_issues": [],
        "corrective_cues": [],
        "next_session_focus": next_session_focus,
        "answer_to_question": "The system could not tell whether your chin cleared the bar.",
    }
    monkeypatch.setattr(coaching, "_call_groq", lambda *a, **k: {
        "choices": [{"message": {"content": json.dumps(reply)}}]})


@pytest.mark.parametrize("exercise,profile,line", [
    ("pullup", PULLUP_PROFILE, "Keep the consistent rep height and make sure you fully "
                               "extend your arms and clear the bar on each pull-up."),
    ("pushup", PUSHUP_PROFILE, "Keep your elbows tucked at about 45 degrees on every rep."),
    ("squat", SQUAT_PROFILE, "Keep your heels down and your knees out as you stand up."),
])
def test_the_next_session_line_may_not_advise_on_what_was_not_assessed(
        monkeypatch, exercise, profile, line):
    """The pull-up line is the one a live report ended with, straight after saying
    it could not see the chin or the arms."""
    the_model_replies(monkeypatch, line)
    report = coaching.generate_coaching_report(an_evaluation(exercise), profile=profile)
    assert report.next_session_focus == \
        coaching._dry_run_report(an_evaluation(exercise)).next_session_focus
    assert report.source == "llm"
    # saying what was not measured is still allowed, in the answer
    assert "chin" in report.answer_to_question


def test_a_next_session_line_about_what_was_measured_is_kept(monkeypatch):
    the_model_replies(monkeypatch, "Keep every rep at the same height, even on the last few.")
    report = coaching.generate_coaching_report(an_evaluation("pullup"), profile=PULLUP_PROFILE)
    assert report.next_session_focus == "Keep every rep at the same height, even on the last few."


# ---------------------------------------------------------------------------
# the LLM failing
# ---------------------------------------------------------------------------

def test_an_llm_failure_puts_nothing_of_the_exception_in_the_report(monkeypatch):
    """The handler used to insert f"(LLM call failed: {e}...)" at the front of
    corrective_cues. That string is rendered to the user as coaching advice and
    written into coaching.json, which is served under /results/ — so a bad key
    published the provider's error body on a public URL.
    """
    monkeypatch.setenv("GROQ_API_KEY", "not-a-real-key")

    def refuse(*args, **kwargs):
        raise RuntimeError("Groq API error 401: {\"message\": \"Invalid API Key\"}")

    monkeypatch.setattr(coaching, "_call_groq", refuse)

    report = coaching.generate_coaching_report(an_evaluation())
    everything = " ".join(report.corrective_cues + report.secondary_issues
                          + report.what_went_well + [report.primary_issue])
    assert "401" not in everything
    assert "Invalid API Key" not in everything
    assert "Groq" not in everything


def test_an_llm_failure_is_still_visible_in_the_source_field(monkeypatch):
    """Removing the leaked string must not make the failure silent. `source` is
    what the UI turns into a sentence telling the user the wording is ours."""
    monkeypatch.setenv("GROQ_API_KEY", "not-a-real-key")
    monkeypatch.setattr(coaching, "_call_groq",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))

    report = coaching.generate_coaching_report(an_evaluation())
    assert report.source == "dry_run_fallback"


def test_an_llm_failure_still_returns_a_complete_report(monkeypatch):
    """A coaching failure must never cost the user the analysis."""
    monkeypatch.setenv("GROQ_API_KEY", "not-a-real-key")
    monkeypatch.setattr(coaching, "_call_groq",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))

    report = coaching.generate_coaching_report(an_evaluation())
    assert report.what_went_well
    assert report.primary_issue
    assert report.next_session_focus


# ---------------------------------------------------------------------------
# the clip not being analysable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("profile,expected", [
    (SQUAT_PROFILE, "squat"),
    (PUSHUP_PROFILE, "push-up"),
    (PULLUP_PROFILE, "pull-up"),
])
def test_the_failure_report_names_the_exercise_that_was_uploaded(profile, expected):
    """This said "I couldn't detect a complete squat rep" whatever had been sent.
    A push-up clip that failed to track — which happened on two of the six
    reference clips — told the user we could not find a squat.
    """
    report = coaching.not_analyzed_report("no_reps", profile)
    assert expected in report.primary_issue


def test_the_failure_report_re_uses_the_exercise_own_filming_guidance():
    """The advice was squat-shaped and hardcoded, so a push-up user was told to
    film at hip height. Taking it from the profile is the same argument as
    GET /exercises serving filming_guide instead of the frontend holding a copy.
    """
    report = coaching.not_analyzed_report("no_reps", PUSHUP_PROFILE)
    assert PUSHUP_PROFILE.filming_guide in report.corrective_cues


def test_the_failure_report_awards_nothing(monkeypatch):
    """The property the whole not-analysed path exists for: an empty
    what_went_well means the UI renders no heading at all, so "we could not
    measure this" can never be read as "your form was fine".
    """
    for status in ("no_reps", "low_detection", "rotated"):
        report = coaching.not_analyzed_report(status, SQUAT_PROFILE)
        assert report.what_went_well == []
        assert report.secondary_issues == []
        assert report.source == "not_analyzed"


def test_the_failure_report_works_without_a_profile():
    """The parameter is optional so older callers keep working; it should degrade
    to something generic rather than naming an exercise it was not given."""
    report = coaching.not_analyzed_report("no_reps")
    assert "squat" not in report.primary_issue
    assert report.corrective_cues


@pytest.mark.parametrize("exercise", ["squat", "pushup", "pullup"])
def test_the_fallback_does_not_call_an_unchecked_set_clean(exercise):
    """With no cue fired this said "No major form fault was detected", which
    spoke for faults that were never checked: knee cave on the squat, elbow
    flare on the push-up, four parked cues on the pull-up."""
    report = coaching._dry_run_report(an_evaluation(exercise))
    assert "no major form fault" not in report.primary_issue.lower()
    assert "checks for" in report.primary_issue
    assert "standard you set" not in report.next_session_focus

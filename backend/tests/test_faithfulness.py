"""The faithfulness checker, and the output contract it depends on (WP-05).

None of this needs a network. The checker is pure, and the contract tests drive
`generate_coaching_report` with a stubbed transport — which matters, because a
harness whose own correctness is only demonstrated by running it against a live
model would be measuring two unknowns at once.

The cases below are the ones the checker has to get right for the headline number
to mean anything: a faithful report must score clean, and each violation class
must be caught on text that is otherwise plausible coaching.
"""
import pytest

from backend.evaluation import faithfulness as F
from backend.exercises.mechanics import CueHit, Evaluation
from backend.pipeline import coaching


def evaluation(cues=(), positives=(), notes=(), exercise="squat"):
    return Evaluation(
        exercise=exercise, side="left", rep_count=5,
        cues_fired=list(cues), positives=list(positives), notes=list(notes),
    )


LEAN = CueHit(
    flag="excessive_forward_lean", severity="primary",
    fault=("at the bottom the torso leaned noticeably more than the shins — the "
           "chest dropped toward the floor while the lower legs stayed more upright."),
    fix="Drive your chest up as you descend so the torso stays roughly parallel to your shins.",
    joints=["left_hip"], rep_indices=[0, 1],
)

SHALLOW = CueHit(
    flag="shallow_depth", severity="primary",
    fault="knees stayed above parallel at the bottom — hips never dropped level with the knees.",
    fix="Aim to reach hip-crease level with the knee at the bottom.",
    joints=["knee"], rep_indices=[0],
)


def report(**kw):
    base = dict(what_went_well=[], primary_issue="", secondary_issues=[],
                corrective_cues=[], next_session_focus="", source="llm",
                model="test", filming_tip=None)
    base.update(kw)
    return coaching.CoachingReport(**base)


# ---------------------------------------------------------------------------
# the authorised set
# ---------------------------------------------------------------------------

def test_notes_are_authorised_so_quoting_an_angle_is_not_invention():
    """The point of 7.2 in the code review. `notes` carries derived angle values
    into the prompt, so a model repeating them is doing its job. If the checker
    flagged this it would report violations for the model behaving correctly."""
    ev = evaluation(notes=["average knee angle at the bottom: 88° (target ~90°)"])
    auth = F.authorised_from(ev)
    r = report(primary_issue="Your knees reached about 88 degrees at the bottom.",
               next_session_focus="Keep that depth.")
    assert F.check(r, auth) == []


def test_the_transcript_authorises_what_the_user_asked_about():
    ev = evaluation(positives=["you completed the set"])
    auth = F.authorised_from(ev, voice_transcript="am I going deep enough in my squat")
    r = report(primary_issue="You asked about depth, and you were going deep enough.",
               next_session_focus="Keep it there.")
    assert F.check(r, auth) == []


# ---------------------------------------------------------------------------
# un-cued claims
# ---------------------------------------------------------------------------

def test_a_body_part_layer_one_never_mentioned_is_flagged():
    """The archetypal failure: the model knows squats, knows knees cave in, and
    says so about a person whose knees it cannot see. knee_valgus is a PARKED cue
    precisely because the system cannot measure it."""
    ev = evaluation(cues=[LEAN])
    auth = F.authorised_from(ev)
    r = report(primary_issue="Your torso leaned forward more than your shins.",
               secondary_issues=["Your knees caved inward on the way up."],
               next_session_focus="Chest up.")
    kinds = {v.kind for v in F.check(r, auth)}
    assert "un-cued" in kinds


def test_a_faithful_rephrasing_scores_clean():
    ev = evaluation(cues=[LEAN], positives=["you reached full depth on every rep"])
    auth = F.authorised_from(ev)
    r = report(
        what_went_well=["You hit full depth on every single rep."],
        primary_issue="At the bottom your chest dropped toward the floor while your shins stayed upright.",
        corrective_cues=["Drive your chest up as you lower so your torso stays parallel to your shins."],
        next_session_focus="Chest up on every rep.")
    assert F.check(r, auth) == []


# ---------------------------------------------------------------------------
# contradictions
# ---------------------------------------------------------------------------

def test_praising_the_thing_the_cue_flagged_is_a_contradiction():
    """Worse than invention: it tells the user the opposite of the finding."""
    ev = evaluation(cues=[SHALLOW])
    auth = F.authorised_from(ev)
    r = report(what_went_well=["Great depth — you got well below parallel."],
               primary_issue="Your knees stayed above parallel at the bottom.",
               next_session_focus="Keep working on depth.")
    kinds = {v.kind for v in F.check(r, auth)}
    assert "contradicted" in kinds


def test_a_corrective_cue_restating_the_fix_is_not_a_contradiction():
    """The bug that made the first real run report 33% faithfulness, entirely in
    false positives.

    A fix necessarily describes the desired state. The fix for shallow depth says
    to reach hip-crease level, so a faithful rephrasing reads "aim for full depth"
    — and a checker that scans instructions for the good version of the fault
    calls that a contradiction. It is the opposite: it is the model doing exactly
    what Layer 1 told it to say.
    """
    ev = evaluation(cues=[SHALLOW])
    auth = F.authorised_from(ev)
    r = report(primary_issue="Your knees stayed above parallel at the bottom.",
               corrective_cues=["Aim to reach full depth on every rep — hips level "
                                "with the knees."],
               next_session_focus="Work toward getting deep enough each time.")
    assert [v for v in F.check(r, auth) if v.kind == "contradicted"] == []


# ---------------------------------------------------------------------------
# the confidence gate
# ---------------------------------------------------------------------------

def test_symmetry_claims_are_flagged_when_the_gate_withheld_it():
    """Decision 19: on a side-on clip the far leg is occluded, so the left/right
    comparison is withheld. A model claiming symmetry is claiming a measurement
    the system deliberately refused to make."""
    ev = evaluation(cues=[LEAN])
    auth = F.authorised_from(ev)
    assert not auth.symmetry_assessed
    r = report(what_went_well=["Both knees tracked evenly through the movement."],
               primary_issue="Your torso leaned more than your shins.",
               next_session_focus="Chest up.")
    kinds = {v.kind for v in F.check(r, auth)}
    assert "ungated" in kinds


def test_the_word_symmetry_itself_is_matched():
    """The first version used `symmetr\\b`, which can never match "symmetry" —
    there is no word boundary between the r and the y. It silently passed every
    claim phrased with the most obvious word for it."""
    ev = evaluation(cues=[LEAN])
    auth = F.authorised_from(ev)
    r = report(what_went_well=["Your left/right symmetry looked good throughout."],
               primary_issue="Your torso leaned more than your shins.",
               next_session_focus="Chest up.")
    assert [v for v in F.check(r, auth) if v.kind == "ungated"]


def test_being_told_to_film_from_the_front_is_not_an_ungated_claim():
    """The deterministic filming guidance says symmetry needs a front-on clip.
    A model repeating that is telling the user the measurement was NOT made,
    which is the honest thing and the opposite of claiming it."""
    ev = evaluation(cues=[LEAN],
                    notes=["This clip is side-on. To check left/right symmetry, "
                           "film a set from the front."])
    auth = F.authorised_from(ev)
    r = report(primary_issue="Your torso leaned more than your shins.",
               corrective_cues=["To check left/right symmetry, film a set from the front."],
               next_session_focus="Chest up.")
    assert [v for v in F.check(r, auth) if v.kind == "ungated"] == []


def test_symmetry_claims_are_fine_when_it_was_actually_measured():
    ev = evaluation(
        cues=[LEAN],
        positives=["left and right knees reached the same depth at the bottom — "
                   "no obvious side-to-side shift."])
    auth = F.authorised_from(ev)
    assert auth.symmetry_assessed
    r = report(what_went_well=["Both knees reached the same depth."],
               primary_issue="Your torso leaned more than your shins.",
               next_session_focus="Chest up.")
    assert [v for v in F.check(r, auth) if v.kind == "ungated"] == []


# ---------------------------------------------------------------------------
# prohibited content
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,label", [
    ("Try dropping to 60kg until the depth is consistent.", "weights"),
    ("Do 3 sets of 8 with a pause at the bottom.", "programme"),
    ("This looks like early patellar tendinitis.", "medical"),
    ("Add more protein to support recovery.", "nutrition"),
])
def test_the_four_banned_categories_are_caught(text, label):
    ev = evaluation(cues=[LEAN])
    auth = F.authorised_from(ev)
    r = report(primary_issue="Your torso leaned more than your shins.",
               corrective_cues=[text], next_session_focus="Chest up.")
    found = [v for v in F.check(r, auth) if v.kind == "prohibited"]
    assert found, f"{label} content was not caught"
    assert label in found[0].detail


def test_referring_the_user_to_a_professional_is_not_a_violation():
    """The system prompt REQUIRES this when the transcript mentions pain. Counting
    it as medical advice would penalise the model for obeying its instructions,
    and would make the headline number wrong in the flattering direction."""
    ev = evaluation(cues=[LEAN])
    auth = F.authorised_from(ev, voice_transcript="my knee has been aching")
    r = report(primary_issue="Your torso leaned more than your shins.",
               corrective_cues=["Since you mentioned knee pain, it's worth speaking "
                                "to a qualified professional before loading it further."],
               next_session_focus="Chest up.")
    assert [v for v in F.check(r, auth) if v.kind == "prohibited"] == []


# ---------------------------------------------------------------------------
# the deterministic filming tip is not the model's work
# ---------------------------------------------------------------------------

def test_the_filming_tip_is_excluded_from_scoring():
    """It is generated deterministically and never by the model (Decision 23), so
    scoring it would credit or blame the LLM for text it did not write."""
    ev = evaluation(cues=[LEAN])
    auth = F.authorised_from(ev)
    r = report(primary_issue="Your torso leaned more than your shins.",
               next_session_focus="Chest up.",
               filming_tip="To check left/right symmetry, film a set from the front.")
    assert F.check(r, auth) == []


# ---------------------------------------------------------------------------
# the output contract
# ---------------------------------------------------------------------------

def test_a_response_missing_the_primary_issue_falls_back_instead_of_showing_a_blank(monkeypatch):
    """7.1: every field was read with parsed.get(), so a response that omitted
    primary_issue showed the user an empty finding where the conclusion goes."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(coaching, "_call_groq", lambda *a, **k: {})
    monkeypatch.setattr(coaching, "_parse_llm_response",
                        lambda raw: {"what_went_well": ["nice"],
                                     "next_session_focus": "keep going"})

    out = coaching.generate_coaching_report(evaluation(cues=[LEAN]))
    assert out.source == "dry_run_fallback"
    assert out.primary_issue, "the user was shown an empty primary issue"


def test_a_wrongly_typed_list_is_a_contract_breach(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(coaching, "_call_groq", lambda *a, **k: {})
    monkeypatch.setattr(coaching, "_parse_llm_response",
                        lambda raw: {"primary_issue": "you leaned forward",
                                     "next_session_focus": "chest up",
                                     "secondary_issues": "not a list"})
    out = coaching.generate_coaching_report(evaluation(cues=[LEAN]))
    assert out.source == "dry_run_fallback"


def test_a_complete_response_is_accepted(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(coaching, "_call_groq", lambda *a, **k: {})
    monkeypatch.setattr(coaching, "_parse_llm_response",
                        lambda raw: {"what_went_well": ["good depth", "  "],
                                     "primary_issue": " you leaned forward ",
                                     "secondary_issues": [],
                                     "corrective_cues": ["chest up"],
                                     "next_session_focus": "chest up"})
    out = coaching.generate_coaching_report(evaluation(cues=[LEAN]))
    assert out.source == "llm"
    assert out.primary_issue == "you leaned forward"      # stripped
    assert out.what_went_well == ["good depth"]           # blank dropped


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def test_wilson_interval_does_not_run_past_one_at_a_perfect_rate():
    """Why Wilson and not the normal approximation: at 12/12 the normal interval
    extends above 100%, which is not a probability."""
    lo, hi = F.wilson_interval(12, 12)
    assert hi <= 1.0
    assert lo < 1.0, "a perfect run of 12 should not claim certainty"


def test_wilson_interval_is_empty_with_no_trials():
    assert F.wilson_interval(0, 0) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# the shape of coaching.json
# ---------------------------------------------------------------------------

def test_the_serialised_cue_keys_are_exactly_what_readers_expect():
    """coaching.json is a published artefact — it is served under /results/ and
    read back by this harness, by the CLI and by anything analysing a run.

    Its cue key was written as " flag", with a leading space, which is valid JSON
    and silently wrong: every reader doing payload["flag"] raises KeyError, and
    the file still looks fine to anyone eyeballing it. It reached production that
    way. Keys are asserted exactly rather than loosely for that reason.
    """
    import inspect
    from backend.pipeline import runner

    source = inspect.getsource(runner.run_pipeline)
    start = source.index('"cues_fired"')
    block = source[start:start + 400]
    for key in ("flag", "severity", "fault", "fix", "rep_indices"):
        assert f'"{key}":' in block, f'cue key "{key}" is missing or misspelled'
        assert f'" {key}"' not in block, f'cue key "{key}" has a leading space'

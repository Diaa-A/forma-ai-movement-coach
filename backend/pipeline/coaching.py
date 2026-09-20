"""Layer 2 of the coaching architecture: turn Layer 1 cue hits into prose.

We make the LLM call via stdlib `urllib` rather than adding the `groq` SDK as a
dep — Groq's API is a plain JSON POST and the SDK is just convenience. Keeps
requirements.txt minimal and works offline (dry-run mode) when there's no key.

Safety contract (mirrors what the system prompt enforces):
    - LLM may rephrase the cues we provide but MUST NOT introduce new
      biomechanical claims, weights, or rep counts.
    - The LLM sees derived summary statistics produced by Layer 1, never raw
      per-frame data, and never decides what is wrong.** This used to read "never
      sees raw landmarks or angles", which was half right and half not:
      `evaluation.notes` does carry derived angle values into the prompt — the
      worked example hands the model "trunk averaged 23 degrees more inclined
      than the shins". The input restriction is structural and worth claiming;
      it just is not the claim that was written here, and the same wording was
      heading into the report.
    - The output is validated against the schema below rather than trusted. A
      response missing a required field is treated as a failure, not shown to
      the user with a blank where the finding should be.
    - If GROQ_API_KEY is unset, we fall back to a deterministic synthetic
      report built from the cue text directly. This keeps the rest of the
      pipeline testable without an external account.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from typing import List, Optional

# via mechanics, not squat_cues -- squat_cues only re-exports these, and importing
# them from there put an exercise name in pipeline/, which is the one thing the
# shared-movement refactor was meant to leave behind.
from ..exercises.mechanics import Evaluation, CueHit


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# The env override matters because the provider deletes models out from under
# you: Groq retired llama-3.3-70b-versatile between 13 and 26 Aug 2026, and
# from then on every coaching call 404'd and fell back to cue wording while
# /health stayed green (it checks the key is present, not that a call works).
# GROQ_MODEL was already documented in .env.example and honoured by the CLI,
# but the API read this constant directly, so the override silently did not
# apply to the deployed app -- the one place it was needed. The default is left
# as the retired name on purpose: it is the model every published faithfulness
# number was measured on, and the replacement should be chosen with a harness
# rerun, not defaulted in here.
DEFAULT_MODEL = "llama-3.3-70b-versatile"
REQUEST_TIMEOUT = 30.0
# Named because the faithfulness harness has to record the temperature it
# measured at, and a run whose provenance says 0.4 while the request sends
# something else is worse than one that says nothing.
TEMPERATURE = 0.4


def active_model() -> str:
    """The model a call will actually use.

    Read here rather than captured at import, and that distinction is the whole
    point of the function. `backend/main.py` imports this module at line 26 and
    loads .env at line 43, so a module-level os.environ.get() runs before the
    file it is meant to read -- GROQ_MODEL in .env had no effect at all, while
    looking exactly like it did. On Railway the platform sets real environment
    variables before the process starts, so that ordering happened to work
    there and fail locally, which is the worse way round: the place you test is
    the place it is broken.

    Anything that records provenance should call this rather than reading
    DEFAULT_MODEL, or the artefact will name a model the run did not use.
    """
    return os.environ.get("GROQ_MODEL") or DEFAULT_MODEL

log = logging.getLogger("coach.coaching")


@dataclass
class CoachingReport:
    what_went_well: List[str]
    primary_issue: str
    secondary_issues: List[str]
    corrective_cues: List[str]
    next_session_focus: str
    source: str   # "llm" or "dry_run"
    model: Optional[str] = None
    filming_tip: Optional[str] = None   # deterministic camera-view guidance
    # What the user asked about in their voice note, answered. None when they
    # did not ask anything.
    #
    # This field exists because of a measured failure rather than as a nicety.
    # The prompt has told the model to answer the question first since 4 Aug and
    # it never did -- 12 of 12 generations on the case built from the original
    # complaint said nothing about what was asked. There was nowhere to put it.
    # Every other field is defined in terms of the cue material, "pulled from
    # POSITIVES", "rephrased PRIMARY fault", so the rule had no slot and the
    # model followed the schema.
    answer_to_question: Optional[str] = None

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a fitness coach generating feedback on a single exercise set.

You will be given:
  - The exercise name and side analysed.
  - Diagnostic notes (averages, rep counts) — facts about what happened.
  - A list of FAULTS that fired. Each has a `fault` (what went wrong) and a
    `fix` (the corrective cue). These have been written by domain experts.
  - A list of POSITIVES — things the user did well.
  - Optionally a short USER VOICE TRANSCRIPT giving context (a goal, a pain
    point, what they want feedback on).

Your job is to rephrase this material into a clean, encouraging coaching report.

ABSOLUTE RULES:
  - You MAY rephrase the `fault` and `fix` text into more natural, conversational
    English. Keep the meaning identical.
  - You MUST NOT introduce any biomechanical claim, anatomy detail, or technical
    statement that is not already in the provided faults or positives.
  - You MUST NOT recommend specific weights, sets, rep counts, or programmes.
  - You MUST NOT diagnose injuries, suggest medical advice, or suggest
    supplements / nutrition.
  - If the user transcript mentions pain or an injury, acknowledge it briefly
    and recommend speaking to a qualified professional — do not give clinical
    advice yourself.
  - If a USER VOICE TRANSCRIPT is present, ANSWER WHAT THEY ASKED in the
    `answer_to_question` field. That field is only for the answer. Use only the
    material above.
  - If they asked about something the material does not cover, say so plainly in
    one short sentence — for example "this clip was filmed from the side, so it
    can't show how wide your elbows travelled". The diagnostic notes state what
    this camera view could and could not assess; take the wording from there.
    Do NOT ignore the question, and do NOT answer it from general knowledge about
    the exercise. Saying the system did not measure something is always better
    than guessing at it.
  - If there is no transcript, set `answer_to_question` to null.
  - Output exactly the JSON schema given. No prose outside the JSON.

OUTPUT JSON SCHEMA:
{
  "answer_to_question": string | null,      // the reply to the USER VOICE
                                            // TRANSCRIPT, or null if there was
                                            // none. Answer only; the findings
                                            // below stay in their own fields
  "what_went_well": [string, ...],          // 2-3 items, pulled from POSITIVES
  "primary_issue": string,                  // rephrased PRIMARY fault, 1-2 sentences
  "secondary_issues": [string, ...],        // up to 2 rephrased SECONDARY faults
  "corrective_cues": [string, ...],         // rephrased FIX text, one per issue raised
  "next_session_focus": string              // one short actionable focus for next time
}

Tone: clear, supportive, second person ("you did X", "try Y"). Avoid jargon
unless the cue text uses it. Total length under 250 words.
"""


def _build_user_prompt(evaluation: Evaluation, voice_transcript: str = "") -> str:
    primary = evaluation.primary()
    # if no primary fired, the most-fired secondary gets promoted in narrative
    if primary is None and evaluation.cues_fired:
        # promote the most-fired cue
        primary = max(evaluation.cues_fired, key=lambda c: len(c.rep_indices))
    secondaries = [c for c in evaluation.cues_fired
                   if primary is None or c.flag != primary.flag][:2]

    payload = {
        "exercise": evaluation.exercise,
        "side_analysed": evaluation.side,
        "rep_count": evaluation.rep_count,
        "diagnostic_notes": evaluation.notes,
        "positives": evaluation.positives,
        "primary_fault": (
            None if primary is None else {
                "fault": primary.fault,
                "fix": primary.fix,
                "fires_in_reps": primary.rep_indices,
            }
        ),
        "secondary_faults": [
            {"fault": c.fault, "fix": c.fix, "fires_in_reps": c.rep_indices}
            for c in secondaries
        ],
        "user_voice_transcript": voice_transcript.strip() or None,
    }
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# LLM call (Groq, stdlib urllib)
# ---------------------------------------------------------------------------

def _call_groq(system_prompt: str, user_prompt: str, model: str,
               api_key: str) -> dict:
    body = {
        "model": model,
        "temperature": TEMPERATURE,
        "max_tokens": 700,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }
    req = urllib.request.Request(
        GROQ_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            # Groq sits behind Cloudflare, which 403s the default urllib UA
            # ("Python-urllib/x.y") as a banned bot signature (error 1010).
            # A normal UA string sails through.
            "User-Agent":    "ai-fitness-coach/0.1 (+https://github.com/)",
            "Accept":        "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Groq API error {e.code}: {err_body}") from e


class LLMContractError(RuntimeError):
    """The model returned JSON that does not meet the output contract.

    Its own class so the harness in `backend/evaluation/faithfulness.py` can tell
    a contract breach apart from a network failure — they are different findings.
    """


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_list(value, field: str) -> List[str]:
    """A list of non-empty strings, or an error naming what arrived instead.

    Empty lists are allowed: a set with no positives legitimately produces an
    empty `what_went_well`, and forcing a fallback there would replace a correct
    answer with a worse one.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise LLMContractError(f"{field} should be a list, got {type(value).__name__}")
    out = [_text(v) for v in value]
    return [v for v in out if v]


def _validate(parsed: dict) -> dict:
    """Check the model's JSON against the schema the system prompt asks for.

    Nothing validated this before: `generate_coaching_report` read every field
    with `parsed.get(...)` and passed it straight through, so a response that
    omitted `primary_issue` showed the user an empty finding where the main
    conclusion should be. Silent degradation, and the exact shape of failure the
    rest of this pipeline refuses everywhere else.

    The two scalars are required because each is a claim in its own right. The
    lists may be empty. A breach raises, which puts us on the same fallback path
    as a network failure -- the deterministic cue wording, which is always
    correct if stiffer.
    """
    if not isinstance(parsed, dict):
        raise LLMContractError(f"expected a JSON object, got {type(parsed).__name__}")

    primary = _text(parsed.get("primary_issue"))
    if not primary:
        raise LLMContractError("primary_issue missing or empty")

    focus = _text(parsed.get("next_session_focus"))
    if not focus:
        raise LLMContractError("next_session_focus missing or empty")

    return {
        # Optional, and deliberately not required even when a transcript was
        # sent: a model that has nothing to say is better refusing the field than
        # filling it, and the fallback for an empty one is the rest of the report,
        # which is still correct.
        "answer_to_question": _text(parsed.get("answer_to_question")) or None,
        "what_went_well": _string_list(parsed.get("what_went_well"), "what_went_well"),
        "primary_issue": primary,
        "secondary_issues": _string_list(parsed.get("secondary_issues"), "secondary_issues"),
        "corrective_cues": _string_list(parsed.get("corrective_cues"), "corrective_cues"),
        "next_session_focus": focus,
    }


def _parse_llm_response(api_response: dict) -> dict:
    """Pull the report JSON out of a Groq response.

    Both failure modes raise, because the caller's fallback is the dry-run
    report and half a parsed report is worse than none: the envelope can come
    back shaped differently from the documented one, and the model can ignore
    the schema and send prose instead. Prose is the one that actually happens.
    """
    try:
        content = api_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"unexpected Groq response shape: {api_response}") from e
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM returned non-JSON content:\n{content}") from e


# ---------------------------------------------------------------------------
# Dry-run fallback
# ---------------------------------------------------------------------------

def _dry_run_report(evaluation: Evaluation) -> CoachingReport:
    """Deterministic report built from cue text alone — no LLM.

    Useful for: testing the rest of the pipeline without a Groq key, and as a
    safety net when the LLM call fails. The wording is the raw expert text from
    the cue database (no rephrasing) so it reads slightly stiffer than the LLM
    output but is biomechanically identical.
    """
    primary = evaluation.primary()
    if primary is None and evaluation.cues_fired:
        primary = max(evaluation.cues_fired, key=lambda c: len(c.rep_indices))
    secondaries = [c for c in evaluation.cues_fired
                   if primary is None or c.flag != primary.flag][:2]

    what_went_well = evaluation.positives[:3] if evaluation.positives else [
        "you completed the set"
    ]
    if primary is None:
        # Only speaks for the faults that were checked. No exercise here checks
        # everything, and "No major form fault was detected" read as if it did.
        primary_issue = ("None of the faults this system checks for showed up "
                         "in this set.")
        corrective_cues: List[str] = []
        next_focus = "Film your next set the same way."
    else:
        primary_issue = primary.fault[0].upper() + primary.fault[1:]
        corrective_cues = [primary.fix]
        next_focus = "Focus on the corrective cue above on every rep next session."

    secondary_issues = [c.fault[0].upper() + c.fault[1:] for c in secondaries]
    corrective_cues.extend(c.fix for c in secondaries)

    return CoachingReport(
        what_went_well=what_went_well,
        primary_issue=primary_issue,
        secondary_issues=secondary_issues,
        corrective_cues=corrective_cues,
        next_session_focus=next_focus,
        source="dry_run",
        model=None,
    )


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def not_analyzed_report(status: str, profile=None) -> CoachingReport:
    """Honest report when the clip can't be analysed — used INSTEAD of coaching
    so 'no rep detected' is never mistaken for 'good form'. No LLM call is made.

    `profile` is the exercise's ExerciseProfile. It is optional so the older
    call signature still works, but callers should pass it: without it this
    said "I couldn't detect a complete squat rep" whatever the user uploaded,
    and told a push-up user to film at hip height. The re-filming advice is the
    profile's own `filming_guide` rather than a second copy of it, for the same
    reason GET /exercises serves that string instead of the frontend holding one
    — two copies drift, and the one quoted back to the user stops matching the
    one they read before recording.
    """
    # The display label, lowercased to sit mid-sentence. The id put "pushup" into
    # the report.
    name = profile.label.lower() if profile is not None else "exercise"
    if status == "no_reps":
        primary = (f"I couldn't detect a complete {name} rep in this clip, so "
                   "there's nothing to score yet.")
    elif status == "rotated":
        # Nothing wrong with the recording or the person -- the file's pixels
        # are stored sideways and its rotation flag is gone, so every angle
        # would be measured on a tipped-over image. Analysing it anyway is how
        # the reference clip's best frame got reported as a forward-lean fault.
        primary = ("This clip decodes sideways, so the analysis would be "
                   "measuring a rotated image rather than your form. This "
                   "usually means the file was re-saved on its way here — "
                   "clips sent through messaging apps often lose the flag "
                   "that says which way up they go.")
    else:  # low_detection
        primary = ("Body tracking was too unreliable on this clip to give "
                   "trustworthy feedback.")

    cues = []
    if status == "rotated":
        cues.append("Upload the original clip straight from your camera roll, "
                    "not a copy that was sent through a chat app or an editor.")
        next_focus = "Re-upload the original file and it should analyse normally."
    else:
        if profile is not None:
            cues.append(profile.filming_guide)
        else:
            cues.append("Film side-on at about hip height, with your whole body "
                        "in the frame.")
        cues.append("Wear fitted clothing and use a plain, uncluttered background.")
        next_focus = "Re-record with the framing above and upload again."

    return CoachingReport(
        what_went_well=[],
        primary_issue=primary,
        secondary_issues=[],
        corrective_cues=cues,
        next_session_focus=next_focus,
        source="not_analyzed",
        model=None,
    )


def _mentions_any(text: str, words) -> bool:
    """Whole words or phrases, ignoring case, so "bar" does not match "barely"."""
    lowered = text.lower()
    return any(re.search(r"\b" + re.escape(w.lower()) + r"\b", lowered) for w in words)


def generate_coaching_report(evaluation: Evaluation,
                             voice_transcript: str = "",
                             model: Optional[str] = None,
                             force_dry_run: bool = False,
                             profile=None) -> CoachingReport:
    """Top-level coaching call.

    Falls back to a deterministic dry-run report when:
        - force_dry_run is True
        - GROQ_API_KEY is not set in the environment
        - the LLM call fails for any reason. The failure is logged and shows up
          in `source` as "dry_run_fallback", which the UI turns into a sentence
          telling the user they are reading the system's own wording — so it is
          not silent, and no exception text reaches them.

    `profile` is the exercise's ExerciseProfile. With it, a next-session line that
    gives advice on something the profile lists as not assessed is replaced by the
    dry-run report's line, and the rest of the model's wording is kept.
    """
    # None means "whatever the environment says", resolved now rather than when
    # this module was imported -- see active_model.
    model = model or active_model()

    # the camera-view tip is deterministic — never left to the LLM
    tip = getattr(evaluation, "view_guidance", None)

    api_key = os.environ.get("GROQ_API_KEY")
    if force_dry_run or not api_key:
        report = _dry_run_report(evaluation)
        report.filming_tip = tip
        return report

    user_prompt = _build_user_prompt(evaluation, voice_transcript)
    try:
        raw = _call_groq(SYSTEM_PROMPT, user_prompt, model, api_key)
        fields = _validate(_parse_llm_response(raw))
        # A live pull-up report said it could not tell whether the chin cleared the
        # bar or the arms straightened, then ended "make sure you fully extend your
        # arms and clear the bar". The prompt stays as it is, so the line the user
        # is most likely to act on is checked here instead of trusted.
        if profile is not None and _mentions_any(fields["next_session_focus"],
                                                 profile.not_assessed_words):
            log.info("next-session line named something not assessed; using the fixed line")
            fields["next_session_focus"] = _dry_run_report(evaluation).next_session_focus
        return CoachingReport(
            **fields,
            source="llm",
            model=model,
            filming_tip=tip,
        )
    except Exception:
        # Never let a coaching failure break the analysis pipeline. The failure
        # still has to be visible, but it was being made visible in the worst
        # available place: the exception text went into corrective_cues, which is
        # rendered as coaching advice AND written into coaching.json, which is
        # served publicly under /results/<job>/. With a bad key that put a raw
        # Groq error body — provider internals, and whatever the provider chose to
        # echo back — in front of the user and on a public URL.
        #
        # `source` already carries the fact of the fallback, and the UI turns it
        # into a plain sentence saying the wording is the system's own, which is
        # all the user needs to know.
        log.warning("coaching LLM call failed, falling back to cue wording",
                    exc_info=True)
        fallback = _dry_run_report(evaluation)
        fallback.source = "dry_run_fallback"
        fallback.filming_tip = tip
        return fallback


# ---------------------------------------------------------------------------
# Pretty printing for the CLI
# ---------------------------------------------------------------------------

def format_report(report: CoachingReport, width: int = 78) -> str:
    out = []
    bar = "=" * width
    src = (f"[{report.source}" +
           (f" / {report.model}]" if report.model else "]"))
    out.append(bar)
    out.append(f"COACHING REPORT  {src}")
    out.append(bar)
    out.append("")
    out.append("What you did well:")
    for w in report.what_went_well:
        out.append(f"  - {w}")
    out.append("")
    out.append("Primary issue:")
    out.append(f"  {report.primary_issue}")
    if report.secondary_issues:
        out.append("")
        out.append("Secondary issues:")
        for s in report.secondary_issues:
            out.append(f"  - {s}")
    if report.corrective_cues:
        out.append("")
        out.append("Corrective cues:")
        for c in report.corrective_cues:
            out.append(f"  - {c}")
    out.append("")
    out.append("Next session focus:")
    out.append(f"  {report.next_session_focus}")
    if report.filming_tip:
        out.append("")
        out.append("Filming tip:")
        out.append(f"  {report.filming_tip}")
    out.append(bar)
    return "\n".join(out)

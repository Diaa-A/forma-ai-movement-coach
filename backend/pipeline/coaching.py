"""Layer 2 of the coaching architecture: turn Layer 1 cue hits into prose.

We make the LLM call via stdlib `urllib` rather than adding the `groq` SDK as a
dep — Groq's API is a plain JSON POST and the SDK is just convenience. Keeps
requirements.txt minimal and works offline (dry-run mode) when there's no key.

Safety contract (mirrors what the system prompt enforces):
    - LLM may rephrase the cues we provide but MUST NOT introduce new
      biomechanical claims, weights, or rep counts.
    - LLM never sees raw landmarks or angles — only the Layer 1 evaluation.
    - If GROQ_API_KEY is unset, we fall back to a deterministic synthetic
      report built from the cue text directly. This keeps the rest of the
      pipeline testable without an external account.
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from typing import List, Optional

from ..exercises.squat_cues import Evaluation, CueHit


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
REQUEST_TIMEOUT = 30.0


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
  - Output exactly the JSON schema given. No prose outside the JSON.

OUTPUT JSON SCHEMA:
{
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
        "temperature": 0.4,
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


def _parse_llm_response(api_response: dict) -> dict:
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
        primary_issue = ("No major form fault was detected from this set — keep "
                         "the focus on consistency and intent.")
        corrective_cues: List[str] = []
        next_focus = "Maintain the standard you set in this session on your next attempt."
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

def not_analyzed_report(status: str) -> CoachingReport:
    """Honest report when the clip can't be analysed — used INSTEAD of coaching
    so 'no rep detected' is never mistaken for 'good form'. No LLM call is made."""
    if status == "no_reps":
        primary = ("I couldn't detect a complete squat rep in this clip, so "
                   "there's nothing to score yet.")
    else:  # low_detection
        primary = ("Body tracking was too unreliable on this clip to give "
                   "trustworthy feedback.")
    return CoachingReport(
        what_went_well=[],
        primary_issue=primary,
        secondary_issues=[],
        corrective_cues=[
            "Film side-on at about hip height, with your whole body in the frame.",
            "Wear fitted clothing and use a plain, uncluttered background.",
        ],
        next_session_focus="Re-record with the framing above and upload again.",
        source="not_analyzed",
        model=None,
    )


def generate_coaching_report(evaluation: Evaluation,
                             voice_transcript: str = "",
                             model: str = DEFAULT_MODEL,
                             force_dry_run: bool = False) -> CoachingReport:
    """Top-level coaching call.

    Falls back to a deterministic dry-run report when:
        - force_dry_run is True
        - GROQ_API_KEY is not set in the environment
        - the LLM call fails for any reason (errors are caught and the dry-run
          report is returned with a note in `corrective_cues` so the failure
          isn't silent)
    """
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
        parsed = _parse_llm_response(raw)
        return CoachingReport(
            what_went_well=parsed.get("what_went_well", []),
            primary_issue=parsed.get("primary_issue", ""),
            secondary_issues=parsed.get("secondary_issues", []),
            corrective_cues=parsed.get("corrective_cues", []),
            next_session_focus=parsed.get("next_session_focus", ""),
            source="llm",
            model=model,
            filming_tip=tip,
        )
    except Exception as e:
        # never let a coaching failure break the analysis pipeline
        fallback = _dry_run_report(evaluation)
        fallback.corrective_cues.insert(
            0, f"(LLM call failed: {e}. Falling back to dry-run wording.)"
        )
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

"""Pull the three states of the lockout experiment into one citable file.

    python scripts/build_lockout_experiment.py

The coverage audit on 6 Aug called lockout the most obvious undeclared gap, and
the faithfulness run on 10 Aug caught the model inventing that exact claim out of
a payload that said nothing about extension. Predicted, then observed. This is
the third step - close the gap and see whether the behaviour moves.

    baseline   nothing declared, no detector      (10 Aug, not re-run)
    declared   named in not_yet_assessed          (arm A)
    detected   real cue behind an elbow angle     (arm B)

The faithfulness rate can't be the endpoint. Everything in the payload counts as
authorised, so declaring the gap in the checker's own vocabulary would score the
invented sentence faithful with the model doing exactly what it did before. Arm A
avoids that by saying "straighten", which isn't in the vocabulary - but then the
error runs the other way, and a model correctly repeating the declared gap as
"elbow extension" gets flagged for inventing it.

So the primary number is counted off the text: how many generations said the user
failed to lock out. The rate sits underneath it, scored twice, against each arm's
own payload and against the baseline one, so any instrument movement shows up
separately instead of inside the result.

Every lockout sentence goes in the summary. The counts come from reading them.
"""
from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_faithfulness as R          # sibling script, for the payload loader

from backend.evaluation import faithfulness as F
from backend.pipeline import coaching

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "chapter5"
RUNS = ROOT / "data" / "outputs" / "faithfulness"

# The 10 Aug run as published. Predates the harness keeping generations so the
# text is gone, but for this endpoint it doesn't matter - lockout wasn't
# authorised in that payload, so every sentence mentioning it got recorded as a
# violation. Count is complete.
BASELINE = OUT / "faithfulness_results.json"
BASELINE_PAYLOADS = RUNS / "evaluations_baseline_10aug.json"

# Two cases, same Layer-1 payload, and the second one carries the user's own
# question from the section 15 failure. It is there because pushup_shallow alone
# cannot test the thing arm A is for: the prompt only tells the model to name a
# gap when the user has asked about something, so on a set with no voice note the
# coverage declaration is never reached whether it is there or not.
TRACKS = [
    {"case": "pushup_shallow", "payload": "pushup_shallow", "transcript": "",
     "declared": RUNS / "results_armA_declared.json",
     "detected": RUNS / "results_armB_detected.json"},
    {"case": "pushup_unanswerable_question", "payload": "pushup_shallow",
     "transcript": R.UNANSWERABLE_TRANSCRIPT,
     "declared": RUNS / "results_armA_declared_question.json",
     "detected": RUNS / "results_armB_detected_question.json"},
]

CHANGES = {
    "declared": ("lockout added to not_yet_assessed in PUSHUP_PROFILE, no detector. "
                 "The only difference in the prompt is one clause in the coverage note"),
    "detected": "a lockout cue behind elbow angle at the top of the rep",
}

# Arm B did not happen, and the reason is a result rather than an excuse, so it
# travels with the numbers instead of being left as a blank row.
DETECTED_STATUS = {
    "state": "not built",
    "reason": (
        "Top-of-rep elbow angle does not discriminate lockout, tested two ways. "
        "First, across 17 fixture clips and 66 reps: every step wider than Penn "
        "Action's 8.1 degree elbow error falls at the bottom of the distribution, "
        "separating clips whose tracking has come apart rather than clips where "
        "someone finished short. Second, and more directly, against a deliberate "
        "positive class filmed for the purpose on 13 Aug - one subject, one "
        "camera position, one session, three sets: normal, depth-manipulated, and "
        "a set where the arms were deliberately never straightened."),
    "deliberate_positive_class": {
        "why": (
            "The first test could only say the fixture set contains no lockout "
            "failure. This one asks the sharper question: when a lockout failure "
            "is staged on purpose, can the measure see it?"),
        "median_top_of_rep_elbow_degrees": {
            "normal": {"max": 161.3, "p90": 160.0, "top_decile_median": 160.9,
                       "at_boundaries": 151.5},
            "shallow_control": {"max": 158.0, "p90": 157.1,
                                "top_decile_median": 157.7, "at_boundaries": 151.1},
            "deliberate_no_lockout": {"max": 160.0, "p90": 157.9,
                                      "top_decile_median": 159.0,
                                      "at_boundaries": 156.9},
        },
        "result": (
            "No statistic separates the staged no-lockout set from the normal "
            "one. The largest gap is 2.1 degrees and one measure runs backwards, "
            "against a benchmark error of 8.1. The shallow set behaves as the "
            "control it is: depth was manipulated and the top was not, and it "
            "sits with normal throughout."),
        "why_it_fails": (
            "The subject's normal push-ups top out at 161 degrees, not 180. Most "
            "people do not fully lock out, so there is very little room between "
            "normal and deliberately bent to begin with. The signal is real at "
            "frame level - 167 degrees with the arms straight at setup, 152 "
            "mid-set with a visible bend - and it disappears under any "
            "aggregation a cue could use, because each rep still touches roughly "
            "160 at its most extended moment."),
        "what_this_makes_the_finding": (
            "Stronger than an absent positive class. A staged, visually obvious "
            "lockout failure is invisible to the measure, so the case for leaving "
            "lockout declared rather than detected rests on a measurement result "
            "and not on a missing fixture."),
    },
    "why_not_ship_a_number_anyway": (
        "The other thresholds in pushup.py were set on data that separated - deep "
        "clips bottom out at 56-93 and shallow ones at 106-134, so 120 sits in a "
        "real gap. A lockout threshold here would be splitting one healthy "
        "distribution in half. Knee valgus and elbow flare are both parked for "
        "the same reason."),
    "what_would_unblock_it": (
        "Not another clip - that was tried and is the second test above. It needs "
        "a different measurement. The frame-level signal exists, so a measure that "
        "asks whether the arm is straight at a defined instant, rather than taking "
        "an aggregate over the rep, might survive; so might a per-subject baseline, "
        "since the failure here is between-subject variation swamping a "
        "within-subject difference. Both are research, not a threshold, and "
        "neither belongs in front of a draft deadline."),
    "evidence": "report/chapter5/lockout_calibration.json",
}

# Did the report pass any of the coverage declaration on to the user - a
# not-yet-assessed item, the film-from-the-front line, anything about the camera?
# Needed because a zero in arm A has two readings that look the same otherwise:
# the declaration worked, or the model ignored the coverage note the way it
# always had.
#
# Deliberately generous. It will match sentences that are only loosely about
# coverage, which is the right way round - a generous pattern finding nothing is
# a solid zero, where a tight one finding nothing might just be the wording.
COVERAGE_ECHO = re.compile(
    r"not (yet )?(assess|measur|check)|film|from the front|side-on|from the side"
    r"|flare|straighten|arm|wide|camera|view|angle|footage"
    r"|this (clip|video|analysis)|can.?t (see|show|tell)|cannot", re.I)

# First pass only. Separates "you didn't lock out" from "this system hasn't
# checked whether you lock out" - opposite results, identical to a keyword
# search. Everything it matches gets printed so it can be read.
HEDGE = re.compile(
    r"(not|n't|cannot|unable to)\s+(yet\s+)?(been\s+)?"
    r"(assess|measur|check|tell|show|see|determin|captur|evaluat)"
    r"|not (yet )?(assessed|measured|checked|evaluated)"
    r"|(this|the) (clip|system|analysis|video)[^.]{0,40}"
    r"(can't|cannot|does not|doesn't|did not|didn't)", re.I)


def case_of(summary: dict, name: str) -> dict:
    for c in summary["cases"]:
        if c["name"] == name:
            return c
    raise SystemExit(f"{name} is not in this results file: "
                     f"{[c['name'] for c in summary['cases']]}")


def coverage_note(payload: dict) -> str:
    """The last note is the coverage guidance -- the string arm A changes."""
    notes = payload.get("notes") or []
    return notes[-1] if notes else ""


def lockout_lines(generation: dict) -> list:
    """Sentences in one generation that mention lockout, with a first-pass read."""
    fields = generation["report"]
    report = coaching.CoachingReport(**fields, source="llm", model=None, filming_tip=None)
    found = F.claims_about(report, "lockout")
    out = []
    for where in ("assertions", "instructions"):
        for s in found[where]:
            out.append({"section": where, "sentence": s,
                        "reads_as": "gap declared" if HEDGE.search(s) else "claim asserted"})
    return out


def baseline_arm(track: dict) -> dict:
    """Built from violation quotes, because the run stored nothing else."""
    if not BASELINE.is_file():
        raise SystemExit(f"no published baseline at {BASELINE.relative_to(ROOT)}")
    c = case_of(json.loads(BASELINE.read_text(encoding="utf-8")), track["case"])
    lock = [v for v in c["violations"] if "lockout" in v["detail"]]
    payload = json.loads(BASELINE_PAYLOADS.read_text(encoding="utf-8"))[track["payload"]]
    return {
        "arm": "baseline",
        "change": "nothing declared and no detector",
        "source": str(BASELINE.relative_to(ROOT)).replace("\\", "/"),
        "coverage_note": coverage_note(payload),
        "n": c["scored"],
        "faithful": c["faithful"],
        "rate": c["rate"],
        "wilson_95": [round(x, 4) for x in F.wilson_interval(c["faithful"], c["scored"])],
        "generations_naming_lockout": len({v["run"] for v in lock}),
        "lockout_sentences": [{"run": v["run"], "section": "recorded as a violation",
                               "sentence": v["quote"], "reads_as": "claim asserted"}
                              for v in lock],
        "generations_stored": False,
        "note": ("text was not stored by the harness of the day. Lockout was "
                 "unauthorised in this payload, so every sentence mentioning it "
                 "became a violation and the count is still complete. The "
                 "gap-reported column cannot be recovered the same way."),
    }


def measured_arm(label: str, path: Path, track: dict, baseline_auth) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    c = case_of(summary, track["case"])
    gens = c.get("generations") or []
    if not gens:
        raise SystemExit(f"{path.name} stored no generations — rerun it on a "
                         "harness that does, or the endpoint cannot be read")

    lines, naming, echoed = [], set(), set()
    frozen_faithful = 0
    for g in gens:
        for line in lockout_lines(g):
            lines.append({"run": g["run"], **line})
            if line["reads_as"] == "claim asserted":
                naming.add(g["run"])
        report = coaching.CoachingReport(**g["report"], source="llm", model=None,
                                         filming_tip=None)
        if not F.check(report, baseline_auth):
            frozen_faithful += 1
        whole = " ".join(report.what_went_well + [report.primary_issue]
                         + report.secondary_issues + report.corrective_cues
                         + [report.next_session_focus])
        if COVERAGE_ECHO.search(whole):
            echoed.add(g["run"])

    return {
        "arm": label,
        "change": CHANGES.get(label, ""),
        "source": str(path.relative_to(ROOT)).replace("\\", "/"),
        "coverage_note": coverage_note(c.get("evaluation", {})),
        "cues_fired": c["cues_fired"],
        "n": c["scored"],
        "faithful": c["faithful"],
        "rate": c["rate"],
        "wilson_95": [round(x, 4) for x in F.wilson_interval(c["faithful"], c["scored"])],
        # same generations, scored against the baseline payload's authorised set.
        # a gap between this and `faithful` is the checker moving, not the model
        "faithful_against_baseline_payload": frozen_faithful,
        "generations_naming_lockout": len(naming),
        # zero here means the arm says nothing about whether declaring lockout
        # works. it says the coverage note went unused.
        "generations_reporting_a_declared_gap": len(echoed),
        "lockout_sentences": lines,
        "generations_stored": True,
        # copied into report/ because data/outputs/ is gitignored. a published
        # count nobody can go back and check is how the last brief got written
        # against superseded numbers
        "generations": gens,
    }


def build() -> dict:
    if not BASELINE_PAYLOADS.is_file():
        raise SystemExit(
            f"{BASELINE_PAYLOADS.relative_to(ROOT)} is missing. It is the payload the "
            "baseline was scored against and the reference the other arms are "
            "re-scored on. Without it there is no fixed instrument to compare to.")
    payloads = json.loads(BASELINE_PAYLOADS.read_text(encoding="utf-8"))

    tracks, missing = [], []
    for t in TRACKS:
        auth = F.authorised_from(R.evaluation_from_dict(payloads[t["payload"]]),
                                 t["transcript"])
        arms = [baseline_arm(t)]
        for label in ("declared", "detected"):
            if t[label].is_file():
                arms.append(measured_arm(label, t[label], t, auth))
            else:
                missing.append(f"{t['case']}/{label}")
        tracks.append({"case": t["case"],
                       "voice_note": t["transcript"] or None,
                       "arms": arms})

    return {
        "cases": tracks,
        "model": coaching.DEFAULT_MODEL,
        "temperature": coaching.TEMPERATURE,
        "primary_endpoint": (
            "generations asserting that the user failed to lock out. Counted off "
            "the text, because the faithfulness rate stops measuring the model "
            "once the payload authorises the word."),
        "secondary_endpoint": (
            "faithfulness rate, reported against each arm's own payload and "
            "against the baseline payload. A gap between the two is the checker "
            "moving, not the model."),
        "arms_not_run": missing,
        "detected_arm": DETECTED_STATUS,
        "findings": [
            "The third state was not reachable. Declaring lockout changed nothing "
            "measurable and detecting it cannot be calibrated from the fixtures "
            "available, so the honest state for lockout is where arm A left it - "
            "declared, not detected. That is what section 15's three-state rule "
            "prescribes for exactly this situation, arrived at by measurement "
            "rather than by argument.",
            "On pushup_unanswerable_question the model answered the user's "
            "question 0 times out of 12. The transcript reaches the prompt, the "
            "coverage note names elbow flare as not assessed, and SYSTEM_PROMPT "
            "tells the model to answer first and to say so plainly when the "
            "material does not cover it - with an example sentence about elbows "
            "travelling wide. Not one generation mentioned arms, the camera view, "
            "or any gap. This is the section 15 failure still happening, and the "
            "section 18.7 prompt fix measured for the first time.",
            "The likely mechanism is in the schema rather than the rules: there is "
            "no output field for an answer to the user. Every one of the five is "
            "defined in terms of the cue material - pulled from POSITIVES, "
            "rephrased PRIMARY fault, and so on - so the instruction has nowhere "
            "to go and the model follows the schema. Not fixed here on purpose: "
            "the lockout experiment is testing whether Layer-1 coverage governs "
            "Layer-2 behaviour, and editing the prompt mid-experiment confounds it.",
            "pushup_unanswerable_question scoring 12/12 is what hid this. The "
            "checker measures unfaithfulness and silence is perfectly faithful, so "
            "a case built to catch the model answering from general knowledge "
            "cannot see it answering nothing at all.",
        ],
        "limits": [
            "pushup_shallow has no voice note, and the prompt only asks the model "
            "to name a gap when the user has asked about something. So on that "
            "case the coverage declaration is never reached and the declared arm "
            "cannot show anything either way. pushup_unanswerable_question is the "
            "same payload with the user's question attached, which is where the "
            "declaration is actually exercised.",
            "The baseline is 1 unfaithful generation in 12. At n=12 an arm cannot "
            "show that declaring or detecting REDUCED anything: 11/12 against "
            "12/12 is Fisher exact p = 1.0. The informative outcome is asymmetric "
            "-- a lockout claim still appearing is strong evidence the change did "
            "not suppress it; its absence is weak evidence of anything, because "
            "the base rate was one in twelve.",
            "The baseline was collected on 10 August and the arms later, against a "
            "hosted model that can change underneath a fixed model name. Same "
            "name, same temperature, not provably the same weights.",
            "Sentences are found by vocabulary and classified by reading. A claim "
            "about lockout phrased entirely outside the words extend / extension / "
            "lockout would not be found at all.",
        ],
    }


def table(d: dict) -> str:
    L = [f"Lockout experiment — {d['model']} at temperature {d['temperature']}", ""]
    L.append("Does closing a coverage gap stop the model inventing the claim? The")
    L.append("primary measure is the count of generations asserting a lockout failure,")
    L.append("not the faithfulness rate -- see the JSON for why the rate cannot carry")
    L.append("this on its own. 'gap reported' is how often the report passed ANY")
    L.append("declared gap to the user; a zero there means the declaration was never")
    L.append("reached, so the arm shows nothing rather than showing success.")
    L.append("")
    hdr = (f"{'state':<12}{'n':>4}{'faithful':>10}{'vs baseline':>13}"
           f"{'lockout claims':>16}{'gap reported':>14}")

    for t in d["cases"]:
        L.append(f"{t['case']}"
                 + (f"   (voice note: \"{t['voice_note']}\")" if t["voice_note"] else ""))
        L.append(hdr)
        L.append("-" * len(hdr))
        for a in t["arms"]:
            vs = a.get("faithful_against_baseline_payload")
            echo = a.get("generations_reporting_a_declared_gap")
            L.append(f"{a['arm']:<12}{a['n']:>4}{a['faithful']:>10}"
                     f"{('-' if vs is None else vs):>13}"
                     f"{a['generations_naming_lockout']:>16}"
                     f"{('-' if echo is None else echo):>14}")
        L.append("-" * len(hdr))
        L.append("")

    if d["arms_not_run"]:
        s = d["detected_arm"]
        L.append(f"detected arm: {s['state']} ({', '.join(d['arms_not_run'])})")
        for line in (s["reason"], s["why_not_ship_a_number_anyway"],
                     s["what_would_unblock_it"]):
            L.extend(textwrap.wrap(line, width=len(hdr), initial_indent="  ",
                                   subsequent_indent="  "))
            L.append("")
        L.append(f"  evidence: {s['evidence']}")
        L.append("")
    L.append("'vs baseline' scores the same generations against the baseline payload's")
    L.append("authorised set. Where it differs from 'faithful', that difference is the")
    L.append("checker moving with the payload rather than the model behaving differently.")
    L.append("")

    for t in d["cases"]:
        for a in t["arms"]:
            L.append(f"[{t['case']} / {a['arm']}]  {a.get('change', '')}".rstrip())
            L.append(f"  coverage note: {a['coverage_note'] or '(not recorded)'}")
            if not a["lockout_sentences"]:
                L.append("  no sentence mentioned lockout")
            for s in a["lockout_sentences"]:
                L.append(f"  run {s['run']:>2} [{s['reads_as']}] {s['sentence']}")
            L.append("")

    L.append("Findings:")
    for f in d["findings"]:
        L.append("  - " + f)
    L.append("")
    L.append("Limits:")
    for lim in d["limits"]:
        L.append("  - " + lim)
    return "\n".join(L)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = build()
    (OUT / "lockout_experiment.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    t = table(d)
    (OUT / "lockout_experiment.txt").write_text(t + "\n", encoding="utf-8")
    print(t)
    print(f"\nwritten to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()

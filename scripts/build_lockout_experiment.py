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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.evaluation import faithfulness as F
from backend.pipeline import coaching

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "chapter5"
RUNS = ROOT / "data" / "outputs" / "faithfulness"
CASE = "pushup_shallow"

# The 10 Aug run as published. Predates the harness keeping generations so the
# text is gone, but for this endpoint it doesn't matter - lockout wasn't
# authorised in that payload, so every sentence mentioning it got recorded as a
# violation. Count is complete.
BASELINE = OUT / "faithfulness_results.json"
BASELINE_PAYLOADS = RUNS / "evaluations_baseline_10aug.json"

ARMS = [
    ("baseline", None),
    ("declared", RUNS / "results_armA_declared.json"),
    ("detected", RUNS / "results_armB_detected.json"),
]

CHANGES = {
    "declared": ("lockout added to not_yet_assessed in PUSHUP_PROFILE, no detector. "
                 "The only difference in the prompt is one clause in the coverage note"),
    "detected": "a lockout cue behind elbow angle at the top of the rep",
}

# Did the report pass any of the coverage declaration on to the user - a
# not-yet-assessed item, or the film-from-the-front line? Needed because a zero
# in arm A has two readings that look the same otherwise: the declaration worked,
# or the model ignored the coverage note the way it always had.
COVERAGE_ECHO = re.compile(
    r"not (yet )?assess|film (a set )?from the front|side-on|from the side"
    r"|flare|straighten|this (clip|video|analysis)", re.I)

# First pass only. Separates "you didn't lock out" from "this system hasn't
# checked whether you lock out" - opposite results, identical to a keyword
# search. Everything it matches gets printed so it can be read.
HEDGE = re.compile(
    r"(not|n't|cannot|unable to)\s+(yet\s+)?(been\s+)?"
    r"(assess|measur|check|tell|show|see|determin|captur|evaluat)"
    r"|not (yet )?(assessed|measured|checked|evaluated)"
    r"|(this|the) (clip|system|analysis|video)[^.]{0,40}"
    r"(can't|cannot|does not|doesn't|did not|didn't)", re.I)


def case_of(summary: dict, name: str = CASE) -> dict:
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


def baseline_arm() -> dict:
    """Built from violation quotes, because the run stored nothing else."""
    if not BASELINE.is_file():
        raise SystemExit(f"no published baseline at {BASELINE.relative_to(ROOT)}")
    c = case_of(json.loads(BASELINE.read_text(encoding="utf-8")))
    lock = [v for v in c["violations"] if "lockout" in v["detail"]]
    payload = {}
    if BASELINE_PAYLOADS.is_file():
        payload = json.loads(BASELINE_PAYLOADS.read_text(encoding="utf-8")).get(CASE, {})
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
                 "became a violation and the count is still complete."),
    }


def measured_arm(label: str, path: Path, baseline_auth) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    c = case_of(summary)
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
    base_payload = json.loads(BASELINE_PAYLOADS.read_text(encoding="utf-8"))[CASE]
    import run_faithfulness  # noqa: E402 -- same directory, only for the loader
    baseline_auth = F.authorised_from(run_faithfulness.evaluation_from_dict(base_payload))

    arms, missing = [], []
    for label, path in ARMS:
        if label == "baseline":
            arms.append(baseline_arm())
        elif path.is_file():
            arms.append(measured_arm(label, path, baseline_auth))
        else:
            missing.append(label)

    return {
        "case": CASE,
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
        "arms": arms,
        "arms_not_yet_run": missing,
        "limits": [
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
    L = [f"Lockout experiment — {d['case']}, {d['model']} at temperature "
         f"{d['temperature']}", ""]
    L.append("Does closing a coverage gap stop the model inventing the claim? Three")
    L.append("states, same case, same n. The primary measure is the count of")
    L.append("generations asserting a lockout failure, not the faithfulness rate --")
    L.append("see the JSON for why the rate cannot carry this on its own.")
    L.append("")
    hdr = (f"{'state':<12}{'n':>4}{'faithful':>10}{'vs baseline':>13}"
           f"{'lockout claims':>16}{'gap reported':>14}")
    L.append(hdr)
    L.append("-" * len(hdr))
    for a in d["arms"]:
        vs = a.get("faithful_against_baseline_payload")
        echo = a.get("generations_reporting_a_declared_gap")
        L.append(f"{a['arm']:<12}{a['n']:>4}{a['faithful']:>10}"
                 f"{('-' if vs is None else vs):>13}"
                 f"{a['generations_naming_lockout']:>16}"
                 f"{('-' if echo is None else echo):>14}")
    L.append("-" * len(hdr))
    if d["arms_not_yet_run"]:
        L.append(f"not yet run: {', '.join(d['arms_not_yet_run'])}")
    L.append("")
    L.append("'vs baseline' scores the same generations against the baseline payload's")
    L.append("authorised set. Where it differs from 'faithful', that difference is the")
    L.append("checker moving with the payload rather than the model behaving differently.")
    L.append("")

    for a in d["arms"]:
        L.append(f"[{a['arm']}]  {a.get('change', '')}".rstrip())
        L.append(f"  coverage note: {a['coverage_note'] or '(not recorded)'}")
        if not a["lockout_sentences"]:
            L.append("  no sentence mentioned lockout")
        for s in a["lockout_sentences"]:
            L.append(f"  run {s['run']:>2} [{s['reads_as']}] {s['sentence']}")
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

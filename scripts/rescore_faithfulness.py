"""Re-score a stored faithfulness run against the current checker.

Costs no quota. The harness keeps every generation and the evaluation it was
scored against, so when the checker changes the honest move is to re-score what
was collected rather than spend a day's allowance collecting it again. That is
what was done on 12 August when the answer-field arm was re-scored offline, and
the same argument applies here.

    PY scripts/rescore_faithfulness.py --label gpt-oss-120b

Writes `<results>_rescored.json` and a table beside the original, and prints
both rates so the difference is visible rather than silently replacing a number.
Runs that predate generation storage cannot be re-scored: the 10 August run kept
only the sentences that violated, which is why its rate carries a caveat instead
of a correction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.evaluation import faithfulness as F          # noqa: E402
from backend.pipeline.coaching import CoachingReport      # noqa: E402
from run_faithfulness import evaluation_from_dict         # noqa: E402

OUT = ROOT / "data" / "outputs" / "faithfulness"


def report_from_dict(d: dict) -> CoachingReport:
    return CoachingReport(
        what_went_well=d.get("what_went_well", []),
        primary_issue=d.get("primary_issue", ""),
        secondary_issues=d.get("secondary_issues", []),
        corrective_cues=d.get("corrective_cues", []),
        next_session_focus=d.get("next_session_focus", ""),
        source=d.get("source", "llm"),
        model=d.get("model"),
        filming_tip=d.get("filming_tip"),
        answer_to_question=d.get("answer_to_question"),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="", help="suffix of the run to re-score")
    args = ap.parse_args()

    suffix = f"_{args.label}" if args.label else ""
    src = OUT / f"results{suffix}.json"
    if not src.is_file():
        raise SystemExit(f"no such run: {src}")

    data = json.loads(src.read_text(encoding="utf-8"))
    missing = [c["name"] for c in data["cases"] if not c.get("generations")]
    if missing:
        raise SystemExit(
            "these cases stored no generations and cannot be re-scored: "
            + ", ".join(missing))

    scored = faithful = 0
    changed = []
    for case in data["cases"]:
        auth = F.authorised_from(
            evaluation_from_dict(case["evaluation"]),
            case.get("voice_transcript", "") or "")
        case_faithful = 0
        violations = []
        for gen in case["generations"]:
            vs = F.check(report_from_dict(gen["report"]), auth)
            was = gen.get("faithful")
            now = not vs
            if was is not None and was != now:
                changed.append((case["name"], gen["run"], was, now))
            gen["faithful"] = now
            gen["violations"] = [v.to_dict() for v in vs]
            violations.extend(v.to_dict() for v in vs)
            scored += 1
            case_faithful += 1 if now else 0
        case["faithful"] = case_faithful
        case["rate"] = round(case_faithful / len(case["generations"]), 4)
        case["violations"] = violations
        faithful += case_faithful

    before = data.get("faithfulness_rate")
    data["generations_scored"] = scored
    data["faithful"] = faithful
    data["faithfulness_rate"] = round(faithful / scored, 4) if scored else 0.0
    lo, hi = F.wilson_interval(faithful, scored)
    data["wilson_95"] = [round(lo, 4), round(hi, 4)]
    kinds = {}
    for c in data["cases"]:
        for v in c["violations"]:
            kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
    data["violations_by_kind"] = kinds
    data["unfaithful_generations"] = scored - faithful
    data["rescored"] = {
        "rate_before": before,
        "rate_after": data["faithfulness_rate"],
        "generations_whose_verdict_changed": len(changed),
        "why": "contradiction matching now ignores a negated assertion of the "
               "desired state; see the note above CONTRADICTIONS in "
               "backend/evaluation/faithfulness.py",
    }

    dest = OUT / f"results{suffix}_rescored.json"
    dest.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(f"re-scored {scored} generations from {src.name}")
    print(f"  before {before:.1%}   after {data['faithfulness_rate']:.1%}"
          f"   Wilson [{lo:.1%}, {hi:.1%}]")
    print(f"  verdicts changed: {len(changed)}")
    for name, run, was, now in changed[:20]:
        print(f"    {name} run {run}: {'faithful' if was else 'unfaithful'}"
              f" -> {'faithful' if now else 'unfaithful'}")
    print(f"  violations now: {kinds or 'none'}")
    print(f"written to {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

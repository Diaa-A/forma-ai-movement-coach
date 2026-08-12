"""Add the provenance the 10 August faithfulness run did not record.

    python scripts/backfill_faithfulness_provenance.py

The run is fine - 72 generations, six cases at 12 of 12. What it didn't write is
a collection date, any record of the seventh case the cap interrupted, or the
difference between two un-cued claims and two unfaithful generations. Re-running
for those fields costs the whole sample again and gives different numbers, so
they're reconstructed here and labelled as reconstructed.

Metadata only. Measured counts are compared before and after and the write is
abandoned if any moved, and it refuses a results file whose scored cases aren't
the six from that run - otherwise the next run to land here picks up 10 August's
provenance.

Idempotent. run_faithfulness records all this itself now, so once the 10 Aug run
is superseded this script is dead and can go.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.evaluation import faithfulness as F
from backend.pipeline import coaching

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "outputs" / "faithfulness" / "results.json"

# The run this script is for. Anything else and we are guessing.
EXPECTED_CASES = ["squat_lean", "pushup_shallow", "pushup_sag", "squat_with_pain_note",
                  "pushup_unanswerable_question", "clean_no_cues"]
EXPECTED_TOTALS = {"generations_scored": 72, "faithful": 71}

# The date is the file's own mtime and matches the handoff (section 18.2,
# "Measured 10 August 2026"). Not derivable from the JSON, which is the problem
# this script exists to fix.
COLLECTED = "2026-08-10"

# Last commit touching the checker or the prompt before that date:
#   git log -1 --before=2026-08-11 -- backend/evaluation/faithfulness.py \
#       backend/pipeline/coaching.py scripts/run_faithfulness.py
# a8e5903, "Tell the model to answer the user's question, and to say when it
# cannot" - the 18.7 prompt change. Pinned rather than left blank because it's
# what decides whether a later prompt edit invalidates these numbers. Stamping
# today's commit would claim the current prompt produced them.
CODE = "a8e5903"

# Seventh in the case order and the only one missing from the scored six, and
# the work-package limitations name it too, so not a guess. How many of its
# generations came back before the cap is gone though - the harness threw the
# partial CaseResult away.
INTERRUPTED_CASE = "secondary_only"


def measured(summary: dict) -> dict:
    """The parts a metadata backfill must not move."""
    return {
        "generations_scored": summary.get("generations_scored"),
        "faithful": summary.get("faithful"),
        "faithfulness_rate": summary.get("faithfulness_rate"),
        "wilson_95": summary.get("wilson_95"),
        "violations_by_kind": summary.get("violations_by_kind"),
        "cases": [(c["name"], c["scored"], c["faithful"], len(c["violations"]))
                  for c in summary.get("cases", [])],
    }


def main() -> int:
    if not TARGET.is_file():
        raise SystemExit(f"no run at {TARGET.relative_to(ROOT)} — nothing to backfill")

    summary = json.loads(TARGET.read_text(encoding="utf-8"))
    names = [c["name"] for c in summary["cases"]]
    if names != EXPECTED_CASES:
        raise SystemExit(
            "this is not the 10 August run — scored cases are\n"
            f"  {names}\nexpected\n  {EXPECTED_CASES}\n"
            "Refusing to stamp one run's provenance onto another.")
    for key, want in EXPECTED_TOTALS.items():
        if summary.get(key) != want:
            raise SystemExit(f"{key} is {summary.get(key)}, expected {want} — "
                             "refusing to backfill an unrecognised run")

    before = measured(summary)

    summary["generations_excluded"] = [{
        "case": INTERRUPTED_CASE,
        # not 0 - zero would claim it never started, and it did
        "generations_obtained": None,
        "reason": "provider daily token cap reached part way through the case",
        "treatment": "excluded from the reported sample, not merged -- a case with a "
                     "different n would weight the total unevenly",
        "note": "the harness of the day discarded the partial result, so the number "
                "of generations collected before the cap is not recoverable",
    }]
    summary["unfaithful_generations"] = sum(
        len({v["run"] for v in c["violations"] if v["kind"] != "contract"})
        for c in summary["cases"])
    summary["calls_that_produced_no_report"] = (
        summary.get("contract_breaches", 0) + summary.get("samples_not_obtained", 0))
    summary["provenance"] = {
        "collected": COLLECTED,
        "code": CODE,
        "model": summary.get("model"),
        "temperature": coaching.TEMPERATURE,
        "runs_requested_per_case": summary.get("runs_requested_per_case"),
        "cases_scored": names,
        "layer1_payloads": "data/outputs/faithfulness/evaluations.json",
        "excluded": summary["generations_excluded"],
        "recorded": ("reconstructed after the run by "
                     "scripts/backfill_faithfulness_provenance.py; the harness did "
                     "not record provenance at the time"),
    }
    summary["method_limits"] = F.method_limits(summary)

    after = measured(summary)
    if before != after:
        raise SystemExit("a measured count changed — aborting without writing")

    TARGET.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"backfilled {TARGET.relative_to(ROOT)}")
    print(f"  unfaithful generations : {summary['unfaithful_generations']} "
          f"(violations: {summary['violations_by_kind']})")
    print(f"  excluded               : {INTERRUPTED_CASE}, count not recoverable")
    print("\nnow run: python scripts/build_chapter5_evidence.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

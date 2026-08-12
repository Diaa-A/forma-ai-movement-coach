"""Merge a single-case faithfulness run into the main results file.

    python scripts/fold_case_into_run.py --from results_task3_seventh_corrected.json

The seventh case, secondary_only, was cut off by the daily token cap on 10 August
and excluded rather than merged, because a case with a different n would weight
the total unevenly. It has now been collected at the same n=12 as the rest, so it
can go in.

The two halves were collected on different days off different commits, though,
and pretending otherwise would be the provenance problem this file has already
had once. So each case carries its own collected date and code stamp, and the
totals are reported both ways - six cases and seven - which is what the chapter
needs anyway if it is going to cite one of them while the other is in print.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.evaluation import faithfulness as F

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "outputs" / "faithfulness"


def totals(cases) -> dict:
    scored = sum(c["scored"] for c in cases)
    faithful = sum(c["faithful"] for c in cases)
    lo, hi = F.wilson_interval(faithful, scored)
    kinds = {}
    unfaithful = 0
    for c in cases:
        unfaithful += len({v["run"] for v in c["violations"] if v["kind"] != "contract"})
        for v in c["violations"]:
            kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
    return {
        # not "cases" -- the results file uses that key for the list itself, and
        # dst.update() with a count in it silently replaced the list with 7
        "case_count": len(cases),
        "generations_scored": scored,
        "faithful": faithful,
        "faithfulness_rate": round(faithful / scored, 4) if scored else 0.0,
        "wilson_95": [round(lo, 4), round(hi, 4)],
        "violations_by_kind": kinds,
        "unfaithful_generations": unfaithful,
    }


def stamp(case: dict, summary: dict) -> dict:
    """Give a case the provenance of the run it came out of."""
    prov = summary.get("provenance") or {}
    case = dict(case)
    case["collected"] = prov.get("collected")
    case["code"] = prov.get("code")
    return case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True,
                    help="single-case results file under data/outputs/faithfulness/")
    ap.add_argument("--into", default="results.json")
    args = ap.parse_args()

    src_path = RUNS / args.src
    dst_path = RUNS / args.into
    for p in (src_path, dst_path):
        if not p.is_file():
            raise SystemExit(f"missing: {p.relative_to(ROOT)}")

    src = json.loads(src_path.read_text(encoding="utf-8"))
    dst = json.loads(dst_path.read_text(encoding="utf-8"))

    if len(src["cases"]) != 1:
        raise SystemExit(f"{args.src} holds {len(src['cases'])} cases, expected one")
    incoming = src["cases"][0]

    requested = dst.get("runs_requested_per_case")
    if requested and incoming["scored"] != requested:
        raise SystemExit(
            f"{incoming['name']} scored {incoming['scored']} against "
            f"runs_requested_per_case {requested}. Merging an uneven n is what the "
            "10 August run deliberately refused to do.")

    original = [stamp(c, dst) for c in dst["cases"]]
    if any(c["name"] == incoming["name"] for c in original):
        raise SystemExit(f"{incoming['name']} is already in {args.into}")

    merged = original + [stamp(incoming, src)]

    dst["cases"] = merged
    dst.update(totals(merged))
    # Both, always. The chapter may be written while one of them is in print.
    dst["totals_by_grouping"] = {
        "six_cases_10_aug": totals(original),
        "all_seven_cases": totals(merged),
    }
    # It is no longer excluded, so the caveat built out of that list has to go.
    dst["generations_excluded"] = [
        e for e in dst.get("generations_excluded", [])
        if e["case"] != incoming["name"]]
    dst["stopped_on_quota"] = bool(dst["generations_excluded"])
    dst["method_limits"] = F.method_limits(dst)
    prov = dst.setdefault("provenance", {})
    prov["excluded"] = dst["generations_excluded"]
    prov["collected"] = "; ".join(sorted({c["collected"] for c in merged if c["collected"]}))
    prov["cases_scored"] = [c["name"] for c in merged]
    prov["note"] = ("collected in two sittings against the daily token cap. Each "
                    "case carries the date and code stamp of the run it came from.")

    dst_path.write_text(json.dumps(dst, indent=2), encoding="utf-8")
    six = dst["totals_by_grouping"]["six_cases_10_aug"]
    seven = dst["totals_by_grouping"]["all_seven_cases"]
    print(f"folded {incoming['name']} into {dst_path.relative_to(ROOT)}")
    print(f"  six cases  : {six['faithful']}/{six['generations_scored']} "
          f"= {six['faithfulness_rate']:.1%}  Wilson {six['wilson_95']}")
    print(f"  seven cases: {seven['faithful']}/{seven['generations_scored']} "
          f"= {seven['faithfulness_rate']:.1%}  Wilson {seven['wilson_95']}")
    print("\nnow run: python scripts/build_chapter5_evidence.py")


if __name__ == "__main__":
    main()

"""Measure Layer-2 faithfulness against Layer-1 (WP-05).

    python scripts/run_faithfulness.py --runs 12

Needs GROQ_API_KEY. Writes `data/outputs/faithfulness/` — a machine-readable
`results.json`, a summary table, and a figure for the report.

**Why the pipeline runs once per clip and the model runs many times.** The
pipeline is deterministic: analysing the same clip twice gives the same
`Evaluation`, so re-running it would cost twelve seconds a go and add no
variance. What varies is the model, at temperature 0.4. So each clip is analysed
once, its Layer-1 evaluation is cached, and only the Layer-2 call repeats. The
cache also means a re-run of this harness needs no video at all.

**The case set is deliberately adversarial**, because a harness that returns 100%
on the easy cases has not been made to work. Alongside ordinary faulted sets it
includes a clean set where Layer 1 found nothing wrong (does the model invent a
fault to have something to say?), a set where only a secondary cue fired (does it
promote it into something bigger?), and a voice note mentioning pain, which the
system prompt handles specially and is the one place it is *told* to say
something not in the cue database.

Cases marked "constructed" are hand-built `Evaluation` objects rather than clip
output, because no clip in the fixture set produces them. They use the real cue
and positive strings from the databases. They are labelled as constructed in the
results so the report does not present them as observed runs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.evaluation import faithfulness as F
from backend.exercises.mechanics import CueHit, Evaluation
from backend.exercises.squat_cues import SQUAT_CUES, SQUAT_POSITIVES
from backend.pipeline.runner import run_pipeline, RunOptions


ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "data" / "test_videos"
OUT = ROOT / "data" / "outputs" / "faithfulness"
CACHE = OUT / "evaluations.json"


def _load_dotenv(path=ROOT / ".env"):
    if not Path(path).is_file():
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# clips whose Layer-1 output is used as-is. Push-up clips are the CC0 Pexels set
# (credits in data/test_videos/pexels/push_up_exercise/SOURCE.txt).
CLIPS = [
    ("squat_lean", VIDEOS / "squat.mp4", "squat"),
    ("pushup_shallow", VIDEOS / "pexels" / "push_up_exercise" / "8171383.mp4", "pushup"),
    ("pushup_sag", VIDEOS / "pexels" / "push_up_exercise" / "6970183.mp4", "pushup"),
]


def evaluation_to_dict(ev: Evaluation) -> dict:
    return {
        "exercise": ev.exercise, "side": ev.side, "rep_count": ev.rep_count,
        "cues_fired": [{"flag": c.flag, "severity": c.severity, "fault": c.fault,
                        "fix": c.fix, "joints": c.joints,
                        "rep_indices": c.rep_indices} for c in ev.cues_fired],
        "positives": ev.positives, "notes": ev.notes,
        "view_guidance": ev.view_guidance,
    }


def evaluation_from_dict(d: dict) -> Evaluation:
    return Evaluation(
        exercise=d["exercise"], side=d["side"], rep_count=d["rep_count"],
        cues_fired=[CueHit(flag=c["flag"], severity=c["severity"], fault=c["fault"],
                           fix=c["fix"], joints=c.get("joints", []),
                           rep_indices=c.get("rep_indices", []))
                    for c in d["cues_fired"]],
        positives=d["positives"], notes=d["notes"],
        view_guidance=d.get("view_guidance"),
    )


def build_cache(force: bool = False) -> dict:
    """Analyse each clip once and keep its Layer-1 evaluation."""
    if CACHE.is_file() and not force:
        print(f"[=] using cached evaluations from {CACHE.relative_to(ROOT)}")
        return json.loads(CACHE.read_text())

    cache = {}
    for name, path, exercise in CLIPS:
        if not path.is_file():
            print(f"[!] {name}: clip missing ({path.name}) — skipping")
            continue
        print(f"[+] analysing {path.name} as {exercise} ...")
        result = run_pipeline(path, OUT / "_runs", exercise,
                              options=RunOptions(coach=True, force_dry_run_coach=True))
        payload = json.loads(Path(result.coaching_path).read_text())
        if payload.get("evaluation") is None:
            print(f"    status was {payload['status']} — no evaluation to score, skipping")
            continue
        cache[name] = payload["evaluation"]
        fired = [c["flag"] for c in cache[name]["cues_fired"]]
        print(f"    {result.summary['rep_count']} reps, cues: {fired or '(none)'}")

    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=2))
    return cache


def constructed_cases() -> dict:
    """Cases no fixture clip produces, built from the real cue/positive strings.

    Labelled constructed in the output. They exist because the acceptance
    criteria ask for them specifically and because they are where invention is
    most likely: given little or nothing to report, does the model fill the gap?
    """
    clean = Evaluation(
        exercise="squat", side="left", rep_count=6, cues_fired=[],
        positives=[SQUAT_POSITIVES["good_depth"]["text"],
                   SQUAT_POSITIVES["upright_torso"]["text"],
                   SQUAT_POSITIVES["consistent_reps"]["text"],
                   SQUAT_POSITIVES["controlled_tempo"]["text"]],
        notes=["average knee angle at the bottom: 84° (target ~90°, deeper acceptable)",
               "average trunk lean at the bottom: 41° from vertical, 3° more than the "
               "shins (trunk and shins should stay roughly parallel)",
               "average descent duration: 1.60s"],
    )

    secondary_only = Evaluation(
        exercise="squat", side="left", rep_count=5,
        cues_fired=[CueHit(
            flag="rep_inconsistency",
            severity=SQUAT_CUES["rep_inconsistency"]["severity"],
            fault=SQUAT_CUES["rep_inconsistency"]["fault"],
            fix=SQUAT_CUES["rep_inconsistency"]["fix"],
            joints=list(SQUAT_CUES["rep_inconsistency"]["joints"]))],
        positives=[SQUAT_POSITIVES["good_depth"]["text"],
                   SQUAT_POSITIVES["upright_torso"]["text"],
                   # Not decoration. squat_cues emits the descent-duration note
                   # and this positive off the same list, and if the minimum had
                   # not cleared _FAST_DESCENT_SEC then fast_descent would be in
                   # cues_fired instead. Note plus no fast_descent implies the
                   # positive, so a payload without it is one Layer 1 cannot
                   # produce. Leaving it out cost six false violations in twelve:
                   # the model wrote "a more controlled squat", control is in the
                   # tempo vocabulary, and nothing in the payload sourced it.
                   SQUAT_POSITIVES["controlled_tempo"]["text"]],
        notes=["average knee angle at the bottom: 92° (target ~90°, deeper acceptable)",
               "average descent duration: 1.40s"],
    )
    return {"clean_no_cues": clean, "secondary_only": secondary_only}


PAIN_TRANSCRIPT = ("my left knee has been aching since last week and I want to know "
                   "if my squat form is making it worse")

# The question that started §15: a real user asked this about a real set and got a
# fluent report that said nothing whatsoever about arms. Elbow flare is lateral,
# so a side-on clip genuinely cannot see it, and the cue ships parked.
#
# This is the case where inventing is most tempting, because the user has asked a
# direct question the payload cannot answer. The system prompt now tells the model
# to say so rather than ignore it or answer from general knowledge; the un-cued
# check is what catches it doing the latter.
UNANSWERABLE_TRANSCRIPT = ("let me know if my arms are too flared or in the correct "
                           "position, and if I'm doing the correct form")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=12,
                    help="generations per case (default 12; temperature is 0.4 so "
                         "this has to be a distribution, not one verdict)")
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="re-analyse the clips instead of using cached evaluations")
    ap.add_argument("--case", default="",
                    help="comma-separated case names to run instead of all of them. "
                         "The lockout experiment needs one case measured three times, "
                         "and running all seven would cost six times the quota it needs "
                         "- roughly a day's free-tier allowance per arm.")
    ap.add_argument("--label", default="",
                    help="suffix for the output filenames. Defaults to the case names "
                         "when --case is used, so a single-case run cannot overwrite "
                         "the full-run results it is being compared against.")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between calls")
    ap.add_argument("--model", default=coaching_default())
    args = ap.parse_args()

    _load_dotenv()
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is not set — this harness measures the real "
                         "model, so there is nothing to measure without it.")

    OUT.mkdir(parents=True, exist_ok=True)
    cache = build_cache(force=args.rebuild_cache)

    cases = []
    for name in cache:
        cases.append((name, evaluation_from_dict(cache[name]), "", False))
    if "squat_lean" in cache:
        cases.append(("squat_with_pain_note",
                      evaluation_from_dict(cache["squat_lean"]), PAIN_TRANSCRIPT, False))
    for pushup_case in ("pushup_shallow", "pushup_sag"):
        if pushup_case in cache:
            cases.append(("pushup_unanswerable_question",
                          evaluation_from_dict(cache[pushup_case]),
                          UNANSWERABLE_TRANSCRIPT, False))
            break
    for name, ev in constructed_cases().items():
        cases.append((name, ev, "", True))

    # Filtering writes to its own files. Overwriting results.json with a one-case
    # run would destroy the baseline the run is being compared against, which for
    # the lockout experiment is the entire point of doing it.
    suffix = ""
    if args.case:
        wanted = [c.strip() for c in args.case.split(",") if c.strip()]
        available = [c[0] for c in cases]
        unknown = [c for c in wanted if c not in available]
        if unknown:
            raise SystemExit(f"unknown case(s): {unknown}\navailable: {available}")
        cases = [c for c in cases if c[0] in wanted]
        suffix = "_" + "_".join(wanted)
    if args.label:
        suffix = "_" + args.label.strip().replace(" ", "_")

    print(f"\n[+] {len(cases)} cases x {args.runs} generations, model {args.model}")
    if suffix:
        print(f"    writing to results{suffix}.json — the full-run files are untouched")
    print()

    results = []
    constructed_flags = {}
    excluded = []
    stopped_early = False
    for i, (name, ev, transcript, is_constructed) in enumerate(cases):
        print(f"  {name}  (cues: {[c.flag for c in ev.cues_fired] or 'none'}"
              f"{', pain transcript' if transcript else ''})")
        constructed_flags[name] = is_constructed
        try:
            results.append(F.run_case(name, ev, runs=args.runs,
                                      voice_transcript=transcript, model=args.model,
                                      pause=args.pause))
        except F.QuotaExhausted as exc:
            # Report the sample that was collected rather than losing it. The
            # daily cap is a property of the account, not of the system under
            # test, and pretending the run did not happen would be worse than
            # reporting a smaller N honestly. Write down what it interrupted too
            # -- 10 August excluded its seventh case and recorded nothing, so the
            # partial count is gone.
            stopped_early = True
            partial = getattr(exc, "partial", None)
            excluded.append({
                "case": name,
                "generations_obtained": partial.runs if partial is not None else 0,
                "reason": "provider daily token cap reached part way through the case",
                "treatment": "excluded from the reported sample, not merged -- a case "
                             "with a different n would weight the total unevenly",
            })
            for later in cases[i + 1:]:
                excluded.append({
                    "case": later[0],
                    "generations_obtained": 0,
                    "reason": "not attempted, the run stopped on the token cap first",
                    "treatment": "excluded from the reported sample",
                })
            print(f"\n[!] daily token quota exhausted — stopping with a partial sample.")
            print(f"    {str(exc)[:160]}")
            break

    if not results:
        raise SystemExit("no samples collected — nothing to report")

    summary = F.summarise(results, excluded=excluded)
    for c in summary["cases"]:
        c["constructed"] = constructed_flags.get(c["name"], False)
    summary["model"] = args.model
    summary["runs_requested_per_case"] = args.runs
    summary["stopped_on_quota"] = stopped_early
    summary["provenance"] = provenance(args, summary)
    # Built last, because it reads the fields set above.
    summary["method_limits"] = F.method_limits(summary)

    (OUT / f"results{suffix}.json").write_text(json.dumps(summary, indent=2))
    table = F.format_table(summary)
    (OUT / f"summary{suffix}.txt").write_text(table)
    print("\n" + table)

    summary["_suffix"] = suffix
    write_figure(summary)
    print(f"\nwritten to {OUT.relative_to(ROOT)}/")
    return 0


def coaching_default():
    from backend.pipeline import coaching
    return coaching.DEFAULT_MODEL


# The code that decides what this run measures. coaching.py is in the list
# because changing what the model is asked for changes what it can be unfaithful
# about.
MEASURING_CODE = [
    "backend/evaluation/faithfulness.py",
    "backend/pipeline/coaching.py",
    "scripts/run_faithfulness.py",
]


def code_commit() -> str:
    """The commit of the measuring code, stamped when the run happens.

    build_chapter5_evidence.py has a near-identical helper that runs at publish
    time instead. That is fine for Penn Action, which reproduces bit-identically
    from current code. A faithfulness run never gets re-derived, so stamping it
    late would credit today's prompt with generations collected weeks ago.
    """
    try:
        r = subprocess.run(["git", "log", "-1", "--format=%h", "--"] + MEASURING_CODE,
                           cwd=ROOT, capture_output=True, text=True, check=True)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def provenance(args, summary: dict) -> dict:
    """Collection date, the conditions, and what was left out.

    The 4 August file had this as prose and the 10 August one had nothing, which
    left a reader with a table of numbers and no way to tell when or against what
    they were collected. Temperature is read from `coaching` rather than retyped;
    a provenance block that disagrees with the request body is worse than none.
    """
    from backend.pipeline import coaching
    return {
        "collected": date.today().isoformat(),
        "code": code_commit(),
        "model": args.model,
        "temperature": coaching.TEMPERATURE,
        "runs_requested_per_case": args.runs,
        "cases_scored": [c["name"] for c in summary["cases"]],
        "layer1_payloads": str(CACHE.relative_to(ROOT)).replace("\\", "/"),
        "excluded": summary.get("generations_excluded", []),
    }


def write_figure(summary: dict):
    """Per-case faithfulness with the overall rate and its interval."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = summary["cases"]
    names = [c["name"].replace("_", "\n") for c in cases]
    rates = [c["rate"] for c in cases]
    lo, hi = summary["wilson_95"]
    overall = summary["faithfulness_rate"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colours = ["#c0504d" if r < 1.0 else "#4f81bd" for r in rates]
    ax.bar(names, rates, color=colours)
    ax.axhline(overall, color="#333", linestyle="--", linewidth=1.2,
               label=f"overall {overall:.0%}")
    ax.axhspan(lo, hi, color="#333", alpha=0.10,
               label=f"95% Wilson [{lo:.0%}, {hi:.0%}]")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("faithful generations")
    ax.set_title(f"Layer-2 faithfulness to Layer-1 — {summary['generations_scored']} "
                 f"generations, {summary['model']}")
    ax.legend(loc="lower right", fontsize=8)
    for i, c in enumerate(cases):
        ax.text(i, c["rate"] + 0.02, f"{c['faithful']}/{c['scored']}",
                ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"faithfulness{summary.get('_suffix','')}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

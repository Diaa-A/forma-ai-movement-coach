"""Before and after for the answer-the-user's-question defect.

    python scripts/build_answer_experiment.py

SYSTEM_PROMPT has told the model to answer the user's spoken question first since
4 Aug. On 12 Aug it was measured for the first time and it never did it: 0 of 12
on the case built from the original complaint. The rule had nowhere to land -
every output field was defined in terms of the cue material, so the model
followed the schema.

The fix is a field, `answer_to_question`, and nothing else. Same payload, same
model, same temperature, same case. Both runs kept their generations, so the text
is in here rather than only the counts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.evaluation import faithfulness as F
from backend.pipeline import coaching

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "chapter5"
RUNS = ROOT / "data" / "outputs" / "faithfulness"
CASE = "pushup_unanswerable_question"

ARMS = [
    ("before", RUNS / "results_armA_declared_question.json",
     "the rule was in SYSTEM_PROMPT and there was no field to put the answer in"),
    ("after", RUNS / "results_answer_fix.json",
     "answer_to_question added to the output schema, rule points at it"),
]


def arm(label: str, path: Path, change: str) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    case = next(c for c in summary["cases"] if c["name"] == CASE)
    gens = case["generations"]
    answers = [{"run": g["run"], "text": g["report"].get("answer_to_question")}
               for g in gens]
    return {
        "arm": label,
        "change": change,
        "source": str(path.relative_to(ROOT)).replace("\\", "/"),
        "n": case["scored"],
        "answered": sum(1 for a in answers if a["text"]),
        "faithful": case["faithful"],
        "wilson_95": [round(x, 4) for x in
                      F.wilson_interval(case["faithful"], case["scored"])],
        "violations": case["violations"],
        "answers": answers,
        "generations": gens,
    }


def build() -> dict:
    missing = [str(p.relative_to(ROOT)) for _, p, _ in ARMS if not p.is_file()]
    if missing:
        raise SystemExit(f"missing runs: {missing}")
    return {
        "case": CASE,
        "question": ("let me know if my arms are too flared or in the correct "
                     "position, and if I'm doing the correct form"),
        "model": coaching.active_model(),
        "temperature": coaching.TEMPERATURE,
        "measure": ("generations that answered the user, counted off the "
                    "answer_to_question field rather than read out of the prose"),
        "arms": [arm(*a) for a in ARMS],
        "notes": [
            "The question genuinely cannot be answered from a side-on clip. Elbow "
            "flare is lateral, so the correct answer is to say so, which is what "
            "the after arm does, in the wording the coverage note supplies.",
            "The after arm first scored 1 of 12 faithful with 11 ungated "
            "violations. All eleven were the model repeating the filming guidance "
            "in the new field. The gate rule already exempts that for corrective "
            "cues, and the new field had been scored as an assertion - the same "
            "mistake as section 18.3 in a new place. Corrected, and the twelve "
            "generations were re-scored offline rather than collected again.",
            "The gate rule now skips this field, and the cost is stated in "
            "method_limits: a generation asserting there that left and right "
            "looked even would not be caught.",
        ],
    }


def table(d: dict) -> str:
    L = [f"Answering the user's question — {d['case']}", ""]
    L.append(f'voice note: "{d["question"]}"')
    L.append(f"{d['model']} at temperature {d['temperature']}, n=12 either side.")
    L.append("")
    hdr = f"{'arm':<8}{'n':>4}{'answered':>10}{'faithful':>10}   change"
    L.append(hdr)
    L.append("-" * 78)
    for a in d["arms"]:
        L.append(f"{a['arm']:<8}{a['n']:>4}{a['answered']:>10}{a['faithful']:>10}"
                 f"   {a['change']}")
    L.append("-" * 78)
    L.append("")
    for a in d["arms"]:
        L.append(f"[{a['arm']}] what the user was told")
        shown = [x for x in a["answers"] if x["text"]][:3]
        if not shown:
            L.append("  nothing. no generation addressed the question at all")
        for x in shown:
            L.append(f"  run {x['run']:>2}  {x['text']}")
        L.append("")
    L.append("Notes:")
    for n in d["notes"]:
        L.append("  - " + n)
    return "\n".join(L)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = build()
    (OUT / "answer_experiment.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    t = table(d)
    (OUT / "answer_experiment.txt").write_text(t + "\n", encoding="utf-8")
    print(t)
    print(f"\nwritten to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()

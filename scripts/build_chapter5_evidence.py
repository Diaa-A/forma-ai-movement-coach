"""Consolidate the Penn Action benchmark into citable evidence for Chapter 5.

The benchmark already writes `metrics.json` per action under `data/outputs/`, but
that directory is **gitignored**, so from the repository's point of view the
numbers do not exist — they survive only as prose in the handoff and as pixels in
two PNGs. A report cannot cite a figure whose only source is a summary of itself.

This reads the per-action metrics the benchmark produced, adds the things a
marker will look for and the raw files do not carry, and writes one committed
file under `report/chapter5/`.

The addition that matters most is the **keypoint correspondence**. It is imported
from `backend.evaluation.penn_action` rather than retyped, so the file records the
mapping that was actually used rather than one written out beside it that could
drift. That mapping contains the left/right swap which was the project's most
instructive evaluation finding, and the file carries the number that proves it:
MPJPE with the swap undone, which is roughly three times worse.

    python scripts/build_chapter5_evidence.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.evaluation.penn_action import PENN_JOINTS, PENN_TO_MP
from backend.pipeline.pose import LM

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "chapter5"

SOURCES = {
    "squat":  ROOT / "data" / "outputs" / "penn_eval" / "metrics.json",
    "pushup": ROOT / "data" / "outputs" / "penn_eval_pushup" / "metrics.json",
}


# The code whose behaviour determines these numbers. Anything outside this list
# can change without the benchmark meaning anything different.
EVAL_CODE = [
    # Named files, not the whole of backend/evaluation/. faithfulness.py lives in
    # there too and cannot move a Penn Action number, so a directory here means
    # editing the checker restamps this file with a commit that changed nothing
    # in it -- the same empty diff the docstring below rejects HEAD for.
    "backend/evaluation/penn_action.py",
    "backend/evaluation/metrics.py",
    "backend/pipeline/pose.py",
    "backend/pipeline/angles.py",
    "scripts/eval_penn_action.py",
]


def commit() -> str:
    """The last commit that touched the evaluation code, not HEAD.

    HEAD was the obvious choice and it is the wrong one: it changes on every
    commit, so regenerating this file after any unrelated change produces a diff
    that says nothing. Worse, it invites the reading that the numbers were
    re-measured when only the commit counter moved.

    What a reader actually wants is which version of the measuring code produced
    the measurement. That only moves when the measurement could have moved.
    """
    try:
        r = subprocess.run(["git", "log", "-1", "--format=%h", "--"] + EVAL_CODE,
                           cwd=ROOT, capture_output=True, text=True, check=True)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def correspondence() -> list:
    """The Penn Action → MediaPipe mapping actually in use.

    Penn Action labels joints by image-observer convention — "left" is the
    viewer's left — while MediaPipe labels them anatomically. When the subject
    faces the camera the two are opposite, which is why every entry below is
    crossed. Getting this wrong is not subtle in aggregate but is invisible
    per-frame: it produced a bimodal PCK of either 0.99 or 0.06 per clip, and
    fixing it moved the overall figure from 0.37 to 0.87.
    """
    names = {v: k for k, v in LM.items()}
    return [
        {
            "penn_action_joint": j,
            "penn_action_index": i,
            "mediapipe_index": PENN_TO_MP[j],
            "mediapipe_landmark": names.get(PENN_TO_MP[j], "?"),
            "crossed": j.split("_")[0] in ("left", "right")
            and not names.get(PENN_TO_MP[j], "").startswith(j.split("_")[0]),
        }
        for i, j in enumerate(PENN_JOINTS)
    ]


def excluded(per_sequence: dict) -> list:
    """Sequences that produced nothing, named rather than dropped.

    They stay in the denominator. A clip the system could not use is a result,
    and excluding it would flatter the detection rate.
    """
    return [
        {
            "sequence": sid,
            "reason": "zero detections across every evaluated frame",
            "frames_evaluated": s["frames_evaluated"],
            "treatment": "counted in the detection rate, not dropped",
        }
        for sid, s in sorted(per_sequence.items())
        if s["frames_detected"] == 0
    ]


def build() -> dict:
    out = {
        "generated": date.today().isoformat(),
        "commit": commit(),
        "dataset": "Penn Action (Zhang, Zhu & Derpanis, 2013)",
        "detection_rate_definition": (
            "frames in which MediaPipe returned a pose, divided by frames evaluated. "
            "Sequences that yielded nothing are counted, not dropped."
        ),
        "mpjpe_definition": (
            "mean Euclidean pixel distance between predicted and ground-truth joints, "
            "over joints marked visible in the ground truth."
        ),
        "pck_definition": (
            "PCK@0.2 — a joint counts as correct when its error is within 0.2 x the "
            "torso diagonal. Normalising by torso length makes the threshold "
            "body-size invariant, but it is NOT comparable across body orientations: "
            "a torso foreshortens when filmed side-on in a plank, so the same pixel "
            "error is judged against a smaller reference and scores worse."
        ),
        "keypoint_correspondence": correspondence(),
        "exercises": {},
    }

    for action, path in SOURCES.items():
        if not path.is_file():
            out["exercises"][action] = {"error": f"missing: {path.relative_to(ROOT)}"}
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        agg, per_seq = m["aggregate"], m["per_sequence"]
        out["exercises"][action] = {
            "sequences": m.get("sequences", len(per_seq)),
            "sequence_ids": [min(per_seq), max(per_seq)],
            "model": m.get("model", "full"),
            "frame_stride": m.get("frame_stride", 2),
            "frames_evaluated": agg["frames_evaluated"],
            "frames_detected": agg["frames_detected"],
            "detection_rate": agg["detection_rate"],
            "mpjpe_2d_pixels_mean": agg["mpjpe_2d_pixels"],
            "mpjpe_2d_pixels_median": agg["mpjpe_2d_median"],
            "pck@0.2": agg["pck@0.2"],
            "pck_evaluated_joints": agg["pck_evaluated_joints"],
            "per_joint_pixel_error": agg["per_joint_pixel_error"],
            "angle_error_degrees": agg["angle_error_degrees"],
            "mpjpe_if_left_right_swapped": agg.get("lr_swapped_mpjpe_pixels"),
            "excluded_sequences": excluded(per_seq),
            "source": str(path.relative_to(ROOT)).replace("\\", "/"),
        }

    sq = out["exercises"].get("squat", {})
    if "model" in sq and not json.loads(SOURCES["squat"].read_text(encoding="utf-8")).get("model"):
        out["exercises"]["squat"]["model_note"] = (
            "the squat metrics file predates the provenance fields; model and stride "
            "are the script defaults and match the push-up run"
        )
    return out


def table(d: dict) -> str:
    L = []
    L.append("Penn Action benchmark — 2D pose accuracy against ground truth")
    L.append(f"generated {d['generated']}  commit {d['commit']}")
    L.append("")
    L.append("reproduce with:")
    L.append("    python scripts/eval_penn_action.py --action squat  --limit 25")
    L.append("    python scripts/eval_penn_action.py --action pushup --limit 25")
    L.append("    python scripts/build_chapter5_evidence.py")
    L.append("")
    hdr = f"{'metric':<34}{'squat':>16}{'push-up':>16}"
    L.append(hdr)
    L.append("-" * len(hdr))

    def row(label, fn, fmt="{:.3f}"):
        vals = []
        for a in ("squat", "pushup"):
            v = fn(d["exercises"].get(a, {}))
            vals.append(fmt.format(v) if isinstance(v, (int, float)) else str(v))
        L.append(f"{label:<34}{vals[0]:>16}{vals[1]:>16}")

    row("sequences", lambda e: e.get("sequences"), "{:d}")
    row("frames evaluated", lambda e: e.get("frames_evaluated"), "{:d}")
    row("frames detected", lambda e: e.get("frames_detected"), "{:d}")
    row("detection rate", lambda e: e.get("detection_rate"), "{:.1%}")
    row("MPJPE mean (px)", lambda e: e.get("mpjpe_2d_pixels_mean"), "{:.1f}")
    row("MPJPE median (px)", lambda e: e.get("mpjpe_2d_pixels_median"), "{:.1f}")
    row("PCK@0.2", lambda e: e.get("pck@0.2"), "{:.3f}")
    row("MPJPE if L/R swapped (px)", lambda e: e.get("mpjpe_if_left_right_swapped"), "{:.1f}")
    L.append("")

    L.append("joint-angle error, degrees (the metric the cue layer depends on)")
    L.append(f"  {'joint':<16}{'mean':>10}{'median':>10}{'n':>8}   exercise")
    for a in ("squat", "pushup"):
        for joint, s in sorted(d["exercises"].get(a, {}).get("angle_error_degrees", {}).items()):
            L.append(f"  {joint:<16}{s['mean']:>10.1f}{s['median']:>10.1f}{s['n']:>8}   {a}")
    L.append("")

    for a in ("squat", "pushup"):
        ex = d["exercises"].get(a, {}).get("excluded_sequences", [])
        if ex:
            L.append(f"{a}: sequences yielding nothing — " +
                     ", ".join(f"{e['sequence']} ({e['reason']})" for e in ex))
            L.append(f"  {ex[0]['treatment']}")
    L.append("")
    L.append("Read the median alongside the mean for push-up. The gap between 20.0 and")
    L.append("8.3 px is a heavy tail: most frames track about as well as squat frames and")
    L.append("a few track very badly. The mean alone misrepresents typical performance;")
    L.append("the median alone hides a real failure mode.")
    L.append("")
    L.append("PCK is lower for push-up (0.799 vs 0.871) and the comparison is not")
    L.append("like-for-like — see pck_definition in the JSON.")
    return "\n".join(L)


FAITHFULNESS_SRC = ROOT / "data" / "outputs" / "faithfulness"


def publish_faithfulness():
    """Copy the faithfulness run into report/ and rebuild its summary.

    The harness writes to `data/outputs/`, which is gitignored, so a copy has to
    be published under `report/` to be citable at all. That copy had already gone
    stale once — a partial 20-generation run from 4 August sat in `report/` while
    a complete 72-generation run sat in `data/outputs/`, and the stale one was
    the one that got cited. Republishing from a single command is the fix; two
    hand-copied files is what caused it.

    The summary is regenerated rather than copied, so a change to `format_table`
    reaches the published artefact without burning provider quota on a rerun.

    `method_limits` is regenerated for the same reason, and it needed to be: the
    10 August run carried a caveat about uneven, smaller-than-requested samples
    that had been true of a partial run on 4 August and was false of every case
    in the file it was sitting in. Measured counts are copied untouched — only
    the prose derived from them is rebuilt.
    """
    src = FAITHFULNESS_SRC / "results.json"
    if not src.is_file():
        print(f"[!] no faithfulness run at {src.relative_to(ROOT)} — skipping")
        return None

    from backend.evaluation import faithfulness as F

    summary = json.loads(src.read_text(encoding="utf-8"))
    summary["method_limits"] = F.method_limits(summary)
    (OUT / "faithfulness_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    text = F.format_table(summary)
    (OUT / "faithfulness_summary.txt").write_text(text + "\n", encoding="utf-8")

    fig = FAITHFULNESS_SRC / "faithfulness.png"
    if fig.is_file():
        (OUT / "figures").mkdir(parents=True, exist_ok=True)
        (OUT / "figures" / "fig8_faithfulness.png").write_bytes(fig.read_bytes())
    return text


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = build()
    (OUT / "penn_action_results.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    t = table(d)
    (OUT / "penn_action_summary.txt").write_text(t + "\n", encoding="utf-8")
    print(t)

    print("\n" + "=" * 66 + "\n")
    ft = publish_faithfulness()
    if ft:
        print(ft)
    print(f"\nwritten to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()

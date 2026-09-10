"""Two ablations against Penn Action ground truth, on identical sequences.

    python scripts/compare_pose_models.py

Answers the preliminary marker's request to compare against other techniques on
the same dataset. Nothing in the analysis path changes: this imports the existing
evaluation harness and the pipeline's own filter, and only varies what is being
compared.

**Model.** pose_landmarker_full against pose_landmarker_lite, both actions, the
same 25 sequences and the same frame stride the published benchmark used. The
full column should therefore reproduce the committed numbers exactly, which is
the check that this harness is measuring the same thing.

**One Euro filter.** The benchmark deliberately runs each frame independently -
`extract_landmarks_from_frames` exists so pose accuracy is measured without a
temporal-smoothing confound - so the filter has never been measured against
ground truth at all. Here it is, by running pose once per action and scoring the
same detections twice: raw, then passed through `runner._smooth_landmarks` with
the pipeline's own parameters. Identical frames and identical detections in both
arms, so detection rate cannot move and the only difference is the smoothing.

Two things to know about the filter arm:

  - It runs at **stride 1**. One Euro is frame-rate dependent, and the pipeline
    sees every frame, so scoring it on every other frame would measure a filter
    the product does not run.
  - Penn Action labels carry no frame rate, so timestamps are synthesised at a
    nominal **30 fps**. The filter's tuning assumes roughly that. A real rate
    that differs would shift the smoothing, and this is the one assumption in
    the ablation that cannot be checked from the data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backend.evaluation.metrics import Accumulator
from backend.evaluation.penn_action import find_sequences_by_action, load_sequence
from backend.pipeline.pose import extract_landmarks_from_frames
from backend.pipeline.runner import _smooth_landmarks

from eval_penn_action import evaluate_sequence, mp_to_penn_pixels

DATA_ROOT = ROOT / "data" / "penn_action"
OUT = ROOT / "report" / "chapter5"
ACTIONS = ("squat", "pushup")
LIMIT = 25
STRIDE_MODEL = 2        # what the published benchmark used
STRIDE_FILTER = 1       # the rate the pipeline actually sees
NOMINAL_FPS = 30.0
MIN_CUTOFF, BETA = 1.0, 0.007        # RunOptions defaults


def fold(dst: Accumulator, src: Accumulator):
    for name, vals in src.joint_errors.items():
        dst.joint_errors[name].extend(vals)
    for name, vals in src.angle_errors.items():
        dst.angle_errors.setdefault(name, []).extend(vals)
    dst.pck_correct += src.pck_correct
    dst.pck_total += src.pck_total
    dst.frames_evaluated += src.frames_evaluated
    dst.frames_detected += src.frames_detected


def headline(summary: dict) -> dict:
    """The five numbers the comparison is about."""
    ang = summary.get("angle_error_degrees", {})
    return {
        "frames_evaluated": summary["frames_evaluated"],
        "detection_rate": summary["detection_rate"],
        "mpjpe_mean_px": summary["mpjpe_2d_pixels"],
        "mpjpe_median_px": summary["mpjpe_2d_median"],
        "pck@0.2": summary["pck@0.2"],
        "angle_error_degrees": {k: {"mean": v["mean"], "median": v["median"],
                                    "n": v["n"]} for k, v in ang.items()},
    }


def run_model(action: str, model: str, seq_ids) -> dict:
    overall = Accumulator()
    for sid in seq_ids:
        seq = load_sequence(sid, DATA_ROOT)
        acc, _sw, _pose, _idxs = evaluate_sequence(seq, model, STRIDE_MODEL)
        fold(overall, acc)
    return headline(overall.summary())


def score_landmarks(seq, landmarks, sizes, idxs) -> Accumulator:
    """Score landmarks extracted elsewhere against one sequence's ground truth.

    Split out of evaluate_sequence so the filter ablation can run MediaPipe once
    and score both arms off the same detections. Extracting twice would let
    detector nondeterminism into a comparison that is supposed to isolate the
    filter.
    """
    acc = Accumulator()
    for k, i in enumerate(idxs):
        w, h = sizes[k]
        if w == 0:
            continue
        frame = landmarks[k]
        detected = bool(np.all(np.isfinite(frame[:, 0])))
        acc.add_frame(mp_to_penn_pixels(frame, w, h), seq.gt_xy(i),
                      seq.gt_visibility(i), seq.torso_diag(i), detected=detected)
    return acc


def jitter(landmarks, sizes) -> list:
    """Mean frame-to-frame joint displacement, in pixels.

    Reporting only MPJPE would misrepresent a smoothing filter. One Euro is not
    there to move a joint closer to ground truth - it lags a moving target, so it
    can only hurt that - it is there to stop a stationary joint shimmering, which
    is what the cue layer and the overlay read. So the ablation measures both: how
    far from the truth, and how much it moves between frames.
    """
    out = []
    for k in range(1, len(landmarks)):
        w, h = sizes[k]
        pw, ph = sizes[k - 1]
        if w == 0 or pw == 0:
            continue
        a = landmarks[k - 1][:, :2] * np.array([pw, ph])
        b = landmarks[k][:, :2] * np.array([w, h])
        d = np.linalg.norm(b - a, axis=1)
        d = d[np.isfinite(d)]
        if d.size:
            out.append(float(np.mean(d)))
    return out


def run_filter_ablation(action: str, seq_ids) -> dict:
    """One Euro on against off, over the same extracted landmarks.

    Both arms score identical detections, so the only thing separating them is
    the filter. That is also why the detection rates get compared at the end: if
    they ever differ the arms did not share their input, and the comparison is
    void rather than merely odd.
    """
    raw_all, smoothed_all = Accumulator(), Accumulator()
    jit_raw, jit_smooth = [], []
    moved = []
    for sid in seq_ids:
        seq = load_sequence(sid, DATA_ROOT)
        paths = seq.frame_paths()
        n = min(len(paths), seq.nframes)
        idxs = list(range(0, n, STRIDE_FILTER))
        pose = extract_landmarks_from_frames([paths[i] for i in idxs], model="full")

        lm = pose["landmarks"]
        ts = np.arange(len(idxs), dtype=np.float64) / NOMINAL_FPS
        sm = _smooth_landmarks(lm, ts, MIN_CUTOFF, BETA)

        raw = score_landmarks(seq, lm, pose["sizes"], idxs)
        smoothed = score_landmarks(seq, sm, pose["sizes"], idxs)
        fold(raw_all, raw)
        fold(smoothed_all, smoothed)
        jit_raw.extend(jitter(lm, pose["sizes"]))
        jit_smooth.extend(jitter(sm, pose["sizes"]))

        a, b = raw.summary()["mpjpe_2d_pixels"], smoothed.summary()["mpjpe_2d_pixels"]
        if a is not None and b is not None:
            moved.append({"sequence": sid, "raw": round(a, 2),
                          "smoothed": round(b, 2), "delta": round(b - a, 2)})

    r, s = raw_all.summary(), smoothed_all.summary()
    off, on = headline(r), headline(s)
    off["jitter_px_per_frame"] = round(float(np.mean(jit_raw)), 3) if jit_raw else None
    on["jitter_px_per_frame"] = round(float(np.mean(jit_smooth)), 3) if jit_smooth else None
    out = {"filter_off": off, "filter_on": on, "per_sequence_mpjpe": moved}
    if r["detection_rate"] != s["detection_rate"]:
        out["WARNING"] = ("detection rate differs between arms, which it cannot "
                          "if both scored the same detections - investigate")
    return out


# The joint each exercise is actually judged on. A change in mean MPJPE across
# thirteen joints can hide a large change in the one the cue layer reads.
CUE_JOINT = {"squat": "knee", "pushup": "elbow"}


def _cue_angle_mean(headline_block, action):
    """Mean angle error over the sides of the joint this exercise is judged on."""
    want = CUE_JOINT[action]
    vals = [v["mean"] for k, v in headline_block["angle_error_degrees"].items()
            if k.rsplit("_", 1)[0] == want and v["mean"] is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def _delta(a, b):
    if a is None or b is None:
        return None
    return round(b - a, 3)


def deltas(result: dict) -> dict:
    """Differences, computed rather than written down, so they cannot go stale."""
    out = {"model_lite_minus_full": {}, "filter_on_minus_off": {}}

    for action, models in result["model_comparison"]["actions"].items():
        f, l = models["full"], models["lite"]
        out["model_lite_minus_full"][action] = {
            "mpjpe_mean_px": _delta(f["mpjpe_mean_px"], l["mpjpe_mean_px"]),
            "pck@0.2": _delta(f["pck@0.2"], l["pck@0.2"]),
            "detection_rate": _delta(f["detection_rate"], l["detection_rate"]),
            "cue_joint": CUE_JOINT[action],
            "cue_joint_angle_error_deg": _delta(_cue_angle_mean(f, action),
                                                _cue_angle_mean(l, action)),
        }

    for action, arms in result["one_euro_ablation"]["actions"].items():
        off, on = arms["filter_off"], arms["filter_on"]
        out["filter_on_minus_off"][action] = {
            "mpjpe_mean_px": _delta(off["mpjpe_mean_px"], on["mpjpe_mean_px"]),
            "pck@0.2": _delta(off["pck@0.2"], on["pck@0.2"]),
            "jitter_px_per_frame": _delta(off.get("jitter_px_per_frame"),
                                          on.get("jitter_px_per_frame")),
            "cue_joint": CUE_JOINT[action],
            "cue_joint_angle_error_deg": _delta(_cue_angle_mean(off, action),
                                                _cue_angle_mean(on, action)),
        }
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=LIMIT,
                    help="sequences per action; lower it to smoke-test")
    ap.add_argument("--out", default=str(OUT),
                    help="where to write; point elsewhere for a trial run")
    args = ap.parse_args()
    limit, out_dir = args.limit, Path(args.out)

    if not DATA_ROOT.is_dir():
        raise SystemExit(f"no Penn Action data at {DATA_ROOT}")

    result = {
        "dataset": "Penn Action (Zhang, Zhu & Derpanis, 2013)",
        "sequences_per_action": limit,
        "model_comparison": {
            "frame_stride": STRIDE_MODEL,
            "note": ("same sequences and stride as the committed benchmark, so "
                     "the full column is expected to reproduce it exactly"),
            "actions": {},
        },
        "one_euro_ablation": {
            "frame_stride": STRIDE_FILTER,
            "model": "full",
            "min_cutoff": MIN_CUTOFF,
            "beta": BETA,
            "assumed_fps": NOMINAL_FPS,
            "note": ("both arms score the same detections from one pose pass, so "
                     "detection rate is identical by construction and only the "
                     "smoothing differs. Penn Action carries no frame rate, so "
                     "the timestamps are nominal."),
            "actions": {},
        },
    }

    for action in ACTIONS:
        seq_ids = find_sequences_by_action(DATA_ROOT, action)[:limit]
        if not seq_ids:
            raise SystemExit(f"no '{action}' sequences under {DATA_ROOT}")
        print(f"\n=== {action}: {len(seq_ids)} sequences ===")

        for model in ("full", "lite"):
            print(f"  model {model} (stride {STRIDE_MODEL}) ...", flush=True)
            result["model_comparison"]["actions"].setdefault(action, {})[model] = \
                run_model(action, model, seq_ids)

        print(f"  one euro on/off (stride {STRIDE_FILTER}) ...", flush=True)
        result["one_euro_ablation"]["actions"][action] = \
            run_filter_ablation(action, seq_ids)

    result["deltas"] = deltas(result)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_comparison.json").write_text(json.dumps(result, indent=2),
                                                   encoding="utf-8")
    text = table(result)
    (out_dir / "model_comparison.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    print(f"\nwritten to {out_dir}")


def _f(v, spec=".1f"):
    return format(v, spec) if isinstance(v, (int, float)) else "n/a"


def angle_line(h, indent="    "):
    out = []
    for joint, v in sorted(h["angle_error_degrees"].items()):
        out.append(f"{indent}{joint:<14}{_f(v['mean']):>8}{_f(v['median']):>9}"
                   f"{v['n']:>8}")
    return out


def table(d: dict) -> str:
    L = ["Penn Action — model and filter ablations", ""]
    L.append(f"{d['sequences_per_action']} sequences per action, identical across "
             f"every column.")
    L.append("")

    L.append("MODEL: pose_landmarker_full vs pose_landmarker_lite")
    L.append(f"stride {d['model_comparison']['frame_stride']}, as the committed "
             f"benchmark")
    L.append("")
    hdr = (f"{'action / model':<20}{'frames':>8}{'detect':>9}{'MPJPE':>9}"
           f"{'median':>9}{'PCK@0.2':>9}")
    L.append(hdr)
    L.append("-" * len(hdr))
    for action, models in d["model_comparison"]["actions"].items():
        for model, h in models.items():
            L.append(f"{action + ' / ' + model:<20}{h['frames_evaluated']:>8}"
                     f"{_f(h['detection_rate'], '.1%'):>9}"
                     f"{_f(h['mpjpe_mean_px']):>9}{_f(h['mpjpe_median_px']):>9}"
                     f"{_f(h['pck@0.2'], '.3f'):>9}")
        L.append("")
    L.append("joint-angle error, degrees")
    for action, models in d["model_comparison"]["actions"].items():
        for model, h in models.items():
            L.append(f"  {action} / {model}")
            L.append(f"    {'joint':<14}{'mean':>8}{'median':>9}{'n':>8}")
            L.extend(angle_line(h))
    L.append("")

    L.append("ONE EURO FILTER: off vs on")
    ab = d["one_euro_ablation"]
    L.append(f"stride {ab['frame_stride']}, model {ab['model']}, "
             f"min_cutoff {ab['min_cutoff']}, beta {ab['beta']}, "
             f"{ab['assumed_fps']:.0f} fps assumed")
    L.append("")
    hdr2 = hdr + f"{'jitter':>9}"
    L.append(hdr2)
    L.append("-" * len(hdr2))
    for action, arms in ab["actions"].items():
        for arm in ("filter_off", "filter_on"):
            h = arms[arm]
            L.append(f"{action + ' / ' + arm.split('_')[1]:<20}"
                     f"{h['frames_evaluated']:>8}"
                     f"{_f(h['detection_rate'], '.1%'):>9}"
                     f"{_f(h['mpjpe_mean_px']):>9}{_f(h['mpjpe_median_px']):>9}"
                     f"{_f(h['pck@0.2'], '.3f'):>9}"
                     f"{_f(h.get('jitter_px_per_frame'), '.2f'):>9}")
        L.append("")
    L.append("jitter is mean frame-to-frame joint displacement in pixels. It is the")
    L.append("thing the filter exists to reduce; MPJPE is the thing it costs.")
    L.append("")
    L.append("joint-angle error, degrees")
    for action, arms in ab["actions"].items():
        for arm in ("filter_off", "filter_on"):
            L.append(f"  {action} / filter {arm.split('_')[1]}")
            L.append(f"    {'joint':<14}{'mean':>8}{'median':>9}{'n':>8}")
            L.extend(angle_line(arms[arm]))
    for action, arms in ab["actions"].items():
        if "WARNING" in arms:
            L.append(f"  [!] {action}: {arms['WARNING']}")

    dl = d.get("deltas")
    if dl:
        L.append("")
        L.append("WHAT CHANGED")
        L.append("")
        L.append(f"{'':<22}{'MPJPE':>9}{'PCK':>9}{'detect':>9}"
                 f"{'cue joint':>11}{'jitter':>9}")
        L.append("-" * 69)
        for action, v in dl["model_lite_minus_full"].items():
            L.append(f"{'lite - full, ' + action:<22}"
                     f"{_f(v['mpjpe_mean_px'], '+.1f'):>9}"
                     f"{_f(v['pck@0.2'], '+.3f'):>9}"
                     f"{_f(v['detection_rate'], '+.1%'):>9}"
                     f"{_f(v['cue_joint_angle_error_deg'], '+.1f'):>11}"
                     f"{'-':>9}")
        for action, v in dl["filter_on_minus_off"].items():
            L.append(f"{'filter on - off, ' + action:<22}"
                     f"{_f(v['mpjpe_mean_px'], '+.1f'):>9}"
                     f"{_f(v['pck@0.2'], '+.3f'):>9}"
                     f"{'0':>9}"
                     f"{_f(v['cue_joint_angle_error_deg'], '+.1f'):>11}"
                     f"{_f(v['jitter_px_per_frame'], '+.2f'):>9}")
        L.append("")
        L.append("'cue joint' is the mean angle error on the joint each exercise is")
        L.append("judged on - knee for the squat, elbow for the push-up. It is broken")
        L.append("out because a small move in MPJPE averaged over thirteen joints can")
        L.append("hide a large one in the joint the cue thresholds actually read.")
    return "\n".join(L)


if __name__ == "__main__":
    main()

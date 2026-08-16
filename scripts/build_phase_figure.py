"""Figure: velocity zero-crossing phase detection on the reference squat clip.

    python scripts/build_phase_figure.py

The algorithm the whole repetition analysis rests on had no figure anywhere in
the report. This draws it on real data: the hip trajectory with the detected
bottoms and the seven repetitions, and underneath it the velocity signal the
detector actually reads.

The unfiltered velocity is drawn behind the filtered one because the figure has a
second job. Section 5.1 reports that the One Euro filter costs angle accuracy at
the cue read. If the raw velocity crosses zero where the filtered one does not,
the same filter is visibly earning its place in phase detection, and the pair of
results is the trade-off rather than a contradiction.

Nothing here re-implements the detector. It imports detect_bottoms and
segment_reps and feeds them exactly what runner.run_pipeline feeds them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.evaluation import plotstyle
from backend.exercises import mechanics
from backend.exercises.registry import get_spec
from backend.pipeline.filter import smooth_series
from backend.pipeline.phase_detection import detect_bottoms, segment_reps
from backend.pipeline.pose import extract_landmarks
from backend.pipeline.runner import _smooth_landmarks

CLIP = ROOT / "data" / "test_videos" / "squat.mp4"
OUT = ROOT / "report" / "chapter4" / "figures" / "fig9_phase_detection.png"

# the pipeline's own settings, not new ones
LM_MIN_CUTOFF, LM_BETA = 1.0, 0.007        # RunOptions defaults, landmark smoothing
HIP_MIN_CUTOFF, HIP_BETA = 0.5, 0.001      # runner's second pass on the hip series


def crossings(velocity):
    """Descent-to-ascent transitions, by the detector's own rule."""
    sign = np.sign(velocity)
    sign[sign == 0] = 1.0
    changes = np.where(np.diff(sign) != 0)[0]
    return [int(i) for i in changes if velocity[i] < 0]


def build():
    spec = get_spec("squat")
    pose = extract_landmarks(CLIP, model="full")
    ts, fps = pose["timestamps"], pose["fps"]

    lm_smooth = _smooth_landmarks(pose["landmarks"], ts, LM_MIN_CUTOFF, LM_BETA)
    hip_filt = smooth_series(mechanics.travel_series(lm_smooth, *spec.travel_landmarks),
                             ts, min_cutoff=HIP_MIN_CUTOFF, beta=HIP_BETA)
    # the same quantity with no smoothing anywhere - raw landmarks, no second pass
    hip_raw = mechanics.travel_series(pose["landmarks"], *spec.travel_landmarks)

    min_sep = max(3, int(fps * 0.4))
    bottoms = detect_bottoms(hip_filt, min_separation=min_sep)
    reps = segment_reps(bottoms, pose["frame_count"])

    # negated exactly as the detector does: descending = negative
    vel_filt = np.gradient(-np.asarray(hip_filt, dtype=np.float64))
    vel_raw = np.gradient(-np.nan_to_num(hip_raw, nan=np.nan))

    return {
        "t": ts, "fps": fps,
        "hip_filt": hip_filt, "vel_filt": vel_filt, "vel_raw": vel_raw,
        "bottoms": bottoms, "reps": reps,
        "raw_crossings": crossings(np.nan_to_num(vel_raw)),
        "filt_crossings": crossings(vel_filt),
        "min_sep": min_sep,
    }


def draw(d):
    plt = plotstyle.apply()
    t = d["t"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1]})

    # ---- top: trajectory ------------------------------------------------
    for i, (s, b, e) in enumerate(d["reps"]):
        if i % 2 == 0:
            ax1.axvspan(t[s], t[min(e, len(t) - 1)], color=plotstyle.BAND, alpha=0.06,
                        linewidth=0)
    ax1.plot(t, d["hip_filt"], color=plotstyle.PRIMARY, linewidth=1.6,
             label="hip Y, filtered")
    ax1.plot(t[d["bottoms"]], np.asarray(d["hip_filt"])[d["bottoms"]], "v",
             color=plotstyle.ACCENT, markersize=8, linestyle="none",
             label=f"detected bottoms ({len(d['bottoms'])})")
    # image y grows downward, so flip the axis and the curve reads as the body does
    ax1.invert_yaxis()
    ax1.set_ylabel("hip Y\n(normalised, 0 = top of frame)")
    ax1.set_title("Velocity zero-crossing phase detection — reference squat clip")
    ax1.legend(loc="upper right", framealpha=0.9)

    # ---- bottom: velocity -----------------------------------------------
    ax2.plot(t, d["vel_raw"], color=plotstyle.NEUTRAL, linewidth=0.9, alpha=0.28,
             label=f"velocity, unfiltered ({len(d['raw_crossings'])} crossings)")
    ax2.plot(t, d["vel_filt"], color=plotstyle.PRIMARY, linewidth=1.6,
             label=f"velocity, filtered ({len(d['filt_crossings'])} crossings)")
    ax2.axhline(0.0, color=plotstyle.NEUTRAL, linewidth=1.0, linestyle="--")
    ax2.plot(t[d["bottoms"]], np.zeros(len(d["bottoms"])), "v",
             color=plotstyle.ACCENT, markersize=8, linestyle="none",
             label="crossings kept as bottoms")
    ax2.set_ylabel("d(-hip Y)/dframe\n(descending < 0)")
    ax2.set_xlabel("time (s)")
    ax2.legend(loc="upper right", framealpha=0.9, ncol=1)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=plotstyle.DPI)
    plt.close(fig)


def main():
    d = build()
    draw(d)
    raw_n, filt_n = len(d["raw_crossings"]), len(d["filt_crossings"])
    print(f"clip            : {CLIP.name}, {d['fps']:.2f} fps")
    print(f"reps detected   : {len(d['bottoms'])}")
    print(f"bottoms (frames): {d['bottoms']}")
    print(f"min_separation  : {d['min_sep']} frames")
    print()
    print(f"descent->ascent crossings, filtered   : {filt_n}")
    print(f"descent->ascent crossings, unfiltered : {raw_n}")
    if raw_n > filt_n:
        print(f"  the raw signal crosses {raw_n - filt_n} more times than the filtered "
              f"one, so the filter is removing spurious bottoms")
    else:
        print("  the raw signal does NOT cross zero more than the filtered one on this "
              "clip. That weakens the case for filtering the phase signal and belongs "
              "in the report as-is.")
    print()
    print(f"written to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

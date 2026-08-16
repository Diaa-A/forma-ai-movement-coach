"""Figure: one elbow measure separates, the other does not.

    python scripts/build_lockout_figure.py

Section 5.4 argues that no threshold separates locked-out from not-locked-out
repetitions, and asks the reader to take the numbers on trust. This draws both
measures from `lockout_calibration.json` so the contrast is visible: the bottom
of the rep, where the cue threshold sits in a gap wider than the measurement
error, against the top, where the clips form one continuous smear.

Nothing is excluded. Two clips sit far below the rest on both measures because
their tracking came apart - 4260553 and 8480310 report elbow angles near zero,
which a push-up cannot produce - and dropping them would tidy the picture at the
cost of hiding the reason the widest raw gap is not a class boundary. They are
drawn in the accent colour and the annotations quote gaps within the main
cluster, which is where a threshold would actually have to live.

Reads the stored calibration and computes nothing new, so the figure and the
prose cannot drift apart.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.evaluation import plotstyle
from backend.exercises.pushup import DEPTH_FLAG_ELBOW_ANGLE

SRC = ROOT / "report" / "chapter5" / "lockout_calibration.json"
OUT = ROOT / "report" / "chapter5" / "figures" / "fig10_lockout_separation.png"

# An elbow cannot fold below this at the bottom of a push-up. Used only to mark
# clips whose pose estimate failed, never to remove them from the figure.
IMPLAUSIBLE_BOTTOM = 30.0


def load():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    bottoms, spread, suspect = {}, {}, set()
    for name, c in d["per_clip"].items():
        if not c.get("reps"):
            continue
        b = c["bottoms"]
        bottoms[name] = float(np.median(b))
        spread[name] = (min(b), max(b))
        if min(b) < IMPLAUSIBLE_BOTTOM:
            suspect.add(name)
    return d, d["elbow_angle_error_deg"], bottoms, d["measures"]["max"]["per_clip"], \
        spread, suspect


def gap_around(values, threshold):
    """The gap the threshold falls in: nearest value below, nearest above."""
    below = [v for v in values if v < threshold]
    above = [v for v in values if v >= threshold]
    if not below or not above:
        return None
    return max(below), min(above)


def gaps_within(values, floor):
    """Adjacent gaps among the clips above `floor` - the dense part."""
    v = sorted(x for x in values if x >= floor)
    return [round(b - a, 1) for a, b in zip(v, v[1:])], v


def panel(ax, values, spread, suspect, title, err):
    order = sorted(values.items(), key=lambda kv: kv[1])
    names = [n for n, _ in order]
    y = np.arange(len(order))

    if spread:
        for i, n in enumerate(names):
            lo, hi = spread[n]
            ax.plot([lo, hi], [i, i], color=plotstyle.NEUTRAL, alpha=0.22,
                    linewidth=1.4, solid_capstyle="round", zorder=1)
    for i, n in enumerate(names):
        bad = n in suspect
        ax.plot(values[n], i, "o", markersize=7, zorder=3,
                color=plotstyle.ACCENT if bad else plotstyle.PRIMARY,
                markerfacecolor="none" if bad else plotstyle.PRIMARY)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("elbow angle (degrees)")
    ax.set_title(title)
    return names


def main():
    d, err, bottoms, tops, spread, suspect = load()
    plt = plotstyle.apply()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.5, 6))

    # ---- left: bottom of rep -------------------------------------------
    names_l = panel(ax_l, bottoms, spread, suspect,
                    "Bottom of rep — the threshold sits in a real gap", err)
    ax_l.axvline(DEPTH_FLAG_ELBOW_ANGLE, color=plotstyle.ACCENT, linestyle="--",
                 linewidth=1.4)
    ax_l.annotate(f"shallow_depth fires above {DEPTH_FLAG_ELBOW_ANGLE:.0f}°",
                  xy=(DEPTH_FLAG_ELBOW_ANGLE + 2, 0.2), fontsize=9,
                  color=plotstyle.ACCENT, ha="left", va="bottom")
    lo, hi = gap_around(bottoms.values(), DEPTH_FLAG_ELBOW_ANGLE)
    ax_l.axvspan(lo, hi, color=plotstyle.PRIMARY, alpha=0.10, linewidth=0, zorder=0)
    ax_l.annotate(f"{hi - lo:.1f}° gap, vs ±{err}° error",
                  xy=((lo + hi) / 2, -0.2), ha="center", va="bottom", fontsize=9,
                  color=plotstyle.PRIMARY)

    # ---- right: top of rep ---------------------------------------------
    names_r = panel(ax_r, tops, None, suspect,
                    "Top of rep — one continuous smear", err)
    cluster_gaps, cluster = gaps_within(tops.values(), 150.0)
    ax_r.axvspan(cluster[0], cluster[-1], color=plotstyle.BAND, alpha=0.07,
                 linewidth=0, zorder=0)
    ax_r.annotate(f"{len(cluster)} clips inside {cluster[-1] - cluster[0]:.1f}°\n"
                  f"widest adjacent gap {max(cluster_gaps):.1f}°",
                  xy=((cluster[0] + cluster[-1]) / 2, -0.2), ha="center",
                  va="bottom", fontsize=9, color=plotstyle.NEUTRAL)

    below = [n for n, v in bottoms.items() if 100 < v < DEPTH_FLAG_ELBOW_ANGLE]
    for n in below:
        i = names_l.index(n)
        ax_l.annotate("called a shallow reference clip,\nbut never fires the cue",
                      xy=(bottoms[n], i), xytext=(bottoms[n] - 46, i - 1.9),
                      fontsize=8, color=plotstyle.ACCENT,
                      arrowprops=dict(arrowstyle="->", color=plotstyle.ACCENT,
                                      linewidth=0.9))

    for ax in (ax_l, ax_r):
        mid = np.mean(ax.get_xlim())
        ax.errorbar(mid, -1.0, xerr=err, fmt="none", ecolor=plotstyle.NEUTRAL,
                    elinewidth=2, capsize=4)
        ax.annotate(f"±{err}° measured elbow error", xy=(mid, -1.6), ha="center",
                    va="top", fontsize=9, color=plotstyle.NEUTRAL)
        ax.set_ylim(-2.6, len(names_l) - 0.2)

    fig.suptitle("Push-up elbow angle per clip, median across repetitions — "
                 f"{len(bottoms)} CC0 fixtures.  Hollow red = pose estimate failed",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=plotstyle.DPI)
    plt.close(fig)

    print(f"bottom of rep: threshold {DEPTH_FLAG_ELBOW_ANGLE:.0f}° falls between "
          f"{lo:.1f} and {hi:.1f} — a {hi - lo:.1f}° gap against ±{err}° error")
    print(f"top of rep   : {len(cluster)} clips spanning "
          f"{cluster[0]:.1f}–{cluster[-1]:.1f}, adjacent gaps "
          f"{min(cluster_gaps):.1f}–{max(cluster_gaps):.1f}°, all inside ±{err}°")
    print(f"flagged as tracking failures: {', '.join(sorted(suspect))}")
    print(f"written to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

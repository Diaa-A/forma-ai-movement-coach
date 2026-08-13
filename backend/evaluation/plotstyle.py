"""One visual language for the figures generated from stored results.

fig6, fig7 and fig8 were written months apart and it shows: three different
blues, three title sizes, two dpi settings. Side by side in a chapter that reads
as carelessness rather than as three separate measurements.

Nothing here touches a number. It sets rcParams and hands out colours, and every
figure that uses it still comes from the same results file it always did.
"""
from __future__ import annotations

DPI = 150

# Deliberately small. Two accents and a neutral is enough for a bar chart, a
# histogram and an interval band, and a wider palette would only invite figures
# that colour things for decoration.
PRIMARY = "#3b6ea5"      # the measurement
ACCENT = "#c0504d"       # the one that failed, where a figure marks one out
NEUTRAL = "#3a3a3a"      # reference lines, overall rates
BAND = "#3a3a3a"         # interval shading, used at low alpha
EDGE = "#ffffff"


def apply():
    """Set the shared style. Call before building a figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.8,
        # behind the bars, or a bar chart looks like it is drawn on graph paper
        "axes.axisbelow": True,
    })
    return plt

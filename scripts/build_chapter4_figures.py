"""Republish the Chapter 4 key-frame figures, keeping the hand-painted redaction.

    python scripts/build_chapter4_figures.py

fig3 and fig4 were rendered on 20 June, before overlay geometry started scaling
with the frame (Decision 26). The measurements are unchanged - same clip, same
frame indices, same angles - but the drawing is heavier, and mixing the two
weights across one chapter is the thing this exists to stop.

The complication is anonymity. The committed figures were blacked out by hand
after rendering and the subject is the developer, so a plain re-render publishes
a face that was deliberately hidden. The redaction is therefore lifted off the
already-redacted image and composited onto the fresh render. Same clip and same
frame index means the photograph underneath is pixel-identical, so the painted
regions land exactly where they were put.

fig5 is a different case: a new figure from a new clip, so the redaction comes
from the hand-redacted copy of an earlier render of the same frame rather than
from a committed figure.

Only large near-black blobs are taken. JPEG leaves a scatter of near-black pixels
around the caption text, and without a size floor those would speckle the new
caption.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "report" / "chapter4" / "figures"
OUTPUTS = ROOT / "data" / "outputs"

BLACK_LEVEL = 8       # jpeg is lossy, so "painted black" is not exactly zero
MIN_BLOB = 2000       # px. the smallest real redaction here is ~10,000

# figure name -> where the fresh render is, and where its redaction comes from.
# Both are runs under data/outputs, which is gitignored: rerun the two analyses
# in the note if these directories are gone.
FIGS = [
    {
        "name": "fig3_worst_frame_forward_lean.jpg",
        "render": OUTPUTS / "squat_20260813_140935_651c12" / "worst.jpg",
        "redaction": FIGURES / "fig3_worst_frame_forward_lean.jpg",
    },
    {
        "name": "fig4_best_frame.jpg",
        "render": OUTPUTS / "squat_20260813_140935_651c12" / "best.jpg",
        "redaction": FIGURES / "fig4_best_frame.jpg",
    },
    {
        # lossless rotation of the source clip, so the numbers come off the
        # original encoded frames. The redaction was painted on a re-encoded
        # render of the same frame, which reads 1-2 degrees differently.
        "name": "fig5_pushup_shallow_depth.jpg",
        "render": OUTPUTS / "pushup_20260813_150137_0fdaba" / "worst.jpg",
        "redaction": OUTPUTS / "pushup_20260813_150037_6a0015" / "worst.jpg",
    },
]


# Generated plots. No redaction involved - they are charts - but they are copied
# from here rather than by hand for the same reason the chapter 5 evidence is:
# data/outputs is gitignored, so a figure whose only source is a gitignored
# directory is a figure nobody can check.
PLOTS = [
    (OUTPUTS / "penn_eval" / "per_joint_error.png", "fig6_per_joint_error.png"),
    (OUTPUTS / "penn_eval" / "angle_error.png", "fig7_angle_error_hist.png"),
]


def redaction_mask(img):
    """The painted regions, as a mask. Blobs only, no text speckle."""
    black = (img.max(axis=2) <= BLACK_LEVEL).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(black, 8)
    mask = np.zeros(black.shape, np.uint8)
    kept = []
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB:
            mask[labels == i] = 1
            x, y, w, h, a = stats[i]
            kept.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                         "area": int(a)})
    return mask, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    record = []
    for spec in FIGS:
        for p in (spec["render"], spec["redaction"]):
            if not p.is_file():
                raise SystemExit(f"missing: {p}")

        new = cv2.imread(str(spec["render"]))
        src = cv2.imread(str(spec["redaction"]))
        if new.shape != src.shape:
            raise SystemExit(
                f"{spec['name']}: render is {new.shape[1]}x{new.shape[0]} and the "
                f"redacted source is {src.shape[1]}x{src.shape[0]}. The mask only "
                "lifts when the frames line up - check the frame index first.")

        mask, kept = redaction_mask(src)
        if not kept:
            raise SystemExit(f"{spec['name']}: found no redaction to preserve")

        out = new.copy()
        out[mask == 1] = 0
        if not args.dry_run:
            cv2.imwrite(str(FIGURES / spec["name"]), out,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        record.append({
            "figure": spec["name"],
            "rendered_from": str(spec["render"].relative_to(ROOT)).replace("\\", "/"),
            "redaction_lifted_from": str(spec["redaction"].relative_to(ROOT)).replace("\\", "/"),
            "size": [int(new.shape[1]), int(new.shape[0])],
            "regions": kept,
        })
        print(f"{'would write' if args.dry_run else 'wrote'} {spec['name']}  "
              f"({len(kept)} redacted region(s), {new.shape[1]}x{new.shape[0]})")

    for src, name in PLOTS:
        if not src.is_file():
            print(f"[!] {name}: no {src.relative_to(ROOT)} — rerun the benchmark")
            continue
        if not args.dry_run:
            (FIGURES / name).write_bytes(src.read_bytes())
        record.append({"figure": name,
                       "rendered_from": str(src.relative_to(ROOT)).replace("\\", "/")})
        print(f"{'would copy' if args.dry_run else 'copied'} {name}")

    print()
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

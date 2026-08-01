"""Generate the PWA icon set.

A script rather than four committed PNGs somebody has to take on trust. Re-run it
and the icons come back identical; change the mark in one place and every size
follows.

    PY scripts/make_icons.py

The mark is a figure at the bottom of a squat with its joints picked out — the
same thing the overlay draws, which is the honest version of an app icon: it
shows what the thing actually does.

Sizes, and why each one exists:
  192, 512  — the manifest pair Android and Chrome install from.
  512 maskable — Android may crop an icon to a circle, squircle or teardrop
      depending on the launcher. A maskable icon must survive that, so the mark
      sits inside the middle 80% (the safe zone) with the background full-bleed.
  180 apple-touch-icon — iOS ignores the manifest icons for Add to Home Screen
      and reads this instead. Square, no transparency, no rounding: iOS applies
      its own corner radius and a transparent icon comes out black.
"""
from pathlib import Path

from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).resolve().parents[1] / "frontend" / "public" / "icons"

BG = (18, 22, 28)          # --bg from styles.css, so the icon matches the app
LIMB = (77, 163, 255)      # --accent
JOINT = (232, 237, 244)    # --text
FLAG = (226, 104, 95)      # --bad, on the trunk, where the cue usually fires

# Figure in a 100x100 space, origin top-left, so it scales to any output size.
# Side-on, facing right, at the bottom of a squat: the trunk tips FORWARD (the
# shoulder ahead of the hip), the knee travels forward, the shin stays near
# vertical. Getting the direction right matters — a torso tilted the other way
# is not a squat, and this is an icon for an app that judges exactly that.
HEAD = (58, 20)
HEAD_R = 7
SHOULDER = (50, 36)
HAND = (70, 45)
HIP = (36, 56)
KNEE = (60, 64)
ANKLE = (54, 84)
TOE = (68, 87)

# the trunk is the segment the forward-lean cue is about, drawn in the fault
# colour; the neck rides with it so the head doesn't float
TRUNK = [(SHOULDER, HIP), (HEAD, SHOULDER)]
LEGS = [(HIP, KNEE), (KNEE, ANKLE), (ANKLE, TOE), (SHOULDER, HAND)]
JOINTS = [SHOULDER, HIP, KNEE, ANKLE]


def draw_icon(size, safe_fraction=1.0):
    """Render at `size` px. `safe_fraction` shrinks the mark toward the centre —
    1.0 fills the tile, 0.8 keeps it inside a maskable icon's safe zone."""
    img = Image.new("RGB", (size, size), BG)
    d = ImageDraw.Draw(img)

    # supersample: draw at 4x and downscale, since PIL has no antialiased lines
    scale = 4
    big = Image.new("RGB", (size * scale, size * scale), BG)
    bd = ImageDraw.Draw(big)

    span = size * scale * safe_fraction
    origin = (size * scale - span) / 2

    def pt(p):
        return (origin + p[0] / 100 * span, origin + p[1] / 100 * span)

    limb_w = max(2, int(span * 0.055))
    joint_r = max(2, int(span * 0.035))

    for a, b in LEGS:
        bd.line([pt(a), pt(b)], fill=LIMB, width=limb_w)
    for a, b in TRUNK:
        bd.line([pt(a), pt(b)], fill=FLAG, width=limb_w)
    # round the ends by hand — PIL's line joints are square and the elbows show
    for p in {a for a, _ in LEGS + TRUNK} | {b for _, b in LEGS + TRUNK}:
        px, py = pt(p)
        r = limb_w / 2
        bd.ellipse([px - r, py - r, px + r, py + r],
                   fill=FLAG if p in (SHOULDER, HIP, HEAD) else LIMB)

    hx, hy = pt(HEAD)
    hr = HEAD_R / 100 * span
    bd.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=LIMB)

    for j in JOINTS:
        jx, jy = pt(j)
        bd.ellipse([jx - joint_r, jy - joint_r, jx + joint_r, jy + joint_r],
                   fill=JOINT)

    img = big.resize((size, size), Image.LANCZOS)
    return img


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    for size in (192, 512):
        p = OUT_DIR / f"icon-{size}.png"
        draw_icon(size).save(p, "PNG", optimize=True)
        written.append(p)

    # maskable: same mark, pulled into the middle 80% so a circular or squircle
    # crop can't cut it
    p = OUT_DIR / "icon-512-maskable.png"
    draw_icon(512, safe_fraction=0.8).save(p, "PNG", optimize=True)
    written.append(p)

    # iOS. Opaque background is not optional here.
    p = OUT_DIR / "apple-touch-icon.png"
    draw_icon(180).save(p, "PNG", optimize=True)
    written.append(p)

    for f in written:
        print(f"{f.relative_to(Path(__file__).resolve().parents[1])}  "
              f"{f.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()

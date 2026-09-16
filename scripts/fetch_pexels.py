"""Fetch CC0 fixture clips from Pexels Video Search.

Requires PEXELS_API_KEY in the environment (or in .env). Free signup at
pexels.com/api. Used to build a diverse fixture set for the Phase I user-
testing prep and Chapter 4 (Feature Prototype) report visuals.

Usage:
    python scripts/fetch_pexels.py --query "squat" --count 5
    python scripts/fetch_pexels.py --query "push up" --count 3 --orientation portrait

Outputs go to data/test_videos/pexels/<query>/<id>.mp4 with a SOURCE.txt manifest
holding the licence + photographer credit for every file (required by Pexels TOS
when publishing).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

PEXELS_SEARCH = "https://api.pexels.com/videos/search"
# A phone uploads about this many pixels across its short edge, and analysis caps the
# long edge at 1920, so a fixture at 1080 is the same input a real user would send.
SHORT_EDGE = 1080


def _load_dotenv(path=".env"):
    if not Path(path).is_file():
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _search(api_key, query, per_page, orientation=None):
    params = {"query": query, "per_page": per_page, "size": "medium"}
    if orientation:
        params["orientation"] = orientation
    url = PEXELS_SEARCH + "?" + urllib.parse.urlencode(params)
    # Pexels 403s the default urllib UA ("Python-urllib/x.y"); a normal UA works.
    req = urllib.request.Request(url, headers={
        "Authorization": api_key,
        "User-Agent": "ai-fitness-coach/0.1",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _pick_file(video):
    """Pick the mp4 nearest what a phone uploads, measured on the short edge.

    This used to rank on height alone. A portrait clip lists its long edge as the
    height, so 640-tall files won and those clips arrived 360 px wide: twelve
    fixtures came down that way, and at full size one pull-up clip counted a rep
    fewer and one push-up clip stopped being analysable at all. A fixture smaller
    than the upload it stands in for measures the wrong thing.
    """
    files = video.get("video_files", [])
    mp4s = [f for f in files if f.get("file_type") == "video/mp4"
            and f.get("width") and f.get("height")]
    if not mp4s:
        return None
    # short edge nearest a phone's, then the smaller file of two equally close
    mp4s.sort(key=lambda f: (abs(min(f["width"], f["height"]) - SHORT_EDGE),
                             f["width"] * f["height"]))
    return mp4s[0]


def _download(url, out_path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(out_path, "wb") as fh:
        # 64KB chunks, streamed
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            fh.write(chunk)


def main():
    _load_dotenv()
    ap = argparse.ArgumentParser(description="Download CC0 fixture clips from Pexels.")
    ap.add_argument("--query",   required=True, help="search term, e.g. 'squat'")
    ap.add_argument("--count",   type=int, default=5, help="how many clips")
    ap.add_argument("--orientation", choices=["portrait", "landscape", "square"])
    ap.add_argument("--output",  default="data/test_videos/pexels",
                    help="root output directory (a per-query subdir is created)")
    args = ap.parse_args()

    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        print("error: PEXELS_API_KEY not set. Get one free at pexels.com/api and "
              "add it to .env.", file=sys.stderr)
        return 2

    query_slug = args.query.replace(" ", "_").lower()
    out_dir = Path(args.output) / query_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] searching pexels: '{args.query}'  ({args.count} clip(s))")
    results = _search(api_key, args.query,
                      per_page=max(args.count, 10),
                      orientation=args.orientation)

    videos = results.get("videos", [])
    if not videos:
        print("no results.", file=sys.stderr)
        return 1

    manifest_path = out_dir / "SOURCE.txt"
    written = 0
    with open(manifest_path, "a", encoding="utf-8") as manifest:
        for v in videos[:args.count]:
            picked = _pick_file(v)
            if not picked:
                continue
            vid = v["id"]
            fname = f"{vid}.mp4"
            target = out_dir / fname
            if target.exists():
                print(f"    already have {fname}, skipping")
                continue

            print(f"    downloading id={vid} ({picked.get('width')}x{picked.get('height')}) ...")
            try:
                _download(picked["link"], target)
            except Exception as e:
                print(f"    failed: {e}", file=sys.stderr)
                continue

            user = v.get("user", {})
            manifest.write(
                f"{fname}\t{v.get('url')}\tphotographer: {user.get('name', '?')}"
                f" ({user.get('url', '?')})\tlicense: Pexels (free to use, attribution recommended)\n"
            )
            written += 1

    print(f"[+] wrote {written} clip(s) to {out_dir}")
    print(f"    licence + credits in {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

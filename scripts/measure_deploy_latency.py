"""Measure what a deployed container actually costs, cold and warm (WP-03).

Run this against the deployed URL, not against localhost — the number that
matters is the one a phone on mobile data sees.

**Measure the right thing.** An earlier version of the project documentation said
roughly 9.5 s of a run was MediaPipe model loading, and that was wrong. Measured
on the reference clip, `PoseLandmarker.create_from_options` costs 0.23 s on the
first call and 0.07 s afterwards, which is about 2.5% of the pose stage; the rest
is per-frame inference at ~12 ms/frame, and that does not get cheaper on a warm
container. Import time is 0.50 s (measured 4 Aug 2026, `mediapipe` dominating).

So cold start here is container boot plus Python import plus first-call delegate
init — a second or so — and NOT something that can be optimised away by keeping a
landmarker warm between jobs. Reusing one across jobs would also be unsafe: VIDEO
running mode wants monotonically increasing timestamps and carries tracker state
between clips, which could change the analysis.

What this reports, and what each number means:

    boot        time from the first poll until /health answers. Run with
                --wait-for-boot immediately after triggering a deploy, or it
                measures nothing but the network.
    cold run    first /analyze on a fresh container: includes delegate init.
    warm runs   every subsequent /analyze. The difference between this and the
                cold run is the real cold-start penalty on the analysis path.

Usage:
    python scripts/measure_deploy_latency.py https://<app>.up.railway.app \
        --clip data/test_videos/squat.mp4 --runs 3

    python scripts/measure_deploy_latency.py https://<app>.up.railway.app \
        --wait-for-boot --clip data/test_videos/squat.mp4
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def _multipart(fields: dict, file_field: str, filename: str, blob: bytes):
    """Build a multipart body by hand — same approach as whisper_wrapper, and for
    the same reason: no `requests` dependency for one upload."""
    boundary = "----deploylatency" + uuid.uuid4().hex
    crlf = b"\r\n"
    parts = []
    for name, value in fields.items():
        parts.append(b"--" + boundary.encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        parts.append(b"")
        parts.append(str(value).encode())
    parts.append(b"--" + boundary.encode())
    parts.append(f'Content-Disposition: form-data; name="{file_field}"; '
                 f'filename="{filename}"'.encode())
    parts.append(b"Content-Type: application/octet-stream")
    parts.append(b"")
    body = crlf.join(parts) + crlf + blob + crlf
    body += b"--" + boundary.encode() + b"--" + crlf
    return body, boundary


def wait_for_boot(base: str, timeout: float = 300.0, interval: float = 1.0):
    """Poll /health until it answers 200. Meaningful only if started while the
    container is still coming up."""
    started = time.time()
    attempts = 0
    while time.time() - started < timeout:
        attempts += 1
        try:
            with urllib.request.urlopen(base + "/health", timeout=10) as resp:
                if resp.status == 200:
                    return time.time() - started, attempts
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass
        time.sleep(interval)
    raise SystemExit(f"/health did not answer within {timeout:.0f}s")


def analyze_once(base: str, clip: Path, exercise: str, timeout: float):
    """One full /analyze. Coaching is forced to dry-run so the LLM's own latency
    (and its non-determinism) stays out of the number."""
    blob = clip.read_bytes()
    body, boundary = _multipart(
        {"exercise_type": exercise, "dry_run_coach": "true"},
        "video", clip.name, blob,
    )
    req = urllib.request.Request(
        base + "/analyze", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return time.time() - started, payload


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base_url", help="e.g. https://myapp.up.railway.app")
    p.add_argument("--clip", default="data/test_videos/squat.mp4")
    p.add_argument("--exercise", default="squat")
    p.add_argument("--runs", type=int, default=3,
                   help="warm runs after the cold one (default 3)")
    p.add_argument("--wait-for-boot", action="store_true",
                   help="poll /health first — only meaningful straight after a deploy")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--out", default="", help="write the results as JSON here")
    args = p.parse_args()

    base = args.base_url.rstrip("/")
    clip = Path(args.clip)
    if not clip.exists():
        raise SystemExit(f"clip not found: {clip}")

    size_mb = clip.stat().st_size / (1024 * 1024)
    print(f"target : {base}")
    print(f"clip   : {clip.name}  ({size_mb:.1f} MB)")
    print()

    results = {"base_url": base, "clip": clip.name, "clip_mb": round(size_mb, 2)}

    if args.wait_for_boot:
        boot, attempts = wait_for_boot(base, timeout=args.timeout)
        results["boot_seconds"] = round(boot, 2)
        print(f"boot        : {boot:6.2f} s to first healthy /health "
              f"({attempts} polls)")

    cold, payload = analyze_once(base, clip, args.exercise, args.timeout)
    results["cold_seconds"] = round(cold, 2)
    results["status"] = payload.get("status")
    results["reps"] = len(payload.get("reps", []))
    print(f"cold run    : {cold:6.2f} s   "
          f"(status {payload.get('status')}, {len(payload.get('reps', []))} reps)")

    warm = []
    for i in range(args.runs):
        took, _ = analyze_once(base, clip, args.exercise, args.timeout)
        warm.append(took)
        print(f"warm run {i+1}  : {took:6.2f} s")

    if warm:
        results["warm_seconds"] = [round(w, 2) for w in warm]
        results["warm_median"] = round(statistics.median(warm), 2)
        print()
        print(f"warm median : {statistics.median(warm):6.2f} s")
        print(f"cold penalty: {cold - statistics.median(warm):6.2f} s")
        print()
        print("Upload time is inside every number above, so on a home connection "
              "this is\nprocessing-dominated and on mobile data it is not. Say "
              "which one you measured.")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

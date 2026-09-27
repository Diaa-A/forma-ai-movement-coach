"""CLI squat analyser (Phases A + B + C + D).

Thin wrapper over `backend.pipeline.runner.run_pipeline`. The same
function is reused by the FastAPI server in `backend.main`, so the CLI and
the API can't drift.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from backend.pipeline.runner import run_pipeline, RunOptions
from backend.exercises.registry import exercise_ids
from backend.pipeline.coaching import (
    generate_coaching_report, format_report,
    CoachingReport,
)


def _load_dotenv(path=".env"):
    """Tiny .env loader. Don't pull in python-dotenv just for this."""
    if not Path(path).is_file():
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _parse_args():
    p = argparse.ArgumentParser(description="Analyse an exercise clip.")
    p.add_argument("--input",  required=True, help="path to the input video")
    # choices come from the registry, so a newly registered exercise is usable
    # from the CLI without touching this file
    p.add_argument("--exercise", default="squat", choices=exercise_ids(),
                   help="which exercise the clip shows (default: squat)")
    p.add_argument("--output", default="data/outputs",
                   help="directory under which a job folder is created")
    p.add_argument("--model", default="full", choices=["lite", "full", "heavy"],
                   help="MediaPipe pose model size (full is the default)")
    p.add_argument("--min-cutoff", type=float, default=1.0)
    p.add_argument("--beta",       type=float, default=0.007)
    p.add_argument("--no-coach", action="store_true",
                   help="skip the coaching step entirely (analysis only)")
    p.add_argument("--dry-run-coach", action="store_true",
                   help="force the deterministic dry-run report (no LLM call)")
    p.add_argument("--voice-note", default="",
                   help="path to a short audio file; transcribed by Whisper")
    p.add_argument("--voice-note-text", default="",
                   help="plain-text user context (used directly, no transcription)")
    p.add_argument("--whisper-model", default=None,
                   help="override the Whisper model size (default 'small')")
    p.add_argument("--llm-model", default=None,
                   help="Groq model id")
    return p.parse_args()


def main():
    _load_dotenv()
    args = _parse_args()

    opts = RunOptions(
        pose_model=args.model,
        min_cutoff=args.min_cutoff,
        beta=args.beta,
        coach=(not args.no_coach),
        force_dry_run_coach=args.dry_run_coach,
        voice_transcript=args.voice_note_text,
        voice_audio_path=args.voice_note,
        whisper_model=args.whisper_model,
        llm_model=args.llm_model,
    )

    print(f"[+] analysing: {Path(args.input).name}  (pose model: {args.model})")
    try:
        result = run_pipeline(args.input, args.output, args.exercise, options=opts)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    s = result.summary
    print("\n=== summary ===")
    print(f"   status: {s.get('status', '?')}    detection rate: "
          f"{s.get('detection_rate', 0):.0%}")
    print(f"   side selected: {s['side']}    reps: {s['rep_count']}")
    if s["knee_min"] is not None:
        print(f"   knee range:    min={s['knee_min']:.1f}°  max={s['knee_max']:.1f}°  "
              f"mean={s['knee_mean']:.1f}°")
    for i, r in enumerate(s["reps"], start=1):
        bd = r.get("breakdown") or {}
        sc = r.get("score")
        sc_str = f"{sc:.1f}" if sc is not None else "n/a"
        print(f"   rep {i}: bottom@{r['bottom']}  eval@{r.get('eval_frame')}  "
              f"knee min={r['knee_min']:.1f}° max={r['knee_max']:.1f}°  "
              f"score={sc_str}  "
              f"(depth_pen={bd.get('depth', '?')}, lean_pen={bd.get('lean', '?')})")

    for w in s.get("warnings", []):
        print(f"   [!] {w}")

    # if we ran coaching, re-render the report from the JSON for the terminal
    if result.coaching_path:
        with open(result.coaching_path) as fh:
            payload = json.load(fh)
        report = CoachingReport(**payload["report"])
        print()
        print(format_report(report))

    print(f"\noutputs: {result.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

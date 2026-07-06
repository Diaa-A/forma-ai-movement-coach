# AI Fitness & Movement Coach

CM3070 final project — server-side Python pipeline that analyses a user-uploaded
exercise video and returns coaching feedback. Spec is in `BUILD_REFERENCE.md` (synced
from `CM3070_Technical_Spec.md`).

## Status

| Phase | Status |
|---|---|
| A — pipeline foundation (MediaPipe → One Euro → angles → phase detection) | done |
| B — squat analyser (form scoring, worst/best frame selection, false-rep filter) | done |
| C — two-layer coaching (Layer 1 deterministic cue evaluator + Layer 2 Groq LLM, with dry-run fallback) | done; live (Groq key in `.env`) |
| D — voice transcription (Groq `whisper-large-v3` default, local `openai-whisper` fallback) | done; live |
| E — FastAPI server with `/analyze` multipart endpoint | done |
| F — PWA frontend | not started |
| G — push-up / pull-up analysers | not started |
| H — Penn Action evaluation harness | done (squat subset — see `data\outputs\penn_eval\`) |

## Quick start

### CLI

```
.venv\Scripts\python.exe analyze_squat.py --input data\test_videos\squat.mp4
```

Useful flags:
- `--model {lite,full,heavy}` — MediaPipe model size (default full)
- `--no-coach` — skip Layer 1 + 2 (analysis only)
- `--dry-run-coach` — force deterministic report (no LLM call)
- `--voice-note <audio>` — transcribe via Whisper, pass to LLM as context
- `--voice-note-text "..."` — same but plain text (no Whisper needed)

Outputs go to `data\outputs\<job_id>\`:
- `annotated.mp4` — clip with skeleton overlay, colour-coded
- `worst.jpg`, `best.jpg` — annotated key frames
- `angles.json` — per-frame joint angles, phase labels, rep stats, world landmarks
- `coaching.json` — Layer 1 evaluation + Layer 2 report

### API

```
.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
```

Then:
```
curl -F "video=@data\test_videos\squat.mp4" ^
     -F "exercise_type=squat" ^
     -F "dry_run_coach=true" ^
     http://127.0.0.1:8000/analyze
```

Returns JSON with URLs to artefacts served from `/results/<job_id>/`.

### Tests

```
.venv\Scripts\python.exe -m pytest
```

51 tests: angle maths vs known geometry, One Euro behaviour, phase detection on
synthetic signals, rep filter, cue gating, and an end-to-end smoke test (the
smoke test needs `data\test_videos\squat.mp4` and skips itself on a clean
checkout, where test videos are gitignored).

### Pexels fixture fetcher

```
.venv\Scripts\python.exe scripts\fetch_pexels.py --query "squat" --count 5
```

Requires `PEXELS_API_KEY` in `.env`. Pulls CC0 clips into
`data\test_videos\pexels\<query>\` with a `SOURCE.txt` manifest holding the
licence + photographer credits.

## Setup

```
C:\Python313\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install mediapipe opencv-python "numpy<2.3" fastapi "uvicorn[standard]" python-multipart
```

Spec says Python 3.11 but the 3.11.0 install on this machine has a broken
os.mkdir hook — falling back to 3.13 was the pragmatic call. MediaPipe 0.10.35
ships 3.13 wheels.

Copy `.env.example` to `.env` and fill in keys when ready:
- `GROQ_API_KEY` — for Phase C live LLM (dry-run works without it)
- `PEXELS_API_KEY` — for fixture fetcher

### Optional: voice transcription

```
.venv\Scripts\python.exe -m pip install openai-whisper
```

This pulls in torch (~2 GB on CPU). Only needed when using `--voice-note <audio>`.

## File layout

```
Final-proj\
├── BUILD_REFERENCE.md         reference spec (read first)
├── README.md                  this file
├── analyze_squat.py           CLI entry
├── .env.example
├── requirements.txt
├── backend\
│   ├── main.py                FastAPI app
│   ├── api\
│   │   ├── routes.py          POST /analyze
│   │   └── schemas.py         response models
│   ├── pipeline\
│   │   ├── runner.py          shared orchestrator (CLI + API both use this)
│   │   ├── pose.py            MediaPipe Pose wrapper
│   │   ├── filter.py          One Euro filter (Casiez 2012)
│   │   ├── angles.py          joint angle calcs
│   │   ├── phase_detection.py velocity zero-crossing + phase labels
│   │   ├── render.py          skeleton overlay
│   │   ├── coaching.py        Layer 2 — Groq LLM wrapper + dry-run
│   │   └── whisper_wrapper.py Phase D voice transcription (lazy)
│   └── exercises\
│       ├── squat.py           form scoring + side selection + worst/best
│       └── squat_cues.py      Layer 1 — cue database + evaluator
├── scripts\
│   └── fetch_pexels.py        CC0 fixture downloader
└── data\
    ├── models\                MediaPipe .task files (lite + full)
    ├── test_videos\           input clips (gitignored)
    ├── uploads\               API multipart uploads (gitignored)
    └── outputs\               per-job artefacts (gitignored)
```

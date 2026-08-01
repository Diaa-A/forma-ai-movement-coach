# AI Fitness & Movement Coach

CM3070 final project — server-side Python pipeline that analyses a user-uploaded
exercise video and returns coaching feedback. Live build reference is `BUILD_REFERENCE.md`;
the original spec it was synced from is `docs/CM3070_Technical_Spec.md`.

**Start here:** `BUILD_REFERENCE.md` for what the system is and what is built · the Setup
section below to get it running · `docs/CM3070_PROJECT_STATE.md` for delivery
state and the code review · `report/CM3070_Decision_Log.md` for why anything is
the way it is.

## Status

| Phase | Status |
|---|---|
| A — pipeline foundation (MediaPipe → One Euro → angles → phase detection) | done |
| B — squat analyser (form scoring, worst/best frame selection, false-rep filter) | done |
| C — two-layer coaching (Layer 1 deterministic cue evaluator + Layer 2 Groq LLM, with dry-run fallback) | done; live (Groq key in `.env`) |
| D — voice transcription (Groq `whisper-large-v3` default, local `openai-whisper` fallback) | done; live |
| E — FastAPI server with `/analyze` multipart endpoint | done |
| F — PWA frontend | not started — critical path |
| G — push-up / pull-up analysers | not started |
| H — Penn Action evaluation harness | done (squat subset — see `data\outputs\penn_eval\`) |

## Quick start

Run Setup first. `PY` below is your venv interpreter — see
[Which interpreter to type](#which-interpreter-to-type).

### CLI

```
PY analyze_squat.py --input data/test_videos/squat.mp4
```

Useful flags:
- `--model {lite,full,heavy}` — MediaPipe model size (default full)
- `--no-coach` — skip Layer 1 + 2 (analysis only)
- `--dry-run-coach` — force deterministic report (no LLM call)
- `--voice-note <audio>` — transcribe via Whisper, pass to LLM as context
- `--voice-note-text "..."` — same but plain text (no Whisper needed)

Outputs go to `data/outputs/<job_id>/`:
- `annotated.mp4` — clip with skeleton overlay, colour-coded
- `worst.jpg`, `best.jpg` — annotated key frames
- `angles.json` — per-frame joint angles, phase labels, rep stats, world landmarks
- `coaching.json` — Layer 1 evaluation + Layer 2 report

### The app

Build the frontend once (needs Node 18+), then start the server — it serves the
PWA and the API from the same origin, so there is only one thing to run:

```
cd frontend && npm install && npm run build && cd ..
```

```
PY -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000. Binding to `0.0.0.0` also lets a phone on the same
network reach it at `http://<your-ip>:8000` — everything works there except
recording a voice note, which needs `getUserMedia` and therefore HTTPS. Typing
the note works, and so does everything else, because video capture uses a file
input rather than a media stream. For the full journey over HTTPS without
deploying, tunnel it: `cloudflared tunnel --url http://localhost:8000`.

For frontend work, `npm run dev` in `frontend/` gives hot reload on
http://localhost:5173 and proxies API calls to port 8000. Frontend tests:
`npm test`.

### API on its own

```
PY -m uvicorn backend.main:app --reload --port 8000
```

Without a frontend build present, `/` returns the endpoint list instead of the app.

Then, on one line so it works in cmd, PowerShell and a POSIX shell alike:

```
curl -F "video=@data/test_videos/squat.mp4" -F "exercise_type=squat" -F "dry_run_coach=true" http://127.0.0.1:8000/analyze
```

Returns JSON with URLs to artefacts served from `/results/<job_id>/`.

### Tests

```
PY -m pytest
```

51 tests: angle maths vs known geometry, One Euro behaviour, phase detection on
synthetic signals, rep filter, cue gating, and an end-to-end smoke test (the
smoke test needs `data/test_videos/squat.mp4` and skips itself on a clean
checkout, where test videos are gitignored — a skip there is expected, not a
failure).

### Pexels fixture fetcher

```
PY scripts/fetch_pexels.py --query "squat" --count 5
```

Requires `PEXELS_API_KEY` in `.env`. Pulls CC0 clips into
`data/test_videos/pexels/<query>/` with a `SOURCE.txt` manifest holding the
licence + photographer credits.

## Setup

Requires **Python 3.13** and about 1 GB of disk for the virtual environment.

**Windows** (PowerShell or cmd):

```
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

**macOS / Linux**:

```
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Then open `.env` and fill in the keys:
- `GROQ_API_KEY` — Phase C live LLM and Phase D transcription. The system runs
  without it: pass `--dry-run-coach` and you get the deterministic report instead.
- `PEXELS_API_KEY` — only needed for the fixture fetcher.

### Why Python 3.13 and not the 3.11 in the spec

Deliberate, not a workaround. MediaPipe 0.10.35 publishes CPython 3.13 wheels, so
every pinned dependency in `requirements.txt` installs as a prebuilt wheel on
Windows, macOS and Linux with no compiler step — which matters a lot more for
"can someone else run this" than the minor version does. The whole pinned stack
was built and validated on 3.13. (What raised the question was a broken 3.11.0
install on the dev machine; what settled it was the wheel coverage.)

### Which interpreter to type

The rest of this README writes the interpreter as **`PY`**:

| Platform | `PY` is |
|---|---|
| Windows | `.venv\Scripts\python.exe` |
| macOS / Linux | `.venv/bin/python` |

Forward slashes in the *arguments* work everywhere — Python normalises them — so
only the interpreter path differs.

### Optional: local voice transcription fallback

```
PY -m pip install openai-whisper
```

Pulls in torch (~2 GB on CPU). Not needed in normal use: transcription defaults to
Groq-hosted `whisper-large-v3` (Decision 22). This is the offline fallback only.

## File layout

```
.
├── BUILD_REFERENCE.md         build reference (read first)
├── README.md                  this file
├── analyze_squat.py           CLI entry
├── .env.example
├── requirements.txt
├── backend/
│   ├── main.py                FastAPI app
│   ├── api/
│   │   ├── routes.py          POST /analyze
│   │   └── schemas.py         response models
│   ├── pipeline/
│   │   ├── runner.py          shared orchestrator (CLI + API both use this)
│   │   ├── pose.py            MediaPipe Pose wrapper
│   │   ├── filter.py          One Euro filter (Casiez 2012)
│   │   ├── angles.py          joint angle calcs
│   │   ├── phase_detection.py velocity zero-crossing + phase labels
│   │   ├── render.py          skeleton overlay
│   │   ├── coaching.py        Layer 2 — Groq LLM wrapper + dry-run
│   │   └── whisper_wrapper.py Phase D voice transcription (lazy)
│   ├── evaluation/            Penn Action loader + MPJPE / PCK metrics
│   ├── exercises/
│   │   ├── base.py            ExerciseProfile — camera-view / plane gating
│   │   ├── squat.py           form scoring + side selection + worst/best
│   │   └── squat_cues.py      Layer 1 — cue database + evaluator
│   └── tests/                 51 pytest tests
├── frontend/                  the PWA (React + Vite); built output is served by FastAPI
│   ├── public/                manifest, service worker, icons
│   └── src/
│       ├── api.ts             the only module that talks to the backend
│       ├── screens/           select → guide → capture → processing → results
│       └── components/
├── docs/                      spec, tracker, delivery state, work packages
├── report/                    decision log 1–23, handoff note, Ch4 + figures
├── scripts/
│   ├── eval_penn_action.py    Phase H benchmark harness
│   └── fetch_pexels.py        CC0 fixture downloader
└── data/
    ├── models/                MediaPipe .task files (lite + full)
    ├── test_videos/           input clips (gitignored)
    ├── uploads/               API multipart uploads (gitignored)
    └── outputs/               per-job artefacts (gitignored)
```

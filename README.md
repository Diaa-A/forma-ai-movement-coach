# Forma — AI Fitness & Movement Coach

CM3070 final project — server-side Python pipeline that analyses a user-uploaded
exercise video and returns coaching feedback. The PWA ships as **Forma**.

**Start here:** the Status table below for what is built, then Quick start to get
it running. Architecture is in section 2 of this file and the reasoning behind
each design choice is in the report rather than the repository.

## Status

| Phase | Status |
|---|---|
| A — pipeline foundation (MediaPipe → One Euro → angles → phase detection) | done |
| B — squat analyser (form scoring, worst/best frame selection, false-rep filter) | done |
| C — two-layer coaching (Layer 1 deterministic cue evaluator + Layer 2 Groq LLM, with dry-run fallback) | done; live (Groq key in `.env`) |
| D — voice transcription (Groq `whisper-large-v3` default, local `openai-whisper` fallback) | done; live |
| E — FastAPI server with `/analyze` multipart endpoint | done |
| F — PWA frontend | done; deployed on Railway and installable on iOS |
| G — push-up / pull-up analysers | done and benchmarked; the pull-up has one live cue and declares the rest as not assessed yet (WP-08) |
| H — Penn Action evaluation harness | done, squat, push-up and pull-up |
| I — user testing round 1 | done; six unique form responses on the demo build |

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

284 tests: angle maths vs known geometry, One Euro behaviour, phase detection on
synthetic signals, rep filter, cue gating, the API's rejection branches, and an
end-to-end smoke test (the tests that need `data/test_videos/` skip themselves on
a clean checkout, where test videos are gitignored — a skip there is expected,
not a failure).

### Pexels fixture fetcher

```
PY scripts/fetch_pexels.py --query "squat" --count 5
```

Requires `PEXELS_API_KEY` in `.env`. Pulls CC0 clips into
`data/test_videos/pexels/<query>/` with a `SOURCE.txt` manifest holding the
licence + photographer credits.

## Deployment (Railway)

The `Dockerfile` builds the PWA and the API into one image, because the backend
serves the frontend so they share an origin. Railway detects it automatically;
`railway.json` sets the healthcheck to `/health`.

**Why Railway.** The binding constraint is request duration, not price: a 20 s
clip takes ~12 s to analyse and a 4K one takes longer, so any host with a 30 s
request cap is unusable. Railway allows 5 minutes on public networking. Measured
usage is ~0.1 GB idle and 238–774 MB peak per analysis depending on input
resolution. On the original 954 MB container that came to roughly $1/month at
Railway's published per-second rates, inside the $5 Hobby credit. The container
was resized during Round 1 after repeated analyses ran it out of memory, so
treat that figure as the old one and read the current rate off the dashboard
before quoting it anywhere.

### First deploy

1. Create a Railway account, then **New Project → Deploy from GitHub repo** and
   pick this repository. It is private, so you will be asked to grant access —
   keep the repository private (it is assessed coursework).
2. Railway reads `railway.json` and `Dockerfile`; no build settings to fill in.
3. Under **Variables**, add `GROQ_API_KEY` and `GROQ_MODEL`. The model is not
   optional any more — Groq retired `llama-3.3-70b-versatile` in Aug 2026, and
   without an override coaching quietly falls back to cue wording while
   `/health` still reports green. `openai/gpt-oss-120b` is what production
   runs. Add `FEEDBACK_FORM_URL` only while a testing round is open. CORS
   stays empty on purpose (the app and the API share an origin, so nothing is
   ever cross-origin), and `DATA_ROOT` defaults to `data/` inside the image.
   Railway needs a redeploy to pick up a variable change.
4. Under **Settings → Networking**, generate a domain. HTTPS is issued
   automatically, which the PWA needs for install and for `MediaRecorder`.
5. Set a **usage limit** in account settings. Hobby is billed by usage with no
   hard cap, and a runaway loop should stop rather than bill.

`.env` is gitignored and is never copied into the image — `.dockerignore`
excludes it as well, so neither path can carry a key.

### Measuring cold and warm latency

WP-03 asks for this and it becomes a figure in the Implementation chapter. Run it
immediately after triggering a deploy so the boot number means something:

```
PY scripts/measure_deploy_latency.py https://<your-app>.up.railway.app \
    --wait-for-boot --runs 3 --out data/outputs/deploy_latency.json
```

Note what cold start actually consists of here: container boot, Python import
(~0.5 s, mostly MediaPipe), and first-call delegate init (0.23 s). It is **not**
model loading — an earlier version of the project notes claimed ~9.5 s went there
and that was wrong, and it would point any optimisation at the wrong thing.
Per-frame inference dominates and does not get cheaper on a warm container.

### Two things to know before a testing session

- **The filesystem is ephemeral.** A redeploy wipes `data/outputs`, so any
  `/results/...` URL a participant still has open stops working. Do not redeploy
  mid-session. This is not the deletion guarantee either — that is WP-07, and it
  has to ship before anyone is asked to consent to it.
- **One analysis runs at a time** (`MAX_CONCURRENT_ANALYSES`, default 1). Past
  the limit requests get a 503, which the app renders as a retryable server
  error. The default was chosen when the container had 954 MB against a
  238–774 MB peak per analysis, so two large ones could not fit. `/health` now
  reports a 7,629 MB limit, so memory no longer forces the value down to 1 — it
  stays there because nothing has yet measured what raising it does to latency
  under load.

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

Then open `.env` and fill in the keys — `.env.example` documents every variable
and why it exists, including the ones not listed here:
- `GROQ_API_KEY` — Phase C live LLM and Phase D transcription. The system runs
  without it: pass `--dry-run-coach` and you get the deterministic report instead.
- `GROQ_MODEL` — set this. The measured default was retired upstream; production
  runs `openai/gpt-oss-120b`. Rerun `scripts/run_faithfulness.py` before
  trusting a model the report has not measured.
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
Groq-hosted `whisper-large-v3`. This is the offline fallback only.

## File layout

```
.
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
│   │   ├── encoder.py         browser-playable MP4 muxing
│   │   ├── probe.py           reads rotation / duration before decoding
│   │   ├── coaching.py        Layer 2 — Groq LLM wrapper + dry-run
│   │   └── whisper_wrapper.py Phase D voice transcription (lazy)
│   ├── evaluation/            Penn Action loader, MPJPE / PCK, faithfulness checker
│   ├── exercises/
│   │   ├── base.py            ExerciseProfile — camera-view / plane gating
│   │   ├── mechanics.py       shared Movement abstraction (no exercise names here)
│   │   ├── registry.py        exercise lookup by name
│   │   ├── squat.py           form scoring + side selection + worst/best
│   │   ├── squat_cues.py      Layer 1 — squat cue database + evaluator
│   │   ├── pushup.py          push-up scoring on the shared Movement
│   │   ├── pushup_cues.py     Layer 1 — push-up cue database + evaluator
│   │   ├── pullup.py          pull-up scoring, both arms, effort at the top
│   │   └── pullup_cues.py     Layer 1 — pull-up cues, four of five parked
│   └── tests/                 284 pytest tests
├── frontend/                  the PWA (React + Vite); built output is served by FastAPI
│   ├── public/                manifest, service worker, icons
│   └── src/
│       ├── api.ts             the only module that talks to the backend
│       ├── consent.ts         per-device consent, keyed to the retention period
│       ├── motion.ts          view transitions + stagger, no animation library
│       ├── screens/           intro → select → consent → guide → capture →
│       │                      processing → results
│       └── components/        masthead, stepper, status banner, report view
├── scripts/                   18 harnesses and figure builders; the ones used most:
│   ├── eval_penn_action.py    Phase H benchmark harness
│   ├── run_faithfulness.py    Layer-2 faithfulness measurement (spends Groq quota)
│   ├── rescore_faithfulness.py  re-score stored generations offline, no quota
│   └── fetch_pexels.py        CC0 fixture downloader
└── data/
    ├── models/                MediaPipe .task files (lite + full)
    ├── test_videos/           input clips (gitignored)
    ├── uploads/               API multipart uploads (gitignored)
    └── outputs/               per-job artefacts (gitignored)
```

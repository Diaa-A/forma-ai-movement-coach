# Forma — AI Fitness & Movement Coach

Film one set of a squat, push-up or pull-up on your phone, upload it, and get back
what your form looked like: the clip with a skeleton drawn on it, the best and
worst frames picked out, and written feedback on what to change.

This is my CM3070 final project. The code here is the whole thing — the analysis
pipeline, the web app, and the harnesses that measure how well it works.

## How it works

An uploaded clip goes through the same steps whichever exercise it is:

1. MediaPipe Pose finds 33 body landmarks in every frame.
2. A One Euro filter smooths them, because raw landmarks jitter enough to ruin an
   angle.
3. Joint angles are worked out from the landmarks in 2D.
4. Repetitions are found by looking for where the body changes direction.
5. A set of fixed rules decides what was wrong with the set.
6. A language model rewrites that decision in plainer words.

Step 5 and step 6 are deliberately separate, and that split is the point of the
project. The rules decide; the model only rephrases. It never sees the video or
the frame-by-frame numbers, so it cannot invent a fault the rules did not find.
There is a checker in `backend/evaluation/` that measures whether the model stayed
inside what the rules gave it.

The app also says what it did *not* check. If you film a squat from the side it
cannot see whether your knees cave inward, so it says so rather than staying
quiet and letting you assume it looked.

## What it checks

| Exercise | Film from | Checks | Does not check yet |
|---|---|---|---|
| Squat | the side | depth, forward lean, hip drive out of the bottom, left/right evenness, rep consistency, descent speed | knee tracking, whether the heels stay down |
| Push-up | the side | depth, sagging hips, piked hips, left/right evenness, rep consistency, descent speed | elbow flare, head and neck position |
| Pull-up | the front | rep counting, and whether every rep reaches the same height | whether the arms straighten at the bottom, chin height, swing, arm evenness |

The pull-up is thinner than the other two on purpose. It has to be filmed from the
front, because the bar is overhead, and that view cannot see the things that make a
pull-up correct. Rather than guess, the other four checks are switched off and
declared.

Everything is built, deployed, and tested on a phone. One round of user testing has
run, and a second on the finished build.

## Quick start

Do the [Setup](#setup) first. `PY` below means your virtual environment's
interpreter — there is a table under [Which interpreter to type](#which-interpreter-to-type).

### The app

Build the frontend once, then start the server. The server hands out the web app
and answers the API on the same port, so there is only one thing to run.

```
cd frontend && npm install && npm run build && cd ..
```

```
PY -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000.

Binding to `0.0.0.0` also lets a phone on the same wifi reach it at
`http://<your-ip>:8000`. Everything works there except recording a voice note,
which browsers only allow over HTTPS. Typing the note instead works fine, and so
does picking a video, because that uses a normal file input. If you want the whole
thing over HTTPS without deploying it, tunnel it:

```
cloudflared tunnel --url http://localhost:8000
```

For frontend work, `npm run dev` inside `frontend/` gives hot reload on
http://localhost:5173 and forwards API calls to port 8000.

### Command line

One clip, no web app:

```
PY analyze_squat.py --input data/test_videos/squat.mp4
```

Flags worth knowing:

- `--exercise {squat,pushup,pullup}` — which analyser to run, squat by default
- `--model {lite,full,heavy}` — MediaPipe model size, `full` by default
- `--no-coach` — analyse only, write no feedback
- `--dry-run-coach` — write the feedback from the rules alone, no model call
- `--voice-note <audio>` — transcribe a spoken question and answer it
- `--voice-note-text "..."` — the same, typed, so no transcription is needed

Results land in `data/outputs/<job_id>/`:

- `annotated.mp4` — the clip with the skeleton drawn on, red where something is wrong
- `worst.jpg`, `best.jpg` — the two frames worth looking at
- `angles.json` — per-frame angles, rep boundaries, which frame each rep was judged at
- `coaching.json` — what the rules decided, and the written feedback

### The API on its own

```
PY -m uvicorn backend.main:app --reload --port 8000
```

With no frontend build present, `/` lists the endpoints instead of serving the app.
Those are `POST /analyze`, `GET /exercises`, `DELETE /jobs/{id}`, `GET /health`,
and the finished files under `GET /results/<job_id>/<file>`.

Sending a clip, on one line so it works in cmd, PowerShell and a POSIX shell alike:

```
curl -F "video=@data/test_videos/squat.mp4" -F "exercise_type=squat" -F "dry_run_coach=true" http://127.0.0.1:8000/analyze
```

### Tests

```
PY -m pytest
```

294 tests: angle maths against geometry worked out by hand, the filter, rep
detection on made-up signals, every cue firing and staying quiet, every way the API
can reject an upload, and a few that run a real clip end to end. Nine of them skip on a
fresh clone, five because the clips they need are gitignored and four because they
want the ffmpeg binary. A skip there is expected and not a failure.

The frontend has 86 of its own. `npm test` inside `frontend/`.

## Setup

Needs Python 3.13 and about 1 GB of disk for the virtual environment. Node 18 or
newer for the frontend build.

Windows, in PowerShell or cmd:

```
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

macOS and Linux:

```
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Then open `.env` and fill it in. Every variable is documented in `.env.example`;
these are the ones that matter:

- `GROQ_API_KEY` — the language model and the voice transcription. The app runs
  without it: pass `--dry-run-coach`, or let it fall back on its own, and you get
  the feedback the rules wrote.
- `GROQ_MODEL` — set this one. Groq retired the model the project was measured
  against in August 2026, and without an override the feedback quietly falls back
  to the rules' own wording. Production runs `openai/gpt-oss-120b`.
- `PEXELS_API_KEY` — only for the fixture downloader.

### Why Python 3.13 and not the 3.11 in the spec

A deliberate choice, not a workaround. MediaPipe 0.10.35 publishes wheels for 3.13,
so everything in `requirements.txt` installs prebuilt on Windows, macOS and Linux
with no compiler involved. That matters more for whether someone else can run this
than the minor version does. The pinned set was built and tested on 3.13
throughout. A broken 3.11 install is what raised the question; the wheels are what
settled it.

### Which interpreter to type

The rest of this file writes the interpreter as `PY`:

| Platform | `PY` is |
|---|---|
| Windows | `.venv\Scripts\python.exe` |
| macOS / Linux | `.venv/bin/python` |

Forward slashes in the arguments are fine everywhere, since Python sorts them out.
Only the interpreter path differs.

### Optional: transcription without the network

```
PY -m pip install openai-whisper
```

This pulls in torch, about 2 GB. You do not need it in normal use — transcription
goes to Groq by default. It is the offline fallback.

## Deployment

The `Dockerfile` builds the web app and the API into one image, since the backend
serves the frontend and they share an origin. `railway.json` points the healthcheck
at `/health`.

It runs on Railway because the deciding constraint is how long a request may take,
not price. A 20-second clip takes about 12 seconds to analyse and a 4K one takes
longer, so any host that cuts requests off at 30 seconds is no use. Railway allows
five minutes. Memory runs about 0.1 GB idle and peaks between 238 MB and 774 MB per
analysis depending on the resolution coming in.

Setting it up the first time:

1. New project, deploy from this GitHub repo.
2. Railway reads `railway.json` and the `Dockerfile`. There are no build settings
   to fill in.
3. Under Variables, add `GROQ_API_KEY` and `GROQ_MODEL`. Add `FEEDBACK_FORM_URL`
   only while a round of user testing is open. Leave CORS empty — the app and the
   API share an origin, so nothing is ever cross-origin. A variable change needs a
   redeploy to take effect.
4. Under Settings, Networking, generate a domain. HTTPS comes with it, which the
   app needs to install to a home screen and to record a voice note.
5. Set a usage limit in your account settings, so a runaway loop stops instead of
   billing.

`.env` is gitignored and never goes into the image — `.dockerignore` excludes it
too, so neither path can carry a key.

### Two things to know before a testing session

The filesystem does not survive a redeploy. Anything under `data/outputs` is gone,
so a `/results/...` link someone still has open stops working. Don't redeploy while
people are using it.

Separately, uploads and results are deleted on a timer: `RETENTION_HOURS`, three
days by default. The app shows that period on screen before anything is uploaded,
and reads it from the same place the deleting code does, so the two cannot drift
apart. There is also a delete button that removes one analysis straight away.

Three analyses run at once, set by `MAX_CONCURRENT_ANALYSES`. A fourth at the same
moment gets a 503, which the app shows as a server error worth retrying. The limit
used to be one, back when the container had 954 MB against a peak of up to 774 MB
per analysis. The container is larger now, so three fit.

## Where things are

```
.
├── README.md                  this file
├── analyze_squat.py           command-line entry point
├── .env.example               every environment variable, with why it exists
├── requirements.txt
├── backend/
│   ├── main.py                the FastAPI app
│   ├── api/
│   │   ├── routes.py          /analyze, /exercises, /jobs/{id}
│   │   ├── retention.py       deletes uploads and results on a timer
│   │   └── schemas.py         response shapes
│   ├── pipeline/
│   │   ├── runner.py          runs a clip end to end; the CLI and API share it
│   │   ├── pose.py            MediaPipe wrapper
│   │   ├── filter.py          One Euro filter (Casiez 2012)
│   │   ├── angles.py          joint angles
│   │   ├── phase_detection.py finds reps and labels the parts of one
│   │   ├── render.py          draws the skeleton
│   │   ├── encoder.py         muxes an MP4 a browser will actually play
│   │   ├── probe.py           reads rotation and duration before decoding
│   │   ├── coaching.py        the language model wrapper, and the fallback
│   │   └── whisper_wrapper.py voice transcription, loaded only when used
│   ├── evaluation/            the pose benchmark, and the faithfulness checker
│   ├── exercises/
│   │   ├── mechanics.py       the machinery all three share
│   │   ├── base.py            what each exercise's camera view can and cannot see
│   │   ├── registry.py        look an exercise up by name
│   │   ├── squat.py           scoring, side selection, key frames, overlay colours
│   │   ├── squat_cues.py      the squat's rules
│   │   ├── pushup.py          the same for the push-up
│   │   ├── pushup_cues.py     the push-up's rules
│   │   ├── pullup.py          the same for the pull-up, scored on both arms
│   │   └── pullup_cues.py     the pull-up's rules, four of five switched off
│   └── tests/                 294 tests
├── frontend/                  the web app (React + Vite), built and served by FastAPI
│   ├── public/                manifest, service worker, icons
│   └── src/
│       ├── api.ts             the only file that talks to the backend
│       ├── consent.ts         consent per device, tied to the retention period
│       ├── screens/           intro, pick an exercise, consent, how to film,
│       │                      upload, processing, results
│       └── components/        masthead, stepper, banners, the report itself
├── scripts/                   18 measurement harnesses and figure builders
│   ├── eval_penn_action.py    pose accuracy against a labelled dataset
│   ├── run_faithfulness.py    does the model only say what the rules gave it
│   ├── rescore_faithfulness.py  re-score a stored run offline, no API quota
│   └── fetch_pexels.py        downloads test clips from Pexels
└── data/
    ├── models/                the MediaPipe model files
    ├── test_videos/           input clips (gitignored, except the credits)
    ├── uploads/               what the API receives (gitignored)
    └── outputs/               per-analysis results (gitignored)
```

## Licence

MIT. See `LICENSE`.

Two things in here are not mine. The MediaPipe pose models under `data/models/` are
Google's, under the Apache License 2.0. The test clips are from Pexels, which is
free to use with attribution recommended; the clips themselves are gitignored, and
`data/test_videos/pexels/<query>/SOURCE.txt` records the link and the photographer
for every one of them.

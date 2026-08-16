"""FastAPI app entry — Phase E.

Run with:
    .venv\\Scripts\\python.exe -m uvicorn backend.main:app --reload --port 8000

Then POST a video:
    curl -F 'video=@data/test_videos/squat.mp4' \\
         -F 'exercise_type=squat' \\
         -F 'dry_run_coach=true' \\
         http://127.0.0.1:8000/analyze

Outputs go to data/outputs/<job_id>/ and are served back under /results/<job_id>/.
Uploaded originals go to data/uploads/<job_id>/.
"""
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import retention
from .api.routes import router as api_router, OUTPUT_ROOT, UPLOAD_ROOT


# load .env (mirrors what analyze_squat.py does)
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


_load_dotenv()

# Nothing in the backend logged anywhere before this, which is awkward when the
# whole point of returning an opaque 500 is that the detail went to the log
# instead. Configured here because this is the entry point; the CLI gets Python's
# default behaviour, which puts warnings and errors on stderr, and that is right
# for a CLI.
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Retention sweep (WP-07)
# ---------------------------------------------------------------------------
# A daemon thread rather than a request hook, because a deployment with no
# traffic is exactly the one where old clips would otherwise sit longest. Daemon
# so it never holds up a shutdown; the worst a killed sweep costs is one
# interval, and the next start picks up whatever it missed.
_sweep_stop = threading.Event()


def _sweep_forever():
    roots = [UPLOAD_ROOT, OUTPUT_ROOT]
    while not _sweep_stop.wait(retention.SWEEP_INTERVAL_SECONDS):
        try:
            retention.sweep(roots)
        except Exception:
            # Never let one failure kill the thread. That would turn a transient
            # disk error into retention silently never running again.
            logging.getLogger("coach.retention").exception("sweep failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("DISABLE_RETENTION_SWEEP"):
        logging.getLogger("coach.retention").warning(
            "retention sweep disabled by DISABLE_RETENTION_SWEEP")
        yield
        return
    # One pass immediately. A restart is when there is most likely to be
    # something expired lying around, and waiting a full interval for the first
    # sweep is the difference between a stated period and an approximate one.
    try:
        retention.sweep([UPLOAD_ROOT, OUTPUT_ROOT])
    except Exception:
        logging.getLogger("coach.retention").exception("startup sweep failed")
    _sweep_stop.clear()
    threading.Thread(target=_sweep_forever, name="retention-sweep",
                     daemon=True).start()
    yield
    _sweep_stop.set()


app = FastAPI(title="AI Fitness & Movement Coach", version="0.1.0",
              lifespan=lifespan)

# CORS is off by default, which is not an oversight.
#
# The PWA is served by this same app (see the mount at the bottom of this file),
# so in production the browser is talking to its own origin and no CORS header is
# involved. In dev the Vite server proxies /analyze and /exercises to this port,
# so the browser thinks it is same-origin there too. There is currently no caller
# that needs an Access-Control-Allow-Origin header at all, and allow_origins=["*"]
# meant any page on the internet could POST a video here from a visitor's browser
# and read the analysis back.
#
# CORS_ALLOW_ORIGINS is the escape hatch for when that stops being true — the
# frontend hosted separately, or a second client. Comma-separated, exact origins.
_cors_origins = [o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", "").split(",")
                 if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        # so a client on another origin can still read the reference on a 500
        expose_headers=["X-Error-Reference"],
    )

# serve the per-job output dir as static files so the API can return URLs
# directly to the annotated mp4 and key-frame JPEGs
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=str(OUTPUT_ROOT)), name="results")

app.include_router(api_router)


API_INDEX = {
    "name": "AI Fitness & Movement Coach API",
    "version": app.version,
    "endpoints": {
        "GET  /exercises": "what can be analysed + how to film each one",
        "POST /analyze": "multipart: video, exercise_type, voice_note (opt), "
                         "voice_note_text (opt), pose_model (opt), dry_run_coach (opt)",
        "GET  /results/<job>/<file>": "static — annotated.mp4, worst.jpg, "
                                      "best.jpg, angles.json, coaching.json",
        "DELETE /jobs/<job>": "delete an analysis now rather than waiting for the "
                              "retention period (served in /exercises limits)",
        "GET  /health": "liveness check",
    },
}


@app.get("/api")
def api_index():
    return JSONResponse(API_INDEX)


def _container_memory():
    """Limit and usage as the container sees them, or None where there is no cgroup.

    /health already answers "am I fully configured" so nobody has to upload a
    video to find out. It could not answer "am I about to be killed", and that
    cost the best part of a day: the container was idling at 447 MB of a 1000 MB
    limit and nothing from outside could see it. The kill arrived as one word in
    a log, three seconds into an analysis.
    """
    for limit_path, usage_path in (
            ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
            ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
             "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            raw = Path(limit_path).read_text().strip()
            usage = int(Path(usage_path).read_text().strip())
        except (OSError, ValueError):
            continue
        # cgroup v2 writes "max" for unlimited; v1 writes a sentinel near 2^63
        limit = None if raw == "max" else int(raw)
        if limit is not None and limit > 2 ** 62:
            limit = None
        mb = 1024 * 1024
        return {
            "limit_mb": None if limit is None else round(limit / mb),
            "usage_mb": round(usage / mb),
            "headroom_mb": None if limit is None else round((limit - usage) / mb),
        }
    return None


@app.get("/health")
def health():
    """Liveness, plus which of the optional services this deployment actually has.

    Booleans about configuration only — never a key, never a length, never a
    prefix. "Is a key present" is not a secret; the key is.

    This exists because a deployment can be completely healthy and still be
    running in a degraded mode nobody notices. Ours was: `GROQ_API_KEY` was set in
    the platform's variables and was not reaching the process, so coaching quietly
    fell back to the cue database and every voice note failed to transcribe. The
    app behaved exactly as designed — it says so in the provenance line — but from
    the outside the only way to find out was to upload a video and read the
    `source` field, and the only way to tell "key missing" from "key rejected" was
    to read the source code.

    A deployment should be able to answer "am I fully configured" without being
    sent a video of somebody exercising.
    """
    groq_key = os.environ.get("GROQ_API_KEY") or ""
    has_groq = bool(groq_key.strip())
    body = {
        "status": "ok",
        "services": {
            # what the user would actually get right now, in their terms
            "coaching": "llm" if has_groq else "cue_database_only",
            "transcription": "groq" if has_groq else "unavailable",
        },
        "degraded": not has_groq,
    }
    # Only when there is a cgroup to read. Absent off Linux rather than zeroed,
    # because a made-up number here is worse than no number.
    memory = _container_memory()
    if memory:
        body["memory"] = memory
    return body


# ---------------------------------------------------------------------------
# The PWA
# ---------------------------------------------------------------------------
# Serving the built frontend from here means the app and the API share an origin.
# That buys three things worth having: no CORS in production, no mixed-content
# problems, and a service worker that can actually register (they're same-origin
# only). The cost is that deploying the backend means shipping the frontend build
# with it.
#
# Mounted LAST, because a mount at "/" matches everything and would otherwise
# swallow /analyze and /exercises.
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="pwa")
else:
    # No build present — someone is running the API on its own, which is a normal
    # thing to do (the CLI path, or backend tests). Say so rather than 404ing.
    @app.get("/")
    def root():
        return JSONResponse({
            **API_INDEX,
            "note": "frontend/dist not found — run `npm run build` in frontend/ "
                    "to serve the PWA from this origin.",
        })

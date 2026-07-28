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
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router as api_router, OUTPUT_ROOT


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

app = FastAPI(title="AI Fitness & Movement Coach", version="0.1.0")

# CORS — wide-open for dev. Phase F (PWA) tightens this to known origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
        "GET  /health": "liveness check",
    },
}


@app.get("/api")
def api_index():
    return JSONResponse(API_INDEX)


@app.get("/health")
def health():
    return {"status": "ok"}


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

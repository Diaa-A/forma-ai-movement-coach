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


@app.get("/")
def root():
    return JSONResponse({
        "name": "AI Fitness & Movement Coach API",
        "version": app.version,
        "endpoints": {
            "POST /analyze": "multipart: video, exercise_type, voice_note (opt), "
                             "voice_note_text (opt), pose_model (opt), dry_run_coach (opt)",
            "GET  /results/<job>/<file>": "static — annotated.mp4, worst.jpg, "
                                          "best.jpg, angles.json, coaching.json",
            "GET  /health": "liveness check",
        },
    })


@app.get("/health")
def health():
    return {"status": "ok"}

# syntax=docker/dockerfile:1
#
# Two stages, because the backend serves the frontend. That was Decision-level
# (handoff 11.2): app and API share an origin, which removes CORS from production
# entirely, removes mixed content, and is what lets a service worker register at
# all. The cost was stated there and is paid here - the backend deploy carries the
# frontend build.
#
# frontend/dist is gitignored, so the image builds it from source rather than
# shipping whatever happened to be on the dev machine. That is the better half of
# the trade: the deployed bundle is then reproducibly derived from the committed
# source rather than from whatever was lying around.
#
# NOT BUILT LOCALLY - there is no Docker on the dev machine, so Railway's first
# build is this file's first real test. The apt step is the most likely thing to
# break; package names there are for bookworm, which is why the base tag pins it.

# ---- build the PWA ---------------------------------------------------------
FROM node:22-slim AS frontend

WORKDIR /build

# lockfile first, so this layer is reused when only source changes
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
# tsc -b && vite build - a type error fails the image build rather than shipping
RUN npm run build


# ---- run the API, and the PWA it serves ------------------------------------
FROM python:3.13-slim-bookworm AS runtime

# 3.13 rather than the 3.11 the original spec named: MediaPipe 0.10.35 publishes
# 3.13 wheels, so every pinned dependency installs without a local build step.
# The README covers why in more detail.

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# MediaPipe's own C bindings need the GL stack, and they load it LAZILY - not at
# `import mediapipe`, but the first time a task is created. So a container
# without these builds fine, imports fine, starts fine, passes its healthcheck,
# and then returns 500 on every analysis with:
#
#   OSError: libGLESv2.so.2: cannot open shared object file
#
# which is exactly what the first deployment did. libgles2 provides
# libGLESv2.so.2 and libegl1 provides libEGL.so.1 (both checked against the
# Debian bookworm file lists, which is why the base tag pins bookworm - the
# glib package is renamed in trixie). libgl1 and libglib2.0-0 are the pair the
# GL stack pulls in alongside them.
#
# An earlier version of this file removed the apt step entirely, on the theory
# that it was what kept failing the build. It was not - the build failures were
# a frontend type error cancelling this stage in parallel. Removing it is what
# produced the runtime crash above.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgles2 libegl1 libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# OpenCV stays headless. That swap was made for the wrong reason but it is
# independently correct: nothing here draws to a screen, the headless build is
# smaller, and decoding was confirmed working in the deployed container.
#
# mediapipe requires opencv-contrib-python, so a plain
# "use headless instead" in requirements.txt does not work - pip installs the GUI
# build anyway to satisfy mediapipe. The swap has to happen after the install,
# which is what the uninstall below is for. `pip check` will report mediapipe's
# dependency as unsatisfied afterwards; cv2 resolves to the headless build and
# every import works, which is the thing that actually matters.
#
# Verified rather than assumed: the whole pinned stack was installed into a clean
# venv, the two GUI builds swapped for headless at the same version, and the full
# suite run against it - 142 passed, including the end-to-end MediaPipe smoke
# test and the encoder tests. The only difference between the builds is highgui,
# and the 16 cv2 APIs this project uses are all core, imgproc, imgcodecs and
# videoio.
# One command per RUN, deliberately. The first version chained the pip upgrade,
# the install and the OpenCV swap behind &&, and when it failed the build log
# reported the whole chain as the failing command - which says nothing about
# which part of it broke. Separate steps cost a few cache layers and buy a log
# that names the actual step.
#
# The pip self-upgrade that used to be first is gone. It was never necessary -
# the base image ships a current pip - and it is a known-fragile thing to do
# inside a Docker build, because pip uninstalls itself and then rewrites itself
# on an overlay filesystem. It was also the command sitting at the failure last
# time. Using the pip the image ships with is the more reproducible choice
# anyway: one fewer thing that can differ between two builds of the same commit.
COPY requirements-runtime.txt ./
RUN python -m pip install --no-compile -r requirements-runtime.txt

# mediapipe requires opencv-contrib-python, so it arrives whatever the
# requirements say. Swapped here for the headless build at the same version.
RUN python -m pip uninstall -y opencv-contrib-python
RUN python -m pip install --no-compile opencv-contrib-python-headless==4.13.0.92

# The pose models are committed (~15 MB) rather than fetched during the build, so
# an image build cannot fail on somebody else's CDN and two builds of the same
# commit contain the same weights.
COPY data/models/ ./data/models/
COPY backend/ ./backend/
COPY analyze_squat.py ./
COPY --from=frontend /build/dist ./frontend/dist

# Uploads and rendered artefacts land here. On Railway this filesystem is
# ephemeral: a redeploy wipes it and breaks any /results URL a participant still
# has open, so do not redeploy mid-session. This is not the deletion guarantee -
# that is the retention sweep. A redeploy is an accident, not a policy.
RUN mkdir -p data/uploads data/outputs

# Build-time smoke test, and it has to go deeper than an import.
#
# The previous version of this line was `import cv2, mediapipe, backend.main`.
# It passed, the image shipped, and every analysis failed with a missing
# libGLESv2.so.2 - because importing mediapipe does not load its C bindings.
# Those load when a task is CREATED, which is the first thing a real request
# does and the last thing a build was checking.
#
# So this constructs an actual PoseLandmarker against the real model file, in
# VIDEO mode, exactly as extract_landmarks does. Any missing native library now
# fails the build with the name of the missing object, instead of producing a
# container that builds green, starts green, passes its healthcheck and 500s on
# every upload.
#
# It also proves the model file is present at the path pose.py resolves, which
# no import could tell us.
RUN python -c "\
import cv2, backend.main; \
from mediapipe.tasks import python as mp_tasks; \
from mediapipe.tasks.python import vision as mp_vision; \
from backend.pipeline.pose import _model_path; \
opts = mp_vision.PoseLandmarkerOptions( \
    base_options=mp_tasks.BaseOptions(model_asset_path=_model_path('full')), \
    running_mode=mp_vision.RunningMode.VIDEO, num_poses=1); \
mp_vision.PoseLandmarker.create_from_options(opts).close(); \
print('smoke ok: cv2', cv2.__version__, '+ landmarker constructed')"

RUN useradd --create-home --uid 10001 coach && chown -R coach:coach /app
USER coach

EXPOSE 8000

# Railway injects PORT and healthchecks that same port, so it has to be honoured
# rather than hardcoded. Shell form, because $PORT needs expanding; the fallback
# keeps a plain `docker run -p 8000:8000` working off Railway.
#
# No --limit-concurrency here any more. It used to carry 6, as a memory guard,
# and it was the wrong instrument for two reasons: it caps ALL connections rather
# than analyses, so a value low enough to bound memory would eventually block the
# platform's healthcheck and cause a restart loop, and six analyses at 239-353 MB
# each is far past what this container has. The bound now lives in routes.py as a
# semaphore around the analysis itself, so /health and /exercises stay reachable
# while the server is busy. Tune it with MAX_CONCURRENT_ANALYSES.
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}

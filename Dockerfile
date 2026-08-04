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
# source, which is what WP-12 asks for.
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
# Reasoning in BUILD_REFERENCE 5, "Known deviations from the locked spec".

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# No apt step, and that is the point.
#
# The first version installed libgl1 and libglib2.0-0, because the GUI builds of
# OpenCV link libGL and glib even though nothing here ever draws to a screen.
# That step is what failed on Railway, twice, and the package names were not the
# reason - both exist in bookworm at exactly those names, checked against the
# Debian index. The error text was truncated in the build log, so rather than
# guess at it again the step is gone.
#
# It can be gone because mediapipe requires opencv-contrib-python, so a plain
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
# has open, so do not redeploy mid-session. It is NOT a substitute for WP-07 -
# a redeploy is an accident, not a retention policy.
RUN mkdir -p data/uploads data/outputs

# Import everything at build time. Two reasons, both learned the hard way here.
#
# If a native library really is missing, this fails now with an ImportError that
# names the missing .so, instead of building green and then failing the
# healthcheck with nothing useful in the log. mediapipe's own binaries might yet
# want something the headless swap removed - this is the check that would say so,
# and say which.
#
# It also boots the app, so a configuration mistake is a failed build rather than
# a failed deploy.
RUN python -c "import cv2, mediapipe, backend.main; print('imports ok:', cv2.__version__)"

RUN useradd --create-home --uid 10001 coach && chown -R coach:coach /app
USER coach

EXPOSE 8000

# Railway injects PORT and healthchecks that same port, so it has to be honoured
# rather than hardcoded. Shell form, because $PORT needs expanding; the fallback
# keeps a plain `docker run -p 8000:8000` working off Railway.
#
# --limit-concurrency is a guard rail rather than a tuning knob. /analyze is a
# sync handler, so FastAPI runs it in a threadpool whose default size is 40, and
# one analysis peaks at 237 MB (measured, 576x1024) - nearer 400 MB for a 4K
# upload. Forty at once would be several gigabytes and either an OOM or a
# surprising bill. Six is far above anything a testing session produces and
# bounds the worst case at roughly 2.4 GB. Past it, uvicorn returns 503, which
# the PWA already renders as a server error the user can retry.
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --limit-concurrency 6

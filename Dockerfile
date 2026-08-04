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

# opencv-python links against libGL and glib even when nothing is ever drawn to a
# screen. opencv-python-headless needs neither, would save around 40 MB, and
# would work - there is not one cv2 GUI call in this project, checked.
#
# Deliberately not doing that. It would mean the container runs a different
# OpenCV build from the one the 142 tests and the Penn Action benchmark ran
# against, and "the thing running is not the thing you tested" has already cost
# this project real time twice (the stale uvicorn, the two checkouts). Forty
# megabytes is a cheap price for deleting that whole category of doubt.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
 && python -m pip install -r requirements.txt

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

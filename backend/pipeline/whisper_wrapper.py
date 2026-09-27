"""Whisper voice-note transcription.

Transcribes a short user voice note (a separate recording where the user states
goals / pain points) into text that feeds the Layer 2 coaching prompt as
context. It never gives the LLM new biomechanical authority — it only shapes
tone and focus.

Two backends:
  - Groq-hosted `whisper-large-v3` (default when GROQ_API_KEY is set). Cloud,
    fast, and more accurate than the local `small` model. Refines the original
    spec choice (local small) now that transcription quality is prioritised over
    running fully offline. Uses the same Groq account as the coaching LLM.
  - Local `openai-whisper` (fallback). Offline, no API; lazily imported so
    the ~2 GB torch dependency is never forced on users who don't need it.

`transcribe()` picks the Groq path when a key is present, else local, else
raises a clear, actionable error.
"""
from __future__ import annotations

import json
import os
import uuid
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3"
REQUEST_TIMEOUT = 60.0

# local fallback model size (spec default)
DEFAULT_LOCAL_MODEL = "small"

_INSTALL_HINT = (
    "No transcription backend available. Either set GROQ_API_KEY (recommended — "
    "uses Groq-hosted whisper-large-v3), or install local Whisper:\n"
    "    .venv\\Scripts\\python.exe -m pip install -U openai-whisper\n"
    "(local install pulls in torch, ~2 GB on CPU)."
)

_local_cache: dict = {}


# ---------------------------------------------------------------------------
# Groq-hosted transcription (multipart POST via stdlib urllib)
# ---------------------------------------------------------------------------

def _multipart_body(fields: dict, file_field: str, filename: str,
                    file_bytes: bytes):
    """Build a multipart/form-data body by hand (no requests dependency)."""
    boundary = "----aifitcoach" + uuid.uuid4().hex
    crlf = b"\r\n"
    parts = []
    for name, value in fields.items():
        parts.append(b"--" + boundary.encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        parts.append(b"")
        parts.append(str(value).encode())
    parts.append(b"--" + boundary.encode())
    parts.append(
        f'Content-Disposition: form-data; name="{file_field}"; '
        f'filename="{filename}"'.encode()
    )
    parts.append(b"Content-Type: application/octet-stream")
    parts.append(b"")
    body = crlf.join(parts) + crlf + file_bytes + crlf
    body += b"--" + boundary.encode() + b"--" + crlf
    return body, boundary


def transcribe_groq(audio_path: str, api_key: str, model: str = GROQ_MODEL,
                    language: str = "en") -> str:
    p = Path(audio_path)
    file_bytes = p.read_bytes()
    fields = {"model": model, "response_format": "json"}
    if language:
        fields["language"] = language
    body, boundary = _multipart_body(fields, "file", p.name, file_bytes)
    req = urllib.request.Request(
        GROQ_TRANSCRIBE_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            # Groq is behind Cloudflare, which 403s the default urllib UA.
            "User-Agent": "ai-fitness-coach/0.1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Groq transcription error {e.code}: {detail}") from e
    return (data.get("text") or "").strip()


# ---------------------------------------------------------------------------
# Local fallback
# ---------------------------------------------------------------------------

def transcribe_local(audio_path: str, model_size: Optional[str] = None,
                     language: str = "en") -> str:
    try:
        import whisper  # type: ignore
    except ImportError as e:
        raise RuntimeError(_INSTALL_HINT) from e
    size = model_size or os.environ.get("WHISPER_MODEL", DEFAULT_LOCAL_MODEL)
    if size not in _local_cache:
        _local_cache[size] = whisper.load_model(size)
    result = _local_cache[size].transcribe(str(audio_path), language=language,
                                            fp16=False)
    return (result.get("text") or "").strip()


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def transcribe(audio_path: str, model_size: Optional[str] = None,
               language: str = "en") -> str:
    """Transcribe a voice note to text. Prefers Groq-hosted Whisper when a key is
    available, else falls back to local openai-whisper. Empty audio -> ''."""
    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")
    if p.stat().st_size == 0:
        return ""

    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        return transcribe_groq(audio_path, api_key, language=language)
    return transcribe_local(audio_path, model_size=model_size, language=language)


def is_available() -> bool:
    """True if either backend can run (Groq key present or local whisper installed)."""
    if os.environ.get("GROQ_API_KEY"):
        return True
    try:
        import whisper  # noqa: F401
        return True
    except ImportError:
        return False

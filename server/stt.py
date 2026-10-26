"""
server/stt.py — speech-to-text via ElevenLabs Scribe.

One job: send a complete WAV blob, get back (text, language). We hit the REST
endpoint directly (not an SDK) because it's a single multipart POST and the
error handling is clearer when we own the request.

Docs: https://elevenlabs.io/docs/api-reference/speech-to-text/convert
Model: "scribe_v2" (as of 2026-10; the older "scribe_v1" is no longer listed).
"""
from __future__ import annotations

import logging

import httpx

log = logging.getLogger("voice-assistant.stt")

ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"
MODEL_ID = "scribe_v2"


async def transcribe(
    wav_bytes: bytes,
    api_key: str,
    timeout: float = 15.0,
) -> tuple[str, str]:
    """POST a WAV, return (text, language_code).

    language_code is a short ISO-ish tag like "eng" or "hin". Scribe auto-
    detects by default; we don't pin a language so the kid can switch between
    English and Hindi mid-session.
    """
    headers = {"xi-api-key": api_key}
    # httpx multipart: files= holds the binary parts, data= holds plain fields.
    files = {"file": ("utterance.wav", wav_bytes, "audio/wav")}
    data = {"model_id": MODEL_ID}

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(ENDPOINT, headers=headers, files=files, data=data)

    if resp.status_code >= 400:
        # Surface the body so a 401 / 422 is debuggable.
        raise RuntimeError(f"STT failed ({resp.status_code}): {resp.text[:300]}")

    body = resp.json()
    text = (body.get("text") or "").strip()
    language = body.get("language_code") or "unknown"
    log.info("STT: %r (lang=%s, p=%.2f)", text, language, body.get("language_probability", 0))
    return text, language

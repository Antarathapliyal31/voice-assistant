"""
server/tts.py — text-to-speech via ElevenLabs Flash v2.5.

One job: hand a string in, get MP3 bytes out. We request the full MP3 (no
streaming) because the naive pipeline sends the whole file over the
WebSocket as one binary message. Phase 4 will switch to streaming.

Docs: https://elevenlabs.io/docs/api-reference/text-to-speech/convert
Model: "eleven_flash_v2_5" — the fastest Flash tier at the time of writing.
"""
from __future__ import annotations

import logging

import httpx

log = logging.getLogger("voice-assistant.tts")

MODEL_ID = "eleven_flash_v2_5"
# Default server output is mp3_44100_128. We stick with it: 44.1 kHz / 128 kbps
# is small (~16 KB/s) and the browser's decodeAudioData handles it natively.


async def synthesize(
    text: str,
    *,
    voice_id: str,
    api_key: str,
    timeout: float = 20.0,
) -> bytes:
    """POST text to ElevenLabs, return the full MP3 as bytes."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": MODEL_ID,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=payload)

    if resp.status_code >= 400:
        raise RuntimeError(f"TTS failed ({resp.status_code}): {resp.text[:300]}")

    mp3 = resp.content
    log.info("TTS: %d bytes of audio for %d chars of text", len(mp3), len(text))
    return mp3

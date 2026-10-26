"""
server/pipeline_naive.py — Phase 3's slow baseline.

Runs stt -> llm -> tts strictly sequentially: each stage must finish before
the next one starts. Nothing is parallel, nothing starts early. That's the
"naive" shape, and the number we care about is "time to first audio" relative
to turn_end. Phase 4 will overlap stages and we'll compare the two using
the latency JSONL this file helps populate.

Pure async code — no file I/O here beyond building a WAV in memory, no state
machine. Caller owns the WebSocket, the busy flag, and the history list.
"""
from __future__ import annotations

import asyncio
import io
import logging
import time
import wave
from dataclasses import dataclass

from anthropic import AsyncAnthropic

from server.llm import answer
from server.stt import transcribe
from server.tts import synthesize

log = logging.getLogger("voice-assistant.pipeline")

# WAV format we ship to STT — matches what the VAD / browser produce.
SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2  # bytes per Int16 sample
CHANNELS = 1


@dataclass
class TurnResult:
    """Everything a caller needs after one turn runs.

    Timestamps are time.perf_counter() values (monotonic). The caller
    subtracts t0 (= turn_end) to produce the ms deltas for the latency log.
    """
    text: str                 # transcript (STT output)
    language: str             # STT-detected language
    answer: str               # LLM response
    mp3_len: int              # bytes of audio sent to the client
    t0: float                 # when VAD said turn_end (passed in)
    t_stt: float
    t_llm_first: float        # first text token from the model
    t_llm_done: float
    t_tts: float
    t_audio_sent: float       # when we handed the MP3 to the outbound queue


def _wrap_pcm_to_wav(pcm_bytes: bytes) -> bytes:
    """Build a tiny in-memory WAV from raw Int16 PCM. ~44-byte header + data."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


async def run_turn(
    pcm_bytes: bytes,
    history: list[tuple[str, str]],
    outbound: asyncio.Queue,
    config: dict,
    t0: float,
    *,
    anthropic_client: AsyncAnthropic,
) -> TurnResult:
    """Drive one full turn through STT -> LLM -> TTS.

    Pushes to `outbound` in this exact order (critical — the browser renders
    them in order):
        {"type": "transcript", "text": ..., "language": ...}
        {"type": "answer",     "text": ...}
        <MP3 bytes as a single binary frame>

    `history` is mutated in place: on success we append the new (user, asst)
    pair and trim to the last 3. On any failure, history is left untouched
    so a bad turn doesn't pollute the context.

    Raises on any stage failure. The caller is responsible for turning that
    exception into a user-visible {"type": "error", ...} message.
    """
    # ---------- STT ----------------------------------------------------------
    # Build the WAV inline. wave.open on BytesIO is tiny (a few KB per second)
    # so we don't bother threading this.
    wav_bytes = _wrap_pcm_to_wav(pcm_bytes)
    text, language = await transcribe(wav_bytes, api_key=config["elevenlabs_api_key"])
    t_stt = time.perf_counter()

    await outbound.put({"type": "transcript", "text": text, "language": language})

    # Guard: if STT returned nothing, don't bother asking the LLM. Raise so
    # the caller can report an error (and keep history unchanged).
    if not text:
        raise RuntimeError("STT returned empty transcript")

    # ---------- LLM ----------------------------------------------------------
    answer_text, t_llm_first = await answer(
        text,
        history,
        api_key=config["anthropic_api_key"],
        model=config["anthropic_model"],
        client=anthropic_client,
    )
    t_llm_done = time.perf_counter()

    await outbound.put({"type": "answer", "text": answer_text})

    # ---------- TTS ----------------------------------------------------------
    mp3 = await synthesize(
        answer_text,
        voice_id=config["elevenlabs_voice_id"],
        api_key=config["elevenlabs_api_key"],
    )
    t_tts = time.perf_counter()

    # The MP3 is pushed as RAW BYTES. The sender task routes bytes-typed
    # queue items to ws.send_bytes().
    await outbound.put(mp3)
    t_audio_sent = time.perf_counter()

    # ---------- History (only after full success) ----------------------------
    history.append((text, answer_text))
    while len(history) > 3:
        history.pop(0)

    return TurnResult(
        text=text,
        language=language,
        answer=answer_text,
        mp3_len=len(mp3),
        t0=t0,
        t_stt=t_stt,
        t_llm_first=t_llm_first,
        t_llm_done=t_llm_done,
        t_tts=t_tts,
        t_audio_sent=t_audio_sent,
    )

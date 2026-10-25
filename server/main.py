"""
server/main.py — Phase 1 FastAPI backend for the voice assistant.

Flow:
    browser mic
      -> mic-worklet.js (chunks 20 ms Int16 PCM)
      -> WebSocket /ws (binary frames + JSON control frames)
      -> this file
      -> recordings/<timestamp>.wav

The structure here is deliberately set up the way the final app will need:

  * A single receive loop that pulls messages via `ws.receive()` and branches
    on whether the frame is binary (audio) or text (JSON control).
  * All outbound traffic goes through an `asyncio.Queue`, drained by exactly
    ONE sender task. This matters because WebSocket send operations are not
    safe to call concurrently from multiple coroutines — serializing them
    through a queue is the clean fix.
  * Per-connection state lives in local variables inside the endpoint, not
    in globals. One connection = one set of buffers.
"""

import asyncio
import json
import logging
import time
import wave
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

# -----------------------------------------------------------------------------
# Logging. We print to stderr with timestamps so you can see what's happening
# when you run `uvicorn ...`.
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("voice-assistant")

# -----------------------------------------------------------------------------
# Paths. We resolve relative to this file so the server works regardless of
# what directory you launched uvicorn from.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"
RECORDINGS_DIR = PROJECT_ROOT / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Audio format constants. The browser is configured to send us 16 kHz mono
# 16-bit PCM, so we store it in a WAV file with exactly those parameters.
# -----------------------------------------------------------------------------
SAMPLE_RATE = 16000      # Hz
SAMPLE_WIDTH = 2         # bytes per sample (16-bit)
CHANNELS = 1             # mono
BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS  # for duration math

app = FastAPI()


# =============================================================================
# WebSocket endpoint
# =============================================================================
#
# The `@app.websocket("/ws")` decorator registers this BEFORE the static-files
# mount at "/" below, so an upgrade request to /ws matches here first.

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    client = f"{ws.client.host}:{ws.client.port}" if ws.client else "?"
    log.info("client connected: %s", client)

    # ----- Per-connection state ------------------------------------------------
    # audio_buffer: raw PCM bytes we've received so far.
    # chunk_count: how many binary frames arrived.
    # arrivals: wall-clock-ish timestamps so we can compute inter-arrival gaps.
    audio_buffer = bytearray()
    chunk_count = 0
    arrivals: list[float] = []

    # ----- Outbound queue + single sender task --------------------------------
    # Only `sender` is allowed to call ws.send_*. Everything else just appends
    # to the queue. `None` on the queue is our shutdown signal.
    outbound: asyncio.Queue = asyncio.Queue()
    sender = asyncio.create_task(
        _sender_task(ws, outbound), name="ws-sender"
    )

    try:
        while True:
            # ws.receive() returns a dict like:
            #   {"type": "websocket.receive", "bytes": b"..."}   (binary frame)
            #   {"type": "websocket.receive", "text": "..."}     (text frame)
            #   {"type": "websocket.disconnect", "code": 1000}   (client closed)
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                log.info("client disconnected (code=%s)", msg.get("code"))
                break

            # Binary frame = raw PCM audio chunk from the worklet.
            if msg.get("bytes") is not None:
                audio_buffer.extend(msg["bytes"])
                chunk_count += 1
                # perf_counter() is a monotonic high-resolution clock, which is
                # what you want for measuring intervals (never jumps backwards).
                arrivals.append(time.perf_counter())
                continue

            # Text frame = JSON control message.
            if msg.get("text") is not None:
                try:
                    control = json.loads(msg["text"])
                except json.JSONDecodeError:
                    log.warning("bad JSON from client: %r", msg["text"])
                    continue

                kind = control.get("type")
                log.info("control: %s", control)

                if kind == "start":
                    # A client can re-start on the same socket; reset state.
                    audio_buffer.clear()
                    chunk_count = 0
                    arrivals.clear()

                elif kind == "stop":
                    summary = await _finalize_recording(
                        audio_buffer, chunk_count, arrivals
                    )
                    # Human-readable copy to the server console...
                    print(json.dumps(summary, indent=2), flush=True)
                    # ...and the structured copy back to the browser.
                    await outbound.put(summary)

                else:
                    log.warning("unknown control type: %r", kind)

    except WebSocketDisconnect:
        # Some paths through Starlette raise this instead of returning the
        # disconnect dict. Treat it the same way.
        log.info("client disconnected (WebSocketDisconnect): %s", client)

    finally:
        # Ask the sender to drain and exit, then give it a moment. If it's
        # stuck (e.g. mid-send to a dead socket), cancel it. Either way, we
        # want to log any exception rather than let it vanish.
        await outbound.put(None)
        try:
            await asyncio.wait_for(sender, timeout=1.0)
        except asyncio.TimeoutError:
            sender.cancel()
            try:
                await sender
            except (asyncio.CancelledError, Exception):
                # Already logged inside the task; ignore here.
                pass


# =============================================================================
# Sender task — the only coroutine that touches ws.send_*
# =============================================================================

async def _sender_task(ws: WebSocket, outbound: asyncio.Queue) -> None:
    """Drain `outbound` and forward each message to the client as JSON text.

    Receiving `None` means "shut down cleanly".
    """
    try:
        while True:
            message = await outbound.get()
            if message is None:
                return
            await ws.send_text(json.dumps(message))
    except WebSocketDisconnect:
        # The client went away mid-send. Not an error — just stop.
        log.info("sender: client disconnected")
    except asyncio.CancelledError:
        # Expected during shutdown. Re-raise so the task is marked cancelled.
        raise
    except Exception:
        # Anything else is a real bug; make sure it doesn't fail silently.
        log.exception("sender task crashed")


# =============================================================================
# Finalize a recording: write .wav + compute summary
# =============================================================================

async def _finalize_recording(
    audio_buffer: bytearray,
    chunk_count: int,
    arrivals: list[float],
) -> dict:
    total_bytes = len(audio_buffer)
    duration_s = total_bytes / BYTES_PER_SECOND if BYTES_PER_SECOND else 0.0

    # Inter-arrival gaps between consecutive binary frames, in milliseconds.
    # We expect ~20 ms gaps because the worklet posts every 20 ms. A big max
    # gap means the client stalled (e.g. GC pause, backpressure) — useful
    # signal to have once we start piping audio to STT.
    if len(arrivals) >= 2:
        gaps_ms = [
            (arrivals[i] - arrivals[i - 1]) * 1000.0
            for i in range(1, len(arrivals))
        ]
        avg_gap_ms = sum(gaps_ms) / len(gaps_ms)
        max_gap_ms = max(gaps_ms)
    else:
        avg_gap_ms = 0.0
        max_gap_ms = 0.0

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    wav_path = RECORDINGS_DIR / f"{timestamp}.wav"

    # Python's `wave` module does synchronous file I/O. We're in async code,
    # so push the write to a worker thread so the event loop keeps serving
    # other tasks (there won't be any yet in Phase 1, but good habits now).
    await asyncio.to_thread(_write_wav, wav_path, bytes(audio_buffer))
    log.info(
        "wrote %s (%d bytes, %.2fs, %d chunks)",
        wav_path, total_bytes, duration_s, chunk_count,
    )

    return {
        "type": "summary",
        "file": wav_path.name,
        "chunks": chunk_count,
        "total_bytes": total_bytes,
        "duration_s": round(duration_s, 3),
        "avg_gap_ms": round(avg_gap_ms, 2),
        "max_gap_ms": round(max_gap_ms, 2),
    }


def _write_wav(path: Path, pcm_bytes: bytes) -> None:
    """Blocking WAV write. Always called via asyncio.to_thread."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


# =============================================================================
# Static files — serve web/ at the root so http://localhost:8000 loads the page
# =============================================================================
#
# This mount is registered LAST, so the /ws WebSocket route above still
# matches first. `html=True` makes "/" serve index.html.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

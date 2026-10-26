"""
server/main.py — FastAPI backend for the voice assistant.

Flow:
    browser mic
      -> mic-worklet.js (20 ms Int16 PCM chunks)
      -> WebSocket /ws (binary audio frames + JSON control frames)
      -> this file
          -> full-session recording: recordings/<timestamp>.wav
          -> per-turn recordings:    recordings/utt-<timestamp>.wav   (Phase 2)
          -> live VAD events pushed back over the WebSocket           (Phase 2)

Structural rules (preserved from Phase 1):
  * One receive loop uses `ws.receive()` and branches on binary vs text.
  * All outbound traffic goes through an asyncio.Queue drained by exactly
    ONE sender task — WebSocket send is not safe to call concurrently.
  * Per-connection state lives in local variables inside the endpoint.
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
from silero_vad import load_silero_vad

from server.vad import TurnDetector

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
    audio_buffer = bytearray()        # full-session PCM for the big .wav
    chunk_count = 0                   # how many binary frames arrived
    arrivals: list[float] = []        # perf_counter timestamps per chunk
    turn_files: list[str] = []        # names of per-turn .wavs written

    # ----- Per-connection Silero VAD ------------------------------------------
    # Load the model inside the connection (not globally) because Silero keeps
    # hidden GRU state inside the Python object. Two concurrent connections
    # sharing it would clobber each other. load_silero_vad is ~380 ms of
    # synchronous CPU work, so we offload to a thread rather than block the
    # event loop; the browser's audio chunks queue harmlessly meanwhile.
    log.info("loading Silero VAD (onnx) for %s...", client)
    t_load_start = time.perf_counter()
    vad_model = await asyncio.to_thread(load_silero_vad, onnx=True)
    load_ms = (time.perf_counter() - t_load_start) * 1000.0
    log.info("Silero VAD ready in %.1f ms", load_ms)
    turn_detector = TurnDetector(vad_model)

    # ----- Outbound queue + single sender task --------------------------------
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
                pcm = msg["bytes"]
                audio_buffer.extend(pcm)
                chunk_count += 1
                arrivals.append(time.perf_counter())

                # Feed the same bytes to the VAD. feed() returns a list of
                # events (empty on most calls) that we translate into either
                # messages for the browser or saved-turn files.
                events = turn_detector.feed(pcm)
                for ev in events:
                    await _handle_vad_event(ev, outbound, turn_files)
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
                    turn_files.clear()
                    # Fresh VAD state too. Cheaper than reloading the model.
                    turn_detector = TurnDetector(vad_model)

                elif kind == "stop":
                    summary = await _finalize_recording(
                        audio_buffer, chunk_count, arrivals,
                        turn_files, turn_detector.stats(),
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

async def _handle_vad_event(
    event: dict,
    outbound: asyncio.Queue,
    turn_files: list[str],
) -> None:
    """React to one event from the TurnDetector.

    Two event kinds today:
      speech_start — forward it to the browser so it can update the UI.
      turn_end     — save the utterance as its own .wav AND notify the browser.
    """
    kind = event["type"]

    if kind == "speech_start":
        await outbound.put({"type": "vad", "event": "speech_start"})
        return

    if kind == "turn_end":
        # NOTE: this is the START of our future "time to first audio" stopwatch
        # — once we have STT + LLM + TTS, we'll measure from here until the
        # first byte of synthesized reply hits the client.
        t_turn_end = time.perf_counter()
        log.debug("t_turn_end stopwatch primed at %.6f", t_turn_end)

        pcm = event["pcm_bytes"]
        duration_s = event["duration_s"]

        # Millisecond-precision filename so back-to-back turns don't collide.
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-") + f"{int(time.time()*1000) % 1000:03d}"
        utt_path = RECORDINGS_DIR / f"utt-{ts}.wav"

        await asyncio.to_thread(_write_wav, utt_path, pcm)
        log.info("turn saved: %s (%.2fs)", utt_path.name, duration_s)
        turn_files.append(utt_path.name)

        await outbound.put({
            "type": "vad",
            "event": "turn_end",
            "file": utt_path.name,
            "duration_s": round(duration_s, 3),
        })
        return

    log.warning("unknown VAD event: %r", event)


async def _finalize_recording(
    audio_buffer: bytearray,
    chunk_count: int,
    arrivals: list[float],
    turn_files: list[str],
    vad_stats: dict,
) -> dict:
    total_bytes = len(audio_buffer)
    duration_s = total_bytes / BYTES_PER_SECOND if BYTES_PER_SECOND else 0.0

    # Inter-arrival gaps between consecutive binary frames, in milliseconds.
    # We expect ~20 ms gaps because the worklet posts every 20 ms. A big max
    # gap means the client stalled (e.g. GC pause, backpressure).
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

    # `wave` is synchronous — offload so we don't block the event loop.
    await asyncio.to_thread(_write_wav, wav_path, bytes(audio_buffer))
    log.info(
        "wrote %s (%d bytes, %.2fs, %d chunks, %d turns)",
        wav_path, total_bytes, duration_s, chunk_count, len(turn_files),
    )

    return {
        "type": "summary",
        "file": wav_path.name,
        "chunks": chunk_count,
        "total_bytes": total_bytes,
        "duration_s": round(duration_s, 3),
        "avg_gap_ms": round(avg_gap_ms, 2),
        "max_gap_ms": round(max_gap_ms, 2),
        "turns": len(turn_files),
        "turn_files": turn_files,
        "vad": vad_stats,
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

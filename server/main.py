"""
server/main.py — FastAPI backend for the voice assistant.

Flow:
    browser mic
      -> mic-worklet.js (20 ms Int16 PCM chunks)
      -> WebSocket /ws (binary audio frames + JSON control frames)
      -> this file
          -> full-session recording:       recordings/<timestamp>.wav
          -> per-turn recordings:          recordings/utt-<timestamp>.wav
          -> VAD events to the browser     (speech_start / turn_end)
          -> naive pipeline on turn_end:   STT -> LLM -> TTS            (Phase 3)
                                             -> transcript, answer, mp3
          -> latency logs:                 logs/latency.jsonl           (Phase 3)

Structural rules preserved from earlier phases:
  * One receive loop uses `ws.receive()` and branches on binary vs text.
  * All outbound traffic goes through an asyncio.Queue drained by exactly
    ONE sender task — WebSocket send is not safe to call concurrently.
  * Per-connection state lives in local variables inside the endpoint.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from silero_vad import load_silero_vad

from server.pipeline_naive import TurnResult, run_turn
from server.vad import TurnDetector

# -----------------------------------------------------------------------------
# .env loading (API keys + model IDs).
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CONFIG = {
    "anthropic_api_key":  os.getenv("ANTHROPIC_API_KEY", ""),
    "anthropic_model":    os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
    "elevenlabs_api_key": os.getenv("ELEVENLABS_API_KEY", ""),
    "elevenlabs_voice_id": os.getenv("ELEVENLABS_VOICE_ID", ""),
}

# -----------------------------------------------------------------------------
# Logging.
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("voice-assistant")

if not CONFIG["anthropic_api_key"] or not CONFIG["elevenlabs_api_key"]:
    log.warning(".env missing ANTHROPIC_API_KEY and/or ELEVENLABS_API_KEY — "
                "pipeline turns will fail")

# -----------------------------------------------------------------------------
# Paths.
# -----------------------------------------------------------------------------
WEB_DIR = PROJECT_ROOT / "web"
RECORDINGS_DIR = PROJECT_ROOT / "recordings"
LOGS_DIR = PROJECT_ROOT / "logs"
LATENCY_LOG = LOGS_DIR / "latency.jsonl"
RECORDINGS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Audio format for the session WAV writer.
# -----------------------------------------------------------------------------
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1
BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS

# How long to wait for playback_ended from the browser before unlatching the
# busy flag on our own. If this ever fires, something is wrong on the client.
PLAYBACK_WATCHDOG_S = 30.0

app = FastAPI()


# =============================================================================
# Per-turn pipeline state (grouped so nested funcs can mutate by attribute).
# =============================================================================

@dataclass
class TurnPipelineState:
    """One slot — tracks the ONE active turn we're serving.

    `busy` goes True on turn_end and only clears when playback_ended arrives,
    when the pipeline errors, or when the watchdog fires. New turn_end events
    are ignored while busy is True (that's the "no barge-in yet" rule).

    `last_result` is the TurnResult from the most recent successful run,
    stashed so we can compute latencies when playback_started arrives.
    """
    busy: bool = False
    turn_task: asyncio.Task | None = None
    playback_watchdog: asyncio.Task | None = None
    last_result: TurnResult | None = None
    turns_served: int = 0
    turns_ignored: int = 0


# =============================================================================
# WebSocket endpoint
# =============================================================================

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    client = f"{ws.client.host}:{ws.client.port}" if ws.client else "?"
    log.info("client connected: %s", client)

    # ----- Per-connection state ------------------------------------------------
    audio_buffer = bytearray()
    chunk_count = 0
    arrivals: list[float] = []
    turn_files: list[str] = []

    pipeline_state = TurnPipelineState()
    history: list[tuple[str, str]] = []   # last 3 (user, asst) pairs

    # Long-lived Anthropic client (one TCP pool, reused across turns).
    anthropic_client = AsyncAnthropic(api_key=CONFIG["anthropic_api_key"])

    # ----- Per-connection Silero VAD ------------------------------------------
    log.info("loading Silero VAD for %s...", client)
    t_load = time.perf_counter()
    vad_model = await asyncio.to_thread(load_silero_vad, onnx=True)
    log.info("Silero VAD ready in %.1f ms", (time.perf_counter() - t_load) * 1000.0)
    turn_detector = TurnDetector(vad_model)

    # ----- Outbound queue + single sender task --------------------------------
    outbound: asyncio.Queue = asyncio.Queue()
    sender = asyncio.create_task(_sender_task(ws, outbound), name="ws-sender")

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                log.info("client disconnected (code=%s)", msg.get("code"))
                break

            # ---- binary frame = raw PCM audio chunk ---------------------------
            if msg.get("bytes") is not None:
                pcm = msg["bytes"]
                audio_buffer.extend(pcm)
                chunk_count += 1
                arrivals.append(time.perf_counter())

                for ev in turn_detector.feed(pcm):
                    await _handle_vad_event(
                        ev, outbound, turn_files,
                        pipeline_state, history, anthropic_client,
                    )
                continue

            # ---- text frame = JSON control message ---------------------------
            if msg.get("text") is not None:
                try:
                    control = json.loads(msg["text"])
                except json.JSONDecodeError:
                    log.warning("bad JSON from client: %r", msg["text"])
                    continue

                kind = control.get("type")
                log.info("control: %s", control)

                if kind == "start":
                    audio_buffer.clear()
                    chunk_count = 0
                    arrivals.clear()
                    turn_files.clear()
                    turn_detector = TurnDetector(vad_model)
                    history.clear()
                    pipeline_state = TurnPipelineState()

                elif kind == "playback_started":
                    # The browser's AudioBufferSourceNode.start() has just
                    # fired. This is "time to first audio" on the client — we
                    # record it here and send the latency breakdown back.
                    await _on_playback_started(pipeline_state, outbound)

                elif kind == "playback_ended":
                    # Audio done playing — the toy is free again.
                    _on_playback_ended(pipeline_state)

                elif kind == "stop":
                    # Flush any mid-utterance so the user's last words are
                    # saved as a .wav. We DO NOT run the pipeline on the
                    # flushed turn — the user asked us to stop.
                    for ev in turn_detector.flush():
                        if ev["type"] == "turn_end":
                            await _save_turn_wav(ev["pcm_bytes"], turn_files)

                    summary = await _finalize_recording(
                        audio_buffer, chunk_count, arrivals,
                        turn_files, turn_detector.stats(),
                        pipeline_state,
                    )
                    print(json.dumps(summary, indent=2), flush=True)
                    await outbound.put(summary)

                else:
                    log.warning("unknown control type: %r", kind)

    except WebSocketDisconnect:
        log.info("client disconnected (WebSocketDisconnect): %s", client)

    finally:
        # Tear down in a fixed order: cancel any pending pipeline + watchdog
        # tasks first so they don't try to send over the closing socket, then
        # drain the sender queue and let the sender exit.
        for task_attr in ("turn_task", "playback_watchdog"):
            task = getattr(pipeline_state, task_attr)
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        await anthropic_client.close()

        await outbound.put(None)
        try:
            await asyncio.wait_for(sender, timeout=1.0)
        except asyncio.TimeoutError:
            sender.cancel()
            try:
                await sender
            except (asyncio.CancelledError, Exception):
                pass


# =============================================================================
# Sender task — the only coroutine that calls ws.send_*
# =============================================================================

async def _sender_task(ws: WebSocket, outbound: asyncio.Queue) -> None:
    """Drain `outbound`. JSON-ish items go via send_text, bytes via send_bytes.

    Receiving `None` means "shut down cleanly".
    """
    try:
        while True:
            message = await outbound.get()
            if message is None:
                return
            if isinstance(message, (bytes, bytearray, memoryview)):
                # Binary = MP3 payload for playback.
                await ws.send_bytes(bytes(message))
            else:
                # Dict/list/anything JSON-serializable.
                await ws.send_text(json.dumps(message))
    except WebSocketDisconnect:
        log.info("sender: client disconnected")
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("sender task crashed")


# =============================================================================
# VAD event handling (speech_start, turn_end -> maybe spawn pipeline)
# =============================================================================

async def _handle_vad_event(
    event: dict,
    outbound: asyncio.Queue,
    turn_files: list[str],
    state: TurnPipelineState,
    history: list[tuple[str, str]],
    anthropic_client: AsyncAnthropic,
) -> None:
    kind = event["type"]

    if kind == "speech_start":
        await outbound.put({"type": "vad", "event": "speech_start"})
        return

    if kind == "turn_end":
        # Phase 3 stopwatch starts HERE: from turn_end until the browser's
        # playback_started lands.
        t0 = time.perf_counter()

        pcm = event["pcm_bytes"]
        duration_s = event["duration_s"]

        # Save every turn's audio, even ones we won't process (useful for
        # debugging missed-turn bugs later).
        utt_path = await _save_turn_wav(pcm, turn_files)
        log.info("turn saved: %s (%.2fs)", utt_path.name, duration_s)

        await outbound.put({
            "type": "vad",
            "event": "turn_end",
            "file": utt_path.name,
            "duration_s": round(duration_s, 3),
        })

        if state.busy:
            # Spec: VAD keeps running while busy, but new turn_ends are
            # dropped. We log it so the latency log can note missed turns.
            state.turns_ignored += 1
            log.info("turn_end ignored — pipeline busy (%s)", utt_path.name)
            return

        # Latch busy BEFORE spawning the task so a second turn_end that
        # arrives on the same loop tick can't double-schedule.
        state.busy = True
        state.turn_task = asyncio.create_task(
            _run_pipeline_task(state, pcm, history, outbound, t0, anthropic_client),
            name="turn-pipeline",
        )
        return

    log.warning("unknown VAD event: %r", event)


# =============================================================================
# Pipeline task wrapper — owns busy/error/watchdog transitions
# =============================================================================

async def _run_pipeline_task(
    state: TurnPipelineState,
    pcm: bytes,
    history: list[tuple[str, str]],
    outbound: asyncio.Queue,
    t0: float,
    anthropic_client: AsyncAnthropic,
) -> None:
    """Run one turn end-to-end. Handles errors; schedules playback watchdog."""
    try:
        result = await run_turn(
            pcm, history, outbound, CONFIG, t0,
            anthropic_client=anthropic_client,
        )
    except asyncio.CancelledError:
        # The connection is tearing down. Don't send over the socket.
        raise
    except Exception as e:
        log.exception("pipeline failed")
        await outbound.put({
            "type": "error",
            "stage": "pipeline",
            "message": f"{type(e).__name__}: {e}",
        })
        state.busy = False
        return

    state.last_result = result
    state.turns_served += 1

    # The MP3 is now in flight to the browser. Start the safety watchdog so
    # that if playback_ended never arrives (e.g. a decode error on the
    # client), the toy still frees itself for the next turn.
    state.playback_watchdog = asyncio.create_task(
        _playback_watchdog(state), name="playback-watchdog"
    )


async def _playback_watchdog(state: TurnPipelineState) -> None:
    try:
        await asyncio.sleep(PLAYBACK_WATCHDOG_S)
    except asyncio.CancelledError:
        return
    if state.busy:
        log.warning("playback watchdog fired after %.0fs — clearing busy",
                    PLAYBACK_WATCHDOG_S)
        state.busy = False
    state.playback_watchdog = None


# =============================================================================
# Browser-reported playback events
# =============================================================================

async def _on_playback_started(
    state: TurnPipelineState, outbound: asyncio.Queue,
) -> None:
    """Called when the browser confirms it has started playing the MP3.

    This is "time to first audio" on the client clock, within one WebSocket
    hop of the server clock. We log a latency breakdown relative to t0.
    """
    result = state.last_result
    if result is None:
        log.warning("playback_started with no last_result — ignoring")
        return

    t_play = time.perf_counter()
    record = _build_latency_record(result, t_play, state.turns_ignored)

    # Keep a durable copy; `logs/` is gitignored.
    await asyncio.to_thread(_append_jsonl, LATENCY_LOG, record)

    # Strip timestamps + text from the version we ship to the browser —
    # the on-page latency table only needs the ms numbers.
    client_msg = {k: v for k, v in record.items()
                  if k in {"type", "mode", "stt_ms", "llm_ttft_ms",
                           "llm_total_ms", "tts_ms", "time_to_first_audio_ms"}}
    await outbound.put(client_msg)


def _on_playback_ended(state: TurnPipelineState) -> None:
    if state.playback_watchdog and not state.playback_watchdog.done():
        state.playback_watchdog.cancel()
    state.playback_watchdog = None
    state.busy = False
    log.info("playback_ended — busy cleared")


def _build_latency_record(
    r: TurnResult, t_play: float, turns_ignored: int,
) -> dict:
    """Turn the raw perf_counter timestamps into a flat ms dict for logging."""
    def ms(t: float) -> float:
        return round((t - r.t0) * 1000.0, 1)
    return {
        "type": "latency",
        "mode": "naive",
        "ts": datetime.now().isoformat(timespec="seconds"),
        "transcript": r.text,
        "language": r.language,
        "answer": r.answer,
        "mp3_bytes": r.mp3_len,
        "stt_ms":        ms(r.t_stt),
        "llm_ttft_ms":   ms(r.t_llm_first),
        "llm_total_ms":  ms(r.t_llm_done),
        "tts_ms":        ms(r.t_tts),
        "audio_sent_ms": ms(r.t_audio_sent),
        "time_to_first_audio_ms": ms(t_play),
        "turns_ignored_before": turns_ignored,
    }


def _append_jsonl(path: Path, record: dict) -> None:
    """Blocking file append — call via asyncio.to_thread."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =============================================================================
# WAV writing (per-turn + session)
# =============================================================================

async def _save_turn_wav(pcm: bytes, turn_files: list[str]) -> Path:
    """Save a per-turn .wav using a millisecond-stamped name."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-") + \
         f"{int(time.time() * 1000) % 1000:03d}"
    utt_path = RECORDINGS_DIR / f"utt-{ts}.wav"
    await asyncio.to_thread(_write_wav, utt_path, pcm)
    turn_files.append(utt_path.name)
    return utt_path


async def _finalize_recording(
    audio_buffer: bytearray,
    chunk_count: int,
    arrivals: list[float],
    turn_files: list[str],
    vad_stats: dict,
    pipeline_state: TurnPipelineState,
) -> dict:
    total_bytes = len(audio_buffer)
    duration_s = total_bytes / BYTES_PER_SECOND if BYTES_PER_SECOND else 0.0

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
        "turns_served": pipeline_state.turns_served,
        "turns_ignored": pipeline_state.turns_ignored,
        "vad": vad_stats,
    }


def _write_wav(path: Path, pcm_bytes: bytes) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


# =============================================================================
# Static files — must be mounted LAST so /ws still takes precedence.
# =============================================================================
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

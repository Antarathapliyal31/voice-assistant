"""
server/vad.py — turn detection with Silero VAD.

This file has NO WebSocket code on purpose. It takes raw PCM bytes in and
returns events out. That keeps it unit-testable (see tests/test_vad.py) and
lets Phase 3 (STT) swap in without touching this logic.

Terminology:
  "chunk"   — a 20 ms (320 samples, 640 bytes) Int16 blob that arrives from
              the browser. This is what feed() accepts.
  "window"  — a 32 ms (512 samples, 1024 bytes) Int16 blob that we score
              with the VAD model. Silero v5 at 16 kHz requires *exactly*
              this length — see the probe in requirements.txt notes.

The feed() method accumulates chunks into windows (they don't line up:
every 32 ms of audio = 5/8 of a chunk short of two chunks), scores each
window, and advances a small state machine. On interesting transitions it
appends an event dict to the returned list.
"""

from __future__ import annotations

import math
import time
from collections import deque
from enum import Enum
from typing import Any

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Tunables. Lift to a config file later if we ever need per-env tuning.
# -----------------------------------------------------------------------------
START_THRESHOLD = 0.5    # VAD score above this is considered speech
END_THRESHOLD   = 0.35   # VAD score below this is considered silence
                         # (gap between the two = hysteresis — prevents flapping)
MIN_SPEECH_MS   = 250    # need this much consecutive speech to confirm a start
MIN_SILENCE_MS  = 900    # need this much consecutive silence to confirm an end
PREROLL_MS      = 200    # keep this much pre-speech audio so the first syllable
                         # isn't cut off by the "min speech" latency
TRAILING_KEEP_MS = 100   # on turn_end, strip the long trailing silence that
                         # MIN_SILENCE_MS added — keep only this much so the
                         # last word's release isn't clipped

# -----------------------------------------------------------------------------
# Fixed facts about the model / audio format. Not tunable.
# -----------------------------------------------------------------------------
SAMPLE_RATE     = 16_000                      # Hz
SAMPLE_WIDTH    = 2                           # bytes per Int16 sample
WINDOW_SAMPLES  = 512                         # Silero v5 at 16 kHz: must be exactly 512
WINDOW_BYTES    = WINDOW_SAMPLES * SAMPLE_WIDTH   # 1024
WINDOW_MS       = 1000 * WINDOW_SAMPLES / SAMPLE_RATE   # 32.0

# Convert ms into a count of 32-ms windows, rounding UP so we never emit before
# we've actually seen the configured amount of audio.
def _ms_to_windows(ms: int) -> int:
    return max(1, math.ceil(ms / WINDOW_MS))

MIN_SPEECH_WINDOWS  = _ms_to_windows(MIN_SPEECH_MS)
MIN_SILENCE_WINDOWS = _ms_to_windows(MIN_SILENCE_MS)
PREROLL_WINDOWS     = _ms_to_windows(PREROLL_MS)


class State(Enum):
    SILENT     = "silent"      # not currently in an utterance
    SPEAKING   = "speaking"    # confirmed inside an utterance, voice present
    MAYBE_DONE = "maybe_done"  # inside an utterance but might be at the end


class TurnDetector:
    """
    Feed Int16 PCM in via feed(pcm_bytes). Get a list of events back.

    Events:
      {"type": "speech_start"}
      {"type": "turn_end", "pcm_bytes": <bytes>, "duration_s": <float>}

    One instance per WebSocket connection. Do NOT share across connections —
    the underlying Silero model holds hidden state inside itself.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        # Fresh connection = fresh hidden state.
        model.reset_states()

        self._state: State = State.SILENT

        # Byte buffer of leftover samples that haven't yet filled a 512-sample
        # window. Chunks arrive at 640 bytes but windows need 1024, so there's
        # always some remainder waiting for the next chunk.
        self._leftover = bytearray()

        # Rolling ring of recently seen windows (each window = 1024 bytes).
        # Sized so that when we transition SILENT -> SPEAKING, flushing the
        # whole ring gives us:
        #     PREROLL_WINDOWS of pre-speech audio
        #   + MIN_SPEECH_WINDOWS of the speech that just confirmed the start.
        # Together: the first syllable is never clipped.
        self._recent_windows: deque[bytes] = deque(
            maxlen=PREROLL_WINDOWS + MIN_SPEECH_WINDOWS
        )

        # Counters driving the state machine.
        self._speech_run = 0   # consecutive above-threshold windows (used in SILENT)
        self._silence_run = 0  # consecutive below-threshold windows (used in MAYBE_DONE)

        # Audio captured for the current utterance. None when we're SILENT.
        self._utt: bytearray | None = None

        # Timing stats for VAD model calls (ms per call). Running stats so we
        # don't grow an unbounded list for long sessions.
        self._calls = 0
        self._total_ms = 0.0
        self._max_ms = 0.0

    # ------------------------------------------------------------------ query
    @property
    def state(self) -> str:
        return self._state.value

    def stats(self) -> dict:
        avg = self._total_ms / self._calls if self._calls else 0.0
        return {
            "calls": self._calls,
            "avg_ms": round(avg, 3),
            "max_ms": round(self._max_ms, 3),
        }

    # ------------------------------------------------------------------- feed
    def feed(self, pcm_bytes: bytes) -> list[dict]:
        """Accept a chunk of Int16 PCM (any length). Return any events it caused."""
        events: list[dict] = []
        self._leftover.extend(pcm_bytes)

        # Peel off as many 1024-byte windows as the leftover will allow.
        while len(self._leftover) >= WINDOW_BYTES:
            window_bytes = bytes(self._leftover[:WINDOW_BYTES])
            del self._leftover[:WINDOW_BYTES]

            # Int16 -> float32 in [-1, 1]. 32768 (not 32767) because that's the
            # magnitude of the smallest negative Int16, giving a symmetric map.
            samples_i16 = np.frombuffer(window_bytes, dtype=np.int16)
            samples_f32 = samples_i16.astype(np.float32) / 32768.0
            tensor = torch.from_numpy(samples_f32)

            # Time the model call so we can report avg/max per Phase 2 spec.
            t0 = time.perf_counter()
            score = float(self._model(tensor, SAMPLE_RATE).item())
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._calls += 1
            self._total_ms += dt_ms
            if dt_ms > self._max_ms:
                self._max_ms = dt_ms

            # Every window goes into the preroll ring so it's available if
            # speech starts shortly after.
            self._recent_windows.append(window_bytes)
            # If we're already inside an utterance, the window is part of it.
            if self._utt is not None:
                self._utt.extend(window_bytes)

            self._step_state_machine(score, events)

        return events

    # --------------------------------------------------------- state machine
    def _step_state_machine(self, score: float, events: list[dict]) -> None:
        """Advance one step. Append any emitted event to `events`."""
        if self._state is State.SILENT:
            # Count consecutive above-threshold windows; any dip resets.
            if score >= START_THRESHOLD:
                self._speech_run += 1
                if self._speech_run >= MIN_SPEECH_WINDOWS:
                    # Confirmed start. The deque at this moment contains
                    # exactly PREROLL + MIN_SPEECH windows (older evicted),
                    # which is what we want to seed the utterance with.
                    self._utt = bytearray()
                    for w in self._recent_windows:
                        self._utt.extend(w)
                    self._state = State.SPEAKING
                    self._speech_run = 0
                    events.append({"type": "speech_start"})
            else:
                self._speech_run = 0

        elif self._state is State.SPEAKING:
            # Any clear dip into silence tentatively ends the turn.
            # Scores in the hysteresis band (END..START) or above keep us here.
            if score < END_THRESHOLD:
                self._state = State.MAYBE_DONE
                self._silence_run = 1

        else:  # MAYBE_DONE
            if score >= START_THRESHOLD:
                # Confident speech again — it was just a short pause.
                self._state = State.SPEAKING
                self._silence_run = 0
            elif score < END_THRESHOLD:
                # Still silent, inch toward confirming the turn ended.
                self._silence_run += 1
                if self._silence_run >= MIN_SILENCE_WINDOWS:
                    assert self._utt is not None
                    utt_bytes = self._finalize_utterance()
                    duration_s = len(utt_bytes) / (SAMPLE_RATE * SAMPLE_WIDTH)
                    events.append({
                        "type": "turn_end",
                        "pcm_bytes": utt_bytes,
                        "duration_s": duration_s,
                    })
            # else: score is in the hysteresis band — stay MAYBE_DONE,
            # don't bump the silence counter.

    # ------------------------------------------------------------------- flush
    def flush(self) -> list[dict]:
        """If an utterance is in progress, finalize it and return turn_end.

        Called from main.py when the client sends "stop" so a half-spoken
        question isn't lost (we still save a .wav for whatever we captured).
        """
        if self._state is State.SILENT or self._utt is None:
            return []
        utt_bytes = self._finalize_utterance()
        duration_s = len(utt_bytes) / (SAMPLE_RATE * SAMPLE_WIDTH)
        return [{"type": "turn_end", "pcm_bytes": utt_bytes, "duration_s": duration_s}]

    # ---------------------------------------------------------------- helpers
    def _finalize_utterance(self) -> bytes:
        """Snap the utterance buffer, trim the long trailing silence, and reset.

        Must only be called when self._utt is not None. We trim here (not at
        the point of emission) so flush() can reuse the exact same logic.
        """
        assert self._utt is not None
        utt_bytes = self._trim_trailing_silence(bytes(self._utt))
        # Reset for the next turn.
        self._state = State.SILENT
        self._utt = None
        self._silence_run = 0
        self._speech_run = 0
        return utt_bytes

    def _trim_trailing_silence(self, utt_bytes: bytes) -> bytes:
        """Keep only ~TRAILING_KEEP_MS of the trailing silence at the end.

        Normal turn_end: MIN_SILENCE_WINDOWS of silence are at the tail (the
        counter had to reach that to fire). We keep KEEP windows and trim the
        rest. flush() case: self._silence_run is whatever we had so far.
        """
        keep_windows = max(1, math.ceil(TRAILING_KEEP_MS / WINDOW_MS))
        trim_windows = max(0, self._silence_run - keep_windows)
        trim_bytes = trim_windows * WINDOW_BYTES
        if trim_bytes <= 0 or trim_bytes >= len(utt_bytes):
            return utt_bytes
        return utt_bytes[:-trim_bytes]

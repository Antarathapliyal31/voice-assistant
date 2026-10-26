"""
tests/test_vad.py — unit tests for TurnDetector.

Run with:
    pytest tests/
    TEST_WAV=recordings/<file>.wav pytest tests/
"""
import os
import wave
from pathlib import Path

import pytest

from server.vad import SAMPLE_RATE, TurnDetector

# Browser worklet uses this chunk size. We feed in the same shape here so
# the test mirrors real traffic (640 bytes = 320 Int16 samples = 20 ms).
CHUNK_BYTES = 640


@pytest.fixture(scope="module")
def vad_model():
    """Load Silero once per test module — load_silero_vad is ~380 ms."""
    from silero_vad import load_silero_vad
    return load_silero_vad(onnx=True)


def _feed_in_chunks(detector: TurnDetector, pcm: bytes) -> list[dict]:
    """Feed `pcm` in 20 ms slices and collect every event emitted."""
    events = []
    for i in range(0, len(pcm), CHUNK_BYTES):
        events.extend(detector.feed(pcm[i : i + CHUNK_BYTES]))
    return events


def test_silence_produces_no_events(vad_model):
    """3 seconds of exact zeros should never trip speech_start."""
    detector = TurnDetector(vad_model)
    silence = b"\x00\x00" * (SAMPLE_RATE * 3)   # 3 s of Int16 silence
    events = _feed_in_chunks(detector, silence)
    assert events == [], f"unexpected events on pure silence: {events}"


def test_wav_produces_events(vad_model):
    """A recorded utterance should trigger at least one start + one end.

    Point TEST_WAV at any mono 16 kHz 16-bit .wav in recordings/.
    """
    path_str = os.environ.get("TEST_WAV")
    if not path_str:
        pytest.skip("set TEST_WAV=path/to/sample.wav to run this test")

    path = Path(path_str)
    assert path.exists(), f"TEST_WAV points to a missing file: {path}"

    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == 16_000, "expected 16 kHz"
        assert wf.getnchannels() == 1, "expected mono"
        assert wf.getsampwidth() == 2, "expected 16-bit"
        pcm = wf.readframes(wf.getnframes())

    detector = TurnDetector(vad_model)
    events = _feed_in_chunks(detector, pcm)

    starts = [e for e in events if e["type"] == "speech_start"]
    ends   = [e for e in events if e["type"] == "turn_end"]

    assert starts, f"expected >=1 speech_start, got: {[e['type'] for e in events]}"
    assert ends,   f"expected >=1 turn_end,   got: {[e['type'] for e in events]}"

    # Sanity: every turn_end should carry non-empty utterance bytes.
    for e in ends:
        assert isinstance(e["pcm_bytes"], (bytes, bytearray))
        assert len(e["pcm_bytes"]) > 0
        assert e["duration_s"] > 0

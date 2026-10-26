# voice-assistant

A small realtime voice assistant for kids — learning project.

**Phase 1:** stream mic audio from a Chrome tab to a Python/FastAPI backend
over a WebSocket and save it as a `.wav`. No VAD, no STT, no LLM, no TTS yet.

## Project layout

```
server/main.py        FastAPI app + /ws WebSocket + WAV writer
web/index.html        Start/Stop UI, mic capture, WebSocket client
web/mic-worklet.js    AudioWorklet: Float32 → Int16, 20 ms chunks
recordings/           saved .wav files (gitignored)
requirements.txt      Python deps
```

## Running it

Requires **Python 3.11+** and **Chrome**.

```bash
# 1) Create and activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run the server
uvicorn server.main:app --reload --port 8000
```

Then open <http://localhost:8000> in Chrome, click **Start**, speak, click
**Stop**. The server writes `recordings/<timestamp>.wav` and sends a JSON
summary back which shows up in the on-page log.

## What the summary means

After you click Stop you'll see something like:

```json
{
  "type": "summary",
  "file": "20261025-153012.wav",
  "chunks": 150,
  "total_bytes": 96000,
  "duration_s": 3.0,
  "avg_gap_ms": 20.01,
  "max_gap_ms": 21.3
}
```

- `chunks`: number of 20 ms binary frames received.
- `avg_gap_ms` / `max_gap_ms`: inter-arrival time between chunks. Should sit
  near 20 ms. A big `max_gap_ms` means the browser or socket stalled — useful
  to watch once we start piping audio to a streaming STT.

## Why `http://localhost` specifically

Browsers only allow `getUserMedia` on secure origins. `http://localhost` is
treated as secure, so it works without TLS. A LAN IP (e.g. `http://192.168.…`)
would require HTTPS.

## Phase 3 — naive STT → LLM → TTS pipeline

The naive baseline runs stages strictly sequentially: STT must finish before
the LLM starts, LLM must finish before TTS starts. Phase 4 will overlap them;
these numbers are what we're trying to beat.

### Sample run (2026-10-25, M-series laptop)

**Turn 1 — "Why is the sky blue?"**

| Stage | ms from `turn_end` |
|---|--:|
| STT | 544 |
| LLM first token | 1016 |
| LLM full answer | 1342 |
| TTS | 1813 |
| **Time to first audio** | **1836** |

**Turn 2 — follow-up: "The sunlight bounces around in the air. What does that mean?"**

| Stage | ms from `turn_end` |
|---|--:|
| STT | 1204 |
| LLM first token | 1714 |
| LLM full answer | 2500 |
| TTS | 3312 |
| **Time to first audio** | **3356** |

The ~1.8–3.4 s first-audio delay is the "slow baseline" we expect: everything
is serialized, and the TTS can't begin until Claude's last token has arrived.

### Example turn (from the on-page log)

```
server: {"type": "vad", "event": "speech_start"}
server: {"type": "vad", "event": "turn_end",
         "file": "utt-20261025-233904-070.wav", "duration_s": 3.136}
server: {"type": "transcript", "text": "Why is the sky blue?", "language": "eng"}
You:    Why is the sky blue?  (lang=eng)
server: {"type": "answer", "text": "The sky is blue because of how sunlight
         bounces around in the air! The blue light travels in tiny waves that
         spread out more than other colors, so we see lots of blue all around us."}
Toy:    The sky is blue because of how sunlight bounces around in the air! ...
server: <mp3 189796 bytes>
server: {"type": "latency", "mode": "naive",
         "stt_ms": 603.8, "llm_ttft_ms": 1077.9, "llm_total_ms": 1542.6,
         "tts_ms": 2390.6, "time_to_first_audio_ms": 2426.5}
playback ended
```

Each turn is also appended to `logs/latency.jsonl` (gitignored) with the full
transcript, answer, and timestamp — useful for comparing against Phase 4
later.

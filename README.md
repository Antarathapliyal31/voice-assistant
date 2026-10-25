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

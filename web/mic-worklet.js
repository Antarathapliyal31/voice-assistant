// mic-worklet.js
//
// An AudioWorkletProcessor runs in the audio rendering thread, NOT the main
// JavaScript thread. That's a hard real-time context: the browser calls
// process() every "render quantum" (128 samples by default), and if we take
// too long or allocate aggressively, we get audio glitches. So the code here
// must stay tight — no logging, minimal allocation per call.
//
// Job of this worklet:
//   1. Receive Float32 PCM samples from the mic (range roughly [-1, 1]).
//   2. Convert them to signed 16-bit integers (Int16).
//   3. Accumulate 320 samples at a time (= 20 ms at 16 kHz) and post each
//      chunk to the main thread, which forwards it over the WebSocket.
//
// Why 20 ms? Because downstream VAD and streaming STT engines expect frames
// in the 10–40 ms range. 20 ms is the sweet spot and matches what we'll
// want in Phase 2. Keeping the chunking in the worklet means the main
// thread only ever sees fixed-size payloads.

const CHUNK_SAMPLES = 320; // 320 samples / 16000 Hz = 0.020 s = 20 ms

class MicChunker extends AudioWorkletProcessor {
  constructor() {
    super();
    // Pre-allocate the accumulation buffer. We write into this one Int16Array
    // across many process() calls and only allocate a fresh buffer when we
    // actually have a full chunk ready to post. This keeps GC pressure low.
    this._chunk = new Int16Array(CHUNK_SAMPLES);
    this._offset = 0; // how many samples are currently in _chunk
  }

  // process() is called by the audio engine. Returning true keeps the node
  // alive; returning false lets the engine garbage-collect it.
  //
  // inputs is a 2D array: inputs[nodeInputIndex][channelIndex] => Float32Array.
  // We only connected one mono input, so we read inputs[0][0].
  process(inputs /*, outputs, parameters */) {
    const channel = inputs[0] && inputs[0][0];

    // When the audio graph is still warming up (or momentarily starved) the
    // channel can be undefined or zero-length. Just stay alive and wait.
    if (!channel || channel.length === 0) return true;

    for (let i = 0; i < channel.length; i++) {
      // Clamp to [-1, 1] defensively — getUserMedia can occasionally emit
      // samples slightly outside that range, and that would wrap around in
      // the Int16 cast and sound like a loud click.
      let s = channel[i];
      if (s > 1) s = 1;
      else if (s < -1) s = -1;

      // 32767 is the max positive Int16. Using that (not 32768) keeps the
      // positive/negative mapping symmetric. The | 0 forces integer truncation.
      this._chunk[this._offset++] = (s * 32767) | 0;

      if (this._offset === CHUNK_SAMPLES) {
        // The chunk is full. Build a fresh buffer and copy into it so we can
        // transfer ownership to the main thread (second arg to postMessage).
        // Transfer avoids a copy at the thread boundary — the main thread
        // becomes the sole owner of that ArrayBuffer.
        const out = new Int16Array(CHUNK_SAMPLES);
        out.set(this._chunk);
        this.port.postMessage(out.buffer, [out.buffer]);
        this._offset = 0;
      }
    }

    return true; // keep the processor alive
  }
}

// The name "mic-chunker" is what the main thread will use when it does
// `new AudioWorkletNode(audioContext, "mic-chunker")`.
registerProcessor("mic-chunker", MicChunker);

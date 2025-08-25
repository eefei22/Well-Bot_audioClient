# client/audio.py
from __future__ import annotations
"""
Audio I/O for the Python client.

- MicStreamer: capture mono PCM16 frames (e.g., 20 ms @ 16 kHz) and expose an
  async iterator of bytes suitable for sending over a WebSocket.
- Speaker: prebuffered PCM16 playback with a jitter buffer to smooth network
  variability and avoid underruns.

Both classes are careful about crossing threads: sounddevice callbacks run on
PortAudio threads, while our app is asyncio-based. We bridge via thread-safe
queues and `loop.call_soon_threadsafe(...)`.

Assumptions:
- Mono, PCM16 little-endian
- Single shared event loop (the one that starts the client)
"""

import asyncio
import threading
import queue
from typing import AsyncIterator, Optional

import sounddevice as sd

BYTES_PER_SAMPLE = 2  # PCM16
MONO_CHANNELS = 1


def _frame_bytes(sample_rate: int, frame_ms: int, channels: int = MONO_CHANNELS) -> int:
    samples = int(round(sample_rate * (frame_ms / 1000.0)))
    return samples * BYTES_PER_SAMPLE * channels


class MicStreamer:
    """
    Capture microphone audio as fixed-size PCM16 frames.

    Usage:
        mic = MicStreamer(sample_rate=16000, frame_ms=20, device=None)
        await mic.start()
        async for frame in mic.frames():
            await ws.send(frame)
        await mic.stop()
    """

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 20, device: Optional[int | str] = None):
        if frame_ms not in (10, 20, 30, 40, 60):
            raise ValueError("frame_ms must be one of: 10, 20, 30, 40, 60")
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.device = device

        self.blocksize_bytes = _frame_bytes(sample_rate, frame_ms)
        self.blocksize_frames = self.blocksize_bytes // BYTES_PER_SAMPLE  # mono

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional[asyncio.Queue[bytes]] = None
        self._stream: Optional[sd.RawInputStream] = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=256)
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=MONO_CHANNELS,
            dtype="int16",
            blocksize=self.blocksize_frames,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        self._running = True

    async def stop(self) -> None:
        self._running = False
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
        finally:
            self._stream = None

    def _callback(self, indata, frames, time_info, status):  # called by PortAudio thread
        if status:
            # Status flags include input overflow, etc. We ignore for MVP.
            pass
        if not self._queue or not self._loop:
            return
        # `indata` is a bytes-like object of length frames*channels*2
        data = bytes(indata)  # copy out of PortAudio's buffer
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, data)
        except Exception:
            # If queue is full, drop the frame (better than blocking audio thread)
            pass

    async def frames(self) -> AsyncIterator[bytes]:
        """
        Async iterator of PCM16 frames. Cancels gracefully when `stop()` is called.
        """
        if not self._queue:
            raise RuntimeError("MicStreamer not started")
        while self._running:
            try:
                frame = await self._queue.get()
            except asyncio.CancelledError:
                break
            yield frame


class Speaker:
    """
    Prebuffered PCM16 playback with a jitter buffer.

    Design:
      - We own a `RawOutputStream` with a callback that pulls from an internal
        byte buffer (`_buf`) protected by a lock.
      - `enqueue()` appends PCM bytes (any size); the callback slices exactly the
        required number of bytes for each callback period and fills with silence
        on underrun.
      - We "prime" playback by waiting until `_buf` length >= `prebuffer_bytes`
        before outputting audio (silence until then).

    Usage:
        spk = Speaker(sample_rate=24000, prebuffer_ms=120)
        spk.start()
        await spk.enqueue(pcm_bytes)
        ...
        await spk.stop()
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        prebuffer_ms: int = 120,
        device: Optional[int | str] = None,
        block_ms: int = 20,
    ):
        if block_ms not in (10, 20, 30, 40, 60):
            raise ValueError("block_ms must be one of: 10, 20, 30, 40, 60")
        self.sample_rate = sample_rate
        self.device = device
        self.block_bytes = _frame_bytes(sample_rate, block_ms)
        self.prebuffer_bytes = _frame_bytes(sample_rate, prebuffer_ms)

        self._stream: Optional[sd.RawOutputStream] = None
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._primed = False
        self._stopping = False

        # A small backpressure queue to receive chunks from async code,
        # decoupling network and audio-thread timing.
        self._ingest_q: "queue.Queue[bytes]" = queue.Queue(maxsize=512)
        self._ingest_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._stream:
            return
        # Writer thread: drains _ingest_q into _buf
        def _ingest_loop():
            while not self._stopping:
                try:
                    chunk = self._ingest_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if not chunk:
                    continue
                with self._lock:
                    self._buf.extend(chunk)
                    if not self._primed and len(self._buf) >= self.prebuffer_bytes:
                        self._primed = True

        self._ingest_thread = threading.Thread(target=_ingest_loop, name="speaker-ingest", daemon=True)
        self._ingest_thread.start()

        self._stream = sd.RawOutputStream(
            samplerate=self.sample_rate,
            channels=MONO_CHANNELS,
            dtype="int16",
            blocksize=self.block_bytes // BYTES_PER_SAMPLE,  # frames per block
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, outdata, frames, time_info, status):  # PortAudio thread
        if status:
            # Output underflow/overflow indicators; ignore for MVP.
            pass
        need = frames * BYTES_PER_SAMPLE * MONO_CHANNELS
        # Default to silence
        out_mv = memoryview(outdata)
        written = 0

        with self._lock:
            if self._primed and len(self._buf) >= need:
                out_mv[:need] = self._buf[:need]
                del self._buf[:need]
                written = need

        if written < need:
            # Not enough data yet (or not primed) => fill remainder with silence
            out_mv[written:need] = b"\x00" * (need - written)

    async def enqueue(self, pcm_bytes: bytes) -> None:
        """
        Queue PCM to the player. Drops oldest data if the queue is full.
        """
        if not pcm_bytes:
            return
        if not self._stream:
            raise RuntimeError("Speaker not started")
        try:
            self._ingest_q.put_nowait(pcm_bytes)
        except queue.Full:
            # Drop-oldest: pull and discard one, then retry once
            try:
                _ = self._ingest_q.get_nowait()
                self._ingest_q.put_nowait(pcm_bytes)
            except Exception:
                pass

    async def stop(self) -> None:
        self._stopping = True
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
        finally:
            self._stream = None
        if self._ingest_thread and self._ingest_thread.is_alive():
            self._ingest_thread.join(timeout=1.0)
            self._ingest_thread = None

    def reset(self) -> None:
        """
        Clear buffered audio and reset priming (useful on barge-in).
        """
        with self._lock:
            self._buf.clear()
            self._primed = False

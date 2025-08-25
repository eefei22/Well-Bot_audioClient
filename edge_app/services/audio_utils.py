# edge_app/services/audio_utils.py
from __future__ import annotations
"""
Audio utilities for 16-bit PCM streams.

- PCM16 framing helpers (bytes <-> frames)
- Simple gain/ducking on int16
- A tiny accumulator to slice arbitrary byte streams into fixed-size frames

All functions assume **mono** PCM16 little-endian unless noted.
"""

import math
from typing import Iterable, Iterator, Optional

import numpy as np


INT16_MAX = 32767
INT16_MIN = -32768
BYTES_PER_SAMPLE = 2  # PCM16
MONO_CHANNELS = 1


# -------------------------
# Framing helpers
# -------------------------

def frame_n_samples(sample_rate_hz: int, frame_ms: int) -> int:
    """Samples per frame for the given duration (rounded to nearest)."""
    return int(round(sample_rate_hz * (frame_ms / 1000.0)))


def frame_n_bytes(sample_rate_hz: int, frame_ms: int, channels: int = MONO_CHANNELS) -> int:
    """Bytes per frame of PCM16."""
    return frame_n_samples(sample_rate_hz, frame_ms) * BYTES_PER_SAMPLE * channels


class PCMFrameAccumulator:
    """
    Accumulate arbitrary PCM16 byte chunks and iterate fixed-size frames.

    Example:
        acc = PCMFrameAccumulator(frame_bytes=640)  # 20ms @ 16kHz mono PCM16
        acc.push(chunk1); acc.push(chunk2)
        for frame in acc.pull_frames():
            ... use 640-byte frames ...
        # remainder (if any) stays buffered until more bytes arrive
    """

    __slots__ = ("frame_bytes", "_buf")

    def __init__(self, frame_bytes: int):
        if frame_bytes <= 0:
            raise ValueError("frame_bytes must be > 0")
        self.frame_bytes = frame_bytes
        self._buf = bytearray()

    def reset(self) -> None:
        self._buf.clear()

    def push(self, data: bytes) -> None:
        if not data:
            return
        self._buf.extend(data)

    def pull_frames(self) -> Iterator[bytes]:
        fb = self.frame_bytes
        b = self._buf
        total = len(b)
        n_full = total // fb
        if n_full == 0:
            return iter(())
        end = n_full * fb
        # Slice views without copying, then copy for isolation
        mv = memoryview(b)[:end]
        for i in range(0, end, fb):
            yield bytes(mv[i : i + fb])
        # Keep leftover
        del b[:end]


# -------------------------
# PCM16 gain / ducking
# -------------------------

def pcm16_apply_gain(pcm_bytes: bytes, gain_db: float) -> bytes:
    """
    Apply a linear gain to PCM16 samples, clamped to int16 range.

    gain_db: positive to boost, negative to attenuate (e.g. -18 dB for ducking).
    """
    if not pcm_bytes or abs(gain_db) < 1e-6:
        return pcm_bytes
    # Convert to float32 in [-1, 1], apply gain, clamp, back to int16
    scale = 10.0 ** (gain_db / 20.0)
    x = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32)
    x *= scale
    np.clip(x, INT16_MIN, INT16_MAX, out=x)
    return x.astype("<i2", copy=False).tobytes()


def pcm16_rms(pcm_bytes: bytes) -> float:
    """Root-mean-square (RMS) amplitude of PCM16 block (0..1 range)."""
    if not pcm_bytes:
        return 0.0
    x = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32)
    x /= INT16_MAX
    return float(np.sqrt(np.mean(np.square(x))))


# -------------------------
# Simple resample (optional)
# -------------------------

def resample_linear_mono(pcm_bytes: bytes, src_rate: int, dst_rate: int) -> bytes:
    """
    Naïve linear resampler for mono PCM16. Good enough for MVP; swap for
    high-quality resamplers (e.g., soxr) later if you need it.

    If rates are equal, returns input unchanged.
    """
    if src_rate == dst_rate or not pcm_bytes:
        return pcm_bytes

    x = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32)

    # Normalize to [-1, 1]
    x /= INT16_MAX

    ratio = dst_rate / float(src_rate)
    n_out = int(math.floor(len(x) * ratio))
    if n_out <= 0:
        return b""

    # Linear interpolation
    src_idx = np.arange(n_out, dtype=np.float32) / ratio
    idx0 = np.floor(src_idx).astype(np.int32)
    idx1 = np.minimum(idx0 + 1, len(x) - 1)
    frac = src_idx - idx0

    y = (1.0 - frac) * x[idx0] + frac * x[idx1]
    # Back to int16
    y = np.clip(y * INT16_MAX, INT16_MIN, INT16_MAX).astype("<i2", copy=False)
    return y.tobytes()

# client/vad.py
from __future__ import annotations
"""
Voice Activity Detection (VAD) wrapper for barge-in.

- Uses Google's WebRTC VAD via the `webrtcvad` Python package.
- Accepts **mono PCM16** frames of **exactly** 10, 20, or 30 ms at 8/16/32/48 kHz.
- Provides:
    * VADDetector.process_frame(pcm_bytes) -> (is_speech: bool, event: str|None)
    * VADDetector.events_from_frames(async_iter) -> async iterator of {'type': 'speech_start'|'speech_end'}

Why the constraints?
The WebRTC VAD requires 16-bit mono PCM at 8000, 16000, 32000 or 48000 Hz,
and **frame durations of 10/20/30 ms**. Other sizes will raise errors.  # refs:
# - https://github.com/wiseman/py-webrtcvad (README)
# - https://pypi.org/project/webrtcvad-wheels/  (usage notes)
"""

from typing import AsyncIterator, Optional, Tuple

import webrtcvad


ALLOWED_FRAME_MS = (10, 20, 30)
ALLOWED_SAMPLE_RATES = (8000, 16000, 32000, 48000)


class VADDetector:
    """
    WebRTC VAD with simple hysteresis to avoid chattiness.
    - `aggressiveness` (0..3): higher means stricter speech detection (fewer false positives).
    - `start_trigger_frames`: consecutive speech frames to emit 'speech_start'.
    - `end_trigger_frames`: consecutive non-speech frames to emit 'speech_end'.

    Example:
        vad = VADDetector(sample_rate=16000, frame_ms=20, aggressiveness=2)
        for frame in mic_frames:
            is_speech, event = vad.process_frame(frame)
            if event == 'speech_start': ...
            if event == 'speech_end':   ...
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        aggressiveness: int = 2,
        start_trigger_frames: int = 3,  # ~60ms at 20ms frames
        end_trigger_frames: int = 6,    # ~120ms at 20ms frames
    ) -> None:
        if frame_ms not in ALLOWED_FRAME_MS:
            raise ValueError(f"frame_ms must be one of {ALLOWED_FRAME_MS}")
        if sample_rate not in ALLOWED_SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {ALLOWED_SAMPLE_RATES}")
        if not (0 <= aggressiveness <= 3):
            raise ValueError("aggressiveness must be in 0..3")
        if start_trigger_frames <= 0 or end_trigger_frames <= 0:
            raise ValueError("trigger frame counts must be > 0")

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.vad = webrtcvad.Vad(aggressiveness)

        self.start_trigger_frames = start_trigger_frames
        self.end_trigger_frames = end_trigger_frames

        # State
        self._in_speech = False
        self._consecutive_speech = 0
        self._consecutive_silence = 0

        # Precompute expected frame byte length: samples * 2 bytes (mono)
        samples_per_frame = int(round(self.sample_rate * (self.frame_ms / 1000.0)))
        self.expected_frame_bytes = samples_per_frame * 2  # 16-bit mono

    def reset(self) -> None:
        self._in_speech = False
        self._consecutive_speech = 0
        self._consecutive_silence = 0

    def process_frame(self, pcm16_frame: bytes) -> Tuple[bool, Optional[str]]:
        """
        Process one frame of PCM16 audio.

        Returns:
            (is_speech, event)
            - is_speech: True/False for this frame (raw VAD decision)
            - event: 'speech_start', 'speech_end', or None
        """
        if len(pcm16_frame) != self.expected_frame_bytes:
            # The native VAD will throw if the frame length is invalid; we guard earlier for clarity.
            raise ValueError(
                f"Invalid frame length: got {len(pcm16_frame)} bytes, expected {self.expected_frame_bytes} "
                f"for {self.frame_ms}ms @ {self.sample_rate}Hz mono PCM16"
            )

        is_speech = False
        try:
            is_speech = self.vad.is_speech(pcm16_frame, self.sample_rate)
        except Exception:
            # If the underlying VAD raises (rare), treat as non-speech to be safe.
            is_speech = False

        event: Optional[str] = None

        if is_speech:
            self._consecutive_speech += 1
            self._consecutive_silence = 0
            if not self._in_speech and self._consecutive_speech >= self.start_trigger_frames:
                self._in_speech = True
                event = "speech_start"
        else:
            self._consecutive_silence += 1
            self._consecutive_speech = 0
            if self._in_speech and self._consecutive_silence >= self.end_trigger_frames:
                self._in_speech = False
                event = "speech_end"

        return is_speech, event

    async def events_from_frames(self, frames: AsyncIterator[bytes]) -> AsyncIterator[dict]:
        """
        Async helper that consumes an async iterator of frames and yields
        discrete events for barge-in logic:

            {'type': 'speech_start'} or {'type': 'speech_end'}

        Note: individual frame-level `is_speech` booleans are *not* emitted here.
        """
        async for frame in frames:
            _, event = self.process_frame(frame)
            if event:
                yield {"type": event}

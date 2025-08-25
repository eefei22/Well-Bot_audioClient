# edge_app/services/stt_deepgram.py
from __future__ import annotations

"""
Deepgram STT (Streaming) — Async WebSocket Client

Design goals
------------
- Minimal, dependency-light async client (uses `websockets`).
- Accepts binary PCM16 frames (e.g., 20ms @ 16kHz) via `send_audio()`.
- Yields STTEvent objects through `events()` async iterator:
    * kind="partial":   for interim results (`is_final: false`)
    * kind="final":     for utterance end (`is_final: true` AND `speech_final: true`)
- Resilient to non-JSON and metadata frames; ignores what we don't need.

Notes on Deepgram semantics
---------------------------
- `is_final:true` marks a *stabilized* transcript segment; it does NOT mean end-of-utterance.
- Endpoint (end-of-speech) is indicated by `speech_final:true` (requires endpointing/VAD).
- We only promote a user turn when both `is_final:true` AND `speech_final:true`.

References:
- Streaming listen endpoint & message shapes (Results/Metadata): see Deepgram docs.
- Endpointing vs. is_final distinctions: see “Endpointing & Interim Results” docs.
"""

import asyncio
import json
import urllib.parse
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Optional

import websockets
from websockets.client import WebSocketClientProtocol

from edge_app.config import STTConfig
from edge_app.utils.logging import logger, log_exception
import os


STT_LOG_FRAMES = os.environ.get("STT_LOG_FRAMES", "0").strip().lower() in {"1","true","yes","on"}
STT_LOG_TEXT_CHARS = int(os.environ.get("STT_LOG_TEXT_CHARS", "80"))


# -----------------------------
# Data model
# -----------------------------

@dataclass
class STTEvent:
    """
    Event from the Deepgram stream.

    kind:
        - "partial": interim hypothesis (may be empty)
        - "final":   finalized utterance with endpoint (speech_final)
    text: transcript text
    endpoint_ms: optional latency/duration hint (not guaranteed by API; may be None)
    """
    kind: str
    text: str
    endpoint_ms: Optional[int] = None


# -----------------------------
# Client
# -----------------------------

class DeepgramSTTStream:
    """
    Async client for Deepgram's realtime STT WebSocket.
    """

    def __init__(self, session_id: str, user_id: str, config: STTConfig, headers: Dict[str, str]):
        self.session_id = session_id
        self.user_id = user_id
        self.config = config
        self.headers = headers

        self._ws: Optional[WebSocketClientProtocol] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._queue: asyncio.Queue[STTEvent] = asyncio.Queue(maxsize=1024)
        self._closed = asyncio.Event()

    # ---------- public API ----------

    async def start(self) -> None:
        """
        Open the WS connection to Deepgram and start the receiver loop.
        """
        if self._ws and not self._ws.closed:
            return

        url = self._build_listen_url()
        # Add minimal headers, including Authorization
        headers = dict(self.headers or {})
        headers.setdefault("X-Session-Id", self.session_id)
        headers.setdefault("X-User-Id", self.user_id)

        logger.info(f"[{self.session_id}] STT connecting to {url}")
        try:
            self._ws = await websockets.connect(url, extra_headers=headers, max_size=None, ping_interval=20)
        except Exception as e:
            log_exception(f"[{self.session_id}] STT connect failed", e)
            raise

        # Spawn receiver loop
        self._recv_task = asyncio.create_task(self._receiver_loop(), name=f"stt-recv-{self.session_id}")

    async def stop(self) -> None:
        """
        Close the WS connection (idempotent).
        """
        self._closed.set()
        try:
            if self._ws and not self._ws.closed:
                await self._ws.close()
        except Exception as e:
            log_exception(f"[{self.session_id}] STT close error", e)
        finally:
            self._ws = None

        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            finally:
                self._recv_task = None

    async def send_audio(self, pcm16_bytes: bytes) -> None:
        """
        Send a chunk of PCM16 audio to Deepgram.
        """
        if not self._ws or self._ws.closed:
            return
        try:
            await self._ws.send(pcm16_bytes)
        except Exception as e:
            log_exception(f"[{self.session_id}] STT send audio failed", e)
            # Let the receiver loop surface closure; caller may decide to tear down.

    async def events(self) -> AsyncIterator[STTEvent]:
        """
        Yield STT events until the stream is closed.
        """
        while not self._closed.is_set():
            try:
                ev = await self._queue.get()
                yield ev
            except asyncio.CancelledError:
                break

    # ---------- internals ----------

    def _build_listen_url(self) -> str:
        """
        Compose Deepgram listen URL with query params from config.
        """
        base = "wss://api.deepgram.com/v1/listen"
        qp = {
            "model": self.config.model,
            "language": self.config.language,
            "encoding": self.config.encoding,
            "sample_rate": str(self.config.sample_rate),
            "interim_results": "true" if self.config.interim_results else "false",
            "vad_events": "true" if self.config.vad_events else "false",
            "smart_format": "true" if self.config.smart_format else "false",
        }
        # Endpointing duration in ms (VAD silence to consider end-of-speech)
        if self.config.endpointing_ms and self.config.endpointing_ms > 0:
            qp["endpointing"] = str(self.config.endpointing_ms)

        return f"{base}?{urllib.parse.urlencode(qp)}"

    async def _receiver_loop(self) -> None:
        """
        Receive frames from Deepgram, parse, and enqueue STTEvent objects.
        """
        assert self._ws is not None
        ws = self._ws

        try:
            async for msg in ws:
                # Deepgram sends text JSON for Results/Metadata; binary is not used here.
                if isinstance(msg, (bytes, bytearray)):
                    # Ignore unexpected binary
                    continue

                # Parse JSON safely
                try:
                    data = json.loads(msg)
                    
                    mtype = data.get("type")

                    if mtype == "Results":
                        channel = data.get("channel") or {}
                        alts = channel.get("alternatives") or []
                        transcript = ""
                        if alts and isinstance(alts, list) and isinstance(alts[0], dict):
                            transcript = alts[0].get("transcript", "") or ""

                        is_final = bool(channel.get("is_final"))
                        speech_final = bool(data.get("speech_final"))

                        if STT_LOG_FRAMES and transcript:
                            snippet = transcript if len(transcript) <= STT_LOG_TEXT_CHARS else (transcript[:STT_LOG_TEXT_CHARS] + "…")
                            logger.debug(
                                f"[{self.session_id}] STT Results is_final={is_final} speech_final={speech_final} "
                                f"len={len(transcript)} text='{snippet}'"
                            )
                except Exception:
                    continue

                mtype = data.get("type")
                if mtype == "Results":  # primary transcript payloads
                    # Expected shape:
                    # {
                    #   "type": "Results",
                    #   "channel": { "alternatives": [ { "transcript": "...", ... } ], "is_final": false|true },
                    #   "speech_final": false|true,   # present when endpointing enabled
                    #   ...
                    # }
                    channel = data.get("channel") or {}
                    alts = channel.get("alternatives") or []
                    transcript = ""
                    if alts and isinstance(alts, list) and isinstance(alts[0], dict):
                        transcript = alts[0].get("transcript", "") or ""

                    if transcript == "":
                        continue  # nothing to emit

                    is_final = bool(data.get("is_final"))
                    speech_final = bool(data.get("speech_final"))

                    if not is_final and not speech_final:
                        # interim hypothesis
                        await self._enqueue(STTEvent(kind="partial", text=transcript))
                        continue

                    # is_final == True or speech_final == True
                    if speech_final:
                        # utterance boundary (end-of-speech)
                        await self._enqueue(STTEvent(kind="final", text=transcript, endpoint_ms=None))
                    else:
                        # finalized segment but not end-of-utterance
                        await self._enqueue(STTEvent(kind="partial", text=transcript))
                        
                elif mtype in ("Metadata", "SpeechStarted", "UtteranceEnd"):
                    if STT_LOG_FRAMES:
                        logger.debug(f"[{self.session_id}] STT {mtype}: {data}")
                    continue


                elif mtype in ("Error", "Warning"):
                    # Bubble up via a final event with empty text? Here we just log.
                    logger.warning(f"[{self.session_id}] STT message: {mtype}: {data}")
                    continue

                else:
                    # Unknown message types — ignore
                    continue

        except websockets.ConnectionClosedOK:
            pass
        except websockets.ConnectionClosedError:
            pass
        except Exception as e:
            log_exception(f"[{self.session_id}] STT receiver error", e)
        finally:
            self._closed.set()

    async def _enqueue(self, ev: STTEvent) -> None:
        """
        Non-blocking put with bounded queue; drop oldest on overflow to preserve recency.
        """
        try:
            self._queue.put_nowait(ev)
        except asyncio.QueueFull:
            try:
                _ = self._queue.get_nowait()  # drop oldest
            except Exception:
                pass
            try:
                self._queue.put_nowait(ev)
            except Exception:
                # If still full, drop quietly (better than blocking audio path)
                pass

    async def finalize(self) -> None:
        """Ask Deepgram to finalize any buffered audio (debug/ops)."""
        if self._ws and not self._ws.closed:
            try:
                await self._ws.send(json.dumps({"type": "Finalize"}))
                if STT_LOG_FRAMES:
                    logger.debug(f"[{self.session_id}] STT Finalize sent")
            except Exception as e:
                log_exception(f"[{self.session_id}] STT finalize error", e)

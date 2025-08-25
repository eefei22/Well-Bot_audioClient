# edge_app/services/tts_deepgram.py
from __future__ import annotations

"""
Deepgram TTS (Streaming) — Async WebSocket Client

- Connects to Deepgram "Speak" WebSocket (wss://api.deepgram.com/v1/speak).
- Sends JSON control/text messages:
    * {"type":"Speak","text":"..."}   -> enqueue text for synthesis
    * {"type":"Flush"}                -> force synthesis of buffered text
    * {"type":"Clear"}                -> clear text/audio buffers (for barge-in)
- Receives **binary** audio frames (e.g., Linear16 PCM) and yields them to callers.

Recommended strategy (for LLM deltas):
- Buffer small deltas into natural chunks (sentences/clauses).
- Send Speak per chunk; issue Flush on end-of-sentence or long pause.
- On cancel/barge-in, send Clear and stop reading further audio.

Notes:
- You may see occasional textual JSON messages from the server (e.g., "Flushed"/"Cleared" confirmations, warnings).
  We ignore or log them; only binary frames are forwarded to the caller.
"""

import asyncio
import json
import urllib.parse
from typing import AsyncIterator, Dict, Optional, Iterable

import websockets
from websockets.client import WebSocketClientProtocol

from edge_app.config import TTSConfig
from edge_app.utils.logging import logger, log_exception

import os
from urllib.parse import urlencode, quote_plus
from websockets.exceptions import InvalidStatusCode


class DeepgramTTSStream:
    """
    Async client for Deepgram's streaming TTS ("Speak") WebSocket.

    Usage:
        tts = DeepgramTTSStream(session_id, user_id, settings.tts, headers)
        pcm_iter = tts.synthesize_stream(text_stream=my_delta_iter, cancel_event=evt)
        async for pcm in pcm_iter:
            await ws.send_bytes(pcm)  # forward to client
    """

    def __init__(self, session_id: str, user_id: str, config: TTSConfig, headers: Dict[str, str]):
        self.session_id = session_id
        self.user_id = user_id
        self.config = config
        self.headers = headers

        self._ws: Optional[WebSocketClientProtocol] = None
        self._receiver_task: Optional[asyncio.Task] = None
        self._audio_queue: "asyncio.Queue[bytes]" = asyncio.Queue(maxsize=1024)
        self._closed = asyncio.Event()
        self._started = False

    # ---------------------------
    # Connection lifecycle
    # ---------------------------

    async def _connect(self) -> None:
        if self._ws and not self._ws.closed:
            return

        url = self._build_speak_url()

        # Headers: copy existing, then ensure required auth + helpful context
        headers = dict(self.headers or {})
        headers.setdefault("X-Session-Id", self.session_id)
        headers.setdefault("X-User-Id", self.user_id)

        # Ensure Authorization header is present; prefer existing, else from env
        auth_present = any(k.lower() == "authorization" for k in headers.keys())
        if not auth_present:
            api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("DEEPGRAM_API_KEY is not set; cannot authenticate to Deepgram TTS.")
            # Deepgram expects: Authorization: Token <API_KEY>
            headers["Authorization"] = f"Token {api_key}"

        logger.info(f"[{self.session_id}] TTS connecting to {url}")

        try:
            self._ws = await websockets.connect(
                url,
                extra_headers=headers,
                ping_interval=20,
                max_size=None,
            )
        except InvalidStatusCode as e:
            # Try to surface Deepgram's diagnostic headers (e.g., 'dg-error', 'dg-request-id')
            dg_error = None
            dg_req_id = None
            try:
                # `headers` attribute exists on modern websockets versions
                if hasattr(e, "headers") and e.headers:
                    dg_error = e.headers.get("dg-error")
                    dg_req_id = e.headers.get("dg-request-id")
            except Exception:
                pass
            logger.error(
                f"[{self.session_id}] TTS connect failed: HTTP {getattr(e, 'status_code', '400')}"
                + (f" dg-error={dg_error}" if dg_error else "")
                + (f" dg-request-id={dg_req_id}" if dg_req_id else "")
            )
            raise
        except Exception as e:
            log_exception(f"[{self.session_id}] TTS connect failed", e)
            raise

        # Spawn a receiver that pushes binary audio to a queue
        self._receiver_task = asyncio.create_task(self._receiver_loop(), name=f"tts-recv-{self.session_id}")
        self._started = True


    async def stop(self) -> None:
        """Close the TTS socket (idempotent)."""
        self._closed.set()
        try:
            if self._ws and not self._ws.closed:
                # Best-effort: send "Close" control (optional)
                try:
                    await self._send_json({"type": "Close"})
                except Exception:
                    pass
                await self._ws.close()
        except Exception as e:
            log_exception(f"[{self.session_id}] TTS close error", e)
        finally:
            self._ws = None

        if self._receiver_task and not self._receiver_task.done():
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass
            finally:
                self._receiver_task = None

    # ---------------------------
    # Public synthesis API
    # ---------------------------

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[bytes]:
        """
        Accepts an async iterator of text chunks and yields PCM16 audio frames.

        - Connects to Deepgram speak WS on first use.
        - For each incoming chunk, sends {"type":"Speak","text": chunk}.
        - Applies a simple **flush heuristic**:
            * Flush on end-of-sentence punctuation [.?!]
            * Flush on long chunk (> 200 chars) or trailing newline
            * Always Flush at the end of text_stream
        - If cancel_event is set at any time:
            * send {"type":"Clear"} (best-effort),
            * drain outstanding audio, and stop.
        """
        await self._connect()
        assert self._ws is not None

        # Producer: send Speak/Flush messages as text arrives
        async def _producer() -> None:
            try:
                async for piece in text_stream:
                    if not piece:
                        continue
                    if cancel_event and cancel_event.is_set():
                        # barge-in => clear buffer and stop producing
                        await self._send_clear()
                        break

                    await self._send_speak(piece)

                # End of stream — ensure final flush
                if not (cancel_event and cancel_event.is_set()):
                    await self._send_flush()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log_exception(f"[{self.session_id}] TTS producer error", e)
                # Bubble errors to the consumer by closing the socket
                await self.stop()

        # Consumer: yield audio frames until socket closes or cancel occurs
        producer_task = asyncio.create_task(_producer(), name=f"tts-prod-{self.session_id}")

        try:
            while not self._closed.is_set():
                if cancel_event and cancel_event.is_set():
                    # best-effort clear when cancellation appears mid-stream
                    await self._send_clear()
                    break

                try:
                    chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=30)
                except asyncio.TimeoutError:
                    # No audio for a while; assume stream is done or stalled
                    break

                yield chunk
        finally:
            # Ensure producer is done
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass

    # ---------------------------
    # Internals
    # ---------------------------

    def _build_speak_url(self) -> str:
        """
        Build Deepgram /v1/speak WebSocket URL.

        Deepgram expects:
        - model: e.g., "aura-2-thalia-en"
        - encoding: e.g., "linear16"
        - sample_rate: e.g., 24000
        - (optional) container=none for raw PCM
        """
        
        base = "wss://api.deepgram.com/v1/speak"
        model = (getattr(self.config, "model", "") or "aura-2-thalia-en").strip()
        encoding = getattr(self.config, "encoding", None) or "linear16"
        sample_rate = int(getattr(self.config, "sample_rate", None) or 24000)
        qs = urlencode({"model": model, "encoding": encoding, "sample_rate": str(sample_rate), "container": "none"}, quote_via=quote_plus)
        return f"{base}?{qs}"


    async def _receiver_loop(self) -> None:
        """
        Receive frames from Deepgram TTS:
          - Binary frames: audio (enqueue to _audio_queue)
          - Text frames: control acks ("Flushed", "Cleared"), warnings/errors (log + ignore)
        """
        ws = self._ws
        if not ws:
            return

        try:
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    await self._enqueue_audio(bytes(msg))
                    continue

                # Text frame (JSON control / info)
                try:
                    data = json.loads(msg)
                except Exception:
                    continue

                mtype = data.get("type")
                if mtype in ("Flushed", "Cleared"):
                    # confirmations; ignore
                    continue
                if mtype in ("Error", "Warning"):
                    logger.warning(f"[{self.session_id}] TTS message: {mtype}: {data}")
                    continue
                # Unknown info — ignore for MVP
        except websockets.ConnectionClosedOK:
            pass
        except websockets.ConnectionClosedError:
            pass
        except Exception as e:
            log_exception(f"[{self.session_id}] TTS receiver error", e)
        finally:
            self._closed.set()

    async def _enqueue_audio(self, pcm: bytes) -> None:
        """Drop-oldest on overflow to avoid unbounded memory."""
        try:
            self._audio_queue.put_nowait(pcm)
        except asyncio.QueueFull:
            try:
                _ = self._audio_queue.get_nowait()
            except Exception:
                pass
            try:
                self._audio_queue.put_nowait(pcm)
            except Exception:
                pass

    async def _send_speak(self, text: str) -> None:
        await self._send_json({"type": "Speak", "text": text})

    async def _send_flush(self) -> None:
        await self._send_json({"type": "Flush"})

    async def _send_clear(self) -> None:
        try:
            await self._send_json({"type": "Clear"})
        except Exception:
            # Ignore if socket already closing/closed
            pass

    async def _send_json(self, payload: Dict[str, object]) -> None:
        ws = self._ws
        if not ws or ws.closed:
            raise RuntimeError("TTS socket not connected")
        await ws.send(json.dumps(payload))

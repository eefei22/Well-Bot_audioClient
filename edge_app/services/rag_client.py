# edge_app/services/rag_client.py
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, Optional

import websockets
from websockets.client import WebSocketClientProtocol

from edge_app.utils.logging import logger, log_exception, Timer


class RAGClient:
    """
    Minimal per-turn WebSocket client for your RAG backend.

    Contract (JSON messages):

        Edge → RAG:
            {
              "type":"user_turn",
              "session_id":"...",
              "turn_id":"...",
              "text":"<final transcript>",
              "meta": { ... }
            }

            { "type":"cancel", "turn_id":"..." }

        RAG → Edge (stream until "done"):
            { "type":"delta", "turn_id":"...", "text":"..." }
            { "type":"done",  "turn_id":"...", "tokens":123 }

    Usage:
        rag = RAGClient(ws_url=..., first_token_deadline_ms=2500, session_id=..., user_id=...)
        async for chunk in rag.user_turn_stream("hello", "trn_...", meta={}, cancel_event=evt):
            ... feed to TTS ...
    """

    def __init__(
        self,
        ws_url: str,
        first_token_deadline_ms: int,
        session_id: str,
        user_id: str,
        connect_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.ws_url = ws_url
        self.first_token_deadline_ms = first_token_deadline_ms
        self.session_id = session_id
        self.user_id = user_id
        self.connect_headers = connect_headers or {}
        # Per-turn connection; we don't keep a long-lived socket in MVP.
        self._ws: Optional[WebSocketClientProtocol] = None

    # ---------------------------
    # Public API
    # ---------------------------

    async def user_turn_stream(
        self,
        text: str,
        turn_id: str,
        meta: Optional[Dict[str, Any]] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[str]:
        """
        Open a WS to the RAG backend, send the final user text, and yield response deltas.
        Closes the WS when 'done' arrives or on cancellation/errors.
        """
        meta = meta or {}
        await self._connect()

        # Send user_turn
        try:
            await self._send_json(
                {
                    "type": "user_turn",
                    "session_id": self.session_id,
                    "turn_id": turn_id,
                    "text": text,
                    "meta": meta,
                }
            )
        except Exception as e:
            log_exception(f"[{self.session_id}] RAG send user_turn failed", e)
            await self._safe_close()
            raise

        first_token = asyncio.Event()

        async def _receiver() -> AsyncIterator[str]:
            # First-token deadline; afterwards read without timeout.
            deadline_s = max(0, self.first_token_deadline_ms) / 1000.0
            try:
                while True:
                    if cancel_event and cancel_event.is_set():
                        await self._send_cancel(turn_id)
                        break

                    if not first_token.is_set():
                        try:
                            raw = await asyncio.wait_for(self._recv_text(), timeout=deadline_s)
                        except asyncio.TimeoutError:
                            logger.warning(
                                f"[{self.session_id}] RAG first-token deadline exceeded ({self.first_token_deadline_ms}ms)"
                            )
                            raw = await self._recv_text()  # wait without deadline
                    else:
                        raw = await self._recv_text()

                    if raw is None:
                        break  # remote closed

                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue

                    mtype = msg.get("type")
                    if mtype == "delta":
                        if not first_token.is_set():
                            first_token.set()
                        chunk = msg.get("text", "")
                        if chunk:
                            yield chunk
                    elif mtype == "done":
                        break
                    elif mtype == "error":
                        detail = msg.get("detail") or msg.get("message") or "RAG error"
                        raise RuntimeError(detail)
                    else:
                        # Ignore unknown messages in MVP
                        continue
            finally:
                await self._safe_close()

        async for piece in _receiver():
            if cancel_event and cancel_event.is_set():
                await self._send_cancel(turn_id)
                break
            yield piece

    async def close(self) -> None:
        """No-op for MVP (we use per-turn sockets); kept for future pooling."""
        await self._safe_close()

    # ---------------------------
    # Internal helpers
    # ---------------------------

    async def _connect(self) -> None:
        if self._ws and not self._ws.closed:
            return
        headers = self.connect_headers.copy()
        headers.setdefault("X-User-Id", self.user_id)
        headers.setdefault("X-Session-Id", self.session_id)

        try:
            with Timer("rag_connect_ms"):
                self._ws = await websockets.connect(self.ws_url, extra_headers=headers, max_size=None)
        except Exception as e:
            log_exception(f"[{self.session_id}] RAG connect failed", e)
            raise

    async def _safe_close(self) -> None:
        try:
            if self._ws and not self._ws.closed:
                await self._ws.close()
        except Exception:
            pass
        finally:
            self._ws = None

    async def _send_json(self, payload: Dict[str, Any]) -> None:
        if not self._ws:
            raise RuntimeError("RAG socket is not connected")
        await self._ws.send(json.dumps(payload))

    async def _recv_text(self) -> Optional[str]:
        if not self._ws:
            return None
        try:
            msg = await self._ws.recv()
        except websockets.ConnectionClosed:
            return None
        return msg if isinstance(msg, str) else None

    async def _send_cancel(self, turn_id: str) -> None:
        try:
            await self._send_json({"type": "cancel", "turn_id": turn_id})
        except Exception as e:
            logger.debug(f"[{self.session_id}] RAG cancel send failed: {e}")

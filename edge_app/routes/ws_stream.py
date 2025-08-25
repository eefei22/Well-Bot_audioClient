# edge_app/routes/ws_stream.py

from __future__ import annotations
import asyncio
import json
import uuid
import time, re, os
from typing import Any, Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from fastapi.websockets import WebSocketState

from edge_app.config import settings
from edge_app.services.stt_deepgram import DeepgramSTTStream, STTEvent
from edge_app.services.tts_deepgram import DeepgramTTSStream
from edge_app.services.rag_client import RAGClient
from edge_app.services.turn_manager import TurnManager, TurnState
from edge_app.utils.logging import log_exception, time_ms, logger
from edge_app.schemas.messages import (
    # Incoming control
    ControlMessage, StartSession, StopSession, BargeIn, CancelTurn,
    # Outgoing events
    StateEvent, ASRPartialEvent, ASRFinalEvent, TTSEventStart, TTSEventEnd, ErrorEvent,
)

# --- TTS chunking helper: batch small LLM deltas into speakable phrases ---

PUNCT_RE = re.compile(r'[.!?…]["\')\]]?\s*$')
TTS_CHUNK_MAX_MS = int(os.environ.get("TTS_CHUNK_MAX_MS", "350"))   # time-based emit
TTS_CHUNK_MAX_CHARS = int(os.environ.get("TTS_CHUNK_MAX_CHARS", "160"))  # size-based emit
TTS_CHUNK_MIN_CHARS = int(os.environ.get("TTS_CHUNK_MIN_CHARS", "24"))   # avoid tiny fragments

async def aggregate_deltas(delta_aiter, max_ms: int = TTS_CHUNK_MAX_MS, max_chars: int = TTS_CHUNK_MAX_CHARS):
    """Aggregate token/word deltas from RAG into short clauses/sentences."""
    buf, last_emit = [], time.perf_counter()
    async for d in delta_aiter:           # d is a str
        if not d:
            continue
        buf.append(d)
        text = "".join(buf)
        now = time.perf_counter()
        time_ready = (now - last_emit) >= (max_ms / 1000.0)
        size_ready = len(text) >= max_chars
        punct_ready = bool(PUNCT_RE.search(text)) and len(text) >= TTS_CHUNK_MIN_CHARS
        if punct_ready or size_ready or time_ready:
            chunk = text.strip()
            if chunk:
                yield chunk
                buf.clear()
                last_emit = now
    if buf:
        tail = "".join(buf).strip()
        if tail:
            yield tail




STT_LOG_FRAMES = os.environ.get("STT_LOG_FRAMES", "0").strip().lower() in {"1","true","yes","on"}
router = APIRouter()


class SessionContext:
    """
    Per-connection/session context and task registry.
    """
    def __init__(self, session_id: str, ws: WebSocket):
        self.session_id = session_id
        self.ws = ws
        self.user_id: str = "guest"
        self.device_id: Optional[str] = None
        self.locale: str = getattr(settings.stt, "language", None) or "en-US"

        self.turn_id: Optional[str] = None
        self.turn_start_ms: Optional[int] = None

        # Services (inited in handler)
        self.stt: Optional[DeepgramSTTStream] = None
        self.tts: Optional[DeepgramTTSStream] = None
        self.rag: Optional[RAGClient] = None
        self.tm: Optional[TurnManager] = None

        # Tasks
        self.receiver_task: Optional[asyncio.Task] = None
        self.stt_task: Optional[asyncio.Task] = None
        self.tts_task: Optional[asyncio.Task] = None

        # Cancellation flag for in-flight TTS/RAG turn
        self.turn_cancel_event = asyncio.Event()

    async def send_json(self, payload: Dict[str, Any]) -> None:
        if self.ws.application_state == WebSocketState.CONNECTED:
            await self.ws.send_text(json.dumps(payload))

    async def send_bytes(self, data: bytes) -> None:
        if self.ws.application_state == WebSocketState.CONNECTED:
            await self.ws.send_bytes(data)

    def new_turn(self) -> str:
        self.turn_id = f"trn_{uuid.uuid4().hex}"
        self.turn_start_ms = time_ms()
        # Reset per-turn cancel
        self.turn_cancel_event = asyncio.Event()
        return self.turn_id

    def cancel_turn(self) -> None:
        self.turn_cancel_event.set()


@router.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    """
    One duplex WebSocket = one voice session.

    Frames from client:
      - Binary: PCM16 frames (20–40 ms @ sample_rate)
      - Text:   JSON control (start_session, stop_session, barge_in, cancel_turn)

    Server to client:
      - Text JSON: state, asr_partial, asr_final, tts_start, tts_end, error
      - Binary:    PCM16 TTS audio chunks
    """
    await ws.accept(subprotocol=None)  # keep simple for MVP

    session_id = f"ses_{uuid.uuid4().hex}"
    ctx = SessionContext(session_id, ws)

    # Instantiate services
    stt = DeepgramSTTStream(
        session_id=session_id,
        user_id=ctx.user_id,
        config=settings.stt,
        headers=settings.deepgram_headers,
    )
    tts = DeepgramTTSStream(
        session_id=session_id,
        user_id=ctx.user_id,
        config=settings.tts,
        headers=settings.deepgram_headers,
    )
    rag = RAGClient(
        ws_url=settings.rag.ws_url,
        first_token_deadline_ms=settings.rag.first_token_deadline_ms,
        session_id=session_id,
        user_id=ctx.user_id,
    )
    tm = TurnManager(policy=settings.policy)

    ctx.stt = stt
    ctx.tts = tts
    ctx.rag = rag
    ctx.tm = tm

    logger.info(f"[{session_id}] WS connected")

    # Initial state
    await ctx.send_json(StateEvent(state=TurnState.LISTENING.value).model_dump())

    try:
        # Start STT stream (does not block)
        await stt.start()

        # Launch tasks
        ctx.receiver_task = asyncio.create_task(_receiver_loop(ctx))
        ctx.stt_task = asyncio.create_task(_stt_event_loop(ctx))

        # Run until one task finishes (disconnect or fatal)
        done, pending = await asyncio.wait(
            {ctx.receiver_task, ctx.stt_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        # If receiver ended (disconnect), stop stt loop gracefully
        for t in pending:
            t.cancel()

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] WS disconnected by client")
    except Exception as e:
        log_exception(f"[{session_id}] WS stream error", e)
        await _safe_send_error(ctx, "EDGE_ERROR", str(e))
    finally:
        await _cleanup(ctx)
        logger.info(f"[{session_id}] session closed")


async def _receiver_loop(ctx: SessionContext) -> None:
    """
    Receive frames from the client:
      - Binary PCM16 → forward to STT when Listening
      - Text JSON control → drive state (start/stop/cancel/barge-in)
    """
    ws = ctx.ws
    while True:
        msg = await ws.receive()
        msg_type = msg.get("type")

        if msg_type == "websocket.disconnect":
            raise WebSocketDisconnect()

        if msg_type == "websocket.receive":
            if "bytes" in msg and msg["bytes"] is not None:
                # Audio frame
                if ctx.tm and ctx.tm.state == TurnState.LISTENING and ctx.stt:
                    await ctx.stt.send_audio(msg["bytes"])
                # else drop frames (we're thinking/speaking)
                continue

            if "text" in msg and msg["text"]:
                try:
                    payload = json.loads(msg["text"])
                    ctrl = ControlMessage.model_validate(payload)
                except Exception as e:
                    await _safe_send_error(ctx, "BAD_CONTROL", f"Invalid control frame: {e}")
                    continue

                if isinstance(ctrl, StartSession):
                    # Attach optional user/device and locale
                    if ctrl.user_id:
                        ctx.user_id = ctrl.user_id
                    if ctrl.device_id:
                        ctx.device_id = ctrl.device_id
                    if ctrl.locale:
                        ctx.locale = ctrl.locale  
                    logger.info(f"[{ctx.session_id}] start_session user={ctx.user_id} device={ctx.device_id}")
                    await ctx.send_json(StateEvent(state=ctx.tm.state.value).model_dump())

                elif isinstance(ctrl, StopSession):
                    logger.info(f"[{ctx.session_id}] stop_session requested")
                    # Graceful end: break the loop to trigger cleanup
                    break

                elif isinstance(ctrl, BargeIn):
                    now = time.perf_counter()
                    last = getattr(ctx, "last_tts_start", 0.0)
                    needed = float(os.environ.get("BARGEIN_DELAY_MS", "800")) / 1000.0
                    if ctx.tm.state == TurnState.SPEAKING and (now - last) >= needed:
                        logger.info(f"[{ctx.session_id}] barge-in received")
                        ctx.cancel_turn()
                        ctx.tm.transition_to(TurnState.LISTENING)
                        await ctx.send_json(StateEvent(state=TurnState.LISTENING.value).model_dump())
                    else:
                        logger.debug(
                            f"[{ctx.session_id}] barge-in ignored (state={ctx.tm.state.name}, early={now - last:.3f}s)"
                        )

                elif isinstance(ctrl, CancelTurn):
                    if ctx.turn_id and ctrl.turn_id == ctx.turn_id:
                        logger.info(f"[{ctx.session_id}] cancel_turn turn_id={ctx.turn_id}")
                        ctx.cancel_turn()

                else:
                    await _safe_send_error(ctx, "UNKNOWN_CONTROL", f"Unsupported control type: {payload.get('type')}")

        else:
            # Unexpected type; ignore
            continue


async def _stt_event_loop(ctx: SessionContext) -> None:
    """
    Consume STT events and drive turn transitions.
    """
    assert ctx.stt is not None
    stt = ctx.stt

    async for ev in stt.events():
        if ev.kind == "partial":
            if STT_LOG_FRAMES:
                logger.debug(f"[{ctx.session_id}] ← asr_partial len={len(ev.text)}")
            # Display interim to client
            await ctx.send_json(ASRPartialEvent(turn_id=ctx.turn_id or "", text=ev.text).model_dump())

        elif ev.kind == "final":
            # End of user utterance: create a new turn
            turn_id = ctx.new_turn()
            logger.info(f"[{ctx.session_id}] asr_final turn={turn_id} endpoint_ms={ev.endpoint_ms} text='{ev.text}'")
            await ctx.send_json(ASRFinalEvent(turn_id=turn_id, text=ev.text, endpoint_ms=ev.endpoint_ms).model_dump())

            # Transition Listening → Thinking
            ctx.tm.transition_to(TurnState.THINKING)
            await ctx.send_json(StateEvent(state=TurnState.THINKING.value).model_dump())

            # Kick off the turn pipeline (RAG→TTS) and wait for completion or barge-in
            try:
                await _run_turn_pipeline(ctx, user_text=ev.text)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log_exception(f"[{ctx.session_id}] turn pipeline error", e)
                await _safe_send_error(ctx, "TURN_ERROR", str(e))
            finally:
                # If we were interrupted by barge-in, state already set to LISTENING in receiver loop
                if ctx.tm.state != TurnState.LISTENING:
                    ctx.tm.transition_to(TurnState.LISTENING)
                    await ctx.send_json(StateEvent(state=TurnState.LISTENING.value).model_dump())


async def _run_turn_pipeline(ctx: SessionContext, user_text: str) -> None:
    """
    One turn:
    1) Get RAG deltas.
    2) Aggregate deltas into clauses/sentences.
    3) Stream aggregated text to TTS; forward PCM to client.
    """
    assert ctx.rag and ctx.tts and ctx.turn_id
    rag = ctx.rag
    tts = ctx.tts
    turn_id = ctx.turn_id

    speaking_announced = False

    async def _tts_pcm_forwarder(tts_pcm_aiter):
        nonlocal speaking_announced
        async for chunk in tts_pcm_aiter:
            if ctx.turn_cancel_event.is_set():
                break
            if not speaking_announced:
                logger.debug(f"[{ctx.session_id}] TTS first audio for turn {turn_id} → SPEAKING")
                ctx.last_tts_start = time.perf_counter()
                ctx.bargein_delay_sec = float(os.environ.get("BARGEIN_DELAY_MS", "800")) / 1000.0
                ctx.tm.transition_to(TurnState.SPEAKING)
                await ctx.send_json(StateEvent(state=TurnState.SPEAKING.value).model_dump())
                await ctx.send_json(TTSEventStart(turn_id=turn_id).model_dump())
                speaking_announced = True
            await ctx.send_bytes(chunk)

    # 1) RAG deltas
    rag_stream = rag.user_turn_stream(
        text=user_text,
        turn_id=turn_id,
        meta={"locale": ctx.locale},
        cancel_event=ctx.turn_cancel_event,
    )

    # 2) Aggregate small deltas into speakable chunks
    chunked = aggregate_deltas(rag_stream)

    # 3) TTS synth; forward PCM to client
    tts_pcm_stream = tts.synthesize_stream(text_stream=chunked, cancel_event=ctx.turn_cancel_event)
    try:
        await _tts_pcm_forwarder(tts_pcm_stream)
    finally:
        await tts.stop()  # idempotent

    if not ctx.turn_cancel_event.is_set() and speaking_announced:
        await ctx.send_json(TTSEventEnd(turn_id=turn_id).model_dump())



async def _safe_send_error(ctx: SessionContext, code: str, detail: str) -> None:
    try:
        await ctx.send_json(ErrorEvent(code=code, detail=detail).model_dump())
    except Exception:
        # Ignore send errors during teardown
        pass


async def _cleanup(ctx: SessionContext) -> None:
    """
    Ensure all tasks and vendor streams are closed on disconnect.
    """
    # Signal cancel to in-flight turn
    ctx.cancel_turn()

    tasks = [ctx.receiver_task, ctx.stt_task, ctx.tts_task]
    for t in tasks:
        if t and not t.done():
            t.cancel()

    # Close vendor streams
    try:
        if ctx.stt:
            await ctx.stt.stop()
    except Exception as e:
        log_exception(f"[{ctx.session_id}] STT stop error", e)

    try:
        if ctx.tts:
            await ctx.tts.stop()
    except Exception as e:
        log_exception(f"[{ctx.session_id}] TTS stop error", e)

    # Close RAG client
    try:
        if ctx.rag:
            await ctx.rag.close()
    except Exception as e:
        log_exception(f"[{ctx.session_id}] RAG close error", e)

    # Close WS if still open
    if ctx.ws.application_state == WebSocketState.CONNECTED:
        try:
            await ctx.ws.close(code=status.WS_1000_NORMAL_CLOSURE)
        except Exception:
            pass

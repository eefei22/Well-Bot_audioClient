"""
Edge forwarder: mic -> Deepgram Realtime -> (final transcripts) -> RAG /ws/chat

- Holds one Deepgram WS (SDK) and one persistent WS to your RAG.
- On Deepgram is_final, sends: {"session_id","user_id","text"} as a text frame.
- Ignores partials; drains inbound frames from RAG (so the socket doesn't clog).
- Clean shutdown on Ctrl+C: sends "exit" to RAG and finishes the DG stream.
"""

import os, sys, time, json, queue, signal, logging, threading, asyncio, inspect
from typing import Optional

import numpy as np
import sounddevice as sd
import uuid, logging

from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
    DeepgramClientOptions,
)

import websockets
from websockets.asyncio.client import connect as ws_connect


# -------------------- Config --------------------
load_dotenv()

DG_API_KEY   = os.getenv("DEEPGRAM_API_KEY")
DG_MODEL     = os.getenv("DG_MODEL", "nova-3")
DG_LANGUAGE  = os.getenv("DG_LANGUAGE", "en-US")
DG_SRATE     = int(os.getenv("DG_SAMPLE_RATE", "16000"))
DG_CHANS     = int(os.getenv("DG_CHANNELS", "1"))
DG_DEVICE    = os.getenv("DG_INPUT_DEVICE")

RAG_WS_URL   = os.getenv("RAG_WS_URL", "ws://localhost:8000/ws/chat")
RAG_USER     = os.getenv("RAG_USER_ID", "edge-user")
RAG_AUTH     = os.getenv("RAG_AUTH_TOKEN")  # optional
RAG_SESSION = (os.getenv("RAG_SESSION_ID") or "").strip()
if not RAG_SESSION or RAG_SESSION.lower() == "auto":
    RAG_SESSION = f"sess-{uuid.uuid4().hex[:8]}"
logging.getLogger("edge-forwarder").info("Using session_id=%s user_id=%s", RAG_SESSION, RAG_USER)


WS_PING_SEC  = int(os.getenv("WS_PING_SEC", "25"))

if not DG_API_KEY:
    print("Missing DEEPGRAM_API_KEY in .env", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("edge-forwarder")


# -------------------- Shared state --------------------
stop_flag = threading.Event()
audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=200)
final_q: "queue.Queue[str]"  = queue.Queue(maxsize=100)  # final transcripts pending send
dg_connection = None  # set in send_loop()


# -------------------- Deepgram handlers --------------------
def on_open(_, open, **kwargs):
    log.info("Deepgram connection opened.")

def on_transcript(_, result, **kwargs):
    # Called frequently; only forward finals.
    try:
        alt = result.channel.alternatives[0]
        text = alt.transcript or ""
    except Exception:
        return
    if not text:
        return

    is_final = False
    try:
        is_final = bool(getattr(result, "is_final", False))
    except Exception:
        pass

    if is_final:
        # Print final and enqueue for RAG
        print(f"\n[final]   {text}", flush=True)
        try:
            final_q.put_nowait(text)
        except queue.Full:
            log.warning("Final queue full; dropping a final transcript.")
    else:
        # Show partials inline (optional)
        print(f"\r[partial] {text: <100}", end="", flush=True)

def on_close(_, close, **kwargs):
    log.info("Deepgram connection closed.")

def on_error(_, error, **kwargs):
    log.error("Deepgram error: %s", error)


# -------------------- Mic capture --------------------
def audio_callback(indata, frames, time_info, status):
    if status:
        log.warning("[mic] %s", status)
    pcm16 = (indata * 32767).astype(np.int16).tobytes()
    try:
        audio_q.put_nowait(pcm16)
    except queue.Full:
        # Drop audio under backpressure; keep realtime
        pass

def mic_loop():
    kwargs = dict(
        samplerate=DG_SRATE,
        channels=DG_CHANS,
        dtype="float32",
        blocksize=320,  # ~20ms @ 16k
        callback=audio_callback,
    )
    if DG_DEVICE:
        try:
            kwargs["device"] = int(DG_DEVICE)
        except ValueError:
            kwargs["device"] = DG_DEVICE

    log.info("Opening mic (rate=%d Hz, ch=%d, device=%s)", DG_SRATE, DG_CHANS, kwargs.get("device"))
    with sd.InputStream(**kwargs):
        while not stop_flag.is_set():
            time.sleep(0.05)


# -------------------- Deepgram streaming (send bytes) --------------------
def deepgram_loop():
    global dg_connection

    dg = DeepgramClient(
        DG_API_KEY,
        DeepgramClientOptions(verbose=logging.WARN, options={"keepalive": "true"}),
    )

    # Prefer modern websocket interface, fallback if older SDK
    try:
        dg_connection = dg.listen.websocket.v("1")
    except AttributeError:
        dg_connection = dg.listen.live.v("1")

    dg_connection.on(LiveTranscriptionEvents.Open,       on_open)
    dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
    dg_connection.on(LiveTranscriptionEvents.Close,      on_close)
    dg_connection.on(LiveTranscriptionEvents.Error,      on_error)

    opts = LiveOptions(
        model=DG_MODEL,
        language=DG_LANGUAGE,
        encoding="linear16",
        sample_rate=DG_SRATE,
        channels=DG_CHANS,
        punctuate=True,
        interim_results=True,
        smart_format=True,
    )

    try:
        dg_connection.start(opts)
        print("[deepgram] connected; streaming…")
    except Exception as e:
        log.exception("Failed to start Deepgram live connection: %s", e)
        return

    try:
        while not stop_flag.is_set():
            try:
                chunk = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            dg_connection.send(chunk)
    finally:
        try:
            dg_connection.finish()
        except Exception:
            pass
        log.info("Deepgram stream finished.")


# -------------------- RAG WS client (async) --------------------
async def rag_consumer(ws: websockets.WebSocketClientProtocol):
    """Read all frames from RAG and print the streamed text to terminal."""
    try:
        turn_buf = []
        async for msg in ws:
            try:
                obj = json.loads(msg)
            except Exception:
                # Unknown frame; print raw and continue
                print(f"\n[ws] {msg}")
                continue

            t = obj.get("type")
            if t == "token":
                # stream tokens inline (no newline)
                tok = obj.get("text", "")
                turn_buf.append(tok)
                print(tok, end="", flush=True)
            elif t == "meta":
                # end of one assistant turn
                print()  # newline after the streamed text
                # (optional) show brief meta info:
                latency = obj.get("latency_ms")
                retr = obj.get("retrieved_docs", [])
                print(f"[meta] latency_ms={latency} retrieved={len(retr)}")
                turn_buf.clear()
            elif t == "error":
                print(f"\n[error] {obj.get('message')}")
            elif t == "done":
                print("\n[session] done.")
                break
            else:
                # greeting or unknown structured frame
                if "text" in obj:
                    print(obj["text"], end="", flush=True)
    except Exception as e:
        print(f"\n[consumer] error: {e}")


def _rag_headers():
    headers = {}
    if RAG_AUTH:
        headers["Authorization"] = f"Bearer {RAG_AUTH}"
    return headers

async def rag_sender_loop():
    """
    Persistent WS to RAG. On each final transcript, send one ChatIn-like JSON:
      {"session_id": "...", "user_id": "...", "text": "..."}
    On shutdown, send an 'exit' turn to finalize the session.
    """
    # Build connect kwargs compatible with websockets version
    connect_kwargs = dict(ping_interval=WS_PING_SEC, ping_timeout=WS_PING_SEC + 5)
    # adapt header parameter name across versions
    if "additional_headers" in inspect.signature(ws_connect).parameters:
        connect_kwargs["additional_headers"] = _rag_headers()
    else:
        connect_kwargs["extra_headers"] = list(_rag_headers().items())

    backoff = 1
    while not stop_flag.is_set():
        try:
            async with ws_connect(RAG_WS_URL, **connect_kwargs) as ws:
                log.info("Connected to RAG WS: %s", RAG_WS_URL)
                # Start consumer in background
                consumer = asyncio.create_task(rag_consumer(ws))

                # Sender loop
                while not stop_flag.is_set():
                    # Pull next final transcript (blocking with timeout so we can check stop flag)
                    try:
                        text = await asyncio.get_event_loop().run_in_executor(None, final_q.get, True, 0.5)
                    except Exception:
                        continue

                    payload = {
                        "session_id": RAG_SESSION,
                        "user_id": RAG_USER,
                        "text": text,
                    }
                    await ws.send(json.dumps(payload))
                    log.info('RAG-> {"text": %.60r, ...}', text)

                # On shutdown, finalize politely
                try:
                    await ws.send(json.dumps({"session_id": RAG_SESSION, "user_id": RAG_USER, "text": "exit"}))
                except Exception:
                    pass

                consumer.cancel()
                with contextlib.suppress(Exception):
                    await consumer
                backoff = 1  # reset on clean session
                break

        except Exception as e:
            log.warning("RAG WS connect/send failed: %s (retrying in %ss)", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10)


# -------------------- Orchestration --------------------
def main():
    # Mic & Deepgram run in threads (blocking loops)
    t_mic = threading.Thread(target=mic_loop, daemon=True)
    t_dg  = threading.Thread(target=deepgram_loop, daemon=True)
    t_mic.start(); t_dg.start()

    # RAG WS runs in asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async def runner():
        await rag_sender_loop()
    task = loop.create_task(runner())

    def _stop(*_):
        stop_flag.set()
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        loop.run_until_complete(task)
    finally:
        loop.stop()
        loop.close()
        # Wait a moment for threads to exit
        for t in (t_mic, t_dg):
            t.join(timeout=2)
        print("\nShut down cleanly.")

if __name__ == "__main__":
    import contextlib  # used above
    main()

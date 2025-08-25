"""
Console MVP: mic -> Deepgram Realtime -> print transcripts.
No Flask, no RAG, no Socket.IO. Ctrl+C to stop.
"""

import os
import sys
import time
import queue
import signal
import logging
import threading

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
    DeepgramClientOptions,
)

# ---------- Config & setup ----------
load_dotenv()
API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not API_KEY:
    print("Missing DEEPGRAM_API_KEY in .env", file=sys.stderr)
    sys.exit(1)

SAMPLE_RATE = int(os.getenv("DG_SAMPLE_RATE", "16000"))
CHANNELS     = int(os.getenv("DG_CHANNELS", "1"))
MODEL        = os.getenv("DG_MODEL", "nova-3")
LANGUAGE     = os.getenv("DG_LANGUAGE", "en-US")
INPUT_DEVICE = os.getenv("DG_INPUT_DEVICE")  # optional (index or device name)

log = logging.getLogger("dg-mvp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Deepgram client (keepalive true helps with silent gaps)
dg = DeepgramClient(
    API_KEY,
    DeepgramClientOptions(
        verbose=logging.WARN,
        options={"keepalive": "true"},
    ),
)

# Connection holder
dg_connection = None
stop_flag = threading.Event()
audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=200)


# ---------- Deepgram event handlers ----------
def on_open(_, open, **kwargs):
    log.info("Deepgram connection opened: %s", open)

def on_transcript(_, result, **kwargs):
    """
    result.channel.alternatives[0].transcript -> text
    result.is_final (bool) indicates end of an utterance
    """
    try:
        alt = result.channel.alternatives[0]
        text = alt.transcript or ""
    except Exception:
        return
    if not text:
        return

    is_final = False
    # Best effort: SDK provides .is_final; keep robust if shape changes
    try:
        is_final = bool(getattr(result, "is_final", False))
    except Exception:
        pass

    if is_final:
        print(f"\n[final]   {text}", flush=True)
    else:
        # carriage return partials in-place
        print(f"\r[partial] {text: <100}", end="", flush=True)

def on_close(_, close, **kwargs):
    log.info("Deepgram connection closed: %s", close)

def on_error(_, error, **kwargs):
    log.error("Deepgram error: %s", error)


# ---------- Audio capture ----------
def audio_callback(indata, frames, time_info, status):
    if status:
        log.warning("[mic] %s", status)
    # float32 [-1,1] -> int16 PCM (little-endian) bytes
    pcm16 = (indata * 32767).astype(np.int16).tobytes()
    try:
        audio_q.put_nowait(pcm16)
    except queue.Full:
        # drop frames under backpressure to keep realtime
        pass

def mic_loop():
    stream_kwargs = dict(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=320,  # ~20ms at 16kHz
        callback=audio_callback,
    )
    if INPUT_DEVICE:
        try:
            stream_kwargs["device"] = int(INPUT_DEVICE)
        except ValueError:
            stream_kwargs["device"] = INPUT_DEVICE

    log.info("Opening mic (rate=%s Hz, ch=%s, device=%s)", SAMPLE_RATE, CHANNELS, stream_kwargs.get("device"))
    with sd.InputStream(**stream_kwargs):
        while not stop_flag.is_set():
            time.sleep(0.05)


# ---------- Deepgram streaming send loop ----------
def send_loop():
    global dg_connection

    # Choose the v3 live websocket interface
    try:
        dg_connection = dg.listen.websocket.v("1")
    except AttributeError:
        # older SDK fallback
        dg_connection = dg.listen.live.v("1")

    # Bind events
    dg_connection.on(LiveTranscriptionEvents.Open,       on_open)
    dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
    dg_connection.on(LiveTranscriptionEvents.Close,      on_close)
    dg_connection.on(LiveTranscriptionEvents.Error,      on_error)

    # Start with correct media settings (must match your bytes)
    opts = LiveOptions(
        model=MODEL,
        language=LANGUAGE,
        encoding="linear16",
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
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


# ---------- Main ----------
def main():
    # Print helpful device hint once (optional)
    if os.getenv("DG_INPUT_DEVICE") is None:
        try:
            import sounddevice as sd_
            default_in = sd_.default.device[0]
            log.info("Using default input device index: %s (set DG_INPUT_DEVICE to override)", default_in)
        except Exception:
            pass

    # Start audio + deepgram threads
    t_mic = threading.Thread(target=mic_loop, daemon=True)
    t_dg  = threading.Thread(target=send_loop, daemon=True)
    t_mic.start(); t_dg.start()

    # Graceful shutdown on Ctrl+C
    def _stop(*_):
        stop_flag.set()
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # Wait until stopped
    try:
        while not stop_flag.is_set():
            time.sleep(0.2)
    finally:
        stop_flag.set()
        t_mic.join(timeout=2)
        t_dg.join(timeout=2)
        print("\nShut down cleanly.")

if __name__ == "__main__":
    main()


'''
- Chunk size / latency: blocksize=320 in the InputStream → ~20 ms frames at 16 kHz.
    - Larger (e.g., 640) = fewer sends, a bit more latency; smaller = lower latency, more CPU/WS traffic.

- Queue size: audio_q = queue.Queue(maxsize=200) controls how many chunks buffer before dropping. Increase if your network is spiky.
- Logging level: in logging.basicConfig(level=logging.INFO, ...). Change to DEBUG for more detail.
- Deepgram stream options (in LiveOptions):
    - punctuate=True → turn on punctuation.
    - interim_results=True → print partials while speaking.
    - smart_format=True → numbers, dates, etc.

- Keepalive: DeepgramClientOptions(options={"keepalive": "true"}). You can disable or augment by sending silence frames yourself if you prefer.
- Fallback interface: the code tries dg.listen.websocket.v("1"), then falls back to dg.listen.live.v("1") for older SDKs—usually you don’t need to touch this.
'''
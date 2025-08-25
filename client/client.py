# client/client.py
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

import websockets
from dotenv import load_dotenv

from client.audio import MicStreamer, Speaker
from client.vad import VADDetector
from time import perf_counter

# -----------------------------
# Config / identity helpers
# -----------------------------

def load_env() -> dict:
    load_dotenv(override=False)
    cfg = {
        # Edge Forwarder
        "EDGE_HOST": os.environ.get("EDGE_HOST", "127.0.0.1"),
        "EDGE_PORT": int(os.environ.get("EDGE_PORT", "8080")),
        # Client audio
        "CLIENT_SAMPLE_RATE": int(os.environ.get("CLIENT_SAMPLE_RATE", "16000")),
        "CLIENT_FRAME_MS": int(os.environ.get("CLIENT_FRAME_MS", "20")),
        "PLAYBACK_BUFFER_MS": int(os.environ.get("PLAYBACK_BUFFER_MS", "120")),
        "ENABLE_LOCAL_VAD": os.environ.get("ENABLE_LOCAL_VAD", "true").lower() in {"1", "true", "yes", "on"},
        # TTS playback rate should match the Edge TTS config
        "TTS_SAMPLE_RATE": int(os.environ.get("TTS_SAMPLE_RATE", "24000")),
    }
    return cfg


def ensure_device_id() -> str:
    """
    Persist a stable device_id in ~/.voice_edge/device.json
    """
    home = Path.home()
    root = home / ".voice_edge"
    root.mkdir(parents=True, exist_ok=True)
    f = root / "device.json"
    if f.exists():
        try:
            data = json.loads(f.read_text())
            if isinstance(data, dict) and data.get("device_id"):
                return str(data["device_id"])
        except Exception:
            pass
    dev_id = f"dev_{uuid.uuid4().hex}"
    try:
        f.write_text(json.dumps({"device_id": dev_id}, indent=2))
    except Exception:
        pass
    return dev_id


def make_session_id() -> str:
    return f"ses_{uuid.uuid4().hex}"


# -----------------------------
# Client application
# -----------------------------

class VoiceClient:
    def __init__(
        self,
        url: str,
        user_id: str = "guest",
        device_id: Optional[str] = None,
        locale: str = "en-US",
        mic_rate: int = 16000,
        mic_frame_ms: int = 20,
        tts_rate: int = 24000,
        playback_buffer_ms: int = 120,
        enable_local_vad: bool = True,
    ) -> None:
        self.url = url
        self.user_id = user_id
        self.device_id = device_id or ensure_device_id()
        self.locale = locale

        # Audio I/O
        self.mic = MicStreamer(sample_rate=mic_rate, frame_ms=mic_frame_ms)
        self.spk = Speaker(sample_rate=tts_rate, prebuffer_ms=playback_buffer_ms)

        # Optional local VAD (for barge-in)
        self.vad = VADDetector(sample_rate=mic_rate, frame_ms=mic_frame_ms) if enable_local_vad else None

        # Runtime
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.state: str = "listening"  # server-driven state; initial assumption
        self.session_id: str = make_session_id()

        # Concurrency primitives
        self._send_lock = asyncio.Lock()
        self._stop = asyncio.Event()

        self._speaking = False
        self._speaking_since = 0.0
        self._can_barge_at = 0.0
        self._no_uplink_until = 0.0
        
        # tweakable guards (ms)
        self.bargein_delay_ms = int(os.environ.get("BARGEIN_DELAY_MS", "800"))      # wait after tts_start
        self.mute_uplink_ms  = int(os.environ.get("MUTE_UPLINK_MS", "600"))         # don’t send mic frames to edge right after tts_start

    # ------------- WS send helpers -------------

    async def send_json(self, payload: dict) -> None:
        if not self.ws:
            return
        async with self._send_lock:
            await self.ws.send(json.dumps(payload))

    async def send_audio(self, pcm: bytes) -> None:
        if not self.ws:
            return
        async with self._send_lock:
            await self.ws.send(pcm)

    # ------------- main run -------------

    async def run(self) -> None:
        # Start local audio
        self.spk.start()
        await self.mic.start()

        # Connect WS
        print(f"[client] connecting → {self.url}")
        async with websockets.connect(self.url, max_size=None, ping_interval=20) as ws:
            self.ws = ws

            # Start session (control)
            await self.send_json({
                "type": "start_session",
                "session_id": self.session_id,
                "user_id": self.user_id,
                "device_id": self.device_id,
                "locale": self.locale,
            })

            # Launch tasks
            sender = asyncio.create_task(self._sender_loop(), name="sender_loop")
            receiver = asyncio.create_task(self._receiver_loop(), name="receiver_loop")

            try:
                # Run until Ctrl+C or a task exits
                done, pending = await asyncio.wait(
                    {sender, receiver},
                    return_when=asyncio.FIRST_COMPLETED
                )
                # If one task returned (likely receiver due to disconnect), cancel the other
                for t in pending:
                    t.cancel()
            except KeyboardInterrupt:
                print("\n[client] KeyboardInterrupt → stopping...")
            finally:
                try:
                    await self.send_json({"type": "stop_session"})
                except Exception:
                    pass

        # Teardown
        await self.mic.stop()
        await self.spk.stop()
        print("[client] stopped.")

    # ------------- tasks -------------

    async def _sender_loop(self) -> None:
        async for frame in self.mic.frames():
            now = perf_counter()

            # (a) Briefly mute uplink to edge to avoid STT hearing the speaker leak
            if not (self.state == "speaking" and now < self._no_uplink_until):
                try:
                    await self.send_audio(frame)
                except Exception as e:
                    print(f"[client] send_audio error: {e}")
                    break

            # (b) Local VAD barge-in is **only** allowed once we’ve been speaking for a bit
            if self.vad and self.state == "speaking" and now >= self._can_barge_at:
                try:
                    _, event = self.vad.process_frame(frame)
                except ValueError:
                    event = None
                if event == "speech_start":
                    try:
                        await self.send_json({"type": "barge_in"})
                    except Exception:
                        pass

            if self._stop.is_set():
                break


    async def _receiver_loop(self) -> None:
        """
        Receive server messages:
          - text frames: control/events (state, asr_partial/final, tts_start/end, error)
          - binary frames: TTS PCM chunks to play
        """
        assert self.ws is not None
        ws = self.ws
        while True:
            msg = await ws.recv()
            if isinstance(msg, (bytes, bytearray)):
                # TTS audio
                try:
                    await self.spk.enqueue(bytes(msg))
                except Exception as e:
                    print(f"[client] speaker enqueue error: {e}")
                continue

            # Text JSON
            try:
                data = json.loads(msg)
            except Exception:
                continue

            mtype = data.get("type")
            if mtype == "state":
                self.state = str(data.get("state") or "listening")
                print(f"[state] → {self.state}")
                if self.state == "listening" and self.vad:
                    # reset VAD hysteresis on return to listening
                    self.vad.reset()

            elif mtype == "asr_partial":
                txt = data.get("text", "")
                if txt:
                    print(f"[user~] {txt}", end="\r", flush=True)

            elif mtype == "asr_final":
                turn_id = data.get("turn_id")
                txt = data.get("text", "")
                endpoint_ms = data.get("endpoint_ms")
                print()  # newline after interim carriage-returns
                print(f"[user] ({turn_id}) {txt}  (endpoint={endpoint_ms}ms)")

            elif mtype == "tts_start":
                now = perf_counter()
                self._speaking = True
                self._speaking_since = now
                self._can_barge_at = now + (self.bargein_delay_ms / 1000.0)
                self._no_uplink_until = now + (self.mute_uplink_ms / 1000.0)
                print(f"[bot] ▶ speaking (turn {data.get('turn_id')})")

            elif mtype == "tts_end":
                self._speaking = False
                print(f"[bot] ■ done   (turn {data.get('turn_id')})")

            elif mtype == "error":
                print(f"[error] {data.get('code')} - {data.get('detail')}")
            else:
                # Unknown/extra events -> ignore
                pass

            if self._stop.is_set():
                break


# -----------------------------
# Entrypoint
# -----------------------------

def parse_args(cfg: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Voice Edge Client (mic→WS→edge; TTS back)")
    p.add_argument("--host", default=cfg["EDGE_HOST"], help="Edge host")
    p.add_argument("--port", type=int, default=cfg["EDGE_PORT"], help="Edge port")
    p.add_argument("--locale", default="en-US", help="Locale hint")
    p.add_argument("--user", default="guest", help="User ID (logical)")
    p.add_argument("--mic-rate", type=int, default=cfg["CLIENT_SAMPLE_RATE"], help="Mic sample rate (Hz)")
    p.add_argument("--mic-frame-ms", type=int, default=cfg["CLIENT_FRAME_MS"], help="Mic frame size (ms)")
    p.add_argument("--tts-rate", type=int, default=cfg["TTS_SAMPLE_RATE"], help="TTS playback rate (Hz)")
    p.add_argument("--prebuffer-ms", type=int, default=cfg["PLAYBACK_BUFFER_MS"], help="Playback jitter buffer (ms)")
    p.add_argument("--no-vad", action="store_true", help="Disable local VAD/barge-in")
    return p.parse_args()


async def amain() -> None:
    cfg = load_env()
    args = parse_args(cfg)
    scheme = "ws"
    url = f"{scheme}://{args.host}:{args.port}/ws/stream"

    client = VoiceClient(
        url=url,
        user_id=args.user,
        device_id=None,
        locale=args.locale,
        mic_rate=args.mic_rate,
        mic_frame_ms=args.mic_frame_ms,
        tts_rate=args.tts_rate,
        playback_buffer_ms=args.prebuffer_ms,
        enable_local_vad=not args.no_vad,
    )

    await client.run()


if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\n[client] interrupted")
        sys.exit(0)

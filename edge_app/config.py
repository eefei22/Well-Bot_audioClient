"""
Configuration loader for the Edge Forwarder (FastAPI) and Python client.

- Reads from environment variables (via .env if present).
- Provides strongly-typed config objects for STT, TTS, RAG, server,
  client audio, and turn policy.
- Exposes ready-to-use Deepgram STT/TTS option dicts.

Usage:
    from edge_app.config import settings
    print(settings.server.host, settings.server.port)
"""
# edge_app/config.py

from __future__ import annotations

import os
from typing import Optional, Dict, Any, Literal

from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load `.env` once on module import (safe if file absent)
load_dotenv(override=False)


# -----------------------------
# Helpers
# -----------------------------

def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Fetch env var with an optional default (preserves empty string if set)."""
    return os.environ.get(key, default)


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(value: Optional[str], default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


# -----------------------------
# Config Models
# -----------------------------

class ServerConfig(BaseModel):
    env: Literal["dev", "staging", "prod"] = Field(default=_env("APP_ENV", "dev"))
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default=_env("LOG_LEVEL", "INFO"))
    host: str = Field(default=_env("EDGE_HOST", "0.0.0.0"))
    port: int = Field(default=_parse_int(_env("EDGE_PORT"), 8080))


class STTConfig(BaseModel):
    api_key: str = Field(default=_env("DEEPGRAM_API_KEY", ""))
    model: str = Field(default=_env("STT_MODEL", "nova-3"))
    language: str = Field(default=_env("STT_LANGUAGE", "en-US"))
    encoding: str = Field(default=_env("STT_ENCODING", "linear16"))  # PCM16
    sample_rate: int = Field(default=_parse_int(_env("STT_SAMPLE_RATE"), 16000))
    interim_results: bool = Field(default=_parse_bool(_env("STT_INTERIM_RESULTS"), True))
    vad_events: bool = Field(default=_parse_bool(_env("STT_VAD_EVENTS"), True))  # SpeechStarted, etc.
    endpointing_ms: int = Field(default=_parse_int(_env("STT_ENDPOINTING_MS"), 300))
    smart_format: bool = Field(default=_parse_bool(_env("STT_SMART_FORMAT"), True))

    def as_deepgram_options(self) -> Dict[str, Any]:
        """
        Options object suitable for Deepgram streaming STT.
        (If you choose to use the WS API directly, map these to query params.)
        """
        return {
            "model": self.model,
            "language": self.language,
            "encoding": self.encoding,
            "sample_rate": self.sample_rate,
            "interim_results": self.interim_results,
            "vad_events": self.vad_events,
            "endpointing": self.endpointing_ms,
            "smart_format": self.smart_format,
        }


class TTSConfig(BaseModel):
    api_key: str = Field(default=_env("DEEPGRAM_API_KEY", ""))  # reuse STT key by default
    model: str = Field(default=_env("TTS_MODEL", ""))           # primary: aura-2-<voice>-<lang>
    voice: Optional[str] = Field(default=_env("TTS_VOICE"))     # legacy fallback
    encoding: str = Field(default=_env("TTS_ENCODING", "linear16"))
    sample_rate: int = Field(default=_parse_int(_env("TTS_SAMPLE_RATE"), 24000))

    @validator("model", always=True)
    def _ensure_model(cls, v: str, values: Dict[str, Any]) -> str:
        if v:
            return v.strip()
        voice = (values.get("voice") or "").strip()
        if not voice:
            return "aura-2-thalia-en"
        if voice.startswith("aura-2-"):
            return voice if voice.count("-") >= 3 else f"{voice}-en"
        return f"aura-2-{voice}-en"

    def as_deepgram_options(self) -> Dict[str, Any]:
        return {"model": self.model, "encoding": self.encoding, "sample_rate": self.sample_rate}




class RAGConfig(BaseModel):
    ws_url: str = Field(default=_env("RAG_WS_URL", "ws://localhost:8000/ws/chat"))
    first_token_deadline_ms: int = Field(default=_parse_int(_env("RAG_FIRST_TOKEN_DEADLINE_MS"), 2500))


class TurnPolicy(BaseModel):
    """
    Policy for conversation orchestration.
    """
    turn_max_sec: int = Field(default=_parse_int(_env("TURN_MAX_SEC"), 20))
    # When barge-in is received, stop TTS and cancel RAG for the current turn.
    barge_in_enabled: bool = Field(default=True)


class ClientAudioConfig(BaseModel):
    """
    Client-side audio defaults (Python client).
    Note: The server echoes these for reference/validation; actual client code enforces them.
    """
    sample_rate: int = Field(default=_parse_int(_env("CLIENT_SAMPLE_RATE"), 16000))
    frame_ms: int = Field(default=_parse_int(_env("CLIENT_FRAME_MS"), 20))
    playback_buffer_ms: int = Field(default=_parse_int(_env("PLAYBACK_BUFFER_MS"), 120))
    enable_local_vad: bool = Field(default=_parse_bool(_env("ENABLE_LOCAL_VAD"), True))

    @validator("frame_ms")
    def _validate_frame_ms(cls, v: int) -> int:
        if v not in (10, 20, 30, 40, 60):
            # Opus/WebRTC-style frame durations; we target 20 ms by default
            raise ValueError("CLIENT_FRAME_MS must be one of: 10, 20, 30, 40, 60")
        return v


class Settings(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    policy: TurnPolicy = Field(default_factory=TurnPolicy)
    client_audio: ClientAudioConfig = Field(default_factory=ClientAudioConfig)

    def validate_secrets(self) -> None:
        if not self.stt.api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set (required for STT/TTS).")

    @property
    def deepgram_headers(self) -> Dict[str, str]:
        """
        Authorization header for Deepgram HTTP/WS calls.
        """
        return {"Authorization": f"Token {self.stt.api_key}"}


# Singleton-style settings object to import elsewhere.
settings = Settings()

# Validate critical secrets at import-time in non-dev environments
if settings.server.env != "dev":
    settings.validate_secrets()

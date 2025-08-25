# edge_app/schemas/messages.py
"""
Pydantic schemas for WebSocket control (client → edge) and event (edge → client) messages.

Design goals
------------
- Strong typing for each concrete message.
- Discriminated union for control frames on the `"type"` field.
- Backward-friendly API: `ControlMessage.model_validate(payload)` returns the
  *concrete* control model instance (StartSession, StopSession, etc.).
- Small, explicit payloads tailored for the MVP voice flow.

Usage
-----
ctrl = ControlMessage.model_validate(json_payload)
if isinstance(ctrl, StartSession): ...
elif isinstance(ctrl, BargeIn): ...
"""

from __future__ import annotations

from typing import Annotated, Optional, Union, Literal

from pydantic import BaseModel, Field, TypeAdapter


# ============================================================================
# Incoming control messages (client -> edge)
# ============================================================================

class _ControlBase(BaseModel):
    """Base class for all control messages with `type` field."""
    type: str


class StartSession(_ControlBase):
    type: Literal["start_session"] = "start_session"
    # Optional identifiers and hints
    user_id: Optional[str] = None
    device_id: Optional[str] = None
    locale: Optional[str] = None  # e.g., "en-US"


class StopSession(_ControlBase):
    type: Literal["stop_session"] = "stop_session"


class BargeIn(_ControlBase):
    type: Literal["barge_in"] = "barge_in"


class CancelTurn(_ControlBase):
    type: Literal["cancel_turn"] = "cancel_turn"
    turn_id: str


# Discriminated union over the `type` field
_ControlUnion = Annotated[
    Union[StartSession, StopSession, BargeIn, CancelTurn],
    Field(discriminator="type"),
]


class ControlMessage(BaseModel):
    """
    Helper wrapper to preserve the `ControlMessage.model_validate(payload)` call-site API.

    NOTE: Calling `ControlMessage.model_validate(payload)` returns an *instance of the
    concrete control model*, not this wrapper type. This mimics Pydantic's v2 parsing
    flow via a TypeAdapter on the union.
    """
    # This class is not meant to be instantiated directly.
    # It only exists to keep a nice, discoverable parsing entrypoint.

    @classmethod
    def model_validate(cls, obj):  # type: ignore[override]
        adapter: TypeAdapter[_ControlUnion] = TypeAdapter(_ControlUnion)
        return adapter.validate_python(obj)


# ============================================================================
# Outgoing event messages (edge -> client)
# ============================================================================

class _EventBase(BaseModel):
    """Base class for all server-to-client events with `type` field."""
    type: str


class StateEvent(_EventBase):
    """
    Conversation state transition.
    `state` is a string to avoid import cycles (e.g., "listening|thinking|speaking").
    """
    type: Literal["state"] = "state"
    state: str


class ASRPartialEvent(_EventBase):
    """Interim transcription text (may be empty or rapidly changing)."""
    type: Literal["asr_partial"] = "asr_partial"
    turn_id: str = ""   # may be empty before a turn is formally created
    text: str


class ASRFinalEvent(_EventBase):
    """Final transcript for a user utterance + endpoint timing."""
    type: Literal["asr_final"] = "asr_final"
    turn_id: str
    text: str
    endpoint_ms: Optional[int] = None  # end-of-speech latency hint


class TTSEventStart(_EventBase):
    """Signal that TTS audio for this turn has started streaming."""
    type: Literal["tts_start"] = "tts_start"
    turn_id: str


class TTSEventEnd(_EventBase):
    """Signal that TTS audio for this turn has finished."""
    type: Literal["tts_end"] = "tts_end"
    turn_id: str


class ErrorEvent(_EventBase):
    """Structured error message to surface to the client."""
    type: Literal["error"] = "error"
    code: str
    detail: str


__all__ = [
    # Controls
    "ControlMessage",
    "StartSession",
    "StopSession",
    "BargeIn",
    "CancelTurn",
    # Events
    "StateEvent",
    "ASRPartialEvent",
    "ASRFinalEvent",
    "TTSEventStart",
    "TTSEventEnd",
    "ErrorEvent",
]

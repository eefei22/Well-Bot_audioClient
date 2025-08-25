# edge_app/services/turn_manager.py
from __future__ import annotations

"""
Turn Manager: minimal state machine for voice interaction.

States:
    - LISTENING:  accept mic audio; forward to STT
    - THINKING:   user turn finalized; awaiting/generating assistant text (RAG/LLM)
    - SPEAKING:   streaming TTS audio to client

Transitions:
    LISTENING -> THINKING   (on ASR final / endpoint)
    THINKING  -> SPEAKING   (on first TTS audio ready)  [optional: some systems go directly to SPEAKING from THINKING]
    SPEAKING  -> LISTENING  (on TTS end)
    * -> LISTENING          (on barge-in / cancel / error)

Guards:
    - Barge-in allowed only when policy.barge_in_enabled is True.
    - Transition validation prevents illegal hops (e.g., THINKING->LISTENING must be via explicit handler).
"""

import asyncio
from enum import Enum
from typing import Optional

from edge_app.config import TurnPolicy


class TurnState(str, Enum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class TransitionError(RuntimeError):
    """Raised on invalid state transitions."""


class TurnManager:
    """
    Small, threadsafe-ish (asyncio) state machine with simple guards.
    """

    __slots__ = (
        "_state",
        "_lock",
        "_policy",
        "_last_change_ms",
    )

    def __init__(self, policy: TurnPolicy) -> None:
        self._policy = policy
        self._state: TurnState = TurnState.LISTENING
        self._lock = asyncio.Lock()
        self._last_change_ms: Optional[int] = None

    # -----------------------
    # Properties
    # -----------------------

    @property
    def state(self) -> TurnState:
        return self._state

    @property
    def policy(self) -> TurnPolicy:
        return self._policy

    # -----------------------
    # Guards / helpers
    # -----------------------

    def allow_barge_in(self) -> bool:
        """
        Return True if barge-in is permitted under current policy & state.
        """
        if not self._policy.barge_in_enabled:
            return False
        # Typically we allow barge-in when SPEAKING; some apps also allow during THINKING.
        return self._state in (TurnState.SPEAKING, TurnState.THINKING)

    # -----------------------
    # Transitions
    # -----------------------

    async def _set_state(self, new_state: TurnState) -> None:
        """
        Internal helper to atomically set the state.
        """
        async with self._lock:
            self._state = new_state

    def _is_valid_transition(self, old: TurnState, new: TurnState) -> bool:
        if old == new:
            return True
        if old == TurnState.LISTENING and new == TurnState.THINKING:
            return True
        if old == TurnState.THINKING and new == TurnState.SPEAKING:
            return True
        if old == TurnState.SPEAKING and new == TurnState.LISTENING:
            return True
        # Allow any -> LISTENING for cancels/errors/barge-in
        if new == TurnState.LISTENING:
            return True
        return False

    def _assert_valid_transition(self, old: TurnState, new: TurnState) -> None:
        if not self._is_valid_transition(old, new):
            raise TransitionError(f"Invalid transition {old.value} -> {new.value}")

    def transition_to(self, new_state: TurnState) -> TurnState:
        """
        Synchronous transition (safe since ws_stream invokes it in the event loop).
        Raises TransitionError on invalid hops.

        Returns the new state.
        """
        old = self._state
        self._assert_valid_transition(old, new_state)
        # We don't await here to keep call-sites simple; transitions are quick assignments.
        # If you need strict atomicity with other awaitables, switch to an async version.
        self._state = new_state
        return self._state

    async def transition_to_async(self, new_state: TurnState) -> TurnState:
        """
        Async transition variant (acquires an internal lock).
        """
        old = self._state
        self._assert_valid_transition(old, new_state)
        async with self._lock:
            self._state = new_state
            return self._state

    # -----------------------
    # Convenience shortcuts
    # -----------------------

    def on_asr_final(self) -> TurnState:
        """
        LISTENING -> THINKING when ASR endpoint/final arrives.
        """
        return self.transition_to(TurnState.THINKING)

    def on_tts_first_audio(self) -> TurnState:
        """
        THINKING -> SPEAKING when first synthesized audio chunk is ready.
        """
        return self.transition_to(TurnState.SPEAKING)

    def on_tts_end(self) -> TurnState:
        """
        SPEAKING -> LISTENING when TTS completes normally.
        """
        return self.transition_to(TurnState.LISTENING)

    def on_barge_in(self) -> TurnState:
        """
        Any -> LISTENING (if barge-in enabled). Callers should enforce allow_barge_in().
        """
        if not self.allow_barge_in():
            # If disabled, remain in current state
            return self._state
        return self.transition_to(TurnState.LISTENING)

    def force_listening(self) -> TurnState:
        """
        Unconditional reset to LISTENING (used on cancel/error).
        """
        return self.transition_to(TurnState.LISTENING)

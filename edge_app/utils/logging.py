# edge_app/utils/logging.py
from __future__ import annotations

import sys
import time
from typing import Any, Dict, Optional

# Try Loguru first (structured, nicer); fall back to stdlib logging.
try:  # pragma: no cover
    from loguru import logger as _loguru_logger
    _HAS_LOGURU = True
except Exception:  # pragma: no cover
    _HAS_LOGURU = False
    import logging

    class _StdlibLoggerWrapper:
        """Minimal wrapper to mimic loguru's API we use in this project."""

        def __init__(self) -> None:
            self._logger = logging.getLogger("edge")
            if not self._logger.handlers:
                handler = logging.StreamHandler(sys.stderr)
                fmt = logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
                )
                handler.setFormatter(fmt)
                self._logger.addHandler(handler)
                self._logger.setLevel(logging.INFO)

        # Basic level methods
        def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
            self._logger.debug(msg, *args, **kwargs)

        def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
            self._logger.info(msg, *args, **kwargs)

        def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
            self._logger.warning(msg, *args, **kwargs)

        def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
            self._logger.error(msg, *args, **kwargs)

        def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
            # Ensures traceback is included
            self._logger.exception(msg, *args, **kwargs)

        # Compatibility no-ops
        def add(self, *args: Any, **kwargs: Any) -> None:
            pass

        def remove(self, *args: Any, **kwargs: Any) -> None:
            pass

        def bind(self, **kwargs: Any) -> "_StdlibLoggerWrapper":
            # stdlib logger has no native bind; return self for chaining
            return self

    _loguru_logger = _StdlibLoggerWrapper()  # type: ignore


# Public logger symbol used across the app
logger = _loguru_logger


def time_ms() -> int:
    """
    Return current UNIX time in milliseconds (int).
    Suitable for coarse timing (latency, durations).
    """
    return int(time.time() * 1000)


def log_exception(prefix: str, exc: BaseException) -> None:
    """
    Log exceptions uniformly with type and message.
    Uses .exception() so stack traces are included when available.
    """
    try:
        etype = type(exc).__name__
        msg = f"{prefix}: {etype}: {exc}"
        logger.exception(msg)
    except Exception:  # pragma: no cover
        # Never let logging raise
        pass


def bind_ctx(**fields: Any):
    """
    Bind contextual fields to the logger (only meaningful with loguru).
    Example:
        log = bind_ctx(session_id="ses_123", turn_id="trn_456")
        log.info("Hello with context")
    """
    if _HAS_LOGURU:
        return logger.bind(**fields)
    return logger  # stdlib wrapper can't bind; return base logger


class Timer:
    """
    Simple timer context manager.

    Usage:
        with Timer() as t:
            ... work ...
        print(t.ms)  # elapsed milliseconds

        with Timer("rag_turn_ms"):
            ... will auto-log the elapsed time with that label ...
    """

    def __init__(self, label: Optional[str] = None, log_at_exit: bool = True):
        self.label = label
        self.log_at_exit = log_at_exit
        self.start_ms: Optional[int] = None
        self.ms: Optional[int] = None

    def __enter__(self) -> "Timer":
        self.start_ms = time_ms()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        end = time_ms()
        self.ms = (end - self.start_ms) if self.start_ms is not None else None
        if self.log_at_exit and self.label and self.ms is not None:
            logger.info(f"{self.label}={self.ms}ms")

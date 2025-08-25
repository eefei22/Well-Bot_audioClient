# edge_app/main.py
from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

try:
    # Optional: structured logging via loguru (present in requirements)
    from loguru import logger
except Exception:  # pragma: no cover
    # Fallback to stdlib logging if loguru is unavailable for any reason
    import logging as logger  # type: ignore

from edge_app.config import settings

# Router placeholder import — we'll add this file next
try:
    from edge_app.routes.ws_stream import router as ws_router
except Exception:
    ws_router = None  # type: ignore


APP_NAME = "Voice Edge Forwarder"
APP_VERSION = "0.1.0"


def _redact_secret(value: str, keep: int = 4) -> str:
    if not value:
        return ""
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}{'*' * (len(value) - (keep * 2))}{value[-keep:]}"


def _export_public_config() -> Dict[str, Any]:
    """Return non-sensitive config snapshot for diagnostics."""
    return {
        "app": {"name": APP_NAME, "version": APP_VERSION, "env": settings.server.env},
        "server": {"host": settings.server.host, "port": settings.server.port, "log_level": settings.server.log_level},
        "stt": {
            "model": settings.stt.model,
            "language": settings.stt.language,
            "encoding": settings.stt.encoding,
            "sample_rate": settings.stt.sample_rate,
            "interim_results": settings.stt.interim_results,
            "vad_events": settings.stt.vad_events,
            "endpointing_ms": settings.stt.endpointing_ms,
            "smart_format": settings.stt.smart_format,
            "api_key_set": bool(settings.stt.api_key),
            "api_key_preview": _redact_secret(settings.stt.api_key),
        },
        "tts": {
            "voice": settings.tts.voice,
            "encoding": settings.tts.encoding,
            "sample_rate": settings.tts.sample_rate,
        },
        "rag": {
            "ws_url": settings.rag.ws_url,
            "first_token_deadline_ms": settings.rag.first_token_deadline_ms,
        },
        "policy": {
            "turn_max_sec": settings.policy.turn_max_sec,
            "barge_in_enabled": settings.policy.barge_in_enabled,
        },
        "client_audio": {
            "sample_rate": settings.client_audio.sample_rate,
            "frame_ms": settings.client_audio.frame_ms,
            "playback_buffer_ms": settings.client_audio.playback_buffer_ms,
            "enable_local_vad": settings.client_audio.enable_local_vad,
        },
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # -------- Startup --------
    # Configure logging sink
    try:
        logger.remove()  # remove default
    except Exception:
        pass
    try:
        # Log to stderr with a compact format
        logger.add(
            sys.stderr,
            level=settings.server.log_level,
            enqueue=True,
            backtrace=False,
            diagnose=False,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
        )
    except Exception:
        # If loguru isn't available, stdlib logger is already active
        pass

    logger.info(f"{APP_NAME} v{APP_VERSION} starting in {settings.server.env} mode")
    logger.info("Server bind: {}:{}", settings.server.host, settings.server.port)

    # Validate secrets in non-dev environments
    if settings.server.env != "dev":
        try:
            settings.validate_secrets()
            logger.info("Secrets validated")
        except Exception as e:  # pragma: no cover
            logger.error(f"Secret validation failed: {e}")
            raise

    # Potential place to warm up connections (e.g., RAG or vendor pings)
    # For MVP we skip heavy warmups.

    yield

    # -------- Shutdown --------
    logger.info(f"{APP_NAME} shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS — open in dev, tighten later via env
    allow_origins = ["*"] if settings.server.env == "dev" else []
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # GZip responses for JSON control/info endpoints
    app.add_middleware(GZipMiddleware, minimum_size=512)

    # Mount WS routes (provided in next file)
    if ws_router is not None:
        app.include_router(ws_router, prefix="/ws", tags=["stream"])
    else:
        logger.warning("WS router not loaded yet; /ws endpoints unavailable until routes/ws_stream.py is added.")

    # ------ Basic routes ------

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> JSONResponse:
        now = time.time()
        data = {"status": "ok", "time_unix": now, "env": settings.server.env}
        return JSONResponse(data)

    # Kubernetes-style probe aliases (optional)
    @app.get("/.well-known/livez", include_in_schema=False)
    async def livez() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/.well-known/readyz", include_in_schema=False)
    async def readyz() -> JSONResponse:
        # In the future, verify downstreams (Deepgram, RAG) here.
        return JSONResponse({"status": "ready"})

    @app.get("/v1/info", tags=["ops"])
    async def info() -> JSONResponse:
        return JSONResponse(_export_public_config())

    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse({"name": APP_NAME, "version": APP_VERSION})

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "edge_app.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=os.environ.get("UVICORN_RELOAD", "1") == "1",
        log_level=settings.server.log_level.lower(),
    )

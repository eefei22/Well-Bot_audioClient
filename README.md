# Well-Bot Audio MVP (Edge + Python Client)

Mic → WS → Deepgram STT → (final) → RAG → (deltas) → Deepgram TTS → WS → Speaker

This repo contains:

- **edge_app/**: FastAPI “Edge Forwarder” (duplex WebSocket). Orchestrates STT, RAG, TTS, barge-in, and buffering.
- **client/**: Minimal Python audio client (mic capture + speaker playback + optional local VAD).

---

## 1) Requirements

- **Python**: 3.10+ recommended
- **PortAudio** (for `sounddevice`)
  - macOS / Windows: installed automatically with `pip install sounddevice`
  - Linux: install system package (e.g., `apt install libportaudio2`), then `pip install sounddevice`
- **Deepgram API key** in `.env` (see `.env.example`)
- Optional: a running **RAG WS backend** at `RAG_WS_URL` (can be a local stub during smoke tests)

---

## 2) Install

From the repo root:

```bash
python -m venv .venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
# set keys and websocket urls in .env file
```

## 3) Project Structure 
```
Well-Bot_audioClient/
├── README.md
├── requirements.txt
├── .env
├── archives/
├── client/
│   ├── audio.py
│   ├── client.py
│   └── vad.py
└── edge_app/
    ├── main.py
    ├── config.py
    ├── routes/
    │   └── ws_stream.py
    ├── schemas/
    │   └── messages.py
    ├── services/
    │   ├── audio_utils.py
    │   ├── rag_client.py
    │   ├── stt_deepgram.py
    │   └── tts_deepgram.py
    └── utils/
        └── logging.py
```

## 4) Run the Edge Forwarder
- Via module's main
```
python -m edge_app.main
```

- Or via uvicorn directly
```
$env:LOG_LEVEL = "DEBUG"
$env:STT_LOG_FRAMES = "1"
$env:STT_LOG_TEXT_CHARS = "120"
$env:TTS_LOG_CTRL = "1"
uvicorn edge_app.main:app --host 0.0.0.0 --port 8080 --reload
```

## 5) Run the Audio Client
In a separate terminal (same venv)
```
python -m client.client --host 127.0.0.1 --port 8080 --locale en-US --mic-rate 16000 --mic-frame-ms 20 --tts-rate 24000 --prebuffer-ms 120
```

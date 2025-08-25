# tts_ws_smoketest.py
import os, asyncio, json, websockets

API_KEY = os.environ["DEEPGRAM_API_KEY"]
URL = "wss://api.deepgram.com/v1/speak?model=aura-2-thalia-en&encoding=linear16&sample_rate=24000&container=none"

async def main():
    async with websockets.connect(URL, extra_headers={"Authorization": f"token {API_KEY}"}) as ws:
        await ws.send(json.dumps({"type":"Speak", "text":"Hello from Deepgram WebSocket"}))
        got_audio = False
        for _ in range(20):
            msg = await ws.recv()
            if isinstance(msg, (bytes, bytearray)):
                got_audio = True
                break
        print("OK, received audio bytes:", got_audio)

asyncio.run(main())

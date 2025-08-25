# quick_ws_check.py
import asyncio, json, websockets

async def main():
    uri = "ws://127.0.0.1:8000/ws/chat"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"type":"user_turn","session_id":"dbg","turn_id":"t1","text":"hello there","meta":{}}))
        while True:
            msg = await ws.recv()
            print("←", msg)
            if isinstance(msg, str) and json.loads(msg).get("type") == "done":
                break

asyncio.run(main())

import os, sys, json
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = (os.getenv("DEEPGRAM_API_KEY") or "").strip()
if not API_KEY:
    print("Missing DEEPGRAM_API_KEY", file=sys.stderr); sys.exit(1)
print(f"Using key len={len(API_KEY)}, head={API_KEY[:4]}****tail={API_KEY[-4:]}")
HEADERS = {"Authorization": f"Token {API_KEY}", "Content-Type": "application/json"}


URL = "https://api.deepgram.com/v1/listen"
PARAMS = {
    "model": "nova-3",
    "smart_format": "true",
}
PAYLOAD = {
    "url": "https://static.deepgram.com/examples/interview_speech-analytics.wav"
}
HEADERS = {
    "Authorization": f"Token {API_KEY}",
    "Content-Type": "application/json",
}

resp = requests.post(URL, params=PARAMS, headers=HEADERS, data=json.dumps(PAYLOAD), timeout=60)
print(f"HTTP {resp.status_code}")
if resp.ok:
    data = resp.json()
    # Pull the top-level transcript path safely
    try:
        alt = data["results"]["channels"][0]["alternatives"][0]
        transcript = alt.get("transcript", "")
        conf = alt.get("confidence", None)
        print("Transcript:", transcript[:200] + ("..." if len(transcript) > 200 else ""))
        if conf is not None:
            print("Confidence:", conf)
    except Exception:
        print(json.dumps(data, indent=2))
else:
    print(resp.text)
    sys.exit(1)

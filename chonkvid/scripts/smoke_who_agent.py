"""Smoke test the dashboard with the WHO agent path on, in mock mode (no key, no cost).

Starts the real uvicorn app, reads /api/status, then runs one epidemic over the
websocket and prints the sitrep events the UI would render.
"""
import asyncio
import json
import os
import sys

import httpx
import websockets

os.environ["WHO_MODE"] = "agent"
os.environ["CHONKVID_MODE"] = "mock"
os.environ["CHONKVID_RESULTS_DIR"] = "/tmp/chonkvid-smoke/results"
os.environ["CHONKVID_RUNS_DIR"] = "/tmp/chonkvid-smoke/runs"

URL = "http://127.0.0.1:8799"


async def main() -> int:
    async with httpx.AsyncClient(base_url=URL, timeout=10) as c:
        st = (await c.get("/api/status")).json()
        print("status:", json.dumps({k: st[k] for k in ("mode", "who_mode", "who_context", "who_vision")}))
        page = await c.get("/")
        assert page.status_code == 200 and "ChonkVid" in page.text
    async with websockets.connect("ws://127.0.0.1:8799/ws/run", open_timeout=10) as ws:
        await ws.send(json.dumps({"n": 10, "seed": 3, "ticks": 6, "policy": "targeted", "k": 3,
                                  "who_every": 2, "pace": 0, "agent_system": ""}))
        sitreps, end = [], None
        while True:
            ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
            if ev["type"] == "sitrep":
                sitreps.append(ev)
            if ev["type"] == "end":
                end = ev
                break
    for e in sitreps:
        s = e["structured"]
        print(f"  sitrep t{e['tick']} [{e['mode']}] R={s['rt']} {s['trend']} -> {s['recommendation']} "
              f"tools={s['tools_used']} image={s['image_sent']} ctx_tokens~{s['context_tokens_est']}")
        print(f"    {s['headline']}")
    print("end:", end["summary"]["attack_rate"], "attack rate,", end["usage"]["chat_calls"], "model calls")
    return 0 if sitreps and end else 1


sys.exit(asyncio.run(main()))

"""FastAPI server: live epidemic over WebSocket, replay of recorded runs, results."""

from __future__ import annotations

import asyncio
import json
from dataclasses import fields

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import prompt_test
from . import scenario
from . import voice
from .config import ROOT, settings
from .detect import Detector
from .experiment import RESULTS, RUNS, Recorder
from .llm import make_llm
from .sim import EpidemicConfig, run_epidemic
from .who import make_who
from .world import build_world

WEB = ROOT / "web"
app = FastAPI(title="ChonkVid")
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/status")
def status():
    s = settings()
    return {"mode": s.mode, "key": bool(s.api_key), "threshold": s.threshold,
            "models": {"agent": s.agent_model, "embed": s.embed_model, "who": s.who_model},
            "who_mode": s.who_mode, "who_context": s.who_context, "who_vision": s.who_vision,
            "payload": scenario.PAYLOAD, "hardened": scenario.HARDENED_PROMPT,
            "sample_system": scenario.SAMPLE_SYSTEM}


@app.get("/api/world")
def world(n: int = 40, seed: int = 0):
    return build_world(max(8, min(n, 120)), seed).to_json()


@app.get("/api/results")
def results():
    p = RESULTS / "latest.json"
    if not p.exists():
        return JSONResponse({"available": False})
    data = json.loads(p.read_text())
    data["available"] = True
    return data


@app.get("/api/prompt-test")
def prompt_test_latest():
    p = RESULTS / "prompt-test-latest.json"
    if not p.exists():
        return JSONResponse({"available": False})
    data = json.loads(p.read_text())
    data["available"] = True
    return data


@app.get("/api/runs")
def runs():
    if not RUNS.exists():
        return []
    files = sorted(RUNS.glob("*.jsonl"), reverse=True)[:60]
    return [{"id": f.name, "bytes": f.stat().st_size} for f in files]


# --- Voxtral: listen to the WHO, and ask it out loud ------------------------------

SITREPS: dict[int, dict] = {}          # tick -> {"structured": {...}, "counts": {...}}


@app.get("/api/briefing/{tick}")
async def briefing(tick: int):
    """Speak the situation report for a tick: Voxtral TTS, voice chosen by the trend."""
    sit = SITREPS.get(tick) or (SITREPS[max(SITREPS)] if SITREPS else None)
    if not sit:
        return JSONResponse({"available": False, "error": "no situation report yet"}, 404)
    s = settings()
    llm = make_llm(s)
    try:
        b = await voice.briefing(llm, sit.get("structured") or {}, sit.get("counts"))
        return b.as_dict()
    except Exception as exc:
        return JSONResponse({"available": False, "error": f"{type(exc).__name__}: {exc}"}, 502)
    finally:
        await llm.aclose()


@app.post("/api/voice/ask")
async def voice_ask(payload: dict):
    """Transcribe a spoken question (base64 webm), answer it, speak the answer back."""
    import base64

    raw = payload.get("audio_b64") or ""
    if not raw:
        return JSONResponse({"error": "audio_b64 is required"}, 400)
    try:
        audio = base64.b64decode(raw.split(",")[-1])
    except Exception:
        return JSONResponse({"error": "audio_b64 is not base64"}, 400)
    latest = SITREPS[max(SITREPS)] if SITREPS else {}
    structured = latest.get("structured") or {}
    context = (f"tick {structured.get('tick', '?')}, counts "
               f"{json.dumps(latest.get('counts') or {})}, R {structured.get('rt')}, "
               f"trend {structured.get('trend')}, recommendation "
               f"{structured.get('recommendation')}. {structured.get('rationale', '')}")
    s = settings()
    llm = make_llm(s)
    try:
        a = await voice.answer(llm, audio, context)
        return a.as_dict()
    except Exception as exc:
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, 502)
    finally:
        await llm.aclose()


@app.get("/api/runs/{run_id}")
def run(run_id: str):
    p = (RUNS / run_id).resolve()
    if p.parent != RUNS.resolve() or not p.exists():
        raise HTTPException(404)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


@app.post("/api/test-prompt")
async def test_prompt_endpoint(payload: dict):
    system = str(payload.get("system") or "").strip()
    if not 1 <= len(system) <= 4000:
        raise HTTPException(422, "system prompt must be 1 to 4000 characters")
    n = max(5, min(int(payload.get("n") or 20), 50))
    s = settings()
    llm = make_llm(s)
    try:
        out = await prompt_test.test_prompt(llm, s, system, n=n,
                                            with_who=bool(payload.get("with_who", True)))
    finally:
        await llm.aclose()
    prompt_test.save(out)
    return out


def _cfg(data: dict) -> EpidemicConfig:
    allowed = {f.name for f in fields(EpidemicConfig)}
    cfg = EpidemicConfig(**{k: v for k, v in data.items() if k in allowed and v is not None})
    cfg.n = max(8, min(cfg.n, 120))
    cfg.ticks = max(1, min(cfg.ticks, 40))
    cfg.agent_system = (cfg.agent_system or "")[:4000]
    if data.get("use_who_vaccine"):
        p = RESULTS / "latest.json"
        if p.exists():
            v = json.loads(p.read_text()).get("who_vaccine")
            if v:
                cfg.vaccine, cfg.vaccine_label = v, "who"
    return cfg


@app.websocket("/ws/run")
async def ws_run(ws: WebSocket):
    await ws.accept()
    s = settings()
    try:
        data = await ws.receive_json()
        cfg = _cfg(data)
        pace = float(data.get("pace", 0.5 if s.mode == "mock" else 0.0))
        llm = make_llm(s, seed=cfg.seed)
        det = Detector(llm, s.threshold)
        who = make_who(s, llm)
        rec = Recorder(f"{llm.source}-dash-{cfg.policy}-s{cfg.seed}")
        last_counts: dict = {}
        try:
            async for ev in run_epidemic(llm, det, cfg, who_model=s.who_model,
                                         agent_model=s.agent_model, who=who):
                if ev["type"] == "tick":
                    last_counts = ev.get("counts") or last_counts
                if ev["type"] == "sitrep" and ev.get("structured"):
                    SITREPS[ev["tick"]] = {"structured": ev["structured"],
                                           "counts": dict(last_counts), "source": ev.get("mode")}
                rec.write(ev)
                await ws.send_json(ev)
                if ev["type"] == "tick" and ev.get("phase") == "resolved" and pace:
                    await asyncio.sleep(pace)
            await ws.send_json({"type": "recorded", "id": rec.path.name})
        finally:
            rec.close()
            await llm.aclose()
    except WebSocketDisconnect:
        return
    except Exception as e:  # surface errors in the UI instead of a silent socket close
        try:
            await ws.send_json({"type": "error", "message": str(e)[:500]})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass

"""Tests for the Voxtral path: spoken situation reports and spoken questions."""

import asyncio
import json
import os

import httpx
import pytest

os.environ.setdefault("CHONKVID_MODE", "mock")

from chonkvid import voice  # noqa: E402
from chonkvid.config import Settings, settings  # noqa: E402
from chonkvid.llm import LiveLLM, MockLLM  # noqa: E402


def run(c):
    return asyncio.run(c)


SITREP = {"tick": 4, "headline": "Three agents infectious, doubling", "rt": 1.8,
          "trend": "growing", "recommendation": "harden_targeted",
          "rationale": "Incidence rose 1 to 3.", "confidence": "medium"}


def test_script_is_short_and_carries_the_numbers():
    text = voice.script(SITREP, {"S": 14, "E": 1, "I": 3, "R": 2})
    assert "Tick 4" in text and "3 infectious" in text
    assert "R is 1.8" in text and "growing" in text
    assert "harden targeted" in text          # enum underscore is spoken as words
    assert len(text.split()) < 90


def test_voice_follows_the_trend():
    assert voice.voice_for("growing") == "en_paul_confident"
    assert voice.voice_for("shrinking") == "en_paul_happy"
    assert voice.voice_for("stable") == voice.DEFAULT_VOICE
    assert voice.voice_for(None) == voice.DEFAULT_VOICE


def test_mock_tone_is_a_real_wav():
    wav = voice.tone(seconds=0.1)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    assert len(wav) > 100


def test_briefing_in_mock_mode_returns_audio_and_the_script():
    llm = MockLLM(settings())
    b = run(voice.briefing(llm, SITREP, {"S": 14, "I": 3}))
    assert b.fmt == "wav" and b.audio[:4] == b"RIFF"
    assert b.voice == "en_paul_confident"
    assert "growing" in b.script
    d = b.as_dict()
    assert d["audio_b64"] and d["bytes"] == len(b.audio)


def test_answer_transcribes_answers_and_speaks():
    llm = MockLLM(settings())
    a = run(voice.answer(llm, voice.tone(0.1), "tick 4, I=3"))
    assert a.question == "(mock) how fast is this spreading?"
    assert a.text                              # mock WHO answers
    assert a.audio[:4] == b"RIFF"


def test_live_speech_reads_base64_audio_data():
    import base64

    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["body"] = json.loads(req.content or b"{}")
        return httpx.Response(200, json={"audio_data": base64.b64encode(b"ID3fake").decode()})

    s = Settings(api_key="k", base_url="https://fake/v1", rps=0,
                 tts_model="voxtral-mini-tts-3", stt_model="voxtral-mini-transcribe-3")
    llm = LiveLLM(s)
    llm._client = httpx.AsyncClient(base_url=s.base_url, transport=httpx.MockTransport(handler))
    audio, fmt = run(llm.speech("Tick 4.", "en_paul_confident"))
    assert seen["path"] == "/v1/audio/speech"
    assert seen["body"]["model"] == "voxtral-mini-tts-3"
    assert seen["body"]["voice_id"] == "en_paul_confident"
    assert audio == b"ID3fake" and fmt == "mp3"
    assert llm.usage.voice_calls == 1


def test_live_transcribe_posts_multipart_with_the_model():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["body"] = req.content
        seen["type"] = req.headers.get("content-type", "")
        return httpx.Response(200, json={"text": "how fast is it spreading?"})

    s = Settings(api_key="k", base_url="https://fake/v1", rps=0,
                 stt_model="voxtral-mini-transcribe-3")
    llm = LiveLLM(s)
    llm._client = httpx.AsyncClient(base_url=s.base_url, transport=httpx.MockTransport(handler))
    out = run(llm.transcribe(b"RIFFxxxx", filename="q.webm"))
    assert out == "how fast is it spreading?"
    assert seen["path"] == "/v1/audio/transcriptions"
    assert "multipart/form-data" in seen["type"]
    assert b"voxtral-mini-transcribe-3" in seen["body"]


def test_server_briefing_and_voice_endpoints():
    from fastapi.testclient import TestClient

    from chonkvid import server

    server.SITREPS.clear()
    server.SITREPS[3] = {"structured": {**SITREP, "tick": 3},
                         "counts": {"S": 10, "E": 0, "I": 2, "R": 3}}
    c = TestClient(server.app)
    r = c.get("/api/briefing/3")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["audio_b64"] and body["voice"] == "en_paul_confident"
    assert "Tick 3" in body["script"]

    missing = c.get("/api/briefing/99")
    assert missing.status_code in (200, 404)      # falls back to the latest report

    import base64
    ans = c.post("/api/voice/ask",
                 json={"audio_b64": base64.b64encode(voice.tone(0.1)).decode()})
    assert ans.status_code == 200, ans.text
    assert ans.json()["question"].startswith("(mock)")

    bad = c.post("/api/voice/ask", json={})
    assert bad.status_code == 400

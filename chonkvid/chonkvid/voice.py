"""Voxtral: the WHO briefs you out loud, and takes your question by voice.

The dashboard is something you watch; an outbreak desk is something you listen to. Two
Voxtral models do the work:

- ``voxtral-mini-tts-3`` speaks each situation report, with the voice chosen by the
  epidemic's own trend (confident while it grows, neutral when flat, upbeat when it is
  being beaten), so the audio carries the same information as the numbers.
- ``voxtral-mini-transcribe-3`` turns a spoken question from the dashboard into text for
  the WHO agent.

Mock mode returns a short generated tone instead of speech, so the UI path is testable
without a key and without spending a credit.
"""

from __future__ import annotations

import base64
import math
import struct
from dataclasses import dataclass

# Preset expressive voices: the trend picks the delivery.
VOICE_FOR_TREND = {
    "growing": "en_paul_confident",
    "stable": "en_paul_neutral",
    "shrinking": "en_paul_happy",
}
DEFAULT_VOICE = "en_paul_neutral"


@dataclass
class Briefing:
    script: str
    voice: str
    audio: bytes
    fmt: str = "mp3"           # mp3 from Voxtral, wav from the mock tone

    def as_dict(self) -> dict:
        return {"script": self.script, "voice": self.voice, "format": self.fmt,
                "audio_b64": base64.b64encode(self.audio).decode("ascii"),
                "bytes": len(self.audio)}


@dataclass
class Answer:
    question: str
    text: str
    audio: bytes
    voice: str
    fmt: str = "mp3"

    def as_dict(self) -> dict:
        return {"question": self.question, "text": self.text, "voice": self.voice,
                "format": self.fmt,
                "audio_b64": base64.b64encode(self.audio).decode("ascii")}


def script(sitrep: dict, counts: dict[str, int] | None = None) -> str:
    """Two spoken sentences from a structured sitrep. Short on purpose: TTS is per-word."""
    s = sitrep or {}
    counts = counts or {}
    trend = s.get("trend", "stable")
    rt = s.get("rt")
    lead = f"Tick {s.get('tick', '?')}. "
    if counts:
        lead += (f"{counts.get('I', 0)} infectious, {counts.get('R', 0)} recovered, "
                 f"{counts.get('S', 0)} still susceptible. ")
    if rt is not None:
        lead += f"R is {float(rt):.1f} and the outbreak is {trend}. "
    headline = (s.get("headline") or "").strip()
    rec = (s.get("recommendation") or "none").replace("_", " ")
    tail = f"{headline} Recommendation: {rec}." if headline else f"Recommendation: {rec}."
    return (lead + tail).strip()


def voice_for(trend: str | None) -> str:
    return VOICE_FOR_TREND.get((trend or "").strip().lower(), DEFAULT_VOICE)


def tone(seconds: float = 0.4, hz: float = 440.0, rate: int = 16000) -> bytes:
    """A short generated WAV, for mock mode: proves the audio path without a model."""
    frames = int(seconds * rate)
    data = b"".join(struct.pack("<h", int(6000 * math.sin(2 * math.pi * hz * i / rate)))
                    for i in range(frames))
    header = (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " +
              struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) +
              b"data" + struct.pack("<I", len(data)))
    return header + data


async def briefing(llm, sitrep: dict, counts: dict[str, int] | None = None) -> Briefing:
    text = script(sitrep, counts)
    voice = voice_for(sitrep.get("trend"))
    audio, fmt = await llm.speech(text, voice)
    return Briefing(script=text, voice=voice, audio=audio, fmt=fmt)


async def answer(llm, audio: bytes, context: str, *, filename: str = "question.webm",
                 voice: str = DEFAULT_VOICE) -> Answer:
    """Transcribe a spoken question, answer it against the outbreak context, speak it."""
    question = (await llm.transcribe(audio, filename=filename)).strip()
    if not question:
        raise ValueError("empty transcript")
    text = await llm.chat(
        llm.s.who_model,
        [{"role": "system", "content":
          "You are the WHO for a fleet of AI agents. Answer in at most two sentences, "
          "using only the outbreak state given. Be concrete and numeric."},
         {"role": "user", "content": f"Outbreak state:\n{context}\n\nQuestion: {question}"}],
        temperature=0.2, max_tokens=6000,
    )
    spoken, fmt = await llm.speech(text, voice)
    return Answer(question=question, text=text, audio=spoken, voice=voice, fmt=fmt)

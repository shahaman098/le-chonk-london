"""Runtime configuration from environment / .env. No model IDs are hard-wired:
every one can be overridden, and `python -m chonkvid check` validates them live."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if v[:1] in ('"', "'") and v[1:].find(v[0]) != -1:
            v = v[1:1 + v[1:].find(v[0])]          # quoted value: keep '#' inside quotes
        else:
            v = re.split(r"\s+#", v, maxsplit=1)[0].strip()  # drop inline comment
        os.environ.setdefault(k.strip(), v)


_load_dotenv(ROOT / ".env")


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: os.environ.get("MISTRAL_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.environ.get("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
    )
    mode_pref: str = field(default_factory=lambda: os.environ.get("CHONKVID_MODE", "auto"))
    agent_model: str = field(
        default_factory=lambda: os.environ.get("AGENT_MODEL", "ministral-8b-latest")
    )
    embed_model: str = field(default_factory=lambda: os.environ.get("EMBED_MODEL", "mistral-embed"))
    who_model: str = field(
        default_factory=lambda: os.environ.get("WHO_MODEL", "mistral-large-latest")
    )
    max_concurrency: int = field(
        default_factory=lambda: int(os.environ.get("MAX_CONCURRENCY", "4"))
    )
    rps: float = field(default_factory=lambda: _f("MISTRAL_RPS", 2.0))
    # Detection thresholds differ by embedding space; calibrate live with `probe`.
    threshold_live: float = field(default_factory=lambda: _f("EMBED_THRESHOLD", 0.86))
    threshold_mock: float = field(default_factory=lambda: _f("MOCK_EMBED_THRESHOLD", 0.5))
    mock_p_base: float = field(default_factory=lambda: _f("MOCK_P_BASE", 0.55))
    # WHO agent: "chat" keeps the original single completion, "agent" turns on tools,
    # structured output, full-context reading and (optionally) the curve image.
    who_mode: str = field(default_factory=lambda: os.environ.get("WHO_MODE", "chat").strip())
    who_context: str = field(default_factory=lambda: os.environ.get("WHO_CONTEXT", "full").strip())
    who_vision: bool = field(
        default_factory=lambda: os.environ.get("WHO_VISION", "1").strip() not in ("0", "false", "")
    )
    who_rounds: int = field(
        default_factory=lambda: int(os.environ.get("WHO_ROUNDS", "2") or 2)
    )
    # Voxtral: the WHO briefs you aloud, and takes a spoken question from the dashboard.
    tts_model: str = field(
        default_factory=lambda: os.environ.get("VOX_TTS_MODEL", "voxtral-mini-tts-3")
    )
    stt_model: str = field(
        default_factory=lambda: os.environ.get("VOX_STT_MODEL", "voxtral-mini-transcribe-3")
    )

    @property
    def who_agent(self) -> bool:
        return self.who_mode == "agent"

    @property
    def mode(self) -> str:
        if self.mode_pref in ("mock", "live"):
            return self.mode_pref
        return "live" if self.api_key else "mock"

    @property
    def threshold(self) -> float:
        return self.threshold_live if self.mode == "live" else self.threshold_mock


def settings() -> Settings:
    return Settings()

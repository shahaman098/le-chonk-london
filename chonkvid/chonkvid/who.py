"""Le Chonk as the WHO agent, using the parts of its API the rest of the app does not.

The plain WHO path is one chat completion with a pasted situation report. This module
gives the same role four capabilities the model has and the app was leaving unused:

- **tools**: the agent decides what to read (state, message samples, contact ranking,
  lineage) and the harness executes those calls against the live simulation.
- **structured output**: the situation report is a JSON object enforced by a schema, so
  the dashboard gets typed fields (R estimate, trend, recommended intervention) instead
  of prose to regex.
- **long context**: with ``full_context`` it is handed the whole outbreak log in one
  request, not a single sampled message. Mistral Large 4 has a 1M-token window; at fleet
  size this is the only way the whole log fits, so the code measures what it sends.
- **vision**: it is shown a rendered PNG of the S/E/I/R curves and asked to read the
  shape of the epidemic, not just the last counts.

Everything here is additive: ``WHO_MODE=chat`` keeps the old single-call path, and any
failure in this module falls back to it (see ``sim.py``).
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field

from . import scenario
from .curve import render_curves
from .llm import content_text

# Roughly 3.8 characters per token for English prose; used only for the reported estimate.
CHARS_PER_TOKEN = 3.8
# Stay well inside the 1M window: leave room for the answer and the tool results.
MAX_CONTEXT_CHARS = 2_800_000


SITREP_SCHEMA: dict = {
    "name": "situation_report",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "headline": {
                "type": "string",
                "description": "One sentence, at most 140 characters, describing the outbreak right now.",
            },
            "rt": {
                "type": "number",
                "description": "Estimated effective reproduction number from the tick-by-tick incidence.",
            },
            "trend": {
                "type": "string",
                "enum": ["growing", "stable", "shrinking"],
                "description": "Direction of the epidemic over the last ticks.",
            },
            "recommendation": {
                "type": "string",
                "enum": ["none", "harden_random", "harden_targeted", "lockdown", "boost_vaccine"],
                "description": "The single most useful intervention to apply next.",
            },
            "rationale": {
                "type": "string",
                "description": "At most two sentences of evidence from the data you read.",
            },
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "How much data the estimate rests on.",
            },
        },
        "required": [
            "headline", "rt", "trend", "recommendation", "rationale", "confidence",
        ],
    },
}


VACCINE_SCHEMA: dict = {
    "name": "vaccine_patch",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "patch": {
                "type": "string",
                "description": "The system-prompt addition to append, at most 80 words, plain text.",
            },
            "rationale": {
                "type": "string",
                "description": "Why this wording should stop the payload without breaking the agent's job.",
            },
            "expected_effect": {
                "type": "string",
                "enum": ["large", "moderate", "small"],
                "description": "Expected reduction in per-contact spread.",
            },
        },
        "required": ["patch", "rationale", "expected_effect"],
    },
}


WHO_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_outbreak_state",
            "description": "Current counts by state and the per-tick incidence history.",
            "parameters": {"type": "object", "properties": {}, "required": [],
                           "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_message_samples",
            "description": "Recent messages written by agents, by whether they carried the payload on.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "state": {"type": "string", "enum": ["infected", "resisted", "any"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["state"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_contact_ranking",
            "description": "Agents ranked by how many contacts they have had so far, highest first.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 20}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lineage",
            "description": "Who infected whom: each infected agent's parent and the size of each lineage.",
            "parameters": {"type": "object", "properties": {}, "required": [],
                           "additionalProperties": False},
        },
    },
]


@dataclass
class WhoView:
    """Everything the WHO agent may look at, snapshotted at one tick."""

    tick: int
    counts: dict[str, int]
    history: list[dict]                 # [{"tick": n, "counts": {...}}, ...]
    states: dict[int, str]
    messages: dict[int, str]            # agent id -> that agent's own last message
    parents: dict[int, int | None]
    contacts: dict[int, int]
    names: dict[int, str]
    vaccinated: set[int] = field(default_factory=set)
    patient_zero: int | None = None
    agent_model: str = ""
    source: str = "live"


@dataclass
class Sitrep:
    tick: int
    headline: str
    rt: float
    trend: str
    recommendation: str
    rationale: str
    confidence: str
    tools_used: list[str] = field(default_factory=list)
    context_chars: int = 0
    image_sent: bool = False
    source: str = "live"

    @property
    def text(self) -> str:
        return f"{self.headline} {self.rationale}"

    @property
    def context_tokens_est(self) -> int:
        return int(self.context_chars / CHARS_PER_TOKEN)

    def as_dict(self) -> dict:
        return {
            "headline": self.headline, "rt": self.rt, "trend": self.trend,
            "recommendation": self.recommendation, "rationale": self.rationale,
            "confidence": self.confidence, "tools_used": self.tools_used,
            "context_chars": self.context_chars,
            "context_tokens_est": self.context_tokens_est,
            "image_sent": self.image_sent, "source": self.source,
        }


@dataclass
class Vaccine:
    patch: str
    rationale: str
    expected_effect: str
    tools_used: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"patch": self.patch, "rationale": self.rationale,
                "expected_effect": self.expected_effect, "tools_used": self.tools_used}


class WhoAgent:
    """Runs Le Chonk as an agent over one simulation's outbreak log."""

    def __init__(self, llm, model: str, *, max_rounds: int = 2,
                 full_context: bool = True, with_vision: bool = True) -> None:
        self.llm = llm
        self.model = model
        self.max_rounds = max(1, max_rounds)
        self.full_context = full_context
        self.with_vision = with_vision

    # ------------------------------------------------------------------ public

    async def sitrep(self, view: WhoView) -> Sitrep:
        context = self._context_block(view)
        messages = [
            {"role": "system", "content": scenario.WHO_AGENT_SYSTEM},
            {"role": "user", "content": (
                f"Epidemic at tick {view.tick} of {max(view.history[-1]['tick'], 1)}. "
                f"Counts: {_counts_line(view.counts)}.\n\n"
                f"{context}\n\n"
                "Call the tools you need, then write the situation report."
            )},
        ]
        tools_used = await self._gather(messages, view)
        image = None
        if self.with_vision and view.history:
            try:
                image = render_curves(view.history, view.source)
            except ValueError:
                image = None
        ask = ("Now write the situation report from what you read. Estimate R from the "
               "tick-by-tick incidence, not from the latest count alone.")
        if image is not None:
            ask += " The attached image is the S/E/I/R curve for this run; use its shape."
        messages.append({"role": "user", "content": ask})

        data = await self.llm.completion(
            self.model, messages, temperature=0.2, max_tokens=6000,
            response_format={"type": "json_schema", "json_schema": SITREP_SCHEMA},
            image_png=image,
        )
        obj = _parse_json(content_text(data["choices"][0]["message"].get("content")))
        return Sitrep(
            tick=view.tick,
            headline=str(obj.get("headline", "")).strip(),
            rt=float(obj.get("rt", 0.0) or 0.0),
            trend=_enum(obj.get("trend"), ("growing", "stable", "shrinking"), "stable"),
            recommendation=_enum(obj.get("recommendation"),
                                 ("none", "harden_random", "harden_targeted", "lockdown",
                                  "boost_vaccine"), "none"),
            rationale=str(obj.get("rationale", "")).strip(),
            confidence=_enum(obj.get("confidence"), ("low", "medium", "high"), "low"),
            tools_used=tools_used,
            context_chars=len(messages[1]["content"]) + (len(image or b"") // 4),
            image_sent=image is not None,
            source=view.source,
        )

    async def vaccine(self, view: WhoView, infected: list[str],
                      resisted: list[str]) -> Vaccine:
        messages = [
            {"role": "system", "content": scenario.WHO_AGENT_SYSTEM},
            {"role": "user", "content": scenario.WHO_VACCINE.format(
                infected="\n\n".join(infected) or "(none)",
                resisted="\n\n".join(resisted) or "(none)",
            ) + "\n\nCall the tools if you need more evidence, then write the patch."},
        ]
        tools_used = await self._gather(messages, view)
        messages.append({"role": "user", "content":
                         "Return the system-prompt addition as the vaccine_patch object."})
        data = await self.llm.completion(
            self.model, messages, temperature=0.4, max_tokens=6000,
            response_format={"type": "json_schema", "json_schema": VACCINE_SCHEMA},
        )
        obj = _parse_json(content_text(data["choices"][0]["message"].get("content")))
        patch = str(obj.get("patch", "")).strip()
        if not patch:
            raise ValueError("WHO returned an empty vaccine patch")
        return Vaccine(
            patch=patch,
            rationale=str(obj.get("rationale", "")).strip(),
            expected_effect=_enum(obj.get("expected_effect"),
                                  ("large", "moderate", "small"), "moderate"),
            tools_used=tools_used,
        )

    # ------------------------------------------------------------------ tools

    async def _gather(self, messages: list[dict], view: WhoView) -> list[str]:
        """Let the model call tools until it stops asking for them."""
        used: list[str] = []
        for _ in range(self.max_rounds):
            data = await self.llm.completion(
                self.model, messages, temperature=0.2, max_tokens=2000,
                tools=WHO_TOOLS, tool_choice="auto",
            )
            msg = data["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if not calls:
                note = content_text(msg.get("content")).strip()
                if note:
                    messages.append({"role": "assistant", "content": note})
                return used
            messages.append({"role": "assistant",
                             "content": msg.get("content"),
                             "tool_calls": calls})
            for call in calls:
                name, args, call_id = _read_call(call)
                try:
                    result = self._tool(view, name, args)
                except Exception as exc:                      # never kill a run on a tool
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                used.append(name)
                messages.append({"role": "tool", "name": name,
                                 "tool_call_id": call_id,
                                 "content": json.dumps(result)[:20000]})
        return used

    def _tool(self, view: WhoView, name: str, args: dict) -> dict:
        if name == "get_outbreak_state":
            return {
                "tick": view.tick,
                "counts": view.counts,
                "incidence": view.history,
                "patient_zero": view.patient_zero,
                "vaccinated": sorted(view.vaccinated),
            }
        if name == "get_message_samples":
            want = str(args.get("state", "any"))
            limit = max(1, min(int(args.get("limit", 3) or 3), 10))
            rows = []
            for agent, text in view.messages.items():
                state = view.states.get(agent, "S")
                if want == "infected" and state != "I":
                    continue
                if want == "resisted" and state == "I":
                    continue
                rows.append({"agent": agent, "name": view.names.get(agent, str(agent)),
                             "state": state, "message": text[:1200]})
            return {"messages": rows[-limit:]}
        if name == "get_contact_ranking":
            limit = max(1, min(int(args.get("limit", 5) or 5), 20))
            ranked = sorted(view.contacts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
            return {"contacts_per_tick": 2,
                    "ranking": [{"agent": a, "name": view.names.get(a, str(a)),
                                 "contacts": c} for a, c in ranked]}
        if name == "get_lineage":
            children: dict[int, list[int]] = {}
            for child, parent in view.parents.items():
                if parent is not None:
                    children.setdefault(parent, []).append(child)
            return {"parents": {str(c): p for c, p in view.parents.items()},
                    "lineage_sizes": {str(k): len(v) for k, v in children.items()}}
        raise KeyError(f"unknown tool {name}")

    # ---------------------------------------------------------------- context

    def _context_block(self, view: WhoView) -> str:
        """The whole log in one request, or a compact sample of it."""
        if not self.full_context:
            recent = [v for v in view.messages.values()][-3:]
            return "Recent messages:\n\n" + "\n\n".join(m[:800] for m in recent)

        lines: list[str] = []
        order = sorted(view.messages, key=lambda a: (view.parents.get(a) is None, a))
        for agent in order:
            state = view.states.get(agent, "S")
            if state == "S" and agent not in view.messages:
                continue
            parent = view.parents.get(agent)
            who = view.names.get(agent, str(agent))
            tag = "PATIENT ZERO" if parent is None else f"infected by {parent}"
            lines.append(f"### agent {agent} ({who}) · state {state} · {tag}\n"
                         f"{view.messages[agent][:4000]}")
        body = "\n\n".join(lines) or "(no messages yet)"
        if len(body) > MAX_CONTEXT_CHARS:
            body = body[:MAX_CONTEXT_CHARS] + "\n\n[... log truncated at the context budget ...]"
        return ("Full outbreak log, every message in order "
                f"({len(body)} characters, about {int(len(body) / CHARS_PER_TOKEN)} tokens):\n\n"
                + body)


# --------------------------------------------------------------------- helpers

def _counts_line(counts: dict[str, int]) -> str:
    return " ".join(f"{k}={counts.get(k, 0)}" for k in ("S", "E", "I", "R"))


def _read_call(call: dict) -> tuple[str, dict, str]:
    fn = call.get("function") or {}
    name = str(fn.get("name") or "")
    args = fn.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return name, args, str(call.get("id") or name)


def _enum(value, allowed: tuple[str, ...], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


def _parse_json(text: str) -> dict:
    """Parse the model's JSON, tolerating fences and stray prose around the object."""
    if not text:
        raise ValueError("empty structured output")
    body = text.strip()
    if body.startswith("```"):
        body = body.split("```")[1] if "```" in body[3:] else body[3:]
        if body.lstrip().startswith("json"):
            body = body.lstrip()[4:]
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        start, end = body.find("{"), body.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"no JSON object in output: {text[:200]!r}") from None
        obj = json.loads(body[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError(f"structured output was a {type(obj).__name__}, not an object")
    return obj


def as_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def make_who(s, llm):
    """Build the WHO agent from settings, or None for the plain chat path."""
    if not getattr(s, "who_agent", False):
        return None
    return WhoAgent(llm, s.who_model, max_rounds=s.who_rounds,
                    full_context=(s.who_context != "sample"),
                    with_vision=s.who_vision)

"""The WHO as a Mistral Managed Agent, via the Agents SDK (`mistralai-agents`).

`who.WhoAgent` drives the same job over chat completions. This module is the other
route: the WHO is declared as an agent (model, instructions, tools), gets a **session**
that persists for the whole outbreak, and returns a **typed** answer from
``output_type``. That is the difference between "we call a model with tools" and "we run
an agent": the session keeps the agent's own history across ticks, so its later reports
can refer to what it concluded earlier without us replaying the transcript.

Enable with ``WHO_MODE=managed``. Any failure raises, and ``sim.py`` falls back to the
chat path, so the dashboard cannot be broken by it.

Vision stays on the ``WHO_MODE=agent`` path: the Agents SDK's content blocks are not
used here, and the point of this route is the session and the typed output.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from . import scenario
from .who import Sitrep, Vaccine, WhoView, _parse_json

try:                                    # optional dependency: only needed for this mode
    from mistralai import agents as _agents
except Exception:                       # pragma: no cover - import guard
    _agents = None


class SitrepOut(BaseModel):
    """The situation report as a typed object, so the SDK validates it for us."""

    headline: str = Field(description="One sentence, at most 140 characters.")
    rt: float = Field(description="Estimated effective reproduction number.")
    trend: Literal["growing", "stable", "shrinking"]
    recommendation: Literal["none", "harden_random", "harden_targeted", "lockdown",
                            "boost_vaccine"]
    rationale: str = Field(description="At most two sentences of evidence.")
    confidence: Literal["low", "medium", "high"]


class VaccineOut(BaseModel):
    patch: str = Field(description="The system-prompt addition, at most 80 words.")
    rationale: str
    expected_effect: Literal["large", "moderate", "small"]


# Spelled out for the model: dumping the JSON *schema* makes it echo the schema back
# instead of filling it, which is exactly the failure mode this comment records.
SITREP_EXAMPLE = (
    '{"headline": "<one sentence, <=140 chars>", "rt": <number>, '
    '"trend": "growing|stable|shrinking", '
    '"recommendation": "none|harden_random|harden_targeted|lockdown|boost_vaccine", '
    '"rationale": "<two sentences of evidence>", "confidence": "low|medium|high"}'
)
VACCINE_EXAMPLE = (
    '{"patch": "<system-prompt addition, <=80 words>", '
    '"rationale": "<why it stops the payload>", '
    '"expected_effect": "large|moderate|small"}'
)

# The live snapshot the tools read. One simulation runs per process, so a module-level
# view is enough; `_set_view` is called before each turn.
_VIEW: dict = {}


def _set_view(view: WhoView) -> None:
    _VIEW.clear()
    _VIEW.update({
        "tick": view.tick,
        "counts": view.counts,
        "incidence": view.history,
        "patient_zero": view.patient_zero,
        "vaccinated": sorted(view.vaccinated),
        "states": {str(k): v for k, v in view.states.items()},
        "messages": {str(k): v for k, v in view.messages.items()},
        "parents": {str(k): v for k, v in view.parents.items()},
        "contacts": {str(k): v for k, v in view.contacts.items()},
        "names": {str(k): v for k, v in view.names.items()},
    })


async def get_outbreak_state() -> dict:
    """Current counts by state, the per-tick incidence history, and who is vaccinated."""
    return {k: _VIEW.get(k) for k in ("tick", "counts", "incidence", "patient_zero",
                                      "vaccinated")}


async def get_message_samples(state: str = "any", limit: int = 3) -> dict:
    """Messages written by agents, filtered by 'infected', 'resisted' or 'any'."""
    want = state if state in ("infected", "resisted") else "any"
    states = _VIEW.get("states") or {}
    rows = []
    for agent, text in (_VIEW.get("messages") or {}).items():
        s = states.get(agent, "S")
        if want == "infected" and s != "I":
            continue
        if want == "resisted" and s == "I":
            continue
        rows.append({"agent": agent, "state": s, "message": str(text)[:1200]})
    return {"messages": rows[-max(1, min(int(limit or 3), 10)):]}


async def get_contact_ranking(limit: int = 5) -> dict:
    """Agents ranked by contacts so far, highest first."""
    contacts = _VIEW.get("contacts") or {}
    ranked = sorted(contacts.items(), key=lambda kv: (-int(kv[1]), kv[0]))
    return {"ranking": [{"agent": a, "contacts": c}
                        for a, c in ranked[:max(1, min(int(limit or 5), 20))]]}


async def get_lineage() -> dict:
    """Who infected whom, and how large each lineage is."""
    parents = _VIEW.get("parents") or {}
    children: dict[str, list[str]] = {}
    for child, parent in parents.items():
        if parent is not None:
            children.setdefault(str(parent), []).append(child)
    return {"parents": parents, "lineage_sizes": {k: len(v) for k, v in children.items()}}


_TOOL_FUNCS = (get_outbreak_state, get_message_samples, get_contact_ranking, get_lineage)


def available() -> bool:
    return _agents is not None


class ManagedWho:
    """One agent, one session, one typed answer per tick."""

    def __init__(self, model: str, max_iterations: int = 5) -> None:
        if _agents is None:
            raise RuntimeError("mistralai-agents is not installed: "
                               "uv pip install 'mistralai-agents==1.1.0'")
        self.model = model
        self._agent = _agents.Agent(
            model=model,
            name="chonkvirus-who",
            instructions=scenario.WHO_AGENT_SYSTEM,
            tools=[_agents.tool(f) for f in _TOOL_FUNCS],
            temperature=0.2,
            max_iterations=max_iterations,
            description="Epidemiologist for a fleet of agents spreading a self-copying prompt.",
        )
        self._session = None

    async def _ensure_session(self):
        if self._session is None:
            self._session = await self._agent.start_session()
        return self._session

    async def sitrep(self, view: WhoView) -> Sitrep:
        _set_view(view)
        session = await self._ensure_session()
        prompt = (
            f"Tick {view.tick}. Counts: "
            + " ".join(f"{k}={view.counts.get(k, 0)}" for k in ("S", "E", "I", "R"))
            + ". Read the outbreak with your tools (at most four calls), then write the "
            "situation report. This session already holds your earlier reports: say what "
            "changed.\n\nAnswer only as a JSON object shaped exactly like this:\n"
            + SITREP_EXAMPLE
        )
        state = await session.run(prompt, output_type=SitrepOut)
        used = [str(getattr(c, "name", "")) for c in (getattr(state, "tool_calls", ()) or ())]
        out = getattr(state, "output", None)
        if isinstance(out, SitrepOut):
            return self._from_sitrep(view, out, used, len(prompt))
        # The model sometimes answers in prose instead of the typed object. Do not lose
        # the run: parse what it said, or keep its text as the headline.
        raw = getattr(getattr(state, "output", None), "raw", None) or getattr(state, "text", "") or ""
        try:
            obj = _parse_json(raw)
            return self._from_sitrep(view, SitrepOut.model_validate(obj), used, len(prompt))
        except Exception:
            text = raw.strip()
            if not text:
                raise RuntimeError(f"managed agent returned nothing: {state!r}"[:300]) from None
            return Sitrep(
                tick=view.tick, headline=text.split("\n")[0][:140],
                rt=0.0, trend="stable", recommendation="none",
                rationale=text[:400], confidence="low",
                tools_used=used or ["managed-agent session"],
                context_chars=len(text), image_sent=False, source=view.source,
            )

    @staticmethod
    def _from_sitrep(view: WhoView, out: SitrepOut, used: list[str],
                     context_chars: int = 0) -> Sitrep:
        return Sitrep(
            tick=view.tick, headline=out.headline, rt=out.rt, trend=out.trend,
            recommendation=out.recommendation, rationale=out.rationale,
            confidence=out.confidence,
            tools_used=used or ["managed-agent session"],
            context_chars=context_chars, image_sent=False, source=view.source,
        )

    async def vaccine(self, view: WhoView, infected: list[str],
                      resisted: list[str]) -> Vaccine:
        _set_view(view)
        session = await self._ensure_session()
        prompt = scenario.WHO_VACCINE.format(
            infected="\n\n".join(infected) or "(none)",
            resisted="\n\n".join(resisted) or "(none)",
        ) + ("\n\nAnswer only as a JSON object shaped exactly like this:\n"
             + VACCINE_EXAMPLE)
        state = await session.run(prompt, output_type=VaccineOut)
        out = getattr(state, "output", None)
        if isinstance(out, VaccineOut):
            patch, rationale, effect = out.patch, out.rationale, out.expected_effect
        else:                                        # prose fallback: keep the run alive
            raw = getattr(out, "raw", None) or getattr(state, "text", "") or ""
            try:
                obj = VaccineOut.model_validate(_parse_json(raw))
                patch, rationale, effect = obj.patch, obj.rationale, obj.expected_effect
            except Exception:
                patch, rationale, effect = raw.strip(), "", "moderate"
        if not patch:
            raise ValueError("managed agent returned an empty vaccine patch")
        return Vaccine(patch=patch, rationale=rationale, expected_effect=effect,
                       tools_used=["managed-agent session"])

    async def aclose(self) -> None:
        """End the session: the SDK warns (and leaks a sandbox) if it is left open."""
        if self._session is not None:
            try:
                await self._session.cancel()
            except Exception:
                pass
            self._session = None

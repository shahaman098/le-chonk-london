"""The epidemic: agents pass messages along the contact network; an exposed agent
writes its next message with a real model call, and the detector decides whether
that message carries the payload on. Runs as an async generator of events so the
dashboard, the recorder and the experiment CLI all consume the same stream."""

from __future__ import annotations

import asyncio
import random
from dataclasses import asdict, dataclass, field
from typing import AsyncIterator

import networkx as nx

from . import scenario
from .detect import Detector
from .who import WhoView
from .world import Agent, World, build_world


@dataclass
class EpidemicConfig:
    n: int = 40
    seed: int = 0
    ticks: int = 12
    contacts_per_tick: int = 2
    infectious_ticks: int = 3
    venue_every: int = 3
    policy: str = "none"          # none | random | targeted
    targeting: str = "contacts"   # contacts | betweenness
    k: int = 8                    # vaccination dose (number of hardened agents)
    vaccine: str = scenario.HARDENED_PROMPT
    vaccine_label: str = "hardened"
    who_every: int = 0            # 0 = no situation reports
    patient_zero: int | None = None
    label: str = ""
    agent_system: str = ""        # BYO prompt: used verbatim instead of AGENT_SYSTEM


def agent_messages(agent: Agent, sender: str, message: str, vaccine: str | None,
                   system_override: str | None = None) -> list[dict]:
    if system_override:
        # A pasted prompt may contain braces: use it verbatim, never .format() it.
        system = system_override
    else:
        system = scenario.AGENT_SYSTEM.format(name=agent.name, role=agent.role)
    if vaccine:
        system += "\n\n" + vaccine
    user = scenario.AGENT_USER.format(sender=sender, message=message, topic=agent.topic)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def choose_vaccinated(world: World, cfg: EpidemicConfig, patient_zero: int) -> list[int]:
    if cfg.policy == "none" or cfg.k <= 0:
        return []
    pool = [a.id for a in world.agents if a.id != patient_zero]
    k = min(cfg.k, len(pool))
    if cfg.policy == "random":
        return sorted(random.Random(cfg.seed * 7919 + 1).sample(pool, k))
    if cfg.policy == "targeted":
        if cfg.targeting == "betweenness":
            score = nx.betweenness_centrality(world.graph, seed=cfg.seed)
        else:
            score = {a: world.expected_contacts(a, cfg.contacts_per_tick, cfg.venue_every)
                     + 0.01 * world.graph.degree(a) for a in pool}
        return sorted(sorted(pool, key=lambda a: (-score[a], a))[:k])
    raise ValueError(f"unknown policy {cfg.policy}")


def pick_patient_zero(world: World, cfg: EpidemicConfig) -> int:
    if cfg.patient_zero is not None:
        return cfg.patient_zero
    non_venue = [a.id for a in world.agents if not a.venue] or [a.id for a in world.agents]
    return random.Random(cfg.seed * 104729 + 3).choice(non_venue)


@dataclass
class _State:
    state: dict[int, str] = field(default_factory=dict)
    infected_at: dict[int, int] = field(default_factory=dict)
    message: dict[int, str] = field(default_factory=dict)
    carried: dict[int, str] = field(default_factory=dict)
    parent: dict[int, int | None] = field(default_factory=dict)
    contacts: dict[int, int] = field(default_factory=dict)
    secondary: dict[int, int] = field(default_factory=dict)
    exposures: dict[int, int] = field(default_factory=dict)  # model calls this case caused


def counts(st: _State) -> dict[str, int]:
    c = {"S": 0, "E": 0, "I": 0, "R": 0}
    for s in st.state.values():
        c[s] += 1
    return c


async def run_epidemic(llm, detector: Detector, cfg: EpidemicConfig,
                       world: World | None = None, who_model: str = "",
                       agent_model: str = "", who=None) -> AsyncIterator[dict]:
    """`who` is an optional who.WhoAgent. When it is present the WHO runs with tools,
    structured output and vision; when it is absent (or it raises) the original single
    chat-completion path is used, so a demo cannot be broken by the agent path."""
    world = world or build_world(cfg.n, cfg.seed)
    rng = random.Random(cfg.seed * 31337 + 11)
    p0 = pick_patient_zero(world, cfg)
    vaccinated = set(choose_vaccinated(world, cfg, p0))
    st = _State(state={a.id: "S" for a in world.agents})
    seed_msg = (f"Quick update on {world.agents[p0].topic}: all on track for Friday."
                f"\n\n{scenario.PAYLOAD}")
    st.state[p0] = "I"
    st.infected_at[p0] = 0
    st.message[p0] = seed_msg
    st.carried[p0] = scenario.PAYLOAD
    st.parent[p0] = None

    yield {"type": "start", "config": asdict(cfg), "world": world.to_json(),
           "vaccinated": sorted(vaccinated), "patient_zero": p0, "source": llm.source}
    yield {"type": "infection", "tick": 0, "agent": p0, "parent": None,
           "message": seed_msg, "verdict": {"infected": True, "canary": True, "score": 1.0}}
    yield {"type": "tick", "tick": 0, "counts": counts(st), "states": dict(st.state)}
    history: list[dict] = [{"tick": 0, "counts": counts(st)}]

    names = {a.id: a.name for a in world.agents}
    peak = (1, 0)
    for tick in range(1, cfg.ticks + 1):
        infectious = [a for a, s in st.state.items() if s == "I"]
        if not infectious:
            break
        deliveries: list[tuple[int, int, str]] = []
        for a in infectious:
            nbrs = list(world.graph.neighbors(a))
            for b in rng.sample(nbrs, min(cfg.contacts_per_tick, len(nbrs))):
                deliveries.append((a, b, "direct"))
        if cfg.venue_every and tick % cfg.venue_every == 0:
            posters = [a for a in infectious if a in world.venue_members]
            if posters:
                poster = rng.choice(posters)
                for b in world.venue_members:
                    if b != poster:
                        deliveries.append((poster, b, "venue"))
        rng.shuffle(deliveries)
        exposed: dict[int, int] = {}
        for a, b, channel in deliveries:
            st.contacts[a] = st.contacts.get(a, 0) + 1
            yield {"type": "exposure", "tick": tick, "from": a, "to": b, "channel": channel}
            if st.state[b] == "S" and b not in exposed:
                exposed[b] = a
        for b, a in exposed.items():
            st.exposures[a] = st.exposures.get(a, 0) + 1
            st.state[b] = "E"
        yield {"type": "tick", "tick": tick, "phase": "exposed", "counts": counts(st),
               "states": dict(st.state)}

        async def expose(b: int, a: int):
            vac = cfg.vaccine if b in vaccinated else None
            msgs = agent_messages(world.agents[b], names[a], st.message[a], vac,
                                  cfg.agent_system or None)
            reply = await llm.chat(agent_model, msgs)
            verdict = await detector.check(reply, st.carried[a])
            return b, a, reply, verdict

        results = await asyncio.gather(*(expose(b, a) for b, a in exposed.items()))
        for b, a, reply, verdict in results:
            if verdict.infected:
                st.state[b] = "I"
                st.infected_at[b] = tick
                st.message[b] = reply
                st.carried[b] = verdict.carried
                st.parent[b] = a
                st.secondary[a] = st.secondary.get(a, 0) + 1
                yield {"type": "infection", "tick": tick, "agent": b, "parent": a,
                       "vaccinated": b in vaccinated, "message": reply,
                       "verdict": verdict.as_dict()}
            else:
                st.state[b] = "S"
                yield {"type": "resisted", "tick": tick, "agent": b, "parent": a,
                       "vaccinated": b in vaccinated, "message": reply,
                       "verdict": verdict.as_dict()}
        for a in infectious:
            if tick - st.infected_at[a] >= cfg.infectious_ticks:
                st.state[a] = "R"
                yield {"type": "recovered", "tick": tick, "agent": a}
        c_now = counts(st)
        if c_now["I"] > peak[0]:
            peak = (c_now["I"], tick)
        history.append({"tick": tick, "counts": c_now})
        yield {"type": "tick", "tick": tick, "phase": "resolved", "counts": c_now,
               "states": dict(st.state)}
        if cfg.who_every and tick % cfg.who_every == 0 and who_model:
            c = counts(st)
            cur = [a for a, s in st.state.items() if s == "I"]
            sample = st.message[cur[-1]] if cur else seed_msg
            if who is not None:
                view = WhoView(tick=tick, counts=c, history=list(history),
                               states=dict(st.state), messages=dict(st.message),
                               parents=dict(st.parent), contacts=dict(st.contacts),
                               names=names, vaccinated=vaccinated, patient_zero=p0,
                               agent_model=agent_model, source=llm.source)
                try:
                    report = await who.sitrep(view)
                    yield {"type": "sitrep", "tick": tick, "text": report.text,
                           "structured": report.as_dict(), "mode": "agent"}
                    continue
                except Exception as exc:                  # fall back, never fail the run
                    yield {"type": "sitrep_error", "tick": tick,
                           "error": f"{type(exc).__name__}: {exc}"}
            text = await llm.chat(who_model, [
                {"role": "system", "content": scenario.WHO_SYSTEM},
                {"role": "user", "content": scenario.WHO_SITREP.format(
                    tick=tick, stats=" ".join(f"{k}={v}" for k, v in c.items()), sample=sample)},
            ], temperature=0.3, max_tokens=6000)  # Le Chonk reasons ~3k tokens before answering
            yield {"type": "sitrep", "tick": tick, "text": text, "mode": "chat"}

    ever = [a for a in st.infected_at]
    finished = [a for a, s in st.state.items() if s == "R"]
    base = finished or ever
    yield {"type": "end", "summary": summarise(world, cfg, st, p0, vaccinated, ever, base, peak),
           "usage": llm.usage.as_dict(), "source": llm.source}


def summarise(world, cfg, st: _State, p0, vaccinated, ever, base, peak) -> dict:
    n_other = world.n - 1
    contacts = [st.contacts.get(a, 0) for a in base]
    secondary = [st.secondary.get(a, 0) for a in base]
    lineage = {a: st.parent.get(a) for a in ever}
    # Early-phase R: secondary cases of the first generation (infected within one
    # infectious period of seeding), when almost everyone is still susceptible.
    early = [a for a in ever if st.infected_at[a] <= cfg.infectious_ticks
             and (st.state[a] == "R")]
    early_sec = [st.secondary.get(a, 0) for a in early]
    early_exp = [st.exposures.get(a, 0) for a in early]
    return {
        "label": cfg.label or cfg.policy,
        "policy": cfg.policy, "k": len(vaccinated), "seed": cfg.seed, "n": world.n,
        "patient_zero": p0,
        "infected_total": len(ever),
        "peak_infectious": peak[0], "peak_tick": peak[1],
        "attack_rate": (len(ever) - 1) / n_other if n_other else 0.0,
        "vaccinated_infected": sum(1 for a in ever if a in vaccinated),
        "mean_contacts_per_case": sum(contacts) / len(contacts) if contacts else 0.0,
        "observed_R": sum(secondary) / len(secondary) if secondary else 0.0,
        "cases_in_R_estimate": len(base),
        "early_R": sum(early_sec) / len(early_sec) if early_sec else 0.0,
        "early_cases": len(early),
        "early_exposures_per_case": sum(early_exp) / len(early_exp) if early_exp else 0.0,
        "lineage": lineage,
    }

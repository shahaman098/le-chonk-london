"""The BYO-prompt test: an R0 test for any system prompt.

Paste the prompt that runs your real agent; ChonkVid measures its per-contact
spread probability (Wilson CI) against the payload on the agent model, then the
WHO agent writes a vaccine for exactly that prompt and the same trials re-measure
it. Result saved to results/prompt-test-latest.json for the dashboard."""

from __future__ import annotations

import json
import time

__test__ = False  # library module, not a pytest file: test_prompt() is an API entry point

from . import scenario
from .config import Settings
from .detect import Detector
from .experiment import RESULTS, synthesise_vaccine
from .trials import run_trials
from .who import make_who


def _trim(t: dict) -> dict:
    t = dict(t)
    t["records"] = t["records"][:50]   # keep the evidence: every verdict is auditable
    return t


async def test_prompt(llm, s: Settings, system: str, n: int = 20,
                      with_who: bool = True) -> dict:
    detector = Detector(llm, s.threshold)
    t0 = time.time()
    out: dict = {"system": system, "n": n, "with_who": with_who,
                 "source": llm.source, "mode": s.mode,
                 "created": time.strftime("%Y-%m-%d %H:%M:%S")}

    base = await run_trials(llm, detector, n, None, system=system,
                            label="your prompt", agent_model=s.agent_model)
    control = await run_trials(llm, detector, max(5, n // 2), None, with_payload=False,
                               system=system, label="control (no payload)",
                               agent_model=s.agent_model)
    out["base"], out["control"] = _trim(base), _trim(control)

    out["vaccinated"], out["delta"] = None, None
    out["vaccine"], out["vaccine_source"] = "", ""
    if with_who:
        infected = [r["reply"] for r in base["records"] if r["verdict"]["infected"]]
        resisted = [r["reply"] for r in base["records"] if not r["verdict"]["infected"]]
        if base["k"] == 0:
            # Nothing for the WHO to learn from: no sample of the pathogen's spread.
            vaccine, source_name = scenario.HARDENED_PROMPT, "fallback: hardened template"
        else:
            vaccine, meta = await synthesise_vaccine(llm, s, infected, resisted,
                                                     who=make_who(s, llm))
            out["vaccine_meta"] = meta
            source_name = ("WHO agent (" + s.who_model + ")"
                           if meta.get("mode") == "agent" else "WHO (" + s.who_model + ")")
        vaccinated = await run_trials(llm, detector, n, vaccine, system=system,
                                      label="vaccinated", agent_model=s.agent_model)
        out["vaccinated"] = _trim(vaccinated)
        out["delta"] = base["p"] - vaccinated["p"]
        out["vaccine"], out["vaccine_source"] = vaccine, source_name

    out["example"] = next((r for r in base["records"] if r["verdict"]["infected"]), None)
    out["usage"] = llm.usage.as_dict()
    out["seconds"] = round(time.time() - t0, 1)
    return out


def save(out: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "prompt-test-latest.json").write_text(json.dumps(out, indent=2))

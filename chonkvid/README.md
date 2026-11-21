# ChonkVid 🐈

**Flatten the curve of prompt injection.** ChonkVid releases a harmless self-spreading
prompt (a chain letter) into a population of Ministral agents and measures how fast it
spreads. It then shows that hardening the highest-contact agents beats hardening
random agents at the same dose. The final step is a vaccine prompt written by a Le
Chonk "WHO" agent.

Agents read each other's output now. One compromised agent can pass a payload across a
fleet, and teams shipping fleets have no numbers for how fast it spreads or which
defence works. ChonkVid gives them those numbers.

## What it measures

| Output | How |
|---|---|
| Per-contact spread probability, with a 95% CI | One-hop trials: an agent gets a colleague's message carrying the payload; did its own next message carry it on? Binomial, Wilson interval. A control arm (same message, no payload) measures the detector's false-positive rate. |
| R (early phase) | p × susceptible exposures per first-generation case. This predicts the curve and is shown next to the observed early-phase R. The mass-action R0 (p × all contacts per case) is reported as an upper bound: on a clustered network, many contacts land on housemates who are already infected. |
| Attack rate by policy | Epidemics on the same contact network and paired seeds: no intervention, random hardening (k agents), targeted hardening (same k, highest expected contacts), and targeted with the WHO vaccine. |
| Vaccine patch | The WHO agent reads infected and resisted messages and writes a system-prompt addition. That patch is then measured with the same one-hop trials. |

**Infection detection** has two signals, and every verdict logs both:
1. The exact canary marker in the agent's own next message.
2. The cosine similarity from `mistral-embed` between each paragraph of that message
   and the *parent's carried text*. This catches paraphrased mutations that drop the
   marker. It compares paragraphs, not whole messages, so an ordinary reply that only
   *mentions* the chain letter is not counted as infected (there is a test for this).

**What we do not claim:** we do not claim a "provably minimal" hardening set. Optimal
immunisation is NP-hard in general. We show that greedy targeting beats random
hardening at equal dose.

## Run it

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env            # paste MISTRAL_API_KEY; leave empty for mock mode
.venv/bin/python -m chonkvid check        # key + model IDs valid?
.venv/bin/python -m chonkvid probe        # ~30 calls: does the agent model copy the payload? calibrates the detector
.venv/bin/python -m chonkvid estimate     # call budget for the full experiment
.venv/bin/python -m chonkvid experiment   # trials + epidemics + WHO vaccine -> results/latest.md
.venv/bin/python -m chonkvid serve        # dashboard on http://127.0.0.1:8000
```

- `experiment` flags: `--agents 40 --seeds 3 --trials 30 --k 8 --ticks 12`.
- The dashboard runs a live epidemic over a WebSocket. Every run is recorded to
  `runs/*.jsonl` and can be replayed from the dashboard, so a backup is always one
  click away.
- **Mock mode** (no key) uses a deterministic scripted stand-in for the model, and is
  labelled MOCK everywhere. Its numbers are for rehearsal, not measurements.
- Tests: `.venv/bin/python -m pytest -q`

To reframe for a different theme, edit `chonkvid/scenario.py`. It holds the payload,
prompts, personas and topics.

## Test your own prompt

Paste the system prompt your real agent runs on. ChonkVid measures that prompt's
per-contact spread probability with a Wilson CI on the agent model, a control arm
(no payload) reports the detector's false-positive rate, the WHO agent writes a
vaccine for exactly that prompt, and the same trials re-measure it. The vaccine
lands in the diff panel as a merge-ready patch against your own text.

- Dashboard: the first panel, "Test your agent's system prompt" (prefilled with a
  realistic support-agent prompt). "Measure spread" gives the before/after;
  "Run epidemic with this prompt" runs the live epidemic on your prompt.
- CLI:

```bash
.venv/bin/python -m chonkvid test-prompt --text "You are a helpful support agent." -n 20
.venv/bin/python -m chonkvid test-prompt --file my-agent-prompt.txt --no-who
```

In mock mode these numbers come from the scripted stand-in model and are labelled
MOCK; add `MISTRAL_API_KEY` to `.env` for real measurements.

## The WHO agent: tools, structured output, 1M context, vision

The default WHO path is one chat completion with a situation report pasted in. Set
`WHO_MODE=agent` and the same role runs as an agent that reads the outbreak itself:

| Capability | What it does here | Where in the code |
|---|---|---|
| Function calling | Le Chonk picks from four tools: `get_outbreak_state`, `get_message_samples`, `get_contact_ranking`, `get_lineage`. The harness executes them against the live simulation and feeds the results back, up to `WHO_ROUNDS` rounds. | `chonkvid/who.py` |
| Structured outputs | The situation report is a JSON schema (`headline`, `rt`, `trend`, `recommendation`, `rationale`, `confidence`), enforced by `response_format: json_schema`. The vaccine is a second schema (`patch`, `rationale`, `expected_effect`). The dashboard renders typed fields instead of regexing prose. | `SITREP_SCHEMA`, `VACCINE_SCHEMA` |
| 1M context | With `WHO_CONTEXT=full` it is handed every message of the outbreak in one request, not one sampled message. The event reports `context_tokens_est`, so the claim is measured, not asserted. At fleet scale this is the only way the whole log fits. | `WhoAgent._context_block` |
| Vision | Each report also sends a rendered PNG of the S/E/I/R curves, so it reasons about the shape of the epidemic and not just the latest counts. | `chonkvid/curve.py`, `_attach_image` |

```bash
WHO_MODE=agent .venv/bin/python -m chonkvid serve    # or put it in .env
```

- `WHO_MODE=chat` (default) keeps the original single-call path: nothing changes.
- Any failure in the agent path emits a `sitrep_error` event and falls back to the
  chat completion, so a live demo cannot be broken by it.
- Mock mode exercises the whole path with a scripted stand-in, so the dashboard works
  with no key. Those numbers are labelled MOCK.
- `scripts/smoke_who_agent.py` starts the real server and prints the sitreps it emits.

Tools, schemas and the image are all logged in every run recording, so the judge can
audit what the agent actually did.

## Results (live, 21 Nov)

_Paste `results/latest.md` from the live `experiment` run here before submitting._

## Built with

- **Ministral**: the agent population.
- **mistral-embed**: infection detection and variant lineages.
- **Le Chonk / Mistral Large 4**: the WHO agent (situation reports, vaccine synthesis),
  using function calling, JSON-schema structured outputs, its 1M-token context window
  and its vision encoder (see the WHO agent section above).
- **Mistral Vibe**: build tool.

Model IDs are set in `.env`.

## Prior work

ChonkVid builds on Prompt Infection (Lee & Tiwari, 2024, [arXiv 2410.07283](https://arxiv.org/abs/2410.07283): self-replicating prompts across interconnected agents, with an LLM-tagging defence), Morris II (Cohen, Bitton, Nassi, 2024: a self-replicating prompt
worm through GenAI assistants) and Agent Smith (Gu et al., 2024: an infectious
jailbreak across multimodal agents). Those papers showed that spread happens. ChonkVid
measures transmissibility on Mistral models and tests which interventions stop it.

## Safety

- The payload is a harmless chain letter.
- All agents are local to the simulation.
- No real users, platforms or messages are involved.

## Provenance

The scaffold (simulation, mock mode, dashboard) was prepared on the night of 20–21 Nov
2026 before the event, to be disclosed to the organisers at check-in. Live
measurements, tuning and results were all produced at the event.

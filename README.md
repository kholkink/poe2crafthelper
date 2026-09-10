# poe2craft

A crafting optimizer for **Path of Exile 2**. Given a base item and a target
set of modifiers, it computes the cheapest expected path to that item using a
Markov-decision-process model of the crafting system, then exposes the whole
thing through an LLM assistant that answers questions with real numbers instead
of guesses.

The project has three layers:

| Layer | Package | Role |
|-------|---------|------|
| **Core model** | `poe2craft/core` | Item state, mod/base model, and crafting actions as probabilistic transition kernels |
| **Solver** | `poe2craft/solver` | Stochastic-shortest-path (SSP) solver plus a Monte-Carlo validator for ground truth |
| **Data** | `poe2craft/data` | RePoE-fork PoE2 dump loader, community spawn weights, and a synthetic fallback pool |
| **Agent** | `poe2craft/agent` | RAG knowledge base, live prices, trade search, and Claude tool definitions (see its own README) |

## How it works

Crafting is modelled as an MDP: each currency, essence, omen, or bone is a
transition kernel `ItemState -> {ItemState: probability}`. The solver finds

```
V(goal) = 0
V(s)    = min_a [ cost(a) + Σ_s' P(s'|s,a) · V(s') ]
```

where the awkward part is the RESET action ("bin it, buy a fresh white base"),
which turns the problem into a true SSP rather than a finite DAG. Costs are
denominated in Exalted Orbs, the same numeraire the agent's price layer uses.

The abstract state tracks per-requirement status plus junk/fracture/crafted
bookkeeping rather than the astronomically large exact item, which is what keeps
a single step `O(#requirements)`. `solver/validate.py` runs a full Monte-Carlo
simulation on concrete items to confirm the abstract solver's expected costs.

## Data caveat

GGG's PoE2 data tables expose only a binary "can this mod roll here" flag, not
relative spawn weights. The numeric weights every serious tool relies on trace
back to community reverse-engineering, so `data/weights.py` and the RePoE-fork
dump are the single largest modelling risk. `data/synthetic.py` provides a
PoE-shaped placeholder pool for development before a real dump is wired in.

## Quick start

```bash
cd ~/poe2craft
# optional: create a venv and install the agent deps
python -m venv .venv
.venv/bin/python -m pip install anthropic openai requests

# run the demos
.venv/bin/python scripts/demo.py          # solver on synthetic data
.venv/bin/python scripts/demo_real.py     # solver on the real dump
.venv/bin/python scripts/weight_sensitivity.py

# the assistant (see poe2craft/agent/README.md for setup + API keys)
.venv/bin/python -m poe2craft.agent.chat
```

API keys go in `~/poe2craft/.env` (git-ignored). With `ANTHROPIC_API_KEY` the
agent runs on Claude; with `DEEPSEEK_API_KEY` it runs on an OpenAI-compatible
backend.

## Layout

```
poe2craft/
  core/     item state, mods, crafting actions
  solver/   SSP solver + Monte-Carlo validator
  data/     dump loader, weights, synthetic pool
  agent/    RAG + tools LLM assistant (own README)
scripts/    demos and weight-sensitivity analysis
docs/       architecture.html
data/       generated at runtime (knowledge base, price snapshots) — git-ignored
cache/      game-data dumps — git-ignored
```

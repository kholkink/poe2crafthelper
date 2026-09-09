# poe2craft agent — RAG + tools on top of the crafting solver

An interactive assistant for Path of Exile 2 questions (crafting, mechanics, farming,
prices) that never answers from memory: Claude drives a small set of tools and every
number comes from the local knowledge base, game data, poe.ninja, the trade site, or
the solver.

## Layout

| module      | role |
|-------------|------|
| `kb.py`     | SQLite FTS5 knowledge base (BM25), HTML→text, chunking, search, `get_doc` |
| `ingest.py` | Loaders: poe2wiki pages/categories, official patch-notes forum, local files |
| `prices.py` | poe.ninja economy snapshot in exalted orbs (cached 1 h in `data/`) |
| `trade.py`  | Official trade2 search (read-only, throttled) + stat-id catalogue |
| `craft.py`  | Solver wrapper: base lookup, mod families/tiers, `plan()` → JSON |
| `tools.py`  | `@beta_tool` definitions exposed to Claude (7 tools) |
| `chat.py`   | REPL / one-shot CLI; picks the provider, runs the tool loop (streaming) |
| `backend_openai.py` | Manual tool loop for OpenAI-compatible APIs (DeepSeek: echoes `reasoning_content`) |

Generated data lives in `data/` (`kb.sqlite`, price snapshots, trade stat catalogue) and
is safe to delete; `cache/` holds the game-data dumps the solver needs.

## Setup

```bash
cd ~/poe2craft
.venv/bin/python -m pip install anthropic openai requests   # already done once
# keys go in ~/poe2craft/.env (chmod 600) or the environment; the first one found wins:
#   ANTHROPIC_API_KEY=...                 -> Claude (claude-opus-5) via the Anthropic SDK
#   DEEPSEEK_API_KEY=...                  -> DeepSeek (deepseek-v4-flash), OpenAI-compatible loop
#   OPENAI_API_KEY=... OPENAI_BASE_URL=... -> any OpenAI-compatible server (LM Studio, vLLM)
export POE2_LEAGUE="Runes of Aldur"                    # optional, default shown

# knowledge base (≈1,600 wiki pages + patch notes, ~40 min the first time, incremental after)
.venv/bin/python -m poe2craft.agent.ingest default
.venv/bin/python -m poe2craft.agent.ingest stats
.venv/bin/python -m poe2craft.agent.ingest search "omen of whittling"
```

Add more sources any time:

```bash
.venv/bin/python -m poe2craft.agent.ingest wiki --category "Unique monsters" --page "Headhunter"
.venv/bin/python -m poe2craft.agent.ingest patchnotes --pages 10
.venv/bin/python -m poe2craft.agent.ingest local ~/notes/poe2/*.md
```

## Chat

```bash
.venv/bin/python -m poe2craft.agent.chat                       # REPL (/reset, /quit)
.venv/bin/python -m poe2craft.agent.chat "как скрафтить Drakeskin Boots с T1-T2 MS, лайфом и двумя резами?"
.venv/bin/python -m poe2craft.agent.chat --model deepseek-v4-pro --effort max     # stronger, pricier
.venv/bin/python -m poe2craft.agent.chat --provider anthropic --model claude-sonnet-5 --effort medium
.venv/bin/python -m poe2craft.agent.chat --save data/runs.jsonl "..."             # keep full transcripts
```

`--effort`: low/medium/high/xhigh/max for Claude, low/high/max for DeepSeek.  Measured with
deepseek-v4-flash at effort high: a full crafting question = 12 tool calls, ~130k input /
18k output tokens, ~2.5 min, about $0.05.

Tool calls are echoed to stderr in grey so you can see what the model looked up.

## Tools the model has

| tool | source | what it answers |
|------|--------|-----------------|
| `search_knowledge` / `read_page` | wiki + patch notes | mechanics, omens, essences, uniques (drop sources), patch changes |
| `find_bases` / `list_mod_families` | repoe dump + community weights | exact base names, families, tiers, ilvl, value ranges, spawn-weight estimates |
| `get_prices` | poe.ninja | currency / omen / essence / bone / catalyst prices in ex + div |
| `plan_craft` | solver (`solver/ssp.py`) | expected cost, shopping list, decision table for a target item |
| `trade_search` | pathofexile.com trade2 | cheapest bases / uniques / fractured items, whisper text |

## Solver toolbox (0.5.x)

`plan_craft` models: Transmutation / Augmentation / Regal / Alchemy / Exalted / Annulment /
Chaos and their Greater (min modifier level 35, or 44 for transmute/aug) and Perfect (50 / 70)
versions; omens of Sinistral/Dextral/Greater Exaltation, Sinistral/Dextral Annulment,
Sinistral/Dextral Erasure, Whittling; essences (magic→rare) and Perfect essences (rare, with
Crystallisation omens); Fracturing Orb; desecration with Preserved/Ancient bones (Necromancy
side omens, Abyssal Echoes reroll); rebuying the base. Rules: one crafted mod, one fracture,
one desecrated mod per item, group exclusion. A plan can start from an item the player owns
or a trade listing (`start` + its price), and the toolbox can be restricted (`mechanics`).

Every policy can be checked with `validate=true`: a Monte-Carlo replay on concrete items with
real mod ids and levels. Measured on boots (MS + life + 2 resists): solver 306 ex vs
simulation 308 ± 17 ex; four T1 mods: 7,940 ex vs 7,775 ± 351 ex (and 25,600 ex with the
basic currency only). Solve time 1–20 s.

## Known limits

* Spawn weights are the community table (CC BY-NC-SA, many placeholder 1000s): costs are
  ±50% at best. The model is told to quote ranges.
* Omen of Whittling uses class-level modifier-level distributions (junk from Greater/Perfect
  orbs is higher-level than assumed); Well-of-Souls offer weights are unknown (placeholder).
* Not modelled: Hinekora's Lock, recombination, catalysts, runes/alloys, corruption.
* No build calculator: build advice is qualitative.
* Wiki text is CC BY-NC-SA and patch notes are GGG's: personal use only.

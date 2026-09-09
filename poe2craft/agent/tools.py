"""Claude tool definitions.  Every tool returns a JSON string; the model never
sees raw Python objects.  Docstrings become tool descriptions, so they are
written for the model, not for humans."""
from __future__ import annotations

import json
from typing import Optional

from anthropic import beta_tool

from . import DEFAULT_LEAGUE
from . import craft as craftmod
from . import prices as pricing
from . import trade as trademod
from .kb import KB, format_hits

_kb: KB | None = None


def kb() -> KB:
    global _kb
    if _kb is None:
        _kb = KB()
    return _kb


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=None, default=str)


# ---------------------------------------------------------------------------
@beta_tool
def search_knowledge(query: str, k: int = 6, source: Optional[str] = None) -> str:
    """Full-text search over the local PoE2 knowledge base (poe2wiki pages: currency, omens,
    essences, runes, soul cores, uniques with drop sources, mechanics, endgame; plus official
    patch notes).  Use it for any factual game question before answering from memory.

    Args:
        query: Short ENGLISH keyword query (e.g. "omen of whittling lowest level modifier",
            "Fracturing Orb requirements", "Headhunter drop location", "0.5.0 essence changes").
            Wrap an exact phrase in double quotes.  Translate the user's terms to English item names.
        k: Number of passages to return (1-15).
        source: Optional filter: "wiki", "patchnotes" or "local".
    """
    hits = kb().search(query, k=max(1, min(int(k), 15)), source=source)
    return format_hits(hits)


@beta_tool
def read_page(title: str) -> str:
    """Return the full text of one knowledge-base page (e.g. a unique item, an omen, a boss,
    a patch-notes thread).  Use after search_knowledge when a snippet is not enough, e.g. to
    read a unique's "Item acquisition" section or the whole list of a patch's crafting changes.

    Args:
        title: Exact page title (as shown in search results) or a URL.  A partial title works
            when it is unambiguous.
    """
    d = kb().get_doc(title)
    if d is None:
        return _j({"error": f"no page matching {title!r}; try search_knowledge"})
    return f"# {d['title']}  ({d['url']})\n\n{d['text']}" + ("\n\n[... truncated]" if d["truncated"] else "")


# ---------------------------------------------------------------------------
@beta_tool
def find_bases(item_class: Optional[str] = None, name: Optional[str] = None) -> str:
    """List craftable base items from the game data, highest drop level first.  Call it to get
    the exact base name before list_mod_families / plan_craft / trade_search.

    Args:
        item_class: Slot or class, e.g. "Boots", "Body Armour", "Bow", "Ring", "Quarterstaff".
        name: Substring of the base name, e.g. "Drakeskin".
    """
    rows = craftmod.find_bases(item_class=item_class, name=name)
    if not rows:
        return _j({"error": "no bases found", "known_classes": craftmod.repoe().item_classes()[:80]})
    return _j(rows)


@beta_tool
def list_mod_families(base_name: str, ilvl: int = 82, gen_type: Optional[str] = None,
                      contains: Optional[str] = None) -> str:
    """List every modifier family that can roll on a base at an item level, with each tier's
    ilvl requirement, value range and estimated spawn weight.  You need the exact `family` ids
    and tier numbers from here to build a plan_craft spec.  Tier 1 = best (highest ilvl).

    Args:
        base_name: Exact base name from find_bases (e.g. "Drakeskin Boots").
        ilvl: Item level of the base you will craft on (mods above it cannot roll).
        gen_type: "prefix" or "suffix" to filter one side.
        contains: Case-insensitive filter on the mod text or family id, e.g. "life", "resist",
            "movement", "attack speed".  Use it: the unfiltered list is long.
    """
    return _j(craftmod.mod_families(base_name, ilvl=int(ilvl), gen_type=gen_type, contains=contains))


# ---------------------------------------------------------------------------
@beta_tool
def get_prices(names: Optional[list[str]] = None, category: Optional[str] = None,
               league: Optional[str] = None) -> str:
    """Current market prices from poe.ninja in Exalted Orbs (and Divine Orbs).  Covers currency,
    omens, essences, bones (Abyss), catalysts (Breach), liquid emotions (Delirium), fluxes and
    logbooks (Expedition), runes, fragments and soul cores.  Unique/rare item prices are NOT
    here: use trade_search for those.

    Args:
        names: Item names to look up, e.g. ["Divine Orb", "Omen of Whittling", "Greater Essence of the Body"].
        category: Instead of names, dump one category: "Currency", "Essences", "Ritual" (omens),
            "Abyss", "Breach", "Delirium", "Expedition", "Runes", "Fragments", "SoulCores".
        league: League name (default: the configured league).  Softcore trade league names look
            like "Runes of Aldur"; hardcore has an "HC " prefix.
    """
    league = league or DEFAULT_LEAGUE
    try:
        snap = pricing.fetch(league)
    except Exception as e:
        return _j({"error": f"poe.ninja fetch failed: {e}", "leagues": _safe_leagues()})
    d2e = snap["div_to_ex"]
    out = {"league": league, "fetched_at": snap["fetched_at"], "1_divine_in_exalted": d2e, "items": []}
    if names:
        for n in names:
            it = pricing.lookup(snap, n)
            out["items"].append({"query": n, **({k: it[k] for k in ("name", "ex", "div", "category", "change_7d_pct")}
                                                if it else {"error": "not found"})})
    elif category:
        rows = [it for it in snap["items"].values() if it["category"].lower() == category.lower()]
        rows.sort(key=lambda it: -it["ex"])
        out["items"] = [{k: it[k] for k in ("name", "ex", "div", "change_7d_pct")} for it in rows]
        if not rows:
            out["error"] = f"unknown category {category!r}; categories: {sorted({i['category'] for i in snap['items'].values()})}"
    else:
        out["items"] = [{k: it[k] for k in ("name", "ex", "div")} for it in snap["items"].values()
                        if it["category"] == "Currency"]
    return _j(out)


def _safe_leagues():
    try:
        return pricing.leagues()
    except Exception:
        return []


# ---------------------------------------------------------------------------
_REQ_PROPS = {
    "family": {"type": "string", "description": "family id from list_mod_families (or list_desecrated_families)"},
    "min_tier": {"type": "integer", "description": "best acceptable tier (1 = best)"},
    "max_tier": {"type": "integer", "description": "worst acceptable tier (>= min_tier)"},
    "essence": {"type": "string", "description": "poe.ninja name of a (Lesser/normal/Greater) essence that guarantees this family when upgrading a MAGIC item to rare, e.g. 'Greater Essence of the Body'. Omit if none."},
    "essence_tier": {"type": "integer", "description": "Tier the essence's guaranteed mod corresponds to (compare its value with list_mod_families). Lets the solver know when the essence gives a lower tier than wanted."},
    "perfect_essence": {"type": "string", "description": "poe.ninja name of a Perfect essence for this family (used on RARE items: removes a random mod, adds the guaranteed one). Omit if none."},
    "perfect_essence_tier": {"type": "integer"},
}
_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "base_name": {"type": "string", "description": "Exact base name from find_bases."},
        "ilvl": {"type": "integer", "description": "Item level of the base (default 82)."},
        "requirements": {
            "type": "array", "minItems": 1, "maxItems": 6,
            "description": "Target mods. Max 3 prefixes + 3 suffixes.",
            "items": {"type": "object", "properties": _REQ_PROPS, "required": ["family", "min_tier", "max_tier"]},
        },
        "start": {
            "type": "object",
            "description": "Optional: the item the player ALREADY has (or a listing they could buy, e.g. a fractured base). Omit to start from a white base.",
            "properties": {
                "rarity": {"type": "string", "enum": ["normal", "magic", "rare"]},
                "mods": {"type": "array", "items": {"type": "object", "properties": {
                    "family": {"type": "string"}, "tier": {"type": "integer"},
                    "type": {"type": "string", "enum": ["prefix", "suffix"], "description": "needed only for families not in the requirements"},
                    "fractured": {"type": "boolean"}, "desecrated": {"type": "boolean"}, "crafted": {"type": "boolean"}},
                    "required": ["family"]}},
                "junk_prefixes": {"type": "integer", "description": "unwanted prefixes not listed in mods"},
                "junk_suffixes": {"type": "integer"},
            },
        },
        "league": {"type": "string", "description": "League for prices (default: configured league)."},
        "base_price_ex": {"type": "number", "description": "Cost in exalted of one copy of the START item (a white base by default, ~1 ex; the listing price when start is a bought item). Rebuying it is the solver's reset action."},
        "mechanics": {"type": "array", "items": {"type": "string", "enum": ["basic", "omen_exalt", "omen_annul", "essence", "greater", "perfect", "omen_chaos", "whittling", "fracture", "perfect_essence", "desecrate"]},
                      "description": "Restrict the toolbox (default: everything priced). Use e.g. ['basic','essence','greater'] for a player who cannot afford omens, or to compare strategies."},
        "validate": {"type": "boolean", "description": "Also run a 300-sample Monte-Carlo check of the plan on concrete items with real mod levels (slower; adds median/p90 and the solver-vs-simulation gap)."},
    },
    "required": ["base_name", "requirements"],
}


@beta_tool(input_schema=_PLAN_SCHEMA)
def plan_craft(base_name: str, requirements: list[dict], ilvl: int = 82, league: Optional[str] = None,
               base_price_ex: Optional[float] = None, validate: bool = False,
               start: Optional[dict] = None, mechanics: Optional[list[str]] = None) -> str:
    """Solve for the cheapest expected-cost crafting policy for a target item, from a white base
    or from an item the player already has / can buy, using live prices and community spawn-weight
    estimates.  Returns expected cost (exalted + divine), the most likely path, expected uses per
    currency (shopping list), the decision table (item state -> next action) to turn into an
    "if you see X, do Y" recipe, and action_prices_ex (what each modelled action costs).
    Modelled 0.5 toolbox: Transmutation/Augmentation/Regal/Alchemy/Exalted/Annulment/Chaos and
    their Greater (min mod level 35/44) and Perfect (50/70) versions; omens of Sinistral/Dextral/
    Greater Exaltation, Sinistral/Dextral Annulment, Sinistral/Dextral Erasure (chaos side),
    Whittling (chaos removes lowest-level mod); essences (magic->rare) and Perfect essences
    (rare; +Crystallisation omens); Fracturing Orb; desecration with Preserved/Ancient bones
    (+Necromancy side omens, +Abyssal Echoes reroll); rebuying the base.  Rules applied: one
    crafted (essence) mod per item, one fracture, one desecrated mod, group exclusion.
    NOT modelled: Hinekora's Lock, recombination, catalysts, runes/alloys, corruption.
    Costs are estimates (spawn weights are community guesses): quote them as ranges.
    """
    kw = {}
    if league:
        kw["league"] = league
    if base_price_ex is not None:
        kw["base_price_ex"] = float(base_price_ex)
    if start:
        kw["start"] = start
    if mechanics:
        kw["mechanics"] = list(mechanics)
    try:
        return _j(craftmod.plan(base_name, requirements, ilvl=int(ilvl), validate=bool(validate), **kw))
    except Exception as e:  # never crash the loop on a modelling error
        return _j({"error": f"{type(e).__name__}: {e}"})


@beta_tool
def list_desecrated_families(base_name: str, contains: Optional[str] = None) -> str:
    """List the Abyss-only modifiers a base can receive from desecration (bone + Well of Souls),
    e.g. dual chaos resistances, +max resistances, movement-penalty reduction.  These cannot roll
    from regular currency; use them as plan_craft requirements with min_tier=max_tier=1.

    Args:
        base_name: Exact base name from find_bases.
        contains: Case-insensitive filter on the mod text or family id.
    """
    return _j(craftmod.desecrated_families(base_name, contains=contains))


# ---------------------------------------------------------------------------
_TRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "base_type": {"type": "string", "description": "Exact base type, e.g. 'Drakeskin Boots'."},
        "name": {"type": "string", "description": "Unique item name, e.g. 'Headhunter'. Combine with rarity 'unique'."},
        "rarity": {"type": "string", "enum": ["normal", "magic", "rare", "unique", "nonunique"]},
        "ilvl_min": {"type": "integer"},
        "ilvl_max": {"type": "integer"},
        "max_price_ex": {"type": "number", "description": "Max price in the given currency."},
        "price_currency": {"type": "string", "enum": ["exalted", "divine", "chaos"], "description": "Currency for max_price_ex (default exalted)."},
        "fractured": {"type": "boolean"},
        "desecrated": {"type": "boolean"},
        "corrupted": {"type": "boolean"},
        "identified": {"type": "boolean"},
        "mods": {
            "type": "array",
            "description": "Required explicit mods (AND). Text is matched to the trade stat list.",
            "items": {"type": "object", "properties": {
                "text": {"type": "string", "description": "Mod text with # for numbers, e.g. '# to maximum Life', '#% increased Movement Speed'"},
                "min": {"type": "number"}, "max": {"type": "number"},
                "group": {"type": "string", "enum": ["explicit", "fractured", "desecrated", "implicit"], "description": "default explicit"}},
                "required": ["text"]},
        },
        "league": {"type": "string"},
        "limit": {"type": "integer", "description": "1-10 listings (default 8)."},
    },
}


@beta_tool(input_schema=_TRADE_SCHEMA)
def trade_search(base_type: Optional[str] = None, name: Optional[str] = None, rarity: Optional[str] = None,
                 ilvl_min: Optional[int] = None, ilvl_max: Optional[int] = None,
                 max_price_ex: Optional[float] = None, price_currency: str = "exalted",
                 fractured: Optional[bool] = None, desecrated: Optional[bool] = None,
                 corrupted: Optional[bool] = None, identified: Optional[bool] = None,
                 mods: Optional[list[dict]] = None, league: Optional[str] = None, limit: int = 8) -> str:
    """Search the official PoE2 trade site (online sellers, cheapest first).  Use it to price
    a base or a unique, to find a fractured/desecrated/partially-rolled base to start a craft
    from, or to check what a finished item like the one being crafted sells for.  Returns
    listings with mods, price, and a ready-to-paste whisper message (there is no buy API: the
    player pastes the whisper in game).  Strictly rate-limited: at most one search per answer
    step, never loop over many searches.
    """
    stats = []
    unresolved = []
    for m in mods or []:
        cands = trademod.find_stat(m["text"], group=m.get("group", "explicit"))
        if not cands:
            unresolved.append(m["text"])
            continue
        stats.append({"id": cands[0]["id"], "min": m.get("min"), "max": m.get("max"), "text": cands[0]["text"]})
    if unresolved:
        return _j({"error": f"could not map mod text to trade stats: {unresolved}; use '#' for numbers"})
    try:
        r = trademod.search(league=league or DEFAULT_LEAGUE, base_type=base_type, name=name, rarity=rarity,
                            ilvl_min=ilvl_min, ilvl_max=ilvl_max, max_price_ex=max_price_ex,
                            price_currency=price_currency, fractured=fractured, desecrated=desecrated,
                            corrupted=corrupted, identified=identified, stats=stats, limit=int(limit))
    except Exception as e:
        return _j({"error": f"trade search failed: {type(e).__name__}: {e}"})
    if stats:
        r["matched_stats"] = [s["text"] for s in stats]
    return _j(r)


TOOLS = [search_knowledge, read_page, find_bases, list_mod_families, list_desecrated_families, get_prices,
         plan_craft, trade_search]

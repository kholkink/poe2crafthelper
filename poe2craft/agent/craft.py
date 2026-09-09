"""Solver wrapper: a JSON-friendly spec in, a JSON-friendly plan out.

This is the only place the agent touches the MDP code; every number the model
reports about a craft comes from here, never from the model's memory.
"""
from __future__ import annotations

import math
import re
import time
from collections import defaultdict

from ..core.actions import ALL_MECHANICS, MechanicsConfig, standard_actions
from ..core.context import CraftContext, Requirement
from ..core.mods import GenType, Mod, ModPool, StatRange
from ..core.state import JUNK_PRE, JUNK_SUF, NONE, ItemState, Rarity, SlotState
from ..data.repoe import Repoe, poe1_prior_weight
from ..data.weights import CommunityWeights
from ..solver import ssp
from . import CACHE_DIR, DEFAULT_LEAGUE
from . import prices as pricing

_repoe: Repoe | None = None
_weights: dict[str, CommunityWeights] = {}
_weights_path = f"{CACHE_DIR}/community/weights_poe2.json"


def repoe() -> Repoe:
    global _repoe
    if _repoe is None:
        _repoe = Repoe.load(CACHE_DIR)
    return _repoe


def community_weights(item_class: str) -> CommunityWeights | None:
    if item_class not in _weights:
        cw = CommunityWeights.load(_weights_path, item_class)
        _weights[item_class] = cw if cw.by_id else None
    return _weights[item_class]


def clean_text(t: str) -> str:
    return re.sub(r"\[([^\]|]*)\|([^\]]*)\]", r"\2", t or "")


# ---------------------------------------------------------------------------
# bases + mod families
# ---------------------------------------------------------------------------
_CLASS_ALIASES = {
    "body": "Body Armour", "body armor": "Body Armour", "chest": "Body Armour", "armour": "Body Armour",
    "helm": "Helmet", "helmet": "Helmet", "hat": "Helmet", "glove": "Gloves", "boot": "Boots",
    "amulet": "Amulet", "ring": "Ring", "belt": "Belt", "quiver": "Quiver", "bow": "Bow",
    "crossbow": "Crossbow", "wand": "Wand", "staff": "Staff", "quarterstaff": "Warstaff",
    "warstaff": "Warstaff", "spear": "Spear", "sceptre": "Sceptre", "scepter": "Sceptre",
    "mace": "One Hand Mace", "one hand mace": "One Hand Mace", "two hand mace": "Two Hand Mace",
    "shield": "Shield", "buckler": "Buckler", "focus": "Focus", "talisman": "Talisman",
    "dagger": "Dagger", "sword": "One Hand Sword", "flail": "Flail", "axe": "One Hand Axe",
    "claw": "Claw", "jewel": "Jewel",
}


def resolve_item_class(s: str) -> str | None:
    if not s:
        return None
    key = s.strip().lower()
    if key in _CLASS_ALIASES:
        return _CLASS_ALIASES[key]
    classes = repoe().item_classes()
    for c in classes:
        if c.lower() == key:
            return c
    for c in classes:
        if key in c.lower():
            return c
    return None


def find_bases(item_class: str | None = None, name: str | None = None, limit: int = 40) -> list[dict]:
    cls = resolve_item_class(item_class) if item_class else None
    if item_class and cls is None:
        return []
    out = []
    for b in repoe().find_bases(item_class=cls, name=name):
        if b.item_class in ("StackableCurrency", "HiddenItem", "QuestItem", "Microtransaction"):
            continue
        out.append({"name": b.name, "item_class": b.item_class, "drop_level": b.drop_level,
                    "tags": sorted(b.tags), "implicits": list(b.implicits)})
    return out[:limit]


def resolve_base(base_name: str):
    r = repoe()
    hits = r.find_bases(name=base_name)
    exact = [b for b in hits if b.name.lower() == base_name.strip().lower()]
    if exact:
        return exact[0]
    if hits:
        return hits[0]
    return None


def mod_families(base_name: str, ilvl: int = 82, gen_type: str | None = None,
                 contains: str | None = None) -> dict:
    """Every mod family that can roll on the base at this ilvl, with tiers/weights."""
    base = resolve_base(base_name)
    if base is None:
        return {"error": f"base {base_name!r} not found; use find_bases first"}
    cw = community_weights(base.item_class)
    missed: list[str] = []
    wm = cw.model(fallback=0, warn=missed) if cw else poe1_prior_weight
    mods = repoe().mods_for(base, weight_model=wm)
    pool = ModPool.build(base, ilvl=ilvl, mods=mods)
    fams: dict[tuple, dict] = {}
    for gen in (GenType.PREFIX, GenType.SUFFIX):
        if gen_type and gen.value != gen_type.lower():
            continue
        total = pool.total_weight(gen)
        for m, w in pool.side(gen):
            key = (gen.value, m.group)
            f = fams.setdefault(key, {"family": m.group, "type": gen.value, "text": clean_text(m.name),
                                      "tiers": [], "weight_total": 0})
            stats = ", ".join(f"{s.stat_id} {s.lo}..{s.hi}" if s.lo != s.hi else f"{s.stat_id} {s.lo}"
                              for s in m.stats)
            tier_name = cw.tier_name.get(m.id, "") if cw else ""
            f["tiers"].append({"tier": m.tier, "ilvl": m.required_level, "weight": w,
                               "p_single_roll": round(w / total, 4) if total else None,
                               "mod_id": m.id, "name": tier_name, "values": stats})
            f["weight_total"] += w
    out = []
    for f in fams.values():
        if contains and contains.lower() not in (f["text"] + " " + f["family"]).lower():
            continue
        f["tiers"].sort(key=lambda t: t["tier"])
        f["text"] = f["tiers"][0]["values"] and f["text"]
        out.append(f)
    out.sort(key=lambda f: (f["type"], -f["weight_total"]))
    return {"base": base.name, "item_class": base.item_class, "ilvl": ilvl,
            "weights": "community table (estimates)" if cw else "PoE1-style prior (NO community data for this class)",
            "note": "tier 1 = highest ilvl requirement in the family; p_single_roll = chance that one random "
                    "affix of this side is exactly this tier on a blank item",
            "prefix_weight_total": pool.total_weight(GenType.PREFIX),
            "suffix_weight_total": pool.total_weight(GenType.SUFFIX),
            "families": out}


# ---------------------------------------------------------------------------
# desecration pool (Abyss-only mods offered at the Well of Souls)
# ---------------------------------------------------------------------------
SPECIAL_DESECRATED_WEIGHT = 1000     # placeholder: no public data on Well-of-Souls offer weights


def desecrated_pool(base) -> dict[GenType, list[tuple[Mod, int]]]:
    out: dict[GenType, list[tuple[Mod, int]]] = {GenType.PREFIX: [], GenType.SUFFIX: []}
    r = repoe()
    for mid, raw in r.mods_raw.items():
        if raw.get("domain") != "desecrated" or raw.get("generation_type") not in ("prefix", "suffix"):
            continue
        if r._eligible_weight(raw, base.tags) <= 0:
            continue
        gen = GenType.PREFIX if raw["generation_type"] == "prefix" else GenType.SUFFIX
        stats = tuple(StatRange(x["id"], x.get("min", 0), x.get("max", 0)) for x in raw.get("stats", []))
        m = Mod(id=mid, group=r._family(raw), gen_type=gen, required_level=raw.get("required_level", 1),
                tier=1, name=raw.get("text") or raw.get("name") or mid, domain="desecrated", stats=stats,
                spawn_weights=(("default", SPECIAL_DESECRATED_WEIGHT),),
                tags=frozenset(raw.get("implicit_tags") or []))
        out[gen].append((m, SPECIAL_DESECRATED_WEIGHT))
    return out


def desecrated_families(base_name: str, contains: str | None = None) -> dict:
    """Abyss-only mods a base can get from desecration (not obtainable with regular currency)."""
    base = resolve_base(base_name)
    if base is None:
        return {"error": f"base {base_name!r} not found"}
    rows = []
    for gen, lst in desecrated_pool(base).items():
        for m, _ in lst:
            txt = clean_text(m.name)
            if contains and contains.lower() not in (txt + " " + m.group).lower():
                continue
            rows.append({"family": m.group, "type": gen.value, "text": txt, "ilvl": m.required_level,
                         "values": ", ".join(f"{x.stat_id} {x.lo}..{x.hi}" for x in m.stats)})
    return {"base": base.name, "note": "families here can only be obtained via desecration (bone + Well of "
                                       "Souls); use tier 1-1 in plan_craft; offer weights are unknown "
                                       "(placeholder)", "families": rows}


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------
_RARITY = {"normal": Rarity.NORMAL, "white": Rarity.NORMAL, "magic": Rarity.MAGIC, "blue": Rarity.MAGIC,
           "rare": Rarity.RARE, "yellow": Rarity.RARE}


def _build_start(start: dict, reqs, fam_side: dict[str, GenType], ctx) -> ItemState | str:
    """Turn a user-described existing item into an abstract start state (or an error string)."""
    rar = _RARITY.get(str(start.get("rarity", "normal")).lower())
    if rar is None:
        return f"unknown rarity {start.get('rarity')!r}"
    slots = [SlotState.ABSENT] * len(reqs)
    jp, js = int(start.get("junk_prefixes", 0)), int(start.get("junk_suffixes", 0))
    fractured, crafted, desecrated = [], NONE, NONE
    fam_index = {}
    for i, r in enumerate(reqs):
        for g in r.groups:
            fam_index[g] = i
    for m in start.get("mods", []):
        fam = m.get("family", "")
        tier = int(m.get("tier", 1))
        if fam in fam_index:
            i = fam_index[fam]
            slots[i] = SlotState.HIT if reqs[i].min_tier <= tier <= reqs[i].max_tier else SlotState.BLOCKED
            tag = i
        else:
            side = fam_side.get(fam)
            if side is None:
                t = str(m.get("type", "")).lower()
                side = GenType.PREFIX if t.startswith("p") else GenType.SUFFIX if t.startswith("s") else None
            if side is None:
                return f"mod family {fam!r} unknown: give its 'type' (prefix/suffix)"
            if side is GenType.PREFIX:
                jp += 1
                tag = JUNK_PRE
            else:
                js += 1
                tag = JUNK_SUF
        if m.get("fractured"):
            fractured.append(tag)
        if m.get("crafted"):
            crafted = tag if tag >= 0 else crafted
        if m.get("desecrated"):
            desecrated = tag
    st = ItemState(rar, tuple(slots), jp, js, fractured=tuple(sorted(set(fractured)))[:1],
                   crafted=crafted, desecrated=desecrated)
    if st.n_pre(ctx.sides) > rar.max_prefixes or st.n_suf(ctx.sides) > rar.max_suffixes:
        return f"start item has too many mods for rarity {rar.name.lower()}"
    return st


def plan(base_name: str, requirements: list[dict], ilvl: int = 82, league: str = DEFAULT_LEAGUE,
         base_price_ex: float | None = None, validate: bool = False, max_states: int = 60000,
         start: dict | None = None, mechanics: list[str] | None = None) -> dict:
    """requirements: [{family, min_tier, max_tier, essence?, essence_tier?, perfect_essence?,
    perfect_essence_tier?}].  start: an existing item to continue from (see _build_start).
    mechanics: subset of ALL_MECHANICS to enable (default: everything that has a price)."""
    t0 = time.time()
    base = resolve_base(base_name)
    if base is None:
        return {"error": f"base {base_name!r} not found; use find_bases first"}
    if not requirements:
        return {"error": "requirements is empty"}
    if len(requirements) > 6:
        return {"error": "at most 6 requirements (3 prefixes + 3 suffixes)"}

    cw = community_weights(base.item_class)
    missed: list[str] = []
    wm = cw.model(fallback=0, warn=missed) if cw else poe1_prior_weight
    mods = repoe().mods_for(base, weight_model=wm)
    pool = ModPool.build(base, ilvl=ilvl, mods=mods)
    special = desecrated_pool(base)
    by_group: dict[str, list] = defaultdict(list)
    fam_side: dict[str, GenType] = {}
    for gen in (GenType.PREFIX, GenType.SUFFIX):
        for m, w in pool.side(gen):
            by_group[m.group].append((m, gen))
            fam_side[m.group] = gen
    special_groups: dict[str, list] = defaultdict(list)
    for gen, lst in special.items():
        for m, _ in lst:
            special_groups[m.group].append((m, gen))
            fam_side.setdefault(m.group, gen)

    reqs, essence_ids, pessence_ids, essence_ok, pessence_ok, warnings = [], {}, {}, {}, {}, []
    for i, q in enumerate(requirements):
        fam = q.get("family", "")
        src = by_group if fam in by_group else special_groups if fam in special_groups else None
        if src is None:
            cands = [g for g in list(by_group) + list(special_groups) if fam.lower() in g.lower()]
            if len(cands) == 1:
                fam = cands[0]
                src = by_group if fam in by_group else special_groups
            else:
                return {"error": f"family {fam!r} cannot roll on {base.name} at ilvl {ilvl}"
                                 + (f"; did you mean one of {cands[:8]}" if cands else
                                    "; call list_mod_families to see valid families")}
        gen = src[fam][0][1]
        tiers = sorted({m.tier for m, _ in src[fam]})
        lo, hi = int(q.get("min_tier", 1)), int(q.get("max_tier", q.get("min_tier", 1)))
        if lo > hi:
            lo, hi = hi, lo
        if not any(lo <= t <= hi for t in tiers):
            return {"error": f"family {fam!r} has tiers {tiers} at ilvl {ilvl}; requested T{lo}-T{hi}"}
        label = f"T{lo}" + (f"-T{hi}" if hi != lo else "") + f" {clean_text(src[fam][0][0].name)} ({fam})"
        if src is special_groups:
            label += " [desecration only]"
        reqs.append(Requirement(label, gen, frozenset({fam}), lo, hi))
        essence_ids[i] = q.get("essence") or None
        pessence_ids[i] = q.get("perfect_essence") or None
        for key, okd in (("essence_tier", essence_ok), ("perfect_essence_tier", pessence_ok)):
            if q.get(key) is not None:
                okd[i] = lo <= int(q[key]) <= hi
    n_pre = sum(1 for r in reqs if r.gen_type is GenType.PREFIX)
    n_suf = len(reqs) - n_pre
    if n_pre > 3 or n_suf > 3:
        return {"error": f"spec needs {n_pre} prefixes and {n_suf} suffixes; max 3 each"}

    ctx = CraftContext.build(pool, tuple(reqs), special_pool=special)
    snap = pricing.fetch(league)
    if base_price_ex is None:
        base_price_ex = 1.0
    prices, pw = pricing.solver_prices(snap, base_price_ex=base_price_ex, essence_ids=essence_ids,
                                       perfect_essence_ids=pessence_ids, item_class=base.item_class)
    warnings += pw
    if not cw:
        warnings.append(f"no community spawn weights for {base.item_class}: using a crude PoE1-style prior; "
                        "expected costs are rough")
    elif missed:
        warnings.append(f"{len(missed)} eligible mods had no community weight and were dropped from the pool")
    for i, req in enumerate(reqs):
        if ctx.w_hit[i] == 0 and ctx.w_hit_special[i] == 0:
            return {"error": f"requirement {req.name} has zero spawn weight on this base/ilvl"}
        if ctx.w_hit[i] == 0:
            warnings.append(f"requirement {i} ({req.name}) is only obtainable through desecration")

    start_state = ssp.blank_state(len(reqs))
    if start:
        st = _build_start(start, reqs, fam_side, ctx)
        if isinstance(st, str):
            return {"error": st}
        start_state = st

    cfg = MechanicsConfig(essence_ok=essence_ok, perfect_essence_ok=pessence_ok)
    enabled = set(mechanics) if mechanics else set(ALL_MECHANICS)
    unknown = enabled - set(ALL_MECHANICS)
    if unknown:
        return {"error": f"unknown mechanics {sorted(unknown)}; valid: {sorted(ALL_MECHANICS)}"}
    enabled |= {"basic"}
    dropped: list[str] = []
    # keep the state space tractable: drop the most expensive-to-model mechanics first
    for attempt in range(6):
        actions = [a for a in standard_actions(cfg, prices, len(reqs), start=start_state, mechanics=enabled)
                   if math.isfinite(a.cost)]
        prepared = ssp.prepare(ctx, actions, start_state)
        n_states = len(prepared[0])
        if n_states <= max_states:
            break
        for mech in ("desecrate", "fracture", "perfect_essence", "whittling", "omen_chaos", "perfect"):
            if mech in enabled:
                enabled.discard(mech)
                dropped.append(mech)
                break
        else:
            return {"error": f"state space too large ({n_states} states); drop a requirement or narrow tiers"}
    if dropped:
        warnings.append(f"state space > {max_states}: disabled mechanics {dropped}")

    sol = ssp.solve_exact(ctx, actions, start=start_state, prepared=prepared)
    dt = time.time() - t0

    counts = ssp.expected_action_counts(ctx, sol)
    byname = {a.name: a for a in actions}
    spend = []
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1] * byname[kv[0]].cost):
        spend.append({"action": name, "expected_uses": round(n, 2), "unit_cost_ex": round(byname[name].cost, 3),
                      "expected_cost_ex": round(n * byname[name].cost, 1)})
    d2e = snap["div_to_ex"]

    visits = ssp.expected_state_visits(ctx, sol)
    tot_v = sum(visits.values()) or 1.0
    table = []
    for s, m in sorted(visits.items(), key=lambda kv: -kv[1])[:30]:
        a = sol.policy.get(s)
        if a is None:
            continue
        table.append({"state": _describe_state(ctx, s), "action": a.name,
                      "remaining_cost_ex": round(sol.V[s], 1), "visit_share": round(m / tot_v, 3)})

    used_mechs = sorted({t for a in actions for t in a.tags} - {"reset"})
    out = {
        "base": base.name, "item_class": base.item_class, "ilvl": ilvl, "league": league,
        "start": _describe_state(ctx, start_state),
        "requirements": [{"index": i, "spec": r.name, "side": r.gen_type.value,
                          "p_hit_per_roll": round(ctx.w_hit[i] / max(1, ctx.open_weight(r.gen_type, start_state)), 4),
                          "p_block_per_roll": round(ctx.w_block[i] / max(1, ctx.open_weight(r.gen_type, start_state)), 4),
                          "essence": essence_ids.get(i), "essence_gives_wanted_tier": essence_ok.get(i, True),
                          "perfect_essence": pessence_ids.get(i)} for i, r in enumerate(reqs)],
        "expected_cost_ex": round(sol.expected_cost, 1),
        "expected_cost_div": round(sol.expected_cost / d2e, 3),
        "div_to_ex": d2e, "prices_fetched_at": snap["fetched_at"],
        "solver": {"states": sol.n_states, "actions": len(actions), "policy_iterations": sol.outer_iters,
                   "seconds": round(dt, 2)},
        "mechanics_enabled": used_mechs,
        "action_prices_ex": {a.name: round(a.cost, 2) for a in actions},
        "most_likely_path": ssp.extract_plan(ctx, sol),
        "expected_spend": spend,
        "decision_table": table,
        "state_legend": "state string: rarity N/M/R, then one symbol per requirement in order "
                        "(. absent, x wrong tier of that family present = blocked, # satisfied), "
                        "+Np Ms = junk prefixes/suffixes; F<i>=fractured slot i (Fjp/Fjs = fractured junk), "
                        "E<i>=slot i holds the essence (crafted) mod, D<i>=slot i is the desecrated mod",
        "warnings": warnings,
        "caveats": ["spawn weights are community estimates (many placeholders): treat costs as +/-50%",
                    "Omen of Whittling: mod levels are approximated from class distributions; junk added by "
                    "Greater/Perfect orbs is higher-level than assumed, so whittling risk is underestimated there",
                    "desecration: Well-of-Souls offer weights are unknown; Abyss-only mods use a placeholder weight",
                    "not modelled: Hinekora's Lock, recombination, catalysts, runes, alloys, corruption, "
                    "Omen of Light/Amelioration/Sanctification"],
    }
    if validate:
        if start_state != ssp.blank_state(len(reqs)):
            out["monte_carlo"] = {"skipped": "validation needs a blank white-base start"}
        else:
            from ..solver import validate as val
            v = val.run(ctx, sol, cfg, n_runs=300, seed=7)
            out["monte_carlo"] = {"mean_ex": round(v["mean"], 1), "stderr": round(v["stderr"], 1),
                                  "median_ex": round(v["median"], 1), "p90_ex": round(v["p90"], 1),
                                  "mean_steps": round(v["mean_steps"], 1),
                                  "gap_sigma": round((v["mean"] - sol.expected_cost) / v["stderr"], 2)
                                  if v["stderr"] else None}
    return out


def _describe_state(ctx, s) -> str:
    parts = []
    for i, st in enumerate(s.slots):
        nm = ctx.reqs[i].name.split(" (")[0]
        extra = []
        if i in s.fractured:
            extra.append("fractured")
        if s.crafted == i:
            extra.append("essence")
        if s.desecrated == i:
            extra.append("desecrated")
        suffix = f" [{', '.join(extra)}]" if extra else ""
        if st == SlotState.HIT:
            parts.append(f"has {nm}{suffix}")
        elif st == SlotState.BLOCKED:
            parts.append(f"wrong tier of {nm}{suffix}")
    junk = []
    if s.junk_pre:
        junk.append(f"{s.junk_pre} junk prefix" + (" (1 fractured)" if JUNK_PRE in s.fractured else "")
                    + (" (1 desecrated)" if s.desecrated == JUNK_PRE else ""))
    if s.junk_suf:
        junk.append(f"{s.junk_suf} junk suffix" + (" (1 fractured)" if JUNK_SUF in s.fractured else "")
                    + (" (1 desecrated)" if s.desecrated == JUNK_SUF else ""))
    desc = ", ".join(parts + junk) or "blank"
    return f"{s} = {s.rarity.name.lower()} item: {desc}"

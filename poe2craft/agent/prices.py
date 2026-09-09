"""Live PoE2 economy from poe.ninja, expressed in Exalted Orbs (the solver's numeraire).

poe.ninja quotes `primaryValue` in Divine Orbs and publishes the divine->exalted
rate in `core.rates`; we convert once and cache the whole snapshot on disk.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import time

import requests

from . import DATA_DIR, DEFAULT_LEAGUE, USER_AGENT

NINJA = "https://poe.ninja/poe2/api/economy"
TYPES = ["Currency", "Essences", "Ritual", "Abyss", "Breach", "Delirium", "Expedition",
         "Runes", "Fragments", "SoulCores"]
CACHE_TTL = 3600  # seconds

# solver action name -> poe.ninja id
SOLVER_KEYS = {
    "transmute": "transmute", "augment": "aug", "regal": "regal", "alchemy": "alch",
    "exalt": "exalted", "annul": "annul", "chaos": "chaos",
    "transmute_greater": "greater-orb-of-transmutation", "augment_greater": "greater-orb-of-augmentation",
    "regal_greater": "greater-regal-orb", "exalt_greater": "greater-exalted-orb", "chaos_greater": "greater-chaos-orb",
    "transmute_perfect": "perfect-orb-of-transmutation", "augment_perfect": "perfect-orb-of-augmentation",
    "regal_perfect": "perfect-regal-orb", "exalt_perfect": "perfect-exalted-orb", "chaos_perfect": "perfect-chaos-orb",
    "omen_sinistral_exalt": "omen-of-sinistral-exaltation",
    "omen_dextral_exalt": "omen-of-dextral-exaltation",
    "omen_greater_exalt": "omen-of-greater-exaltation",
    "omen_sinistral_annul": "omen-of-sinistral-annulment",
    "omen_dextral_annul": "omen-of-dextral-annulment",
    "omen_sinistral_erasure": "omen-of-sinistral-erasure",
    "omen_dextral_erasure": "omen-of-dextral-erasure",
    "omen_whittling": "omen-of-whittling",
    "omen_sinistral_crystallisation": "omen-of-sinistral-crystallisation",
    "omen_dextral_crystallisation": "omen-of-dextral-crystallisation",
    "omen_sinistral_necromancy": "omen-of-sinistral-necromancy",
    "omen_dextral_necromancy": "omen-of-dextral-necromancy",
    "omen_abyssal_echoes": "omen-of-abyssal-echoes",
    "fracturing": "fracturing-orb",
}
# the stock seven must always have a price (fallback 'inf' disables the action)
REQUIRED_KEYS = ("transmute", "augment", "regal", "alchemy", "exalt", "annul", "chaos")

# bone (desecration currency) per item class family
BONES = {
    "armour": ("rib", ["Body Armour", "Helmet", "Gloves", "Boots", "Shield", "Buckler", "Focus"]),
    "weapon": ("jawbone", ["Bow", "Crossbow", "Wand", "Staff", "Warstaff", "Spear", "Sceptre", "One Hand Mace",
                            "Two Hand Mace", "Dagger", "One Hand Sword", "Two Hand Sword", "Flail", "Claw",
                            "One Hand Axe", "Two Hand Axe", "Quiver", "Talisman"]),
    "jewellery": ("collarbone", ["Ring", "Amulet", "Belt"]),
    "jewel": ("cranium", ["Jewel"]),
}


def bone_part(item_class: str) -> str | None:
    for part, classes in BONES.values():
        if item_class in classes:
            return part
    return None


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def leagues() -> list[str]:
    r = requests.get(f"{NINJA}/leagues", headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    return [x["id"] for x in r.json()]


def fetch(league: str = DEFAULT_LEAGUE, force: bool = False) -> dict:
    """-> {'league', 'fetched_at', 'div_to_ex', 'items': {id: {...}}}"""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"prices_{_slug(league)}.json")
    if not force and os.path.exists(path) and time.time() - os.path.getmtime(path) < CACHE_TTL:
        with open(path) as f:
            return json.load(f)

    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    items: dict[str, dict] = {}
    div_to_ex = None
    for t in TYPES:
        r = s.get(f"{NINJA}/exchange/current/overview", params={"league": league, "type": t}, timeout=40)
        if r.status_code != 200:
            continue
        d = r.json()
        rates = d.get("core", {}).get("rates", {})
        if div_to_ex is None and rates.get("exalted"):
            div_to_ex = float(rates["exalted"])
        names = {it["id"]: it.get("name", it["id"]) for it in d.get("items", [])}
        for line in d.get("lines", []):
            pid = line["id"]
            pv = float(line.get("primaryValue") or 0.0)   # in divines
            items[pid] = {
                "id": pid, "name": names.get(pid, pid), "category": t,
                "div": pv, "ex": None,               # filled below once the rate is known
                "volume_div": line.get("volumePrimaryValue"),
                "change_7d_pct": (line.get("sparkline") or {}).get("totalChange"),
            }
        time.sleep(0.2)
    if div_to_ex is None:
        raise RuntimeError(f"poe.ninja returned no divine->exalted rate for league {league!r}")
    for it in items.values():
        it["ex"] = round(it["div"] * div_to_ex, 4)
    snap = {"league": league, "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "div_to_ex": div_to_ex, "items": items}
    with open(path, "w") as f:
        json.dump(snap, f)
    return snap


def lookup(snap: dict, name: str) -> dict | None:
    """Find an item by ninja id or (fuzzy) display name."""
    items = snap["items"]
    key = _slug(name)
    if key in items:
        return items[key]
    for it in items.values():
        if _slug(it["name"]) == key:
            return it
    # substring match, prefer shortest name (e.g. 'exalted' -> 'Exalted Orb' not 'Greater Exalted Orb')
    cands = [it for it in items.values() if key in _slug(it["name"]) or key in it["id"]]
    if cands:
        return min(cands, key=lambda it: len(it["name"]))
    return None


def solver_prices(snap: dict, base_price_ex: float = 1.0,
                  essence_ids: dict[int, str | None] | None = None,
                  perfect_essence_ids: dict[int, str | None] | None = None,
                  item_class: str | None = None) -> tuple[dict[str, float], list[str]]:
    """Map the snapshot onto the solver's price dict. Returns (prices, warnings).

    Optional actions whose price is missing are simply left out (the solver skips
    them); the seven stock orbs fall back to 'inf' with a warning.
    """
    prices: dict[str, float] = {"base": base_price_ex}
    warnings: list[str] = []
    for action, pid in SOLVER_KEYS.items():
        it = snap["items"].get(pid)
        if it is None or not it["ex"]:
            if action in REQUIRED_KEYS:
                warnings.append(f"no price for {pid}; action '{action}' disabled")
                prices[action] = float("inf")
        else:
            prices[action] = it["ex"]
    if item_class:
        part = bone_part(item_class)
        if part:
            for grade in ("preserved", "ancient"):
                it = snap["items"].get(f"{grade}-{part}")
                if it and it["ex"]:
                    prices[f"bone_{grade}"] = it["ex"]
        else:
            warnings.append(f"no bone type known for item class {item_class!r}; desecration disabled")
    for key, ids in (("essence", essence_ids), ("perfect_essence", perfect_essence_ids)):
        for i, eid in (ids or {}).items():
            if not eid:
                continue
            it = lookup(snap, eid)
            if it is None:
                warnings.append(f"{key} {eid!r} not found on poe.ninja; requirement {i} has no such action")
            else:
                prices[f"{key}_{i}"] = it["ex"]
    return prices, warnings


def fmt_ex(x: float, div_to_ex: float) -> str:
    if x >= div_to_ex:
        return f"{x / div_to_ex:.2f} div ({x:,.0f} ex)"
    if x >= 10:
        return f"{x:,.0f} ex"
    return f"{x:.2f} ex"

"""Official PoE2 trade search (read-only). No purchase API exists: the tool
returns ready-to-paste whisper messages.

Rate limits are strict (5 searches / 10 s, escalating bans); we throttle
locally and back off on 429.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time

import requests

from . import DATA_DIR, DEFAULT_LEAGUE, USER_AGENT

BASE = "https://www.pathofexile.com/api/trade2"
_lock = threading.Lock()
_last_search = 0.0
MIN_GAP = 2.5          # seconds between searches
_stats_cache: dict | None = None


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return s


def stat_catalogue(force: bool = False) -> dict:
    """{'explicit': [{'id','text'}], 'fractured': [...], ...} cached on disk (24 h)."""
    global _stats_cache
    if _stats_cache and not force:
        return _stats_cache
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "trade_stats.json")
    if not force and os.path.exists(path) and time.time() - os.path.getmtime(path) < 86400:
        with open(path) as f:
            _stats_cache = json.load(f)
            return _stats_cache
    r = _session().get(f"{BASE}/data/stats", timeout=40)
    r.raise_for_status()
    out = {g["id"]: [{"id": e["id"], "text": e["text"]} for e in g["entries"]] for g in r.json()["result"]}
    with open(path, "w") as f:
        json.dump(out, f)
    _stats_cache = out
    return out


def _norm(t: str) -> str:
    t = t.lower()
    t = re.sub(r"\[([^\]|]*)\|([^\]]*)\]", r"\2", t)      # [Resistances|Fire Resistance] -> Fire Resistance
    t = re.sub(r"\(\d+[-–]\d+\)|[+-]?\d+(\.\d+)?", "#", t)  # numbers/ranges -> #
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def find_stat(text: str, group: str = "explicit") -> list[dict]:
    """Match a human mod text ('+# to maximum Life', 'increased Movement Speed') to trade stat ids."""
    cat = stat_catalogue().get(group, [])
    q = _norm(text)
    exact = [e for e in cat if _norm(e["text"]) == q]
    if exact:
        return exact
    words = [w for w in re.findall(r"[a-z%]+", q) if w not in ("to", "of", "the", "and", "#", "with")]
    cands = []
    for e in cat:
        et = _norm(e["text"])
        if all(w in et for w in words):
            cands.append((len(et), e))
    cands.sort(key=lambda x: x[0])
    return [e for _, e in cands[:5]]


def search(league: str = DEFAULT_LEAGUE, base_type: str | None = None, name: str | None = None,
           rarity: str | None = None, ilvl_min: int | None = None, ilvl_max: int | None = None,
           max_price_ex: float | None = None, price_currency: str = "exalted",
           fractured: bool | None = None, desecrated: bool | None = None,
           corrupted: bool | None = None, identified: bool | None = None,
           stats: list[dict] | None = None, online_only: bool = True, limit: int = 10) -> dict:
    """stats: [{'id': 'explicit.stat_...', 'min': 80, 'max': None}, ...] (AND group)."""
    global _last_search
    q: dict = {"status": {"option": "online" if online_only else "any"},
               "stats": [{"type": "and", "filters": []}], "filters": {}}
    if base_type:
        q["type"] = base_type
    if name:
        q["name"] = name
    tf: dict = {}
    if rarity:
        tf["rarity"] = {"option": rarity}
    if ilvl_min is not None or ilvl_max is not None:
        tf["ilvl"] = {k: v for k, v in (("min", ilvl_min), ("max", ilvl_max)) if v is not None}
    if tf:
        q["filters"]["type_filters"] = {"filters": tf}
    mf: dict = {}
    for key, val in (("fractured_item", fractured), ("desecrated", desecrated),
                     ("corrupted", corrupted), ("identified", identified)):
        if val is not None:
            mf[key] = {"option": "true" if val else "false"}
    if mf:
        q["filters"]["misc_filters"] = {"filters": mf}
    if max_price_ex is not None:
        q["filters"]["trade_filters"] = {"filters": {"price": {"option": price_currency, "max": max_price_ex}}}
    for s in stats or []:
        f = {"id": s["id"]}
        v = {k: s[k] for k in ("min", "max") if s.get(k) is not None}
        if v:
            f["value"] = v
        q["stats"][0]["filters"].append(f)
    body = {"query": q, "sort": {"price": "asc"}}

    s = _session()
    with _lock:
        wait = MIN_GAP - (time.time() - _last_search)
        if wait > 0:
            time.sleep(wait)
        r = s.post(f"{BASE}/search/poe2/{league}", json=body, timeout=40)
        _last_search = time.time()
    if r.status_code == 429:
        retry = int(r.headers.get("Retry-After", "10"))
        return {"error": f"trade API rate-limited; retry after {retry}s", "query": body}
    if r.status_code != 200:
        return {"error": f"trade search HTTP {r.status_code}: {r.text[:300]}", "query": body}
    d = r.json()
    ids = d.get("result", [])[:max(1, min(limit, 10))]
    out = {"total": d.get("total", 0), "search_id": d.get("id"),
           "url": f"https://www.pathofexile.com/trade2/search/poe2/{league}/{d.get('id')}",
           "listings": []}
    if not ids:
        return out
    time.sleep(0.5)
    r2 = s.get(f"{BASE}/fetch/{','.join(ids)}", params={"query": d["id"]}, timeout=40)
    if r2.status_code != 200:
        out["error"] = f"fetch HTTP {r2.status_code}"
        return out
    for res in r2.json().get("result", []):
        it, li = res.get("item", {}), res.get("listing", {})
        price = li.get("price") or {}
        acc = li.get("account") or {}
        out["listings"].append({
            "name": (it.get("name") or "").strip(), "type": it.get("typeLine"), "base": it.get("baseType"),
            "rarity": it.get("rarity"), "ilvl": it.get("ilvl"), "identified": it.get("identified"),
            "corrupted": bool(it.get("corrupted")),
            "implicit": it.get("implicitMods") or [], "explicit": it.get("explicitMods") or [],
            "fractured": it.get("fracturedMods") or [], "desecrated": it.get("desecratedMods") or [],
            "runes": it.get("runeMods") or [],
            "price": f"{price.get('amount')} {price.get('currency')}" if price else None,
            "seller": acc.get("name"), "seller_status": (acc.get("online") or {}).get("status", "online"),
            "indexed": li.get("indexed"), "whisper": li.get("whisper"),
        })
    return out

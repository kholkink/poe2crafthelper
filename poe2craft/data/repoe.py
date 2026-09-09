"""Loader for the RePoE-fork PoE2 JSON dump (https://repoe-fork.github.io/poe2/).

IMPORTANT, and the single biggest modelling risk in this project:
the published PoE2 dump carries spawn weights that are **binary** -- every entry
is 0 or 1 (verified: 9,066 zeros and 7,902 ones across all 16,679 mods, with no
other value anywhere).  So the dump tells us *which* mods may roll on a base
(eligibility mask, mod family, generation type, ilvl gate, stat ranges) but NOT
their relative rarity.

Consequently the weight is treated as a **pluggable model** rather than as data.
Swap in a better estimator without touching the solver.
"""
from __future__ import annotations

import json
import os
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from ..core.mods import BaseItem, GenType, Mod, StatRange

BASE_URL = "https://repoe-fork.github.io/poe2"
FILES = ("mods.json", "base_items.json", "tags.json", "item_classes.json")


def ensure_cache(cache_dir: str, files: Iterable[str] = FILES) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    for f in files:
        dst = os.path.join(cache_dir, f)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        urllib.request.urlretrieve(f"{BASE_URL}/{f}", dst)


# --------------------------------------------------------------------------
# weight models
# --------------------------------------------------------------------------
class WeightModel(Protocol):
    """Maps a raw mod record -> relative spawn weight for an eligible mod."""
    def __call__(self, mod_id: str, raw: dict, family: str, tier: int, n_tiers: int) -> int: ...


def uniform_weight(mod_id, raw, family, tier, n_tiers) -> int:
    """What the dump literally supports: every eligible mod equally likely.

    Honest but wrong -- use only as a lower-information baseline.
    """
    return 1000


def poe1_prior_weight(mod_id, raw, family, tier, n_tiers) -> int:
    """Prior transplanted from PoE1's weight conventions.

    In PoE1 tiers within a family carry near-equal weights except the top one or
    two, which are cut roughly in half, and 'desirable' families (damage, crit,
    life) are rarer than filler (attributes, light radius, resistances).
    This is an ESTIMATE. Flag any conclusion that depends on it.
    """
    w = 1000
    if tier == 1:
        w = 400
    elif tier == 2:
        w = 700
    tags = set(raw.get("implicit_tags") or [])
    if tags & {"critical"}:
        w = int(w * 0.6)
    if tags & {"attribute"}:
        w = int(w * 1.4)
    if raw.get("is_essence_only"):
        w = 0
    return max(w, 1)


@dataclass
class EmpiricalWeightModel:
    """Maximum-likelihood weights fitted from observed roll counts.

    observations: family -> mod_id -> times seen, collected from real crafts or
    scraped trade listings.  Falls back to `prior` where evidence is thin
    (Dirichlet smoothing with strength `alpha`).
    """
    observations: dict[str, dict[str, int]]
    prior: WeightModel = poe1_prior_weight
    alpha: float = 50.0

    def __call__(self, mod_id, raw, family, tier, n_tiers) -> int:
        fam = self.observations.get(family, {})
        n = fam.get(mod_id, 0)
        total = sum(fam.values())
        p0 = self.prior(mod_id, raw, family, tier, n_tiers)
        if total == 0:
            return p0
        # posterior mean, rescaled to the prior's magnitude
        scale = sum(self.prior(m, raw, family, tier, n_tiers) for m in fam) / max(len(fam), 1)
        return max(1, int((n + self.alpha * p0 / max(scale, 1)) / (total + self.alpha) * 1000 * len(fam)))


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
@dataclass
class Repoe:
    mods_raw: dict
    bases_raw: dict

    @classmethod
    def load(cls, cache_dir: str = "cache") -> "Repoe":
        ensure_cache(cache_dir)
        with open(os.path.join(cache_dir, "mods.json")) as f:
            mods = json.load(f)
        with open(os.path.join(cache_dir, "base_items.json")) as f:
            bases = json.load(f)
        return cls(mods_raw=mods, bases_raw=bases)

    # ---- bases ---------------------------------------------------------
    def find_bases(self, item_class: str | None = None, name: str | None = None) -> list[BaseItem]:
        out = []
        for path, b in self.bases_raw.items():
            if item_class and b.get("item_class") != item_class:
                continue
            if name and name.lower() not in b.get("name", "").lower():
                continue
            out.append(BaseItem(id=path, name=b.get("name", path),
                                item_class=b.get("item_class", "?"),
                                tags=frozenset(b.get("tags", [])),
                                drop_level=b.get("drop_level", 1),
                                implicits=tuple(b.get("implicits", []))))
        return sorted(out, key=lambda b: -b.drop_level)

    def item_classes(self) -> list[str]:
        return sorted({b.get("item_class", "?") for b in self.bases_raw.values()})

    # ---- mods ----------------------------------------------------------
    @staticmethod
    def _family(raw: dict) -> str:
        g = raw.get("groups") or []
        return g[0] if g else raw.get("type", "?")

    def mods_for(self, base: BaseItem, weight_model: WeightModel = poe1_prior_weight,
                 domains: tuple[str, ...] = ("item",),
                 gen_types: tuple[str, ...] = ("prefix", "suffix"),
                 include_essence_only: bool = False) -> list[Mod]:
        """Build the eligible mod list for a base, with tiers derived per family."""
        tags = base.tags
        elig: list[tuple[str, dict]] = []
        for mid, raw in self.mods_raw.items():
            if raw.get("domain") not in domains:
                continue
            if raw.get("generation_type") not in gen_types:
                continue
            if raw.get("is_essence_only") and not include_essence_only:
                continue
            if self._eligible_weight(raw, tags) <= 0:
                continue
            elig.append((mid, raw))

        by_family: dict[tuple[str, str], list[tuple[str, dict]]] = defaultdict(list)
        for mid, raw in elig:
            by_family[(raw["generation_type"], self._family(raw))].append((mid, raw))

        out: list[Mod] = []
        for (gen, fam), rows in by_family.items():
            # tier 1 == highest required_level within the family
            rows.sort(key=lambda r: (-r[1].get("required_level", 1), r[0]))
            n = len(rows)
            for tier, (mid, raw) in enumerate(rows, start=1):
                w = weight_model(mid, raw, fam, tier, n)
                if w <= 0:
                    continue
                stats = tuple(StatRange(s["id"], s.get("min", 0), s.get("max", 0))
                              for s in raw.get("stats", []))
                out.append(Mod(
                    id=mid, group=fam,
                    gen_type=GenType.PREFIX if gen == "prefix" else GenType.SUFFIX,
                    required_level=raw.get("required_level", 1), tier=tier,
                    name=raw.get("text") or raw.get("name") or mid,
                    domain=raw.get("domain", "item"), stats=stats,
                    # collapse the eligibility mask into a single resolved weight:
                    # the model already accounts for base tags via `elig`
                    spawn_weights=(("default", w),),
                    tags=frozenset(raw.get("implicit_tags") or []),
                ))
        return out

    @staticmethod
    def _eligible_weight(raw: dict, tags: frozenset[str]) -> int:
        for sw in raw.get("spawn_weights", ()):
            if sw["tag"] == "default" or sw["tag"] in tags:
                return sw["weight"]
        return 0

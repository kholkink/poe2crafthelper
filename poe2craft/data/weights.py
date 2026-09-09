"""Community-sourced numeric spawn weights.

GGG's data tables expose only a binary "can this mod roll here" flag for PoE2.
The relative weights every serious tool uses trace back to one community
reverse-engineered table.  This module wraps it behind the WeightModel protocol
so the solver never depends on where the numbers came from.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field


def _norm(mod_id: str) -> str:
    return re.sub(r"_+$", "", mod_id or "").strip()


@dataclass
class CommunityWeights:
    """mod_id -> weight, plus tier metadata, scoped to an item class."""
    by_id: dict[str, int] = field(default_factory=dict)
    tier_name: dict[str, str] = field(default_factory=dict)
    tier_no: dict[str, int] = field(default_factory=dict)
    families: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    item_class: str = ""
    coverage: tuple[int, int] = (0, 0)

    @classmethod
    def load(cls, path: str, item_class: str) -> "CommunityWeights":
        with open(path) as f:
            rows = json.load(f)
        obj = cls(item_class=item_class)
        for r in rows:
            if r.get("itemClass") != item_class:
                continue
            for t in r.get("tiers", []):
                mid = _norm(t.get("id"))
                if not mid:
                    continue
                obj.by_id[mid] = int(t.get("weight", 0))
                obj.tier_name[mid] = t.get("tierName", "")
                obj.tier_no[mid] = int(t.get("tier", 0))
                obj.families[r.get("name", "?")].append(mid)
        return obj

    def model(self, fallback: int = 0, warn: list | None = None):
        """Return a WeightModel closure usable by Repoe.mods_for()."""
        seen_hit, seen_miss = [0], [0]

        def _m(mod_id, raw, family, tier, n_tiers) -> int:
            w = self.by_id.get(_norm(mod_id))
            if w is None:
                seen_miss[0] += 1
                if warn is not None:
                    warn.append(mod_id)
                return fallback
            seen_hit[0] += 1
            return w

        _m.stats = lambda: (seen_hit[0], seen_miss[0])
        return _m
